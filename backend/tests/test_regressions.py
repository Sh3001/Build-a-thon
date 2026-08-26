"""Regression tests -- one per bug found after the first end-to-end run.

Each of these failed before its fix. They exist so the same defect cannot return quietly.
"""
from __future__ import annotations

import pytest

from backend.app.agents.diagnose import diagnose
from backend.app.agents.graph import RecoveryAgent
from backend.app.agents.strategy import select_intervention
from backend.app.config import MAX_AUTO_RECOVERY_AMOUNT_USD, SEED
from backend.app.models.enums import FailureCategory, InterventionType, PolicyDecision
from backend.app.models.schemas import AgentState, ProposedAction, Transaction
from backend.app.policies.engine import PolicyContext, validate
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.results import CaseOutcome, bootstrap_incremental
from backend.app.tools.executor import ActionExecutor
from simulation.payment_gateway import PaymentGateway


def txn(code="closed_account", amount=500.0, **kw):
    base = dict(customer_id="c1", transaction_id="t1", amount=amount, currency="USD",
                payment_method="ach", failure_code=code, avg_transaction_value=400.0,
                previous_success_rate=0.6, failure_count=1, days_since_failure=1.0,
                preferred_channel="email")
    return Transaction(**{**base, **kw})


# ---------------------------------------------------------------- BUG: dead route-around
def test_unreachable_account_is_still_recoverable_by_other_means():
    """`recoverable` conflated "cannot be debited" with "cannot ever be collected", so
    closed accounts were written off before the payment-link route was considered."""
    for code in ("closed_account", "invalid_account"):
        d = diagnose(txn(code))
        assert d.recoverable is True, f"{code} was written off entirely"
        assert d.retry_viable is False, f"{code} must not be retryable"


def test_risk_cases_remain_unrecoverable():
    d = diagnose(txn("suspected_fraud"))
    assert d.recoverable is False and d.retry_viable is False


def test_dead_instruments_are_not_retry_viable():
    for code in ("expired_card", "invalid_payment_method"):
        d = diagnose(txn(code))
        assert d.retry_viable is False
        assert d.recoverable is True, "a dead card can still be paid via a link"


def test_closed_account_gets_a_payment_link_not_an_immediate_stop():
    """The regression itself: every closed account previously took zero actions."""
    t = txn("closed_account", amount=500.0)
    s = AgentState(transaction_id="t1", customer_id="c1", amount=500.0,
                   failure_code="closed_account", transaction=t, expected_recovery=20.0)
    a = select_intervention(s, diagnose(t))
    assert a.action is InterventionType.SEND_PAYMENT_LINK


def test_unreachable_accounts_take_action_at_scale():
    txns = [t for t in to_transactions(load_split("test"))
            if t["failure_code"] in ("closed_account", "invalid_account")][:60]
    assert txns
    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=SEED)))
    states = [agent.run(t) for t in txns]
    acted = [s for s in states if s.actions_taken]
    assert acted, "no unreachable-account case took any action"

    # The precise claim: a case is only left alone when it is below the value floor.
    # Anything skipped for another reason is the old bug returning.
    for s in states:
        if not s.actions_taken:
            assert "does not justify" in s.stop_reason, (
                f"{s.transaction_id} was abandoned for the wrong reason: {s.stop_reason}")
        else:
            assert s.actions_taken == ["send_payment_link"], (
                f"unexpected action on an unreachable account: {s.actions_taken}")


def test_low_value_cases_are_still_declined():
    """The fix must not turn the value floor off entirely."""
    t = txn("closed_account", amount=0.5)
    s = AgentState(transaction_id="t1", customer_id="c1", amount=0.5,
                   failure_code="closed_account", transaction=t, expected_recovery=0.02)
    assert select_intervention(s, diagnose(t)).action is InterventionType.STOP


# ---------------------------------------------------------------- BUG: gate soundness
def test_a_rewritten_action_is_re_validated():
    """A MODIFY rule could synthesise an action the REJECT rules never saw, letting the
    same case be escalated twice."""
    s = AgentState(transaction_id="t1", customer_id="c",
                   amount=MAX_AUTO_RECOVERY_AMOUNT_USD + 900, failure_code="bank_timeout",
                   expected_recovery=100.0, actions_taken=["escalate_case"])
    ctx = PolicyContext(hours_since_last_attempt=999,
                        already_executed=frozenset({"escalate_case"}))
    r = validate(s, ProposedAction(action=InterventionType.STOP), ctx)
    assert r.decision is PolicyDecision.REJECT
    assert r.effective_action is None
    assert "R-IDEMPOTENT" in r.rules_fired


def test_a_legitimate_rewrite_still_succeeds():
    s = AgentState(transaction_id="t2", customer_id="c",
                   amount=MAX_AUTO_RECOVERY_AMOUNT_USD + 900, failure_code="bank_timeout",
                   expected_recovery=100.0)
    r = validate(s, ProposedAction(action=InterventionType.STOP),
                 PolicyContext(hours_since_last_attempt=999))
    assert r.decision is PolicyDecision.MODIFY
    assert r.effective_action.action is InterventionType.ESCALATE_CASE


# ---------------------------------------------------------------- BUG: audit fidelity
def test_a_genuine_zero_is_recorded_as_zero_not_null():
    """`round(x, 6) or None` turned 0.0 into NULL -- and 0.0 is exactly what a risk case
    carries, so the audit log silently lost real values."""
    txns = {t["transaction_id"]: t for t in to_transactions(load_split("test"))}
    risk = next(t for t in txns.values() if t["failure_code"] == "suspected_fraud")
    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=SEED)))
    s = agent.run(risk)
    evs = [e for e in s.audit_events if e.agent_decision == "calculate_expected_recovery"]
    assert evs
    assert evs[0].expected_recovery == 0.0, "a real 0.0 was persisted as None"
    assert evs[0].expected_recovery is not None


# ---------------------------------------------------------------- BUG: model selection
def test_the_shipped_model_is_the_one_that_won_validation():
    """Asserting an algorithm the numbers do not support is a reporting bug."""
    from backend.app.ml.scorer import get_scorer
    s = get_scorer()
    if not s.is_trained:
        pytest.skip("model not trained")
    meta = s.metadata
    cands = meta.get("candidates")
    if not cands:
        pytest.skip("model predates selection metadata")
    best = max(cands, key=cands.get)
    assert meta["algorithm"] == best, (
        f"shipped {meta['algorithm']} but {best} scored higher on validation")
    assert best in s.model_version


# ---------------------------------------------------------------- bootstrap
def _oc(tid, rec, amt, strategy):
    from backend.app.models.enums import FailureCode
    return CaseOutcome(transaction_id=tid, failure_code=FailureCode.BANK_TIMEOUT,
                       failure_category=FailureCategory.TEMPORARY, amount_usd=100.0,
                       strategy=strategy, recovered=rec, amount_recovered=amt)


def test_bootstrap_interval_brackets_the_point_estimate():
    base = [_oc(f"t{i}", i % 4 == 0, 100.0 if i % 4 == 0 else 0.0, "baseline") for i in range(200)]
    agent = [_oc(f"t{i}", i % 2 == 0, 100.0 if i % 2 == 0 else 0.0, "recoverai") for i in range(200)]
    bs = bootstrap_incremental(base, agent, n_boot=400)
    point = sum(c.amount_recovered for c in agent) - sum(c.amount_recovered for c in base)
    assert bs["incremental_revenue"]["p05"] <= point <= bs["incremental_revenue"]["p95"]
    assert bs["excludes_zero"] is True
    assert bs["share_positive"] > 0.95


def test_bootstrap_detects_no_real_difference():
    """If the arms are identical the interval must contain zero."""
    base = [_oc(f"t{i}", i % 3 == 0, 100.0 if i % 3 == 0 else 0.0, "baseline") for i in range(200)]
    agent = [_oc(f"t{i}", i % 3 == 0, 100.0 if i % 3 == 0 else 0.0, "recoverai") for i in range(200)]
    bs = bootstrap_incremental(base, agent, n_boot=400)
    assert bs["excludes_zero"] is False
    assert bs["incremental_revenue"]["p50"] == 0.0


def test_bootstrap_is_paired_not_independent():
    """Pairing is what removes between-case variance; losing it would widen the interval
    enough to hide a real effect."""
    base = [_oc(f"t{i}", False, 0.0, "baseline") for i in range(300)]
    agent = [_oc(f"t{i}", True, 100.0, "recoverai") for i in range(300)]
    bs = bootstrap_incremental(base, agent, n_boot=400)
    assert bs["paired"] is True
    # Every case improves by exactly 100, so a paired resample has zero variance.
    assert bs["incremental_revenue"]["p05"] == bs["incremental_revenue"]["p95"] == 30000.0
