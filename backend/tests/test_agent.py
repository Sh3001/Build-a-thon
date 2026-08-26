"""Phase 6 tests -- agent behaviour and the safety properties of the whole loop."""
from __future__ import annotations

import pytest

from backend.app.agents.diagnose import diagnose
from backend.app.agents.graph import RecoveryAgent
from backend.app.agents.runner import run_agent_batch, to_outcome
from backend.app.agents.strategy import PAYDAY_TARGET_HOURS, select_intervention
from backend.app.config import MAX_AGENT_STEPS, MAX_AUTO_RECOVERY_AMOUNT_USD, MAX_RETRIES, SEED
from backend.app.models.enums import CaseStatus, FailureCategory, InterventionType, category_of
from backend.app.models.schemas import AgentState, Transaction
from backend.app.services.dataio import load_split, to_transactions
from backend.app.tools.executor import ActionExecutor
from simulation.payment_gateway import PaymentGateway


def txn(tid="t1", code="bank_timeout", amount=100.0, **kw):
    base = dict(customer_id="c1", transaction_id=tid, amount=amount, currency="USD",
                payment_method="card", failure_code=code, avg_transaction_value=100.0,
                previous_success_rate=0.7, failure_count=1, days_since_failure=1.0,
                preferred_channel="sms", customer_tenure=400, overdue_days=3)
    return {**base, **kw}


def agent(seed=SEED):
    return RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=seed)))


@pytest.fixture(scope="module")
def sample():
    return to_transactions(load_split("test"))[:150]


# ------------------------------------------------------------------ diagnosis
def test_diagnosis_overrules_a_misleading_code():
    d = diagnose(Transaction(**txn(code="insufficient_funds", failure_count=6,
                                   previous_success_rate=0.05)))
    assert d.root_cause.value == "multiple_declines"
    assert d.confidence < 0.9 and "chronic" in d.rationale


def test_diagnosis_marks_risk_as_unrecoverable():
    d = diagnose(Transaction(**txn(code="suspected_fraud")))
    assert d.recoverable is False and d.category is FailureCategory.RISK_COMPLIANCE


# ------------------------------------------------------------------ strategy
def test_strategy_repairs_a_dead_card_before_retrying():
    s = AgentState(transaction_id="t", customer_id="c", amount=100.0,
                   failure_code="expired_card", expected_recovery=50.0,
                   transaction=Transaction(**txn(code="expired_card")))
    a = select_intervention(s, diagnose(s.transaction))
    assert a.action is InterventionType.REQUEST_PAYMENT_METHOD_UPDATE


def test_strategy_defers_insufficient_funds_into_the_payday_window():
    """The core timing decision: wait for the account to refill instead of burning
    attempts in the first 72 hours."""
    t = Transaction(**txn(code="insufficient_funds", days_since_failure=1.0))
    s = AgentState(transaction_id="t", customer_id="c", amount=100.0,
                   failure_code="insufficient_funds", expected_recovery=50.0,
                   transaction=t, elapsed_hours=24.0,
                   actions_taken=["send_reminder"], contact_count=1)
    a = select_intervention(s, diagnose(t))
    assert a.action is InterventionType.RETRY_PAYMENT
    assert a.delay_hours == pytest.approx(PAYDAY_TARGET_HOURS - 24.0)


def test_strategy_retries_a_transient_fault_immediately():
    t = Transaction(**txn(code="bank_timeout"))
    s = AgentState(transaction_id="t", customer_id="c", amount=100.0,
                   failure_code="bank_timeout", expected_recovery=50.0, transaction=t)
    a = select_intervention(s, diagnose(t))
    assert a.action is InterventionType.RETRY_PAYMENT and a.delay_hours == 0.0


def test_strategy_routes_around_an_unreachable_account():
    t = Transaction(**txn(code="closed_account"))
    s = AgentState(transaction_id="t", customer_id="c", amount=100.0,
                   failure_code="closed_account", expected_recovery=50.0, transaction=t)
    a = select_intervention(s, diagnose(t))
    assert a.action is InterventionType.SEND_PAYMENT_LINK


def test_strategy_escalates_above_the_amount_ceiling():
    amt = MAX_AUTO_RECOVERY_AMOUNT_USD + 1000
    t = Transaction(**txn(amount=amt))
    s = AgentState(transaction_id="t", customer_id="c", amount=amt,
                   failure_code="bank_timeout", expected_recovery=amt / 2, transaction=t)
    assert select_intervention(s, diagnose(t)).action is InterventionType.ESCALATE_CASE


def test_strategy_advances_when_an_action_is_blocked():
    """A refused action must not be proposed again forever."""
    t = Transaction(**txn(code="expired_card"))
    s = AgentState(transaction_id="t", customer_id="c", amount=100.0,
                   failure_code="expired_card", expected_recovery=50.0, transaction=t,
                   blocked_actions=["request_payment_method_update"])
    a = select_intervention(s, diagnose(t))
    assert a.action is not InterventionType.REQUEST_PAYMENT_METHOD_UPDATE


# ------------------------------------------------------------------ the loop
def test_every_case_reaches_a_terminal_state(sample):
    a = agent()
    for t in sample[:60]:
        s = a.run(t)
        assert s.is_terminal, f"{t['transaction_id']} did not terminate ({s.status})"
        assert s.stop_reason


def test_risk_cases_are_escalated_and_never_touched(sample):
    """The hardest safety guarantee: no automated action on fraud or compliance, ever."""
    a = agent()
    seen = 0
    for t in sample:
        if category_of(t["failure_code"]) is not FailureCategory.RISK_COMPLIANCE:
            continue
        seen += 1
        s = a.run(t)
        assert s.status is CaseStatus.ESCALATED
        assert s.attempt_count == 0, "a risk case was retried"
        assert set(s.actions_taken) <= {"escalate_case"}
    assert seen > 0, "no risk cases in the sample"


def test_retry_limit_is_never_exceeded(sample):
    a = agent()
    for t in sample:
        assert a.run(t).attempt_count <= MAX_RETRIES


def test_a_recovered_case_stops_immediately(sample):
    a = agent()
    for t in sample:
        s = a.run(t)
        if s.status is CaseStatus.RECOVERED:
            assert s.action_result is not None and s.action_result.amount_recovered > 0
            assert s.amount_recovered == s.amount_usd


def test_expired_cards_are_never_retried_before_repair(sample):
    a = agent()
    for t in sample:
        if t["failure_code"] != "expired_card":
            continue
        s = a.run(t)
        if s.attempt_count > 0:
            assert s.instrument_fixed, "retried a dead card that was never replaced"


def test_step_ceiling_bounds_the_loop(sample):
    a = agent()
    for t in sample[:60]:
        assert a.run(t).step_count <= MAX_AGENT_STEPS


def test_agent_is_deterministic(sample):
    a1, a2 = agent(), agent()
    for t in sample[:40]:
        s1, s2 = a1.run(t), a2.run(t)
        assert (s1.status, s1.amount_recovered, s1.actions_taken) == \
               (s2.status, s2.amount_recovered, s2.actions_taken)


def test_every_case_produces_an_audit_trail(sample):
    a = agent()
    for t in sample[:30]:
        s = a.run(t)
        assert len(s.audit_events) >= 5
        decisions = [e.agent_decision for e in s.audit_events]
        for stage in ("load_transaction", "score_recovery", "diagnose_root_cause",
                      "calculate_expected_recovery", "select_intervention",
                      "validate_policy", "monitor_outcome"):
            assert stage in decisions, f"{stage} missing from the audit trail"


def test_no_action_runs_without_policy_approval(sample):
    """Swept over real cases: every executed action was preceded by an approval."""
    a = agent()
    for t in sample[:80]:
        s = a.run(t)
        approvals = [e for e in s.audit_events
                     if e.agent_decision == "validate_policy" and e.policy_result in ("approve", "modify")]
        executions = [e for e in s.audit_events
                      if e.agent_decision == "execute_action" and e.action not in ("wait", "stop")]
        assert len(executions) <= len(approvals)


# ------------------------------------------------------------------ batch
def test_batch_produces_comparable_outcomes(sample):
    outcomes, rep, states = run_agent_batch(sample, PaymentGateway(seed=SEED))
    assert len(outcomes) == len(sample) == len(states)
    assert rep["cases"] == len(sample)
    assert rep["risk_actions_taken"] == 0, "the agent acted on a risk case"
    assert rep["revenue_recovered"] >= 0


def test_outcome_conversion_is_faithful(sample):
    a = agent()
    s = a.run(sample[0])
    o = to_outcome(s)
    assert o.transaction_id == s.transaction_id
    assert o.recovered == (s.status is CaseStatus.RECOVERED)
    assert o.retries == s.attempt_count
    assert o.actions == s.actions_taken
