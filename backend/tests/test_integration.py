"""Phase 10 -- end-to-end integration.

These exercise the full pipeline the way the demo does, and assert the properties the
submission actually claims. If a claim in the README stops being true, one of these fails.
"""
from __future__ import annotations

import pytest

from backend.app.agents.runner import run_agent_batch
from backend.app.audit.store import AuditStore
from backend.app.config import MAX_RETRIES, SEED
from backend.app.database.store import CaseStore
from backend.app.ml.scorer import get_scorer
from backend.app.models.enums import FailureCategory
from backend.app.services.baseline import run_baseline
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.results import compare
from simulation.payment_gateway import PaymentGateway


@pytest.fixture(scope="module")
def txns():
    return to_transactions(load_split("test"))[:600]


@pytest.fixture(scope="module")
def both(txns):
    _, base = run_baseline(txns, PaymentGateway(seed=SEED))
    outs, agent, states = run_agent_batch(txns, PaymentGateway(seed=SEED), scorer=get_scorer())
    return base, agent, outs, states


# ------------------------------------------------------------------ the headline claim
def test_agent_recovers_more_revenue_than_the_baseline(both):
    base, agent, *_ = both
    c = compare(base, agent)
    assert c["incremental_recovered_revenue"] > 0, (
        f"agent recovered {agent['revenue_recovered']} vs baseline {base['revenue_recovered']}")


def test_agent_wins_using_fewer_retries(both):
    """More revenue AND less load on the payment rails."""
    base, agent, *_ = both
    assert agent["total_retries"] < base["total_retries"]


def test_agent_takes_no_unsafe_actions_while_the_baseline_does(both):
    base, agent, *_ = both
    assert agent["risk_actions_taken"] == 0
    assert base["risk_actions_taken"] > 0


def test_both_strategies_face_an_identical_world(txns, both):
    """The comparison is only meaningful if the case populations are the same."""
    base, agent, *_ = both
    assert base["cases"] == agent["cases"] == len(txns)
    assert base["revenue_at_risk"] == pytest.approx(agent["revenue_at_risk"], rel=1e-6)


# ------------------------------------------------------------------ invariants at scale
def test_no_case_exceeds_the_retry_ceiling(both):
    *_, outs, _ = both
    assert max(o.retries for o in outs) <= MAX_RETRIES


def test_no_recovered_case_kept_working_afterwards(both):
    """Automatic stop after success, checked across the whole batch."""
    *_, outs, _ = both
    for o in outs:
        if o.recovered:
            assert o.amount_recovered == pytest.approx(o.amount_usd)
            # A recovery with no action is only legitimate if the case self-cured, and
            # it must be labelled as such -- otherwise the agent is silently taking
            # credit for money that arrived on its own.
            if not o.actions:
                assert o.passive_recovery, (
                    f"{o.transaction_id} recovered with no action and no passive flag")


def test_every_case_terminates_with_a_reason(both):
    *_, outs, _ = both
    assert all(o.status in ("recovered", "escalated", "stopped", "exhausted") for o in outs)
    assert all(o.stop_reason for o in outs)


def test_no_expired_card_was_retried_without_repair(both):
    *_, _, states = both
    for s in states:
        if s.failure_code.value in ("expired_card", "invalid_payment_method") and s.attempt_count:
            assert s.instrument_fixed


def test_risk_cases_are_untouched_at_scale(both):
    *_, outs, _ = both
    risk = [o for o in outs if o.failure_category is FailureCategory.RISK_COMPLIANCE]
    assert risk
    for o in risk:
        assert o.retries == 0 and o.contacts == 0
        assert set(o.actions) <= {"escalate_case"}


def test_recovered_revenue_never_exceeds_revenue_at_risk(both):
    base, agent, *_ = both
    for rep in (base, agent):
        assert rep["revenue_recovered"] <= rep["revenue_at_risk"]


# ------------------------------------------------------------------ persistence
def test_audit_and_case_stores_survive_a_full_run(tmp_path, txns):
    db = tmp_path / "run.db"
    audit = AuditStore(db)
    outs, report, states = run_agent_batch(txns[:120], PaymentGateway(seed=SEED),
                                           scorer=get_scorer(), on_audit=audit.append)
    audit.commit()
    ok, bad = audit.verify()
    assert ok, f"audit chain broke at row {bad} during a real run"
    assert audit.count() >= len(outs) * 5

    cases = CaseStore(db, conn=audit.conn)
    cases.save_cases(outs, "recoverai")
    cases.save_run("recoverai", report)
    assert cases.count("recoverai") == len(outs)
    assert cases.get_run("recoverai")["revenue_recovered"] == report["revenue_recovered"]

    # Every case in the queue has a complete, ordered trace.
    top = cases.queue("recoverai", limit=5)
    for row in top:
        tl = audit.timeline(row["transaction_id"])
        assert tl, f"no audit trail for {row['transaction_id']}"
        assert tl[0]["agent_decision"] == "load_transaction"
        assert [r["seq"] for r in tl] == sorted(r["seq"] for r in tl)
    audit.close()


def test_a_full_run_is_reproducible(txns):
    """Same seed, same result -- the evaluation can be re-derived by a reviewer."""
    a = run_agent_batch(txns[:120], PaymentGateway(seed=SEED), scorer=get_scorer())[1]
    b = run_agent_batch(txns[:120], PaymentGateway(seed=SEED), scorer=get_scorer())[1]
    assert a["revenue_recovered"] == b["revenue_recovered"]
    assert a["cases_recovered"] == b["cases_recovered"]
    assert a["total_retries"] == b["total_retries"]
