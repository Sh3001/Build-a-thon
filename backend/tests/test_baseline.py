"""Phase 2 tests -- baseline behaviour, simulator invariants, and metric arithmetic."""
from __future__ import annotations

import pytest

from backend.app.config import ACTION_COST_USD
from backend.app.models.enums import FailureCategory, FailureCode
from backend.app.services.baseline import BaselineConfig, run_baseline, run_case
from backend.app.services.results import CaseOutcome, compare, summarize
from simulation.payment_gateway import PaymentGateway


def txn(tid="t1", code="bank_timeout", amount=100.0, psr=0.7, fc=1):
    return {"transaction_id": tid, "customer_id": "c1", "failure_code": code,
            "amount_usd": amount, "amount": amount, "currency": "USD",
            "previous_success_rate": psr, "failure_count": fc}


# ----------------------------------------------------------------- simulator
def test_hard_failures_can_never_be_retried_into_success():
    """The premise of the whole system: brute force cannot beat diagnosis."""
    gw = PaymentGateway(seed=3)
    for code in ("invalid_account", "closed_account", "suspected_fraud",
                 "compliance_hold", "high_risk_transaction"):
        for i in range(60):
            r = gw.retry_payment(txn(f"{code}_{i}", code), hours_since_failure=24 * (i % 5 + 1),
                                 attempt=(i % 3) + 1)
            assert not r.success and r.probability == 0.0, f"{code} recovered by bare retry"


def test_expired_card_needs_the_instrument_replaced():
    gw = PaymentGateway(seed=3)
    t = txn("ec1", "expired_card")
    assert gw.retry_payment(t, 24, 1).probability == 0.0
    gw._instrument_fixed["ec1"] = True                      # simulate a successful update
    assert gw.retry_payment(t, 24, 1).probability > 0.0


def test_temporary_failures_decay_with_time():
    gw = PaymentGateway(seed=3)
    early = gw.retry_payment(txn("d1", "bank_timeout"), 1, 1).probability
    late = gw.retry_payment(txn("d1", "bank_timeout"), 24 * 20, 1).probability
    assert early > late * 3, f"decay too weak: {early:.3f} vs {late:.3f}"


def test_insufficient_funds_improves_over_the_first_week():
    """The payday interaction -- waiting is the correct move here, unlike a timeout."""
    gw = PaymentGateway(seed=3)
    day1 = gw.retry_payment(txn("f1", "insufficient_funds"), 24, 1).probability
    day6 = gw.retry_payment(txn("f1", "insufficient_funds"), 24 * 6, 1).probability
    assert day6 > day1, f"payday effect missing: {day1:.3f} -> {day6:.3f}"


def test_simulator_is_deterministic():
    a = PaymentGateway(seed=11).retry_payment(txn("x"), 24, 1)
    b = PaymentGateway(seed=11).retry_payment(txn("x"), 24, 1)
    assert (a.success, a.probability) == (b.success, b.probability)


def test_contact_fatigue_reduces_response():
    gw = PaymentGateway(seed=5)
    t = txn("fat1", "insufficient_funds")
    first = gw.customer_responds_to_reminder(t, 24, "sms", 0).probability
    for n in range(1, 4):
        gw.customer_responds_to_reminder(t, 24, "sms", n)
    later = gw.customer_responds_to_reminder(t, 24, "sms", 9).probability
    assert later < first, f"fatigue not applied: {first:.3f} -> {later:.3f}"


# ----------------------------------------------------------------- baseline
def test_baseline_never_exceeds_max_retries():
    gw = PaymentGateway(seed=9)
    cfg = BaselineConfig(max_retries=3)
    for i in range(200):
        out = run_case(txn(f"m{i}", "closed_account"), gw, cfg)   # never succeeds
        assert out.retries <= 3


def test_baseline_stops_immediately_on_success():
    gw = PaymentGateway(seed=2)
    outs = [run_case(txn(f"s{i}", "network_error"), gw, BaselineConfig()) for i in range(150)]
    won = [o for o in outs if o.recovered]
    assert won, "expected some recoveries on network_error"
    for o in won:
        assert o.retries == o.actions.count("retry_payment")
        assert o.recovery_hours is not None
        # A win ends the case: no attempt may follow the successful one.
        assert o.retries <= 3


def test_baseline_respects_the_24h_interval():
    gw = PaymentGateway(seed=2)
    outs = [run_case(txn(f"i{i}", "network_error"), gw, BaselineConfig()) for i in range(200)]
    for o in outs:
        if o.recovery_hours is not None:
            assert o.recovery_hours in (24.0, 48.0, 72.0)


def test_baseline_is_deterministic():
    t = [txn(f"d{i}", "insufficient_funds") for i in range(120)]
    _, a = run_baseline(t, PaymentGateway(seed=21))
    _, b = run_baseline(t, PaymentGateway(seed=21))
    assert a["revenue_recovered"] == b["revenue_recovered"]
    assert a["cases_recovered"] == b["cases_recovered"]


def test_baseline_takes_unsafe_actions_on_risk_cases():
    """Documents the behaviour RecoverAI must eliminate: the naive strategy retries
    fraud and compliance holds because nothing stops it."""
    t = [txn(f"r{i}", "suspected_fraud") for i in range(50)]
    _, rep = run_baseline(t, PaymentGateway(seed=4))
    assert rep["risk_actions_taken"] > 0
    assert rep["cases_recovered"] == 0


def test_baseline_costs_are_accounted():
    t = [txn(f"c{i}", "closed_account") for i in range(10)]
    outs, rep = run_baseline(t, PaymentGateway(seed=4))
    assert rep["total_cost"] == pytest.approx(10 * 3 * ACTION_COST_USD["retry_payment"], rel=1e-6)


# ----------------------------------------------------------------- metrics
def _oc(**kw):
    base = dict(transaction_id="t", failure_code=FailureCode.BANK_TIMEOUT,
                failure_category=FailureCategory.TEMPORARY, amount_usd=100.0)
    return CaseOutcome(**{**base, **kw})


def test_summarize_arithmetic():
    outs = [
        _oc(transaction_id="a", recovered=True, amount_recovered=100.0, recovery_hours=24, retries=1, cost=0.02),
        _oc(transaction_id="b", recovered=False, retries=3, cost=0.06),
    ]
    rep = summarize(outs, "x")
    assert rep["revenue_at_risk"] == 200.0
    assert rep["revenue_recovered"] == 100.0
    assert rep["recovery_rate"] == 0.5
    assert rep["value_recovery_rate"] == 0.5
    assert rep["total_retries"] == 4
    assert rep["net_recovered"] == pytest.approx(100.0 - 0.08)


def test_summarize_handles_empty():
    assert summarize([], "x")["cases"] == 0


def test_compare_computes_incremental_revenue():
    base = summarize([_oc(transaction_id="a", recovered=True, amount_recovered=100.0, cost=0.02)], "baseline")
    agent = summarize([_oc(transaction_id="a", recovered=True, amount_recovered=150.0, cost=0.10)], "agent")
    c = compare(base, agent)
    assert c["incremental_recovered_revenue"] == 50.0
    assert c["recovery_uplift_pct"] == 50.0
    assert c["incremental_roi"] == pytest.approx(50.0 / 0.08, rel=1e-3)
