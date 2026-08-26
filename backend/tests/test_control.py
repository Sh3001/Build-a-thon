"""Tests for the no-touch control arm and causal attribution.

The control arm is what turns "money recovered" into "money we caused". These tests
assert that the counterfactual is real, shared across arms, and correctly subtracted.
"""
from __future__ import annotations

import pytest

from backend.app.agents.runner import run_agent_batch
from backend.app.config import SEED
from backend.app.models.enums import FailureCategory, category_of
from backend.app.services.baseline import run_baseline
from backend.app.services.control import run_control
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.results import compare_to_control
from simulation.payment_gateway import PaymentGateway


@pytest.fixture(scope="module")
def txns():
    return to_transactions(load_split("test"))[:500]


@pytest.fixture(scope="module")
def arms(txns):
    _, ctrl = run_control(txns, PaymentGateway(seed=SEED))
    _, base = run_baseline(txns, PaymentGateway(seed=SEED))
    _, agent, _ = run_agent_batch(txns, PaymentGateway(seed=SEED))
    return ctrl, base, agent


# ------------------------------------------------------------------ the counterfactual
def test_control_takes_no_action_and_spends_nothing(arms):
    ctrl, *_ = arms
    assert ctrl["total_retries"] == 0
    assert ctrl["total_contacts"] == 0
    assert ctrl["total_actions"] == 0
    assert ctrl["total_cost"] == 0.0
    assert ctrl["risk_actions_taken"] == 0


def test_control_still_recovers_something(arms):
    """If the control recovered nothing, the counterfactual would be vacuous and every
    dollar the agent collects would look causal."""
    ctrl, *_ = arms
    assert ctrl["cases_recovered"] > 0
    assert ctrl["revenue_recovered"] > 0
    assert 0.05 < ctrl["recovery_rate"] < 0.35, "implausible spontaneous recovery rate"


def test_every_control_recovery_is_flagged_passive(arms):
    ctrl, *_ = arms
    assert ctrl["passive_recoveries"] == ctrl["cases_recovered"]
    assert ctrl["caused_recoveries"] == 0


def test_self_cure_is_identical_across_arms(txns):
    """The arms must be a counterfactual over ONE population, not three populations."""
    a = PaymentGateway(seed=SEED)
    b = PaymentGateway(seed=SEED)
    for t in txns[:120]:
        assert a.self_cure_hour(t) == b.self_cure_hour(t)


def test_self_cure_is_independent_of_our_actions(txns):
    """Working a case must not change whether it would have self-cured."""
    gw = PaymentGateway(seed=SEED)
    t = txns[0]
    before = gw.self_cure_hour(t)
    gw.retry_payment(t, 24, 1)
    gw.customer_responds_to_reminder(t, 24, "sms", 0)
    assert gw.self_cure_hour(t) == before


def test_temporary_failures_self_cure_more_than_persistent_ones(txns):
    gw = PaymentGateway(seed=SEED)
    rates = {}
    for cat in (FailureCategory.TEMPORARY, FailureCategory.PERSISTENT):
        cs = [t for t in to_transactions(load_split("test"))
              if category_of(t["failure_code"]) is cat]
        cured = sum(1 for t in cs if gw.self_cure_hour(t) is not None)
        rates[cat] = cured / len(cs)
    assert rates[FailureCategory.TEMPORARY] > rates[FailureCategory.PERSISTENT] + 0.10


# ------------------------------------------------------------------ attribution
def test_agent_beats_the_no_touch_control(arms):
    """The claim that actually matters: the agent causes recovery, not just observes it."""
    ctrl, _, agent = arms
    c = compare_to_control(ctrl, agent)
    assert c["incremental_recovered_revenue"] > 0
    assert c["incremental_cases_recovered"] > 0


def test_not_all_recovered_revenue_is_causal(arms):
    """Guards against the headline silently reverting to gross revenue."""
    ctrl, _, agent = arms
    c = compare_to_control(ctrl, agent)
    assert 0.0 < c["share_of_revenue_that_is_causal"] < 1.0
    assert c["incremental_recovered_revenue"] < agent["revenue_recovered"], (
        "the agent is claiming credit for money the control also collected")


def test_agent_is_more_causal_than_the_baseline(arms):
    ctrl, base, agent = arms
    ca = compare_to_control(ctrl, agent)
    cb = compare_to_control(ctrl, base)
    assert ca["incremental_recovered_revenue"] > cb["incremental_recovered_revenue"]
    assert ca["share_of_revenue_that_is_causal"] > cb["share_of_revenue_that_is_causal"]


def test_an_arm_that_does_nothing_shows_zero_impact(arms):
    """Sanity check on the arithmetic: control vs itself must be exactly zero."""
    ctrl, *_ = arms
    c = compare_to_control(ctrl, ctrl)
    assert c["incremental_recovered_revenue"] == 0.0
    assert c["incremental_cases_recovered"] == 0


def test_arms_recover_at_least_what_the_control_does(arms):
    """An active arm should never do worse than leaving the case alone."""
    ctrl, base, agent = arms
    assert agent["revenue_recovered"] >= ctrl["revenue_recovered"]
    assert base["revenue_recovered"] >= ctrl["revenue_recovered"]


def test_passive_and_caused_recoveries_sum_to_the_total(arms):
    for rep in arms:
        assert rep["passive_recoveries"] + rep["caused_recoveries"] == rep["cases_recovered"]


def test_risk_cases_recover_only_passively_under_the_agent(arms):
    """The agent never touches a risk case, so any recovery there must be spontaneous."""
    _, _, agent = arms
    risk = agent["by_category"].get("RISK_COMPLIANCE")
    if risk and risk["cases_recovered"]:
        assert agent["risk_actions_taken"] == 0
