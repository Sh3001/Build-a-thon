"""The baseline ladder.

Each arm isolates one ingredient, so the tests check that each arm actually does the
thing it is named for -- an arm that quietly behaves like its neighbour makes the whole
comparison meaningless.
"""
from __future__ import annotations

import pytest

from backend.app.ml.scorer import get_scorer
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.strategies import (
    SMART_SCHEDULE,
    BaselineSuite,
    run_smart_retry,
)
from simulation.payment_gateway import PaymentGateway

pytestmark = pytest.mark.skipif(
    not (load_split.__module__ and __import__("backend.app.config", fromlist=["DATA_PROCESSED"])
         .DATA_PROCESSED.joinpath("test.csv").exists()),
    reason="no processed dataset -- run scripts/generate_dataset.py first")


@pytest.fixture(scope="module")
def txns():
    return to_transactions(load_split("test"))[:300]


def test_smart_retry_never_attempts_a_structurally_impossible_retry(txns):
    """The whole claim of the smart arm is that knowing this is worth money."""
    outcomes, _ = run_smart_retry(txns, PaymentGateway(seed=1))
    impossible = {"expired_card", "invalid_payment_method", "invalid_account",
                  "closed_account", "suspected_fraud", "high_risk_transaction",
                  "compliance_hold"}
    for o in outcomes:
        if o.failure_code.value in impossible:
            assert o.retries == 0, f"retried {o.failure_code.value}"


def test_smart_retry_takes_no_action_on_risk_cases(txns):
    outcomes, report = run_smart_retry(txns, PaymentGateway(seed=1))
    assert report["risk_actions_taken"] == 0


def test_smart_retry_uses_fewer_attempts_than_the_naive_arm(txns):
    """Diagnosis should cost less, not more."""
    suite = BaselineSuite(seed=1)
    arms = suite.run(txns, arms=["naive_retry", "smart_retry"])
    assert arms["smart_retry"].report["total_retries"] < \
        arms["naive_retry"].report["total_retries"]


def test_every_schedule_entry_is_within_the_horizon():
    for code, schedule in SMART_SCHEDULE.items():
        for offset in schedule:
            assert 0 <= offset <= 30 * 24, f"{code} schedules a retry past the horizon"


def test_the_control_arm_takes_no_action(txns):
    arms = BaselineSuite(seed=1).run(txns, arms=["control"])
    report = arms["control"].report
    assert report["total_retries"] == 0
    assert report["total_contacts"] == 0
    assert report["total_cost"] == 0.0


def test_targeting_arms_share_one_contact_budget(txns):
    """"Targeted better" must not secretly mean "contacted more".

    The invariant is on the *budget*, not on realised contacts. Two arms select the same
    number of cases; how many of those actually get contacted differs because a case that
    settles on the earlier retry never reaches its contact. Asserting equality of
    realised contacts would be asserting that the arms pick the same cases, which is the
    opposite of what these arms are for.
    """
    scorer = get_scorer()
    if not scorer.is_trained:
        pytest.skip("no trained model")
    budget = int(round(len(txns) * 0.2))
    suite = BaselineSuite(seed=1, scorer=scorer, budget_fraction=0.2)
    arms = suite.run(txns, arms=["ml_probability", "expected_value", "smart_retry"])
    for name in ("ml_probability", "expected_value"):
        contacts = arms[name].report["total_contacts"]
        assert contacts <= budget, f"{name} contacted {contacts} on a budget of {budget}"
    # And the arms genuinely differ: identical selections would make the ladder useless.
    assert arms["ml_probability"].report["revenue_recovered"] != \
        arms["expected_value"].report["revenue_recovered"]


def test_arms_run_on_independent_gateways(txns):
    """Sharing one gateway would let an earlier arm's contact fatigue leak into a later
    one, making arm order part of the result."""
    suite = BaselineSuite(seed=1)
    first = suite.run(txns, arms=["naive_retry"])["naive_retry"].report
    second = suite.run(txns, arms=["naive_retry"])["naive_retry"].report
    assert first["revenue_recovered"] == second["revenue_recovered"]


def test_every_arm_reports_the_same_population(txns):
    suite = BaselineSuite(seed=1, scorer=get_scorer())
    arms = suite.run(txns)
    counts = {name: r.report["cases"] for name, r in arms.items()}
    assert len(set(counts.values())) == 1, f"arms saw different populations: {counts}"
