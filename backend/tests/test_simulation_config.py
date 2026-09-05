"""The configurable simulator.

Two properties matter more than the rest: the defaults must reproduce the published
world exactly, and no scenario may break the invariant that brute force cannot beat
diagnosis.
"""
from __future__ import annotations

import pytest

from simulation.config import (
    STRUCTURAL_ZEROS,
    InvalidScenario,
    SimulationConfig,
    load_all,
)
from simulation.payment_gateway import (
    BASE_RETRY,
    CHANNEL_REACH,
    PASSIVE_CURE_RATE,
    PaymentGateway,
)


def test_the_defaults_reproduce_the_published_constants():
    """A configuration system that changes the baseline while introducing itself is
    impossible to trust."""
    cfg = SimulationConfig()
    assert cfg.base_retry == BASE_RETRY
    assert cfg.passive_cure_rate == PASSIVE_CURE_RATE
    assert cfg.channel_reach == CHANNEL_REACH


def test_the_default_gateway_behaves_identically_with_and_without_a_config():
    txn = {"transaction_id": "t1", "customer_id": "c1", "failure_code": "bank_timeout",
           "amount_usd": 100.0, "failure_count": 1, "previous_success_rate": 0.5}
    implicit = PaymentGateway(seed=42)
    explicit = PaymentGateway(seed=42, config=SimulationConfig())
    a = implicit.retry_payment(txn, 24.0, 1)
    b = explicit.retry_payment(txn, 24.0, 1)
    assert (a.success, a.probability) == (b.success, b.probability)


@pytest.mark.parametrize("code", sorted(STRUCTURAL_ZEROS))
def test_no_scenario_may_make_a_dead_instrument_retryable(code):
    """Allowing this would produce a world where brute force beats diagnosis, which
    invalidates every comparison the simulator is used for."""
    with pytest.raises(InvalidScenario, match="structural impossibility"):
        SimulationConfig.from_dict({"name": "bad", "base_retry": {code: 0.4}})


def test_every_shipped_scenario_is_valid():
    configs = load_all()
    assert len(configs) >= 5
    for name, cfg in configs.items():
        cfg.validate()
        assert cfg.description, f"scenario {name} has no description"


def test_a_scenario_overlays_rather_than_replaces():
    """Listing one key changes one thing."""
    default = SimulationConfig()
    pessimistic = SimulationConfig.load("pessimistic")
    assert pessimistic.passive_cure_rate["insufficient_funds"] != \
        default.passive_cure_rate["insufficient_funds"]
    assert pessimistic.passive_cure_rate["closed_account"] == \
        default.passive_cure_rate["closed_account"]


def test_an_unknown_scenario_name_is_refused():
    """A sweep that thinks it ran `pessimistic` and ran `default` reports a robustness
    result it never tested."""
    with pytest.raises(InvalidScenario, match="unknown scenario"):
        SimulationConfig.load("does_not_exist")


def test_a_typo_in_a_parameter_name_is_refused():
    with pytest.raises(InvalidScenario, match="unknown parameters"):
        SimulationConfig.from_dict({"name": "typo", "fatigue_rnage": [0.1, 0.2]})


def test_out_of_range_probabilities_are_refused():
    with pytest.raises(InvalidScenario):
        SimulationConfig.from_dict({"name": "bad", "channel_reach": {"email": 1.4}})
    with pytest.raises(InvalidScenario):
        SimulationConfig.from_dict({"name": "bad", "fatigue_range": [0.9, 0.2]})


def test_scenarios_actually_change_behaviour():
    """A configuration that is loaded and ignored is worse than none."""
    txn = {"transaction_id": "t1", "customer_id": "c1", "failure_code": "bank_timeout",
           "amount_usd": 100.0, "failure_count": 1, "previous_success_rate": 0.5}
    default = PaymentGateway(seed=7).retry_payment(txn, 120.0, 1).probability
    fast = PaymentGateway(seed=7, config=SimulationConfig.load("fast_decay")) \
        .retry_payment(txn, 120.0, 1).probability
    assert fast < default, "fast_decay must close the transient window sooner"


def test_every_scenario_actually_changes_an_outcome():
    """A configuration that is loaded and ignored is worse than one that does not exist.

    This test exists because two parameters -- `delivery_failure_rate` and
    `gateway_unavailable_rate` -- were declared in the scenario file, validated by the
    loader, and read by nothing. `unreliable_rails` therefore produced numbers identical
    to `default` while claiming to test unreliable rails, which is a silently wrong
    result rather than a missing feature.

    Rather than assert each parameter individually, this runs every scenario end to end
    and requires that each produces a distinguishable world.
    """
    from backend.app.tools.executor import ActionExecutor

    txn = {"transaction_id": "t1", "customer_id": "c1", "failure_code": "bank_timeout",
           "amount_usd": 100.0, "failure_count": 1, "previous_success_rate": 0.5}
    signatures: dict[str, tuple] = {}
    for name, cfg in load_all().items():
        gw = PaymentGateway(seed=11, config=cfg)
        ex = ActionExecutor(gateway=gw)
        # A signature broad enough to catch a change anywhere in the parameter set.
        outages = sum(1 for i in range(200)
                      if gw.retry_payment({**txn, "transaction_id": f"t{i}"},
                                          24.0, 1).unavailable)
        # Fatigue only bites on a *later* contact -- the multiplier is `fatigue ** n` and
        # n is zero on the first touch. Probing a fresh gateway would make `low_fatigue`
        # look identical to `default` and hide a wiring gap behind a weak probe.
        fatigued = PaymentGateway(seed=11, config=cfg)
        fatigued.customer_responds_to_reminder(txn, 24.0, "email", 0)
        fatigued.customer_responds_to_reminder(txn, 24.0, "email", 1)
        third = fatigued.customer_responds_to_reminder(txn, 24.0, "email", 2)

        signatures[name] = (
            round(PaymentGateway(seed=11, config=cfg)
                  .retry_payment(txn, 120.0, 1).probability, 6),
            round(third.probability, 6),
            round(cfg.passive_cure_rate["insufficient_funds"], 6),
            round(ex.notifier.failure_rate, 4),
            outages,
        )

    baseline = signatures["default"]
    identical = [n for n, sig in signatures.items() if n != "default" and sig == baseline]
    assert not identical, (
        f"these scenarios are indistinguishable from `default`, so they are testing "
        f"nothing: {identical}. Either a declared parameter is not wired to anything, or "
        f"the scenario does not change what it claims to.")


def test_an_outage_is_not_a_decline():
    """The payment's state is unknown, so the caller must re-present under the same
    idempotency key rather than treat the attempt as refused."""
    cfg = SimulationConfig.from_dict({"name": "always_down", "gateway_unavailable_rate": 1.0})
    gw = PaymentGateway(seed=3, config=cfg)
    res = gw.retry_payment(
        {"transaction_id": "t1", "customer_id": "c", "failure_code": "bank_timeout",
         "amount_usd": 100.0, "failure_count": 1, "previous_success_rate": 0.5}, 24.0, 1)
    assert res.unavailable and not res.success
    assert "state unknown" in res.detail


def test_an_explicit_notifier_is_not_silently_overridden():
    """The failure-injection tests pass `NotificationService(failure_rate=1.0)` and mean
    it; adopting the scenario over the top would make them untrustworthy."""
    from backend.app.tools.executor import ActionExecutor
    from simulation.notification_service import NotificationService

    ex = ActionExecutor(gateway=PaymentGateway(config=SimulationConfig.load("unreliable_rails")),
                        notifier=NotificationService(failure_rate=1.0))
    assert ex.notifier.failure_rate == 1.0


def test_the_fingerprint_ties_a_result_to_the_world_that_produced_it():
    a = SimulationConfig()
    b = SimulationConfig.load("pessimistic")
    assert a.fingerprint() != b.fingerprint()
    assert SimulationConfig().fingerprint() == a.fingerprint()
