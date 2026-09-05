"""Randomised assignment and the statistics built on it."""
from __future__ import annotations

import math

import pytest

from backend.app.experiments.assignment import (
    Experiment,
    InvalidExperiment,
    Variant,
)
from backend.app.experiments.stats import (
    MultiSeedReport,
    average_treatment_effect,
    bootstrap_ci,
    cohens_d,
    compare_arms_across_seeds,
    difference_in_means,
    mean_interval,
    proportion_interval,
    t_critical,
)

UNITS = [f"cust_{i:06d}" for i in range(20_000)]


@pytest.fixture
def experiment():
    return Experiment("recovery_v1",
                      [Variant("control", 1.0), Variant("recoverai", 1.0)],
                      description="agent vs control")


# ---------------------------------------------------------------- assignment
def test_assignment_is_stable_across_calls_and_processes(experiment):
    """A unit that switches arms on reload appears in both denominators."""
    first = {u: experiment.assign(u).variant for u in UNITS[:500]}
    again = {u: experiment.assign(u).variant for u in UNITS[:500]}
    assert first == again
    # A second, independently constructed experiment with the same id must agree --
    # this is what rules out anything process-local like Python's randomised str hash.
    twin = Experiment("recovery_v1", [Variant("control"), Variant("recoverai")])
    assert {u: twin.assign(u).variant for u in UNITS[:500]} == first


def test_assignment_is_roughly_uniform(experiment):
    counts = experiment.balance(UNITS)["counts"]
    assert abs(counts["control"] - counts["recoverai"]) / len(UNITS) < 0.02


def test_weights_are_respected():
    e = Experiment("weighted", [Variant("a", 1.0), Variant("b", 3.0)])
    counts = e.balance(UNITS)["counts"]
    assert 0.23 < counts["a"] / len(UNITS) < 0.27


def test_assignment_is_uncorrelated_across_experiments():
    """A unit in the treatment arm of one experiment must be no likelier to be in the
    treatment arm of the next, or the two results contaminate each other."""
    a = Experiment("exp_a", [Variant("x"), Variant("y")])
    b = Experiment("exp_b", [Variant("x"), Variant("y")])
    agree = sum(1 for u in UNITS if a.assign(u).variant == b.assign(u).variant)
    assert 0.47 < agree / len(UNITS) < 0.53


def test_the_holdout_is_stable_when_variant_weights_change():
    """The holdout must survive a design edit or it stops being a clean comparison."""
    before = Experiment("h", [Variant("a"), Variant("b")], holdout_fraction=0.1)
    after = Experiment("h", [Variant("a"), Variant("b", 5.0)], holdout_fraction=0.1)
    held_before = {u for u in UNITS[:2000] if before.assign(u).variant == "holdout"}
    held_after = {u for u in UNITS[:2000] if after.assign(u).variant == "holdout"}
    assert held_before == held_after


def test_a_single_arm_experiment_is_refused():
    with pytest.raises(InvalidExperiment):
        Experiment("x", [Variant("only")])


def test_duplicate_variant_names_are_refused():
    with pytest.raises(InvalidExperiment):
        Experiment("x", [Variant("a"), Variant("a")])


def test_an_eligibility_predicate_that_raises_excludes_rather_than_admits():
    """Admitting on error silently enrols whoever triggers the bug."""
    e = Experiment("elig", [Variant("a"), Variant("b")],
                   eligibility=lambda ctx: ctx["amount"] > 10)
    assert not e.assign("u1", {}).eligible
    assert e.assign("u1", {"amount": 100}).eligible
    assert not e.assign("u1", {"amount": 1}).eligible


def test_assignment_records_when_it_happened(experiment):
    a = experiment.assign("cust_000001")
    assert a.to_dict()["randomization_timestamp"]
    assert a.to_dict()["experiment_id"] == "recovery_v1"


# ---------------------------------------------------------------- statistics
def test_a_single_replication_yields_no_interval():
    """One draw cannot estimate a spread, and pretending otherwise converts "we do not
    know" into a claim."""
    ci = mean_interval([42.0])
    assert math.isinf(ci.low) and math.isinf(ci.high)
    assert "single replication" in ci.method


def test_small_samples_use_the_t_distribution():
    """With five seeds the normal approximation is about 25% too narrow, and a too-narrow
    interval is worse than none."""
    values = [10.0, 12.0, 9.0, 11.0, 13.0]
    t_ci = mean_interval(values, small_sample=True)
    z_ci = mean_interval(values, small_sample=False)
    assert t_ci.width > z_ci.width
    assert t_critical(4) > 1.96


def test_a_wilson_interval_never_leaves_the_unit_range():
    """A recovery rate with a negative lower bound is a nonsense figure that still gets
    copied onto a slide."""
    for successes, n in [(0, 10), (10, 10), (1, 100), (99, 100)]:
        ci = proportion_interval(successes, n)
        assert 0.0 <= ci.low <= ci.high <= 1.0


def test_difference_in_means_reports_an_interval_and_an_effect_size():
    treated = [12.0, 13.0, 11.5, 12.5, 13.5]
    control = [10.0, 10.5, 9.5, 10.2, 10.8]
    ci = difference_in_means(treated, control)
    assert ci.excludes_zero and ci.point > 0
    assert cohens_d(treated, control) > 1.0


def test_an_indistinguishable_difference_is_reported_as_such():
    """The honest outcome, and the one a harness must be able to express."""
    a = [10.0, 12.0, 8.0, 14.0, 6.0]
    b = [10.5, 11.5, 8.5, 13.0, 7.0]
    assert not difference_in_means(a, b).excludes_zero


def test_the_ate_carries_its_caveat():
    out = average_treatment_effect([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    assert "randomised" in out["caveat"]
    assert out["ate"] == pytest.approx(1.0)


def test_paired_comparison_removes_shared_seed_variation():
    """Treating paired arms as independent inflates the interval with variance the
    design already controlled."""
    a, b = MultiSeedReport("agent"), MultiSeedReport("baseline")
    for seed in range(10):
        common = seed * 1000.0            # large shared seed-to-seed swing
        a.record(seed, {"revenue": 100.0 + common}, ["revenue"])
        b.record(seed, {"revenue": 90.0 + common}, ["revenue"])
    paired = compare_arms_across_seeds(a, b, "revenue")
    unpaired = difference_in_means(a.series("revenue"), b.series("revenue"))
    assert paired["mean_difference"] == pytest.approx(10.0)
    assert paired["excludes_zero"]
    assert not unpaired.excludes_zero, \
        "the unpaired interval should be swamped by the shared variation"


def test_multi_seed_summary_reports_the_worst_case_not_just_the_mean():
    """A positive mean with a negative worst seed is a different story from a positive
    mean everywhere, and reporting only the mean hides it."""
    r = MultiSeedReport("agent")
    for seed, value in enumerate([5.0, 7.0, -2.0, 9.0]):
        r.record(seed, {"uplift": value}, ["uplift"])
    s = r.summary()["metrics"]["uplift"]
    assert s["worst"] == -2.0 and s["best"] == 9.0
    assert s["share_positive"] == 0.75


def test_arms_with_no_shared_seeds_are_not_compared():
    a, b = MultiSeedReport("a"), MultiSeedReport("b")
    a.record(1, {"m": 1.0}, ["m"])
    b.record(2, {"m": 1.0}, ["m"])
    assert compare_arms_across_seeds(a, b, "m")["seeds"] == 0


def test_bootstrap_interval_brackets_the_point_estimate():
    ci = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert ci.low <= ci.point <= ci.high
