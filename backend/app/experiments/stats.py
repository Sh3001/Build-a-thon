"""Statistics for arm comparisons, and for aggregating across seeds.

The existing paired bootstrap is good and stays. What was missing is everything needed to
report a *distribution* rather than a point:

* a single run reports one number. A simulation's headline is a draw from a distribution,
  and quoting the draw as the result is the most common way a simulated finding turns out
  not to replicate;
* no standard error, no effect size, no test. "RecoverAI recovered more" is not a claim
  until it comes with an interval that excludes zero;
* no multi-seed aggregation at all.

Everything here is plain arithmetic over lists -- no SciPy. The normal-approximation
critical values are tabulated, and where a t-distribution would be more correct on a
small number of seeds, that is said rather than papered over.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean, median, pstdev, stdev

#: Two-sided normal critical values. A table rather than a dependency; these are the only
#: three levels anything here reports.
Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}

#: Two-sided t critical values at 95%, by degrees of freedom. Used when aggregating over
#: a small number of seeds, where the normal approximation is meaningfully too narrow --
#: with 5 seeds it understates the interval by about 25%, which is exactly the range this
#: harness runs in.
T95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}


def t_critical(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in T95:
        return T95[df]
    for k in sorted(T95):
        if df < k:
            return T95[k]
    return Z[0.95]


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    level: float = 0.95
    method: str = ""

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    @property
    def width(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict:
        return {"point": round(self.point, 6), "low": round(self.low, 6),
                "high": round(self.high, 6), "level": self.level,
                "excludes_zero": self.excludes_zero, "method": self.method}

    def __str__(self) -> str:
        return (f"{self.point:,.2f} "
                f"[{int(self.level * 100)}% CI {self.low:,.2f} to {self.high:,.2f}]")


# ---------------------------------------------------------------- effects
def mean_interval(values: Sequence[float], level: float = 0.95,
                  small_sample: bool = True) -> Interval:
    """Interval for a mean over independent replications (one per seed).

    Uses the t-distribution by default. With five or ten seeds -- the realistic range for
    a simulation sweep -- the normal approximation is visibly too narrow, and an interval
    that is too narrow is worse than none: it converts "we do not know" into a claim.
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, level, "no data")
    m = mean(xs)
    if n == 1:
        return Interval(m, float("-inf"), float("inf"), level,
                        "single replication: no interval is estimable from one draw")
    sd = stdev(xs)
    se = sd / math.sqrt(n)
    crit = t_critical(n - 1) if small_sample else Z.get(level, Z[0.95])
    method = f"t({n - 1})" if small_sample else "normal"
    return Interval(m, m - crit * se, m + crit * se, level,
                    f"{method} over {n} independent seeds")


def difference_in_means(treated: Sequence[float], control: Sequence[float],
                        level: float = 0.95) -> Interval:
    """Welch interval for a difference in means. Welch rather than pooled because the
    arms genuinely have different variances -- a strategy that acts changes the spread of
    outcomes, not only the centre, and assuming equal variance would be assuming away
    part of the effect."""
    t, c = [float(x) for x in treated], [float(x) for x in control]
    if len(t) < 2 or len(c) < 2:
        return Interval(mean(t) - mean(c) if t and c else 0.0,
                        float("-inf"), float("inf"), level,
                        "too few replications for an interval")
    diff = mean(t) - mean(c)
    vt, vc = stdev(t) ** 2 / len(t), stdev(c) ** 2 / len(c)
    se = math.sqrt(vt + vc)
    if se == 0:
        return Interval(diff, diff, diff, level, "zero variance in both arms")
    df = (vt + vc) ** 2 / (vt ** 2 / (len(t) - 1) + vc ** 2 / (len(c) - 1))
    crit = t_critical(int(df))
    return Interval(diff, diff - crit * se, diff + crit * se, level,
                    f"Welch t, df~{df:.1f}")


def cohens_d(treated: Sequence[float], control: Sequence[float]) -> float:
    """Standardised effect size. Reported alongside the interval because a difference can
    be statistically distinguishable from zero and too small to care about -- with enough
    simulated cases, everything is significant."""
    t, c = [float(x) for x in treated], [float(x) for x in control]
    if len(t) < 2 or len(c) < 2:
        return 0.0
    pooled = math.sqrt(((len(t) - 1) * stdev(t) ** 2 + (len(c) - 1) * stdev(c) ** 2)
                       / (len(t) + len(c) - 2))
    return 0.0 if pooled == 0 else (mean(t) - mean(c)) / pooled


def proportion_interval(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson interval for a rate.

    Wilson, not Wald: the normal-approximation interval is badly wrong near 0 and 1 and
    can produce a lower bound below zero, which for a recovery rate is a nonsense figure
    that nonetheless gets copied into a slide.
    """
    if n == 0:
        return Interval(0.0, 0.0, 0.0, level, "no observations")
    z = Z.get(level, Z[0.95])
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), level, "Wilson")


def two_proportion_interval(s1: int, n1: int, s2: int, n2: int,
                            level: float = 0.95) -> Interval:
    """Interval for a difference in rates (arm 1 minus arm 2)."""
    if n1 == 0 or n2 == 0:
        return Interval(0.0, float("-inf"), float("inf"), level, "empty arm")
    p1, p2 = s1 / n1, s2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = Z.get(level, Z[0.95])
    d = p1 - p2
    return Interval(d, d - z * se, d + z * se, level, "normal approximation on rates")


def average_treatment_effect(outcomes_treated: Sequence[float],
                             outcomes_control: Sequence[float],
                             level: float = 0.95) -> dict:
    """ATE with its interval and effect size.

    Only causal if assignment was randomised. This function cannot check that, so it does
    not claim it -- the caller stamps the result with a `Provenance`, and the report reads
    the provenance rather than the function name.
    """
    ci = difference_in_means(outcomes_treated, outcomes_control, level)
    return {
        "ate": round(ci.point, 6),
        "ci_low": round(ci.low, 6), "ci_high": round(ci.high, 6),
        "level": level, "excludes_zero": ci.excludes_zero,
        "cohens_d": round(cohens_d(outcomes_treated, outcomes_control), 4),
        "n_treated": len(outcomes_treated), "n_control": len(outcomes_control),
        "mean_treated": round(mean(outcomes_treated), 6) if outcomes_treated else 0.0,
        "mean_control": round(mean(outcomes_control), 6) if outcomes_control else 0.0,
        "caveat": "causal only if assignment was randomised; see the result's provenance",
    }


def bootstrap_ci(values: Sequence[float], statistic=mean, n_boot: int = 2000,
                 level: float = 0.95, seed: int = 20260822) -> Interval:
    """Percentile bootstrap for an arbitrary statistic."""
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return Interval(statistic(xs) if xs else 0.0, float("-inf"), float("inf"),
                        level, "too few observations")
    rng = random.Random(seed)
    n = len(xs)
    draws = sorted(statistic([xs[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    lo = draws[int((1 - level) / 2 * n_boot)]
    hi = draws[min(int((1 + level) / 2 * n_boot), n_boot - 1)]
    return Interval(statistic(xs), lo, hi, level, f"percentile bootstrap, {n_boot} draws")


# ---------------------------------------------------------------- multi-seed
@dataclass
class SeedResults:
    """One metric across many seeds. The unit of a multi-seed claim."""
    metric: str
    by_seed: dict[int, float] = field(default_factory=dict)

    def add(self, seed: int, value: float) -> None:
        self.by_seed[seed] = float(value)

    @property
    def values(self) -> list[float]:
        return [self.by_seed[s] for s in sorted(self.by_seed)]

    def summary(self, level: float = 0.95) -> dict:
        xs = self.values
        if not xs:
            return {"metric": self.metric, "seeds": 0}
        ci = mean_interval(xs, level)
        return {
            "metric": self.metric,
            "seeds": len(xs),
            "mean": round(mean(xs), 4),
            "median": round(median(xs), 4),
            "std": round(stdev(xs), 4) if len(xs) > 1 else 0.0,
            "population_std": round(pstdev(xs), 4),
            "ci_low": round(ci.low, 4), "ci_high": round(ci.high, 4),
            "ci_level": level, "ci_method": ci.method,
            "excludes_zero": ci.excludes_zero,
            # The two numbers a reader should look at before the mean. If the worst seed
            # is negative, "mean uplift is positive" is not the whole story, and reporting
            # only the mean would hide the case where the strategy sometimes loses money.
            "worst": round(min(xs), 4),
            "best": round(max(xs), 4),
            "share_positive": round(sum(1 for x in xs if x > 0) / len(xs), 4),
        }


@dataclass
class MultiSeedReport:
    """Every metric, across every seed, for one arm."""
    arm: str
    metrics: dict[str, SeedResults] = field(default_factory=dict)

    def record(self, seed: int, report: dict, keys: Sequence[str]) -> None:
        for key in keys:
            value = report.get(key)
            if value is None:
                continue
            self.metrics.setdefault(key, SeedResults(key)).add(seed, float(value))

    def summary(self, level: float = 0.95) -> dict:
        return {"arm": self.arm,
                "metrics": {k: v.summary(level) for k, v in sorted(self.metrics.items())}}

    def series(self, metric: str) -> list[float]:
        return self.metrics[metric].values if metric in self.metrics else []


def compare_arms_across_seeds(a: MultiSeedReport, b: MultiSeedReport, metric: str,
                              level: float = 0.95) -> dict:
    """Paired comparison of two arms over the same seeds.

    Paired because the arms ran on the same population under the same seed, so seed-to-
    seed variation is common to both and differencing removes it. Treating the two series
    as independent samples would inflate the interval with variance the design already
    controlled -- and would sometimes turn a real effect into "not distinguishable".
    """
    shared = sorted(set(a.metrics.get(metric, SeedResults(metric)).by_seed)
                    & set(b.metrics.get(metric, SeedResults(metric)).by_seed))
    if not shared:
        return {"metric": metric, "seeds": 0,
                "note": "no seeds in common; the arms were not run on the same draws"}
    deltas = [a.metrics[metric].by_seed[s] - b.metrics[metric].by_seed[s] for s in shared]
    ci = mean_interval(deltas, level)
    return {
        "metric": metric, "arm": a.arm, "versus": b.arm, "seeds": len(shared),
        "paired": True,
        "mean_difference": round(ci.point, 4),
        "ci_low": round(ci.low, 4), "ci_high": round(ci.high, 4),
        "ci_method": ci.method, "excludes_zero": ci.excludes_zero,
        "worst_seed_difference": round(min(deltas), 4),
        "best_seed_difference": round(max(deltas), 4),
        "seeds_where_better": sum(1 for d in deltas if d > 0),
    }
