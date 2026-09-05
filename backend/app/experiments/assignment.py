"""Deterministic randomised assignment.

Two requirements that pull in opposite directions, and both are non-negotiable:

**Random**, or the arms differ by something other than the treatment and every
comparison is confounded.

**Stable**, or a customer who reloads the page moves between arms, sees two different
treatments, and appears in both denominators. Assignment by `random.random()` at decision
time satisfies the first and catastrophically fails the second.

The resolution is a hash: `SHA256(experiment_id | unit_id | salt)` mapped onto [0, 1) and
bucketed. It is uniform, it is uncorrelated across experiments (the experiment id is in
the hash, so a unit in the treatment arm of one experiment is no likelier to be in the
treatment arm of the next), and it is a pure function -- the same unit gets the same arm
forever, on any machine, with no state to store.

Deliberately *not* `hash()`: Python's string hash is randomised per process, so it would
reassign every unit on restart.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime


class InvalidExperiment(ValueError):
    """A design that cannot support the inference it is being asked for."""


@dataclass(frozen=True)
class Variant:
    name: str
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class Assignment:
    unit_id: str
    experiment_id: str
    variant: str
    bucket: float
    eligible: bool = True
    reason: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {"unit_id": self.unit_id, "experiment_id": self.experiment_id,
                "variant": self.variant, "bucket": round(self.bucket, 6),
                "eligible": self.eligible, "reason": self.reason,
                "randomization_timestamp": self.at.isoformat()}


@dataclass
class Experiment:
    """One randomised comparison.

    `holdout_fraction` carves off units that receive nothing at all, regardless of
    variant. It is separate from a control *variant* because the two answer different
    questions: a control variant measures "this treatment against that treatment", a
    holdout measures "any of this against nothing", and quoting one as the other is the
    most common way an experiment result gets overstated.
    """
    experiment_id: str
    variants: Sequence[Variant]
    salt: str = "recoverai"
    holdout_fraction: float = 0.0
    #: Units failing this predicate are excluded and recorded as ineligible. Eligibility
    #: is evaluated BEFORE assignment so it cannot depend on the arm -- post-assignment
    #: filtering is how a clean randomisation quietly becomes a biased one.
    eligibility: object = None
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.variants) < 2:
            raise InvalidExperiment(
                f"{self.experiment_id}: an experiment needs at least two variants; "
                f"one arm is a rollout, not a comparison")
        if any(v.weight <= 0 for v in self.variants):
            raise InvalidExperiment("variant weights must be positive")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise InvalidExperiment("holdout_fraction must be in [0, 1)")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise InvalidExperiment(f"duplicate variant names: {names}")

    # ------------------------------------------------------------------ hashing
    def bucket(self, unit_id: str, namespace: str = "assign") -> float:
        """Uniform in [0, 1), stable across processes and machines."""
        digest = hashlib.sha256(
            f"{self.experiment_id}|{namespace}|{unit_id}|{self.salt}".encode()).digest()
        # 8 bytes gives 2^-64 granularity: far finer than any bucketing this will do, and
        # cheap. Dividing by 2**64 keeps the value strictly below 1.0.
        return int.from_bytes(digest[:8], "big") / 2 ** 64

    def assign(self, unit_id: str, context: dict | None = None) -> Assignment:
        if self.eligibility is not None:
            try:
                ok = bool(self.eligibility(context or {}))
            except Exception as exc:
                # An eligibility predicate that raises must exclude the unit, not admit
                # it: admitting on error would silently enrol whoever triggers the bug.
                return Assignment(unit_id, self.experiment_id, "excluded", 0.0, False,
                                  f"eligibility check raised {type(exc).__name__}")
            if not ok:
                return Assignment(unit_id, self.experiment_id, "excluded", 0.0, False,
                                  "not eligible for this experiment")

        # A separate hash namespace, so changing variant weights does not reshuffle who
        # is in the holdout -- the holdout must be stable across design edits or it stops
        # being a clean "nothing at all" comparison.
        if self.holdout_fraction > 0 \
                and self.bucket(unit_id, "holdout") < self.holdout_fraction:
            return Assignment(unit_id, self.experiment_id, "holdout",
                              self.bucket(unit_id), True, "global holdout")

        b = self.bucket(unit_id)
        total = sum(v.weight for v in self.variants)
        cursor = 0.0
        for v in self.variants:
            cursor += v.weight / total
            if b < cursor:
                return Assignment(unit_id, self.experiment_id, v.name, b)
        return Assignment(unit_id, self.experiment_id, self.variants[-1].name, b)

    def assign_all(self, unit_ids: Iterable[str],
                   contexts: dict[str, dict] | None = None) -> list[Assignment]:
        ctx = contexts or {}
        return [self.assign(u, ctx.get(u)) for u in unit_ids]

    # ------------------------------------------------------------------ diagnostics
    def balance(self, unit_ids: Sequence[str]) -> dict:
        """Realised arm sizes against intended ones, with a chi-square statistic.

        Worth checking every run rather than trusting the hash: a salt collision or an
        accidentally correlated unit-id scheme (sequential ids assigned by signup date,
        say) shows up here as imbalance and nowhere else.
        """
        assignments = self.assign_all(unit_ids)
        counts: dict[str, int] = {}
        for a in assignments:
            counts[a.variant] = counts.get(a.variant, 0) + 1

        enrolled = [a for a in assignments if a.eligible and a.variant != "holdout"]
        n = len(enrolled)
        total_w = sum(v.weight for v in self.variants)
        chi2 = 0.0
        expected: dict[str, float] = {}
        for v in self.variants:
            e = n * (v.weight / total_w)
            expected[v.name] = round(e, 2)
            if e > 0:
                chi2 += (counts.get(v.name, 0) - e) ** 2 / e
        return {
            "experiment_id": self.experiment_id,
            "n_units": len(unit_ids),
            "counts": dict(sorted(counts.items())),
            "expected": expected,
            "chi_square": round(chi2, 4),
            "degrees_of_freedom": max(len(self.variants) - 1, 1),
            # A rough flag, not a test. 3.84 is the 5% critical value at 1 df; with more
            # arms it is conservative, which is the right direction for a warning.
            "imbalance_warning": chi2 > 3.84 * max(len(self.variants) - 1, 1),
        }

    def describe(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "salt": self.salt,
            "holdout_fraction": self.holdout_fraction,
            "variants": [{"name": v.name, "weight": v.weight,
                          "description": v.description} for v in self.variants],
        }
