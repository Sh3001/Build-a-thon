"""Where a number came from.

The single most misleading thing this project could do is present a simulated
intervention effect as evidence about the real world. The README is careful about it in
prose; prose is not enforcement, and the moment two result files sit in the same
directory someone will average them.

So provenance is a value carried by the result itself. Every experiment artefact records
one of these, `combine()` refuses to merge incompatible kinds, and `claim_strength()`
returns the strongest sentence the evidence actually supports -- which is the sentence
the report is then obliged to use.
"""
from __future__ import annotations

from enum import Enum


class Provenance(str, Enum):
    """The three kinds of evidence, in ascending order of what they license."""

    #: Historical records of what happened. Supports prediction claims only: nothing was
    #: assigned, so any "effect" here is confounded by whatever drove assignment.
    REAL_OBSERVATIONAL = "REAL_OBSERVATIONAL_DATA"

    #: Outcomes produced by our own simulator. The treatment effect is real *within the
    #: simulator*, which makes it a statement about the model, not about customers.
    SIMULATED_INTERVENTION = "SIMULATED_INTERVENTION_DATA"

    #: Treatment assigned at random, outcomes observed. The only kind that licenses an
    #: unqualified causal claim -- and only over the population that was randomised.
    RANDOMIZED_EXPERIMENT = "RANDOMIZED_EXPERIMENT_DATA"


#: What may be said out loud, per kind. Used verbatim in reports so the caveat cannot be
#: dropped by whoever writes the summary.
CLAIM: dict[Provenance, str] = {
    Provenance.REAL_OBSERVATIONAL: (
        "observational result -- real data, but treatment was not assigned, so this "
        "supports predictive claims only and no causal claim"),
    Provenance.SIMULATED_INTERVENTION: (
        "simulation result -- the treatment effect holds inside the simulator and is "
        "not evidence about real customers"),
    Provenance.RANDOMIZED_EXPERIMENT: (
        "randomized experiment result -- causal within the randomised population"),
}

#: Which kinds may be pooled. Randomised and simulated results describe different worlds;
#: averaging them produces a number that describes neither.
COMPATIBLE: dict[Provenance, frozenset[Provenance]] = {
    p: frozenset({p}) for p in Provenance
}


class ProvenanceConflict(ValueError):
    """Raised when results of different provenance would be pooled."""


def combine(*kinds: Provenance) -> Provenance:
    """The provenance of a result derived from several inputs. Refuses to mix."""
    unique = set(kinds)
    if not unique:
        raise ProvenanceConflict("no provenance given")
    if len(unique) == 1:
        return unique.pop()
    raise ProvenanceConflict(
        "refusing to combine results of different provenance: "
        + ", ".join(sorted(k.value for k in unique))
        + ". These describe different worlds; report them side by side instead.")


def claim_strength(kind: Provenance) -> str:
    return CLAIM[kind]


def label(payload: dict, kind: Provenance) -> dict:
    """Stamp a result dict. Kept as a helper so the two keys are spelled identically in
    every artefact and a reader can grep for one of them."""
    return {**payload, "provenance": kind.value, "claim": CLAIM[kind]}
