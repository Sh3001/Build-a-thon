"""Choosing the action with the highest expected incremental profit.

    best_action = argmax_a  E[profit | a]  subject to the policy engine

The word doing the work is *incremental*. Ranking by `P(recover | action)` selects the
customers who were going to pay anyway; ranking by the difference selects the ones whose
payment the action actually causes. Those are different people, and only the second set
is worth spending on.

## Where the probabilities come from, and how much to trust them

This is the part to read sceptically, so it is stated plainly:

`P(recover | no action)` -- the counterfactual. Taken from the uplift model's control arm
when one is loaded. That estimate is only as causal as the data it was fitted on: on
randomised data it is a real counterfactual; on the simulator it is the simulator's
counterfactual; on observational data it is a conditional mean wearing a counterfactual's
clothes. `EffectEstimate.provenance` records which, and it is carried all the way into
the audit row.

`P(recover | action)` -- per action. **No per-action randomised data exists in this
project.** The uplift model has two arms, treated and control, so it can say what
*intervening* is worth but not what *sending an SMS instead of a payment link* is worth.
The per-action term is therefore a configured relative-effectiveness prior conditioned on
the root cause, declared in `config/effects.yaml` and labelled `PRIOR` in every estimate
it produces.

Calling that prior a measurement would be the single most misleading thing this module
could do, so it does not: `EffectEstimate.is_measured` is False for every per-action
figure until a multi-armed randomised experiment supplies one, and
`scripts/run_experiment.py --arms` is the harness that would collect it.

## What this module does not do

It does not execute, and it does not decide safety. It returns a ranked list of
candidates; the policy engine still has the last word, and a candidate with the highest
profit in the world is refused if a rule says so. Profit never outbids a limit.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.adapters.processor_codes import CanonicalCause
from backend.app.config import ROOT
from backend.app.decision.economics import CostModel, get_cost_model
from backend.app.domain.money import Money
from backend.app.models.enums import (
    CONTACT_ACTIONS,
    EXECUTABLE_ACTIONS,
    FailureCategory,
    FailureCode,
    InterventionType,
    category_of,
)
from backend.app.models.schemas import AgentState, Diagnosis, ProposedAction

DEFAULT_EFFECTS_PATH = ROOT / "config" / "effects.yaml"

#: How an estimate was obtained. Carried on every number this module produces.
MEASURED = "MEASURED_UPLIFT_MODEL"
PRIOR = "CONFIGURED_PRIOR"


@dataclass(frozen=True)
class EffectEstimate:
    """One probability, with its provenance attached.

    A bare float loses the only thing a reader needs in order to know how much weight to
    put on it. Bundling the two means an estimate cannot be passed around stripped of the
    caveat that applies to it.
    """
    value: float
    provenance: str
    detail: str = ""

    @property
    def is_measured(self) -> bool:
        return self.provenance == MEASURED


@dataclass(frozen=True)
class Candidate:
    """One scored action. Every intermediate term is kept so the decision can be
    explained from stored facts rather than reconstructed from a total."""
    action: InterventionType
    p_no_action: EffectEstimate
    p_with_action: EffectEstimate
    incremental_probability: float
    expected_gross_revenue: Money
    expected_incremental_revenue: Money
    processing_cost: Money
    action_cost: Money
    expected_profit: Money
    rationale: str = ""
    feasible: bool = True
    infeasible_reason: str = ""

    @property
    def profit_usd(self) -> float:
        return self.expected_profit.as_float()

    def explain(self) -> str:
        """The decision as prose, generated from the stored numbers.

        Deliberately not written by a language model: an explanation that can invent a
        figure is worse than no explanation, because it is trusted.
        """
        if not self.feasible:
            return f"{self.action.value}: not available -- {self.infeasible_reason}"
        return (
            f"{self.action.value}: P(recover) rises from "
            f"{self.p_no_action.value:.1%} to {self.p_with_action.value:.1%} "
            f"({self.incremental_probability:+.1%}), worth "
            f"{self.expected_incremental_revenue} incrementally; "
            f"costs {self.action_cost} to run plus {self.processing_cost} in fees "
            f"if it lands, for an expected profit of {self.expected_profit}. "
            f"P(recover | action) provenance: {self.p_with_action.provenance}."
        )


#: Relative effectiveness of each action, by canonical cause. **A prior, not a
#: measurement.** Read as: "given that intervening at all lifts recovery by U, this
#: action captures this share of U for this cause." Values above 1.0 mean the action is
#: more effective than the average intervention for that cause -- a method-update request
#: on a dead card, for instance, is the only thing that can work at all.
#:
#: Zero means the action cannot work for that cause and is excluded from the argmax,
#: which is a *structural* claim about the rails (you cannot debit a closed account),
#: not a statistical one, and is the one part of this table that is not a guess.
DEFAULT_EFFECTS: dict[CanonicalCause, dict[InterventionType, float]] = {
    CanonicalCause.NETWORK_FAILURE: {
        InterventionType.RETRY_PAYMENT: 1.30,
        InterventionType.SEND_REMINDER: 0.25,
        InterventionType.SEND_PAYMENT_LINK: 0.55,
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 0.10,
    },
    CanonicalCause.TEMPORARY_DECLINE: {
        InterventionType.RETRY_PAYMENT: 1.10,
        InterventionType.SEND_REMINDER: 0.35,
        InterventionType.SEND_PAYMENT_LINK: 0.65,
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 0.20,
    },
    CanonicalCause.INSUFFICIENT_FUNDS: {
        InterventionType.RETRY_PAYMENT: 0.85,
        InterventionType.SEND_REMINDER: 0.70,      # warn before re-presenting
        InterventionType.SEND_PAYMENT_LINK: 0.90,  # let them choose the timing
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 0.25,
    },
    CanonicalCause.LIMIT_EXCEEDED: {
        InterventionType.RETRY_PAYMENT: 0.45,
        InterventionType.SEND_REMINDER: 0.30,
        InterventionType.SEND_PAYMENT_LINK: 1.05,  # a different rail is the whole fix
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 0.60,
    },
    CanonicalCause.EXPIRED_PAYMENT_METHOD: {
        InterventionType.RETRY_PAYMENT: 0.0,       # structural: the instrument is dead
        InterventionType.SEND_REMINDER: 0.20,
        InterventionType.SEND_PAYMENT_LINK: 0.85,
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 1.25,
    },
    CanonicalCause.INVALID_PAYMENT_METHOD: {
        InterventionType.RETRY_PAYMENT: 0.0,       # structural
        InterventionType.SEND_REMINDER: 0.15,
        InterventionType.SEND_PAYMENT_LINK: 0.80,
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 1.20,
    },
    CanonicalCause.ACCOUNT_CLOSED: {
        InterventionType.RETRY_PAYMENT: 0.0,       # structural: cannot be debited
        InterventionType.SEND_REMINDER: 0.20,
        InterventionType.SEND_PAYMENT_LINK: 1.00,
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 0.55,
    },
    CanonicalCause.DO_NOT_HONOR: {
        InterventionType.RETRY_PAYMENT: 0.20,
        InterventionType.SEND_REMINDER: 0.25,
        InterventionType.SEND_PAYMENT_LINK: 0.95,
        InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: 0.50,
    },
    # Risk and compliance are not an optimisation problem. Every automated action scores
    # zero so the argmax cannot select one even before the policy engine refuses it --
    # two independent mechanisms, because one of them being wrong should not be enough.
    CanonicalCause.FRAUD_SUSPECTED: {},
    CanonicalCause.COMPLIANCE_HOLD: {},
    CanonicalCause.UNKNOWN: {},
}

#: P(recover | no action), by cause. The counterfactual baseline used when no uplift
#: control model is loaded. A prior; the control arm of a run measures the real one.
DEFAULT_PASSIVE: dict[CanonicalCause, float] = {
    CanonicalCause.NETWORK_FAILURE: 0.26,
    CanonicalCause.TEMPORARY_DECLINE: 0.17,
    CanonicalCause.INSUFFICIENT_FUNDS: 0.15,
    CanonicalCause.LIMIT_EXCEEDED: 0.13,
    CanonicalCause.EXPIRED_PAYMENT_METHOD: 0.07,
    CanonicalCause.INVALID_PAYMENT_METHOD: 0.05,
    CanonicalCause.ACCOUNT_CLOSED: 0.01,
    CanonicalCause.DO_NOT_HONOR: 0.04,
    CanonicalCause.FRAUD_SUSPECTED: 0.02,
    CanonicalCause.COMPLIANCE_HOLD: 0.05,
    CanonicalCause.UNKNOWN: 0.10,
}

#: Internal failure code -> canonical cause, for looking the tables up. Derived from the
#: adapter's mapping rather than restated, so the two cannot drift.
def _cause_for(code: FailureCode) -> CanonicalCause:
    from backend.app.adapters.processor_codes import TO_FAILURE_CODE
    for cause, mapped in TO_FAILURE_CODE.items():
        if mapped is code:
            return cause
    cat = category_of(code)
    if cat is FailureCategory.TEMPORARY:
        return CanonicalCause.NETWORK_FAILURE
    if cat is FailureCategory.RISK_COMPLIANCE:
        return CanonicalCause.FRAUD_SUSPECTED
    if cat is FailureCategory.PERSISTENT:
        return CanonicalCause.DO_NOT_HONOR
    return CanonicalCause.UNKNOWN


@dataclass
class EffectModel:
    """Per-action effect priors plus, optionally, a fitted uplift model."""
    effects: dict[CanonicalCause, dict[InterventionType, float]] = \
        field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_EFFECTS.items()})
    passive: dict[CanonicalCause, float] = field(default_factory=lambda: dict(DEFAULT_PASSIVE))
    #: A fitted `UpliftModel`. When present its control arm supplies the counterfactual
    #: and its treated arm supplies the size of the average treatment effect; the table
    #: above then apportions that effect across actions.
    uplift: Any = None
    #: Whether the uplift model was fitted on randomised assignment. Only the caller
    #: knows, so only the caller can set it -- and it defaults to False, because assuming
    #: causal validity you have not verified is how a prototype becomes a wrong claim.
    uplift_is_randomised: bool = False

    @classmethod
    def load(cls, path: Path | str | None = None, uplift: Any = None,
             uplift_is_randomised: bool = False) -> EffectModel:
        p = Path(path or DEFAULT_EFFECTS_PATH)
        model = cls(uplift=uplift, uplift_is_randomised=uplift_is_randomised)
        if not p.exists():
            return model
        import yaml
        raw = yaml.safe_load(p.read_text()) or {}
        for cause_name, actions in (raw.get("effects") or {}).items():
            cause = CanonicalCause(cause_name)
            model.effects[cause] = {
                InterventionType(a): float(v) for a, v in (actions or {}).items()}
        for cause_name, v in (raw.get("passive_recovery") or {}).items():
            model.passive[CanonicalCause(cause_name)] = float(v)
        return model

    # ------------------------------------------------------------------ estimates
    def p_no_action(self, state: AgentState, cause: CanonicalCause,
                    features: Any = None) -> EffectEstimate:
        if self.uplift is not None and features is not None:
            try:
                _, p_c = self.uplift.predict_arms(features)
                return EffectEstimate(
                    float(p_c[0]),
                    MEASURED if self.uplift_is_randomised else PRIOR,
                    "uplift-model control arm"
                    + ("" if self.uplift_is_randomised else
                       " (fitted on non-randomised data: a conditional mean, not a "
                       "verified counterfactual)"))
            except Exception:
                pass
        return EffectEstimate(
            self.passive.get(cause, 0.10), PRIOR,
            f"configured passive-recovery prior for {cause.value}")

    def average_treatment_effect(self, state: AgentState, cause: CanonicalCause,
                                 features: Any = None) -> EffectEstimate:
        """How much intervening at all is worth for this case."""
        if self.uplift is not None and features is not None:
            try:
                u = float(self.uplift.predict_uplift(features)[0])
                return EffectEstimate(
                    u, MEASURED if self.uplift_is_randomised else PRIOR,
                    "uplift-model treatment effect"
                    + ("" if self.uplift_is_randomised else
                       " (fitted on non-randomised data)"))
            except Exception:
                pass
        # Fall back to the scorer's own probability against the passive prior. This is a
        # weak estimate and is labelled as one.
        base = self.passive.get(cause, 0.10)
        return EffectEstimate(
            max(0.0, state.recovery_probability - base), PRIOR,
            "scorer probability minus the passive prior; no treatment model loaded")

    def relative_effectiveness(self, cause: CanonicalCause,
                               action: InterventionType) -> float:
        return self.effects.get(cause, {}).get(action, 0.0)


@dataclass
class ProfitOptimizer:
    """Scores candidate actions and returns them ranked. Never executes."""
    costs: CostModel = field(default_factory=get_cost_model)
    effects: EffectModel = field(default_factory=EffectModel.load)
    #: Per-touch multiplier on the *goodwill* cost. Contact number three annoys more than
    #: contact number one, and pricing that is what stops the optimiser from spending an
    #: un-priced resource.
    fatigue_growth: float = 1.6

    def candidates(self, state: AgentState, dx: Diagnosis,
                   allowed: Sequence[InterventionType] | None = None,
                   features: Any = None) -> list[Candidate]:
        cause = _cause_for(dx.root_cause)
        pool = list(allowed) if allowed is not None else sorted(
            EXECUTABLE_ACTIONS, key=lambda a: a.value)

        p_base = self.effects.p_no_action(state, cause, features)
        ate = self.effects.average_treatment_effect(state, cause, features)
        amount = Money.from_major(state.amount_usd)

        out: list[Candidate] = []
        for action in pool:
            out.append(self._score(state, dx, cause, action, p_base, ate, amount))
        out.sort(key=lambda c: (c.feasible, c.expected_profit.minor), reverse=True)
        return out

    def _score(self, state: AgentState, dx: Diagnosis, cause: CanonicalCause,
               action: InterventionType, p_base: EffectEstimate, ate: EffectEstimate,
               amount: Money) -> Candidate:
        zero = Money.zero(amount.currency)

        if action is InterventionType.ESCALATE_CASE:
            # Escalation buys a human decision, not a probability. Scoring it against the
            # same objective would let a cost model decide whether a fraud case gets
            # looked at, which is not a trade the optimiser is allowed to make.
            cost = self.costs.action_cost(action)
            return Candidate(
                action=action, p_no_action=p_base,
                p_with_action=EffectEstimate(p_base.value, PRIOR, "not modelled"),
                incremental_probability=0.0, expected_gross_revenue=zero,
                expected_incremental_revenue=zero, processing_cost=zero,
                action_cost=cost, expected_profit=-cost,
                rationale="routes the case to a person; not scored on expected profit "
                          "because whether a human looks at a case is not a price decision")

        rel = self.effects.relative_effectiveness(cause, action)
        if rel <= 0.0:
            return Candidate(
                action=action, p_no_action=p_base,
                p_with_action=EffectEstimate(p_base.value, PRIOR, "structurally ineffective"),
                incremental_probability=0.0, expected_gross_revenue=zero,
                expected_incremental_revenue=zero, processing_cost=zero,
                action_cost=zero, expected_profit=zero, feasible=False,
                infeasible_reason=(
                    f"{action.value} cannot affect a {cause.value} failure "
                    f"(structural, not statistical: the rails make it impossible)"))

        incremental = max(0.0, min(ate.value * rel, 1.0 - p_base.value))
        p_action = min(1.0, p_base.value + incremental)

        gross = amount.scale(p_action)
        incremental_revenue = amount.scale(incremental)

        fatigue = (self.fatigue_growth ** state.contact_count) \
            if action in CONTACT_ACTIONS else 1.0
        cost = self.costs.action_cost(action, fatigue_multiplier=fatigue)

        # The fee is charged against the *incremental* recovery, not the gross one: fees
        # on money that would have arrived anyway are not caused by this action, and
        # charging them here would make every action look unprofitable on a case with a
        # high passive-recovery rate.
        incremental_processing = self.costs.processing_cost(incremental_revenue) \
            + self.costs.chargeback_cost(incremental_revenue)
        profit = incremental_revenue - incremental_processing - cost

        return Candidate(
            action=action, p_no_action=p_base,
            p_with_action=EffectEstimate(
                p_action, ate.provenance,
                f"{ate.detail}; apportioned to {action.value} by the configured "
                f"relative-effectiveness prior ({rel:.2f}x) for {cause.value}"),
            incremental_probability=incremental,
            expected_gross_revenue=gross,
            expected_incremental_revenue=incremental_revenue,
            processing_cost=incremental_processing, action_cost=cost,
            expected_profit=profit,
            rationale=f"{cause.value} x {action.value}: effectiveness {rel:.2f}x")

    def best(self, state: AgentState, dx: Diagnosis,
             allowed: Sequence[InterventionType] | None = None,
             features: Any = None) -> Candidate | None:
        """The highest-profit feasible candidate, or None if none clears zero.

        Returning None for "nothing is worth doing" is the point. A recovery system whose
        optimiser always names an action will always act, and most of the value in
        dunning comes from the cases you leave alone.
        """
        ranked = [c for c in self.candidates(state, dx, allowed, features)
                  if c.feasible and c.expected_profit.is_positive]
        return ranked[0] if ranked else None

    def to_proposal(self, candidate: Candidate) -> ProposedAction:
        """Turn a scored candidate into a proposal for the policy engine."""
        return ProposedAction(
            action=candidate.action, reason=candidate.explain(), source="rules",
            expected_profit_usd=candidate.profit_usd)
