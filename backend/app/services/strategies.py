"""The seven arms.

Two arms existed: a no-touch control and a fixed three-retry baseline. That is enough to
show the agent beats *doing nothing* and beats *the dumbest possible thing*, and it is
not enough to show the agent's intelligence is what does the work. The interesting
question is not "does RecoverAI beat a 24-hour retry loop" -- almost anything does -- but
"does it beat a well-tuned rule set, and does it beat plain ML targeting?"

So there are seven, and each isolates one ingredient:

    1  control            no action                       what arrives on its own
    2  naive_retry        fixed 24h x3, everyone          the untuned dunning system
    3  smart_retry        cause-aware timing, no ML       does *diagnosis* pay?
    4  ml_probability     contact the likeliest to pay    does *prediction* pay?
    5  expected_value     rank by amount x P(recover)     does *ranking by money* pay?
    6  uplift             rank by amount x uplift(x)      does *causal* targeting pay?
    7  recoverai          the full agent                  does the whole loop pay?

Reading the ladder is the point. 3-vs-2 prices diagnosis. 4-vs-3 prices the model.
6-vs-5 prices the counterfactual -- and it is the comparison that most often disappoints,
because an uplift model trained on non-randomised data is a ranking dressed as a cause.

All seven run over the same transactions against the same seeded simulator, so the only
difference between two columns is the strategy. Arms 4-6 share a contact budget so that
"targeted better" cannot secretly mean "contacted more".
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from backend.app.config import (
    ACTION_COST_USD,
    MAX_RETRIES,
    MIN_RETRY_INTERVAL_HOURS,
    RECOVERY_HORIZON_DAYS,
)
from backend.app.models.enums import (
    FailureCategory,
    FailureCode,
    category_of,
)
from backend.app.services.baseline import BaselineConfig, run_baseline
from backend.app.services.control import run_control
from backend.app.services.results import CaseOutcome, summarize
from simulation.payment_gateway import PaymentGateway

#: Contact budget for the targeting arms, as a share of the population. Shared so a
#: targeting arm cannot win by simply contacting more people -- which is what would
#: happen with per-arm thresholds, and would make the comparison meaningless.
DEFAULT_CONTACT_BUDGET_FRACTION = 0.35

#: Cause-aware retry schedule for `smart_retry`, in hours after the original failure.
#: Encodes what the diagnosis layer knows and the naive arm does not: transient faults
#: decay so retry immediately; insufficient funds improves until payday so wait; a dead
#: instrument or a closed account cannot be retried at all, so do not.
SMART_SCHEDULE: dict[str, list[float]] = {
    "bank_timeout": [0.5, 24.0, 96.0],
    "network_error": [0.5, 24.0, 96.0],
    "processor_unavailable": [1.0, 24.0, 96.0],
    "temporary_decline": [6.0, 48.0, 144.0],
    "insufficient_funds": [132.0, 300.0],      # payday, then next payday
    "payment_limit_exceeded": [48.0, 168.0],   # after the limit window resets
    "multiple_declines": [72.0],
    # Structural zeros: no schedule at all. Retrying these is guaranteed waste, and the
    # whole claim of the smart arm is that knowing so is worth money.
    "expired_card": [],
    "invalid_payment_method": [],
    "invalid_account": [],
    "closed_account": [],
    "suspected_fraud": [],
    "high_risk_transaction": [],
    "compliance_hold": [],
}


def _blank(txn: dict, strategy: str) -> CaseOutcome:
    code = FailureCode(txn["failure_code"])
    return CaseOutcome(
        transaction_id=txn["transaction_id"], customer_id=txn.get("customer_id", ""),
        amount_usd=float(txn["amount_usd"]), failure_code=code,
        failure_category=category_of(code), strategy=strategy,
        status="stopped", stop_reason="no action taken")


def _settle(out: CaseOutcome, hours: float, reason: str) -> CaseOutcome:
    out.recovered = True
    out.amount_recovered = out.amount_usd
    out.recovery_hours = hours
    out.status = "recovered"
    out.stop_reason = reason
    return out


# ---------------------------------------------------------------- arm 3
def run_smart_retry(transactions: Sequence[dict], gateway: PaymentGateway | None = None,
                    horizon_days: int = RECOVERY_HORIZON_DAYS
                    ) -> tuple[list[CaseOutcome], dict]:
    """Cause-aware retry timing. No model, no customer contact, no LLM.

    This is the arm that matters most for an honest reading of the results. It has all of
    the domain knowledge and none of the machine learning, so whatever it recovers is the
    part of the agent's lift that a competent engineer could have written as a lookup
    table -- and only the gap above *this* line is attributable to the model.
    """
    gw = gateway or PaymentGateway()
    horizon = horizon_days * 24
    outcomes: list[CaseOutcome] = []

    for txn in transactions:
        out = _blank(txn, "smart_retry")
        code = str(txn["failure_code"])
        schedule = SMART_SCHEDULE.get(code, [24.0, 72.0])
        start = float(txn.get("days_since_failure", 0.0)) * 24.0

        if not schedule:
            out.stop_reason = f"{code}: no retry can succeed; not attempted"
            # Still check the counterfactual -- otherwise this arm would under-report
            # recoveries that happen anyway and look worse than it is.
            cured = gw.self_cure_hour(txn, horizon)
            if cured is not None:
                out.passive_recovery = True
                _settle(out, cured, "self-cured with no intervention")
            outcomes.append(out)
            continue

        for attempt, offset in enumerate(schedule, start=1):
            hours = start + offset
            if hours > horizon:
                out.stop_reason = "horizon reached"
                break
            cured = gw.check_self_cure(txn, hours)
            if cured is not None:
                out.passive_recovery = True
                _settle(out, hours, "self-cured before the scheduled retry")
                break
            res = gw.retry_payment(txn, hours_since_failure=hours, attempt=attempt)
            out.retries += 1
            out.actions.append("retry_payment")
            out.cost += ACTION_COST_USD["retry_payment"]
            if out.failure_category is FailureCategory.RISK_COMPLIANCE:
                out.risk_actions += 1
            if res.success:
                _settle(out, hours, "payment succeeded")
                break
        else:
            out.stop_reason = "cause-aware retry schedule exhausted"

        out.cost = round(out.cost, 4)
        outcomes.append(out)

    return outcomes, summarize(outcomes, "smart_retry")


# ---------------------------------------------------------------- arms 4-6
@dataclass
class TargetingArm:
    """A ranked-contact strategy: score every case, contact the top `budget`.

    The three targeting arms differ *only* in `score`. Everything else -- what a contact
    is, when it lands, the budget, the retry policy for the untargeted remainder -- is
    identical, which is what makes the three columns comparable.
    """
    name: str
    score: Callable[[dict], float]
    #: Contacts land here, in hours after the original failure. One touch per case: these
    #: arms are targeting studies, not sequence studies, and giving them a sequence would
    #: confound "who to contact" with "how often".
    contact_at_hours: float = 48.0
    budget_fraction: float = DEFAULT_CONTACT_BUDGET_FRACTION
    #: The untargeted remainder still gets the cheap, safe thing a real system would do.
    #: Without it these arms would be "contact 35% and abandon 65%", which no operator
    #: would ship and which would flatter them against the retry baselines.
    fallback_retries: int = 1

    def run(self, transactions: Sequence[dict], gateway: PaymentGateway,
            horizon_days: int = RECOVERY_HORIZON_DAYS) -> tuple[list[CaseOutcome], dict]:
        horizon = horizon_days * 24
        ranked = sorted(transactions, key=self.score, reverse=True)
        budget = int(round(len(ranked) * self.budget_fraction))
        targeted = {t["transaction_id"] for t in ranked[:budget]}

        outcomes: list[CaseOutcome] = []
        for txn in transactions:
            out = _blank(txn, self.name)
            code = str(txn["failure_code"])
            risky = out.failure_category is FailureCategory.RISK_COMPLIANCE
            start = float(txn.get("days_since_failure", 0.0)) * 24.0

            # A retry first, where one can work at all. Cheap, and it is what every real
            # system does before spending a contact.
            hours = start
            settled = False
            if not risky and code not in ("expired_card", "invalid_payment_method",
                                          "invalid_account", "closed_account"):
                for attempt in range(1, self.fallback_retries + 1):
                    hours = start + MIN_RETRY_INTERVAL_HOURS * attempt
                    if hours > horizon:
                        break
                    if gateway.check_self_cure(txn, hours) is not None:
                        out.passive_recovery = True
                        _settle(out, hours, "self-cured before the retry")
                        settled = True
                        break
                    res = gateway.retry_payment(txn, hours_since_failure=hours,
                                                attempt=attempt)
                    out.retries += 1
                    out.actions.append("retry_payment")
                    out.cost += ACTION_COST_USD["retry_payment"]
                    if res.success:
                        _settle(out, hours, "payment succeeded")
                        settled = True
                        break

            if not settled and txn["transaction_id"] in targeted and not risky:
                hours = max(hours, start + self.contact_at_hours)
                if hours <= horizon:
                    if gateway.check_self_cure(txn, hours) is not None:
                        out.passive_recovery = True
                        _settle(out, hours, "self-cured before the contact")
                    else:
                        channel = str(txn.get("preferred_channel", "email"))
                        res = gateway.customer_pays_via_link(txn, hours, channel, nonce=0)
                        out.contacts += 1
                        out.actions.append("send_payment_link")
                        out.cost += ACTION_COST_USD["send_payment_link"]
                        if res.success:
                            _settle(out, hours, "customer paid via link")
                        else:
                            out.stop_reason = "contacted; customer did not pay"
            elif not settled:
                if risky:
                    out.stop_reason = "risk/compliance: no automated action"
                else:
                    out.stop_reason = "not selected for contact within budget"

            if not out.recovered and not out.stop_reason:
                out.stop_reason = "no recovery within the horizon"
            out.cost = round(out.cost, 4)
            outcomes.append(out)

        return outcomes, summarize(outcomes, self.name)


#: Scores are computed once per (population, model) and reused across seeds. Scoring is a
#: property of the case and the model, not of the random draw, so recomputing it per seed
#: is pure waste -- and it was the dominant cost of a multi-seed sweep: a per-row
#: `DataFrame([txn])` round trip through XGBoost, 1,842 times per arm per seed.
#:
#: Keyed on the model version and the population identity so a retrained model or a
#: different split cannot silently reuse a stale table.
_SCORE_CACHE: dict[tuple, dict[str, float]] = {}
#: Bound on distinct populations held. In practice a sweep uses one or two, but an
#: unbounded module-level dict in a long-running API process is a leak, and this one holds
#: a float per case.
_SCORE_CACHE_MAX = 8


def _remember(key: tuple, table: dict[str, float]) -> dict[str, float]:
    if len(_SCORE_CACHE) >= _SCORE_CACHE_MAX:
        _SCORE_CACHE.pop(next(iter(_SCORE_CACHE)))
    _SCORE_CACHE[key] = table
    return table


def _population_key(transactions: Sequence[dict]) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in transactions:
        h.update(str(t["transaction_id"]).encode())
    return h.hexdigest()[:16]


def _batch_probabilities(scorer, transactions: Sequence[dict]) -> dict[str, float]:
    """P(recovery) for the whole population in one vectorised pass."""
    import pandas as pd
    key = ("probability", str(getattr(scorer, "model_version", "?")),
           _population_key(transactions))
    if key not in _SCORE_CACHE:
        frame = pd.DataFrame(list(transactions))
        probs = scorer.predict_proba(frame)
        return _remember(key, {str(t["transaction_id"]): float(p)
                               for t, p in zip(transactions, probs)})
    return _SCORE_CACHE[key]


def _batch_incremental(targeter, transactions: Sequence[dict]) -> dict[str, float]:
    """`amount x uplift(x)` for the whole population in one pass.

    Falls back to expected recovery where no uplift model is loaded, which is what
    `Targeter.rank` does -- the fallback is preserved so this arm degrades the same way
    the production ranker does rather than silently becoming a different strategy.
    """
    import pandas as pd
    key = ("incremental", str(getattr(targeter, "mode", "?")),
           _population_key(transactions))
    if key not in _SCORE_CACHE:
        ranked = targeter.rank(pd.DataFrame(list(transactions)))
        out: dict[str, float] = {}
        for row in ranked.to_dict("records"):
            value = row.get("incremental_value")
            # NaN check: `value != value` is True only for NaN, which is what pandas
            # produces for a missing column rather than None.
            out[str(row["transaction_id"])] = (
                float(value) if value is not None and value == value
                else float(row.get("expected_recovery", 0.0)))
        return _remember(key, out)
    return _SCORE_CACHE[key]


def _score_ml_probability(scorer, transactions: Sequence[dict]) -> Callable[[dict], float]:
    """Arm 4: contact whoever is likeliest to pay. The intuitive strategy, and the wrong
    one -- it ranks a customer who was always going to pay at the very top."""
    table = _batch_probabilities(scorer, transactions)
    return lambda txn: table.get(str(txn["transaction_id"]), 0.0)


def _score_expected_value(scorer, transactions: Sequence[dict]) -> Callable[[dict], float]:
    """Arm 5: rank by amount x P(recover). Correct for forecasting cash, still wrong for
    spending a contact budget -- it carries the same sure-thing problem, weighted by size."""
    table = _batch_probabilities(scorer, transactions)
    return lambda txn: float(txn["amount_usd"]) * table.get(
        str(txn["transaction_id"]), 0.0)


def _score_uplift(targeter, transactions: Sequence[dict]) -> Callable[[dict], float]:
    """Arm 6: rank by amount x uplift(x). The only one of the three that tries to select
    people whose payment the contact *causes*."""
    table = _batch_incremental(targeter, transactions)
    return lambda txn: table.get(str(txn["transaction_id"]), 0.0)


# ---------------------------------------------------------------- the suite
@dataclass
class ArmResult:
    name: str
    outcomes: list[CaseOutcome]
    report: dict
    description: str = ""


@dataclass
class BaselineSuite:
    """Runs every non-agent arm over one population. The agent arm is run separately by
    `agents.runner`, because it needs an audit sink and a policy engine and the baselines
    deliberately have neither."""
    seed: int = 20260822
    horizon_days: int = RECOVERY_HORIZON_DAYS
    budget_fraction: float = DEFAULT_CONTACT_BUDGET_FRACTION
    scorer: object | None = None
    targeter: object | None = None
    #: Fresh gateway per arm, identically seeded. Sharing one would let an earlier arm's
    #: contact-fatigue and instrument-repair state leak into a later one, and the arm
    #: order would silently become part of the result.
    gateway_factory: Callable[[], PaymentGateway] | None = None
    config: object | None = None

    def _gateway(self) -> PaymentGateway:
        if self.gateway_factory is not None:
            return self.gateway_factory()
        if self.config is not None:
            return PaymentGateway(seed=self.seed, config=self.config)
        return PaymentGateway(seed=self.seed)

    def run(self, transactions: Sequence[dict],
            arms: Iterable[str] | None = None) -> dict[str, ArmResult]:
        wanted = set(arms) if arms is not None else set(self.available())
        out: dict[str, ArmResult] = {}

        if "control" in wanted:
            o, r = run_control(list(transactions), self._gateway(), self.horizon_days)
            out["control"] = ArmResult("control", o, r,
                                       "no action; measures what arrives on its own")

        if "naive_retry" in wanted:
            o, r = run_baseline(list(transactions), self._gateway(),
                                BaselineConfig(retry_interval_hours=MIN_RETRY_INTERVAL_HOURS,
                                               max_retries=MAX_RETRIES,
                                               horizon_days=self.horizon_days))
            for c in o:
                c.strategy = "naive_retry"
            r = summarize(o, "naive_retry")
            out["naive_retry"] = ArmResult("naive_retry", o, r,
                                           "fixed 24h x3 on every case; no diagnosis")

        if "smart_retry" in wanted:
            o, r = run_smart_retry(list(transactions), self._gateway(), self.horizon_days)
            out["smart_retry"] = ArmResult(
                "smart_retry", o, r,
                "cause-aware retry timing, no model: the domain-knowledge-only arm")

        if "ml_probability" in wanted and self.scorer is not None:
            arm = TargetingArm("ml_probability",
                               _score_ml_probability(self.scorer, transactions),
                               budget_fraction=self.budget_fraction)
            o, r = arm.run(list(transactions), self._gateway(), self.horizon_days)
            out["ml_probability"] = ArmResult("ml_probability", o, r,
                                              "contact the top P(recover) within budget")

        if "expected_value" in wanted and self.scorer is not None:
            arm = TargetingArm("expected_value",
                               _score_expected_value(self.scorer, transactions),
                               budget_fraction=self.budget_fraction)
            o, r = arm.run(list(transactions), self._gateway(), self.horizon_days)
            out["expected_value"] = ArmResult("expected_value", o, r,
                                              "contact the top amount x P(recover)")

        if "uplift" in wanted and self.targeter is not None and \
                getattr(self.targeter, "has_uplift", False):
            arm = TargetingArm("uplift", _score_uplift(self.targeter, transactions),
                               budget_fraction=self.budget_fraction)
            o, r = arm.run(list(transactions), self._gateway(), self.horizon_days)
            out["uplift"] = ArmResult("uplift", o, r,
                                      "contact the top amount x uplift(x)")

        return out

    @staticmethod
    def available() -> list[str]:
        return ["control", "naive_retry", "smart_retry", "ml_probability",
                "expected_value", "uplift"]
