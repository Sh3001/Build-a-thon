"""Phase 2 -- shared outcome record and metric aggregation.

The baseline and RecoverAI both emit `CaseOutcome` and are both summarised by
`summarize()`. Having exactly one implementation of the business metrics is what stops
the headline comparison from quietly measuring two different things.
"""
from __future__ import annotations

from statistics import mean
from typing import Iterable

from pydantic import BaseModel, Field

from backend.app.models.enums import FailureCategory, FailureCode, category_of


class CaseOutcome(BaseModel):
    """What one strategy did to one case, and what it got."""
    transaction_id: str
    customer_id: str = ""
    amount_usd: float = 0.0
    failure_code: FailureCode
    failure_category: FailureCategory
    strategy: str = "unknown"

    recovered: bool = False
    amount_recovered: float = 0.0
    recovery_hours: float | None = None

    retries: int = 0
    contacts: int = 0
    actions: list[str] = Field(default_factory=list)
    cost: float = 0.0

    status: str = "stopped"
    #: Recovered without our intervention -- counted as revenue, never as impact.
    passive_recovery: bool = False
    stop_reason: str = ""
    escalated: bool = False
    policy_blocked: int = 0
    #: Actions taken against a RISK_COMPLIANCE case -- these should always be zero.
    risk_actions: int = 0

    recovery_probability: float | None = None
    expected_recovery: float | None = None


def summarize(outcomes: Iterable[CaseOutcome], strategy: str = "") -> dict:
    o = list(outcomes)
    if not o:
        return {"strategy": strategy, "cases": 0}

    at_risk = sum(c.amount_usd for c in o)
    recovered = sum(c.amount_recovered for c in o)
    won = [c for c in o if c.recovered]
    times = [c.recovery_hours for c in won if c.recovery_hours is not None]

    by_cat: dict[str, dict] = {}
    for cat in FailureCategory:
        cs = [c for c in o if c.failure_category is cat]
        if cs:
            by_cat[cat.value] = _cell(cs)

    by_code: dict[str, dict] = {}
    for code in {c.failure_code for c in o}:
        by_code[code.value] = _cell([c for c in o if c.failure_code is code])

    by_action: dict[str, dict] = {}
    for c in o:
        # Attribute a win to the action that immediately preceded it.
        if c.recovered and c.actions:
            k = c.actions[-1]
            e = by_action.setdefault(k, {"wins": 0, "amount_recovered": 0.0})
            e["wins"] += 1
            e["amount_recovered"] = round(e["amount_recovered"] + c.amount_recovered, 2)
        for a in c.actions:
            e = by_action.setdefault(a, {"wins": 0, "amount_recovered": 0.0})
            e["uses"] = e.get("uses", 0) + 1

    # Cumulative recovery by day since the original failure. Computed here so the
    # baseline and the agent are bucketed identically.
    horizon = 31
    per_day = [0.0] * horizon
    cases_day = [0] * horizon
    for c in won:
        if c.recovery_hours is None:
            continue
        d = min(int(c.recovery_hours // 24), horizon - 1)
        per_day[d] += c.amount_recovered
        cases_day[d] += 1
    cum, series = 0.0, []
    for d in range(horizon):
        cum += per_day[d]
        series.append({"day": d, "revenue": round(per_day[d], 2),
                       "cases": cases_day[d], "cumulative_revenue": round(cum, 2)})

    return {
        "strategy": strategy or (o[0].strategy if o else ""),
        "cases": len(o),
        "recovery_over_time": series,
        "revenue_at_risk": round(at_risk, 2),
        "revenue_recovered": round(recovered, 2),
        "cases_recovered": len(won),
        "passive_recoveries": sum(1 for c in won if c.passive_recovery),
        "caused_recoveries": sum(1 for c in won if not c.passive_recovery),
        "passive_revenue": round(sum(c.amount_recovered for c in won if c.passive_recovery), 2),
        "recovery_rate": round(len(won) / len(o), 4),
        "value_recovery_rate": round(recovered / at_risk, 4) if at_risk else 0.0,
        "avg_recovery_hours": round(mean(times), 2) if times else None,
        "median_recovery_hours": round(sorted(times)[len(times) // 2], 2) if times else None,
        "total_retries": sum(c.retries for c in o),
        "total_contacts": sum(c.contacts for c in o),
        "total_actions": sum(len(c.actions) for c in o),
        "cases_escalated": sum(1 for c in o if c.escalated),
        "cases_stopped": sum(1 for c in o if not c.recovered and not c.escalated),
        "policy_blocked_actions": sum(c.policy_blocked for c in o),
        "risk_actions_taken": sum(c.risk_actions for c in o),
        "total_cost": round(sum(c.cost for c in o), 2),
        "net_recovered": round(recovered - sum(c.cost for c in o), 2),
        "cost_per_100_recovered": round(sum(c.cost for c in o) * 100 / recovered, 4) if recovered else None,
        "by_category": by_cat,
        "by_failure_code": by_code,
        "by_action": by_action,
    }


def _cell(cs: list[CaseOutcome]) -> dict:
    at_risk = sum(c.amount_usd for c in cs)
    rec = sum(c.amount_recovered for c in cs)
    return {
        "cases": len(cs),
        "revenue_at_risk": round(at_risk, 2),
        "revenue_recovered": round(rec, 2),
        "cases_recovered": sum(1 for c in cs if c.recovered),
        "recovery_rate": round(sum(1 for c in cs if c.recovered) / len(cs), 4),
        "value_recovery_rate": round(rec / at_risk, 4) if at_risk else 0.0,
    }


def bootstrap_incremental(baseline_outcomes: list[CaseOutcome],
                          agent_outcomes: list[CaseOutcome],
                          n_boot: int = 2000, seed: int = 20260822) -> dict:
    """Confidence interval on the headline metric.

    Cases are resampled *paired* -- the same case index is drawn from both arms -- because
    the two strategies were run on the same population. Resampling them independently
    would inflate the variance with between-case differences that the design removes.

    This matters: recovery amounts are lognormal, so the total is driven by whether a few
    large invoices happened to land. A point estimate alone overstates what one run of
    1,842 cases can tell you.
    """
    import random

    by_id = {c.transaction_id: c for c in baseline_outcomes}
    pairs = [(by_id[a.transaction_id], a) for a in agent_outcomes if a.transaction_id in by_id]
    if not pairs:
        return {}

    rng = random.Random(seed)
    n = len(pairs)
    deltas, rate_deltas = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b_rev = a_rev = 0.0
        b_won = a_won = 0
        for i in idx:
            b, a = pairs[i]
            b_rev += b.amount_recovered
            a_rev += a.amount_recovered
            b_won += b.recovered
            a_won += a.recovered
        deltas.append(a_rev - b_rev)
        rate_deltas.append((a_won - b_won) / n)

    deltas.sort()
    rate_deltas.sort()

    def q(xs, p):
        return round(xs[min(int(p * len(xs)), len(xs) - 1)], 4)

    return {
        "n_boot": n_boot,
        "paired": True,
        "incremental_revenue": {"p05": q(deltas, 0.05), "p50": q(deltas, 0.50),
                                "p95": q(deltas, 0.95)},
        "incremental_recovery_rate": {"p05": q(rate_deltas, 0.05), "p50": q(rate_deltas, 0.50),
                                      "p95": q(rate_deltas, 0.95)},
        #: Share of resamples in which the agent beat the baseline at all.
        "share_positive": round(sum(1 for d in deltas if d > 0) / len(deltas), 4),
        "excludes_zero": bool(q(deltas, 0.05) > 0),
    }


def compare_to_control(control: dict, arm: dict) -> dict:
    """Impact against doing nothing. This is the causal number.

    `revenue_recovered` alone is not impact -- some of it would have arrived regardless.
    Subtracting the untouched control arm is what separates money the strategy *caused*
    from money it merely *observed*.
    """
    inc = arm["revenue_recovered"] - control["revenue_recovered"]
    return {
        "control_recovered": control["revenue_recovered"],
        "arm_recovered": arm["revenue_recovered"],
        "incremental_recovered_revenue": round(inc, 2),
        "control_recovery_rate": control["recovery_rate"],
        "arm_recovery_rate": arm["recovery_rate"],
        "recovery_rate_delta_pp": round(
            (arm["recovery_rate"] - control["recovery_rate"]) * 100, 2),
        "incremental_cases_recovered": arm["cases_recovered"] - control["cases_recovered"],
        "share_of_revenue_that_is_causal": round(inc / arm["revenue_recovered"], 4)
                                           if arm["revenue_recovered"] else None,
        "cost": arm["total_cost"],
        "roi_vs_control": round(inc / arm["total_cost"], 2) if arm["total_cost"] else None,
        "cost_per_100_incremental": round(arm["total_cost"] * 100 / inc, 4) if inc > 0 else None,
    }


def compare(baseline: dict, agent: dict) -> dict:
    """Incremental performance of the agent over the baseline. This is the primary
    business metric, so it is computed in exactly one place."""
    inc_rev = agent["revenue_recovered"] - baseline["revenue_recovered"]
    inc_cases = agent["cases_recovered"] - baseline["cases_recovered"]
    extra_cost = agent["total_cost"] - baseline["total_cost"]
    return {
        "baseline_recovered": baseline["revenue_recovered"],
        "agent_recovered": agent["revenue_recovered"],
        "incremental_recovered_revenue": round(inc_rev, 2),
        "recovery_uplift_pct": round(
            (agent["revenue_recovered"] / baseline["revenue_recovered"] - 1) * 100, 2
        ) if baseline["revenue_recovered"] else None,
        "baseline_recovery_rate": baseline["recovery_rate"],
        "agent_recovery_rate": agent["recovery_rate"],
        "recovery_rate_delta_pp": round(
            (agent["recovery_rate"] - baseline["recovery_rate"]) * 100, 2),
        "incremental_cases_recovered": inc_cases,
        "incremental_cost": round(extra_cost, 2),
        "incremental_roi": round(inc_rev / extra_cost, 2) if extra_cost > 0 else None,
        "baseline_retries": baseline["total_retries"],
        "agent_retries": agent["total_retries"],
        "baseline_risk_actions": baseline["risk_actions_taken"],
        "agent_risk_actions": agent["risk_actions_taken"],
        "baseline_avg_recovery_hours": baseline["avg_recovery_hours"],
        "agent_avg_recovery_hours": agent["avg_recovery_hours"],
    }
