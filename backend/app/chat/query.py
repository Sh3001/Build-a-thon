"""The closed set of questions the assistant can answer, and the code that answers them.

The design mirrors the agent's: **the model proposes, deterministic code disposes.** A
language model maps a question onto one of these intents and fills in typed parameters; it
never computes, formats, or repeats a figure. Every number in every answer comes from SQL
run here.

That is the whole point. A chatbot that generates numbers will eventually generate a wrong
one, and a wrong number stated confidently over financial data is worse than no chatbot.
Here the model's only power is choosing *which* query runs - an intent outside this enum
fails to parse, exactly as an action outside `InterventionType` does.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.app.models.enums import FailureCode


class Intent(str, Enum):
    CASE_DETAIL = "case_detail"
    CASE_TRACE = "case_trace"
    COUNT_CASES = "count_cases"
    SUM_REVENUE = "sum_revenue"
    TOP_CASES = "top_cases"
    ARM_COMPARISON = "arm_comparison"
    FAILURE_BREAKDOWN = "failure_breakdown"
    POLICY_ACTIVITY = "policy_activity"
    DLQ_STATUS = "dlq_status"
    UNKNOWN = "unknown"


class Query(BaseModel):
    """What the router must produce. Anything outside this shape is refused."""

    intent: Intent
    transaction_id: str | None = None
    status: str | None = Field(default=None, description="recovered|escalated|stopped|exhausted")
    failure_code: FailureCode | None = None
    limit: int = Field(default=5, ge=1, le=25)
    order_by: str = Field(default="expected_recovery")
    group_by: str = Field(default="failure_code", description="failure_code|failure_category")


class Answer(BaseModel):
    text: str                       #: rendered from `data`, never from a model
    intent: Intent
    query: dict
    data: dict = Field(default_factory=dict)
    case_ids: list[str] = Field(default_factory=list)   #: deep-linkable
    source: str = "deterministic"   #: which router picked the intent


def _money(v: float | None) -> str:
    return "-" if v is None else f"${v:,.2f}"


class ChatEngine:
    """Runs a resolved `Query` against the stores. No model is involved past this point."""

    def __init__(self, cases, audit=None, dlq=None):
        self.cases, self.audit, self.dlq = cases, audit, dlq

    # ------------------------------------------------------------------ entry
    def run(self, q: Query, source: str = "deterministic") -> Answer:
        fn = {
            Intent.CASE_DETAIL: self._case_detail,
            Intent.CASE_TRACE: self._case_trace,
            Intent.COUNT_CASES: self._count_cases,
            Intent.SUM_REVENUE: self._sum_revenue,
            Intent.TOP_CASES: self._top_cases,
            Intent.ARM_COMPARISON: self._arm_comparison,
            Intent.FAILURE_BREAKDOWN: self._failure_breakdown,
            Intent.POLICY_ACTIVITY: self._policy_activity,
            Intent.DLQ_STATUS: self._dlq_status,
        }.get(q.intent, self._unknown)
        ans = fn(q)
        ans.source = source
        ans.query = q.model_dump(mode="json", exclude_none=True)
        return ans

    # ------------------------------------------------------------------ intents
    def _unknown(self, q: Query) -> Answer:
        return Answer(
            intent=Intent.UNKNOWN, query={},
            text=("I can answer questions about cases, recovered revenue, failure causes, "
                  "policy decisions, the dead letter queue, and the comparison between the "
                  "control, baseline and RecoverAI arms. Try: \"how much did we recover "
                  "from expired cards?\" or \"show me the top 5 escalated cases\"."))

    def _case_detail(self, q: Query) -> Answer:
        if not q.transaction_id:
            return Answer(intent=q.intent, query={}, text="Which transaction id?")
        row = self.cases.get_case(q.transaction_id)
        if row is None:
            return Answer(intent=q.intent, query={},
                          text=f"No case {q.transaction_id} in this run.")
        base = self.cases.get_case(q.transaction_id, "baseline")
        text = (
            f"{row['transaction_id']} - {str(row['failure_code']).replace('_', ' ')}, "
            f"{_money(row['amount_usd'])} at risk. Status **{row['status']}**, recovered "
            f"{_money(row['amount_recovered'])} after {row['retries']} retries and "
            f"{row['contacts']} contacts. Recommended action was "
            f"`{row['recommended_action'] or '-'}`."
        )
        if base:
            text += (f" Under the baseline strategy the same case ended **{base['status']}** "
                     f"with {_money(base['amount_recovered'])} recovered on "
                     f"{base['retries']} retries.")
        return Answer(intent=q.intent, query={}, text=text,
                      data={"case": row, "baseline": base},
                      case_ids=[row["transaction_id"]])

    def _case_trace(self, q: Query) -> Answer:
        if not q.transaction_id or self.audit is None:
            return Answer(intent=q.intent, query={}, text="Which transaction id?")
        rows = self.audit.timeline(q.transaction_id)
        if not rows:
            return Answer(intent=q.intent, query={},
                          text=f"No audit trail for {q.transaction_id}.")
        blocked = [r for r in rows if r["policy_result"] == "reject"]
        # Withheld is not refused, and reporting it as such would tell an operator a case
        # was dropped when it is in fact sitting in their own queue awaiting them.
        withheld = [r for r in rows if r["policy_result"] == "human_review"]
        acted = [r for r in rows if r["action"] and r["policy_result"] == "approve"]
        text = (f"{q.transaction_id} has {len(rows)} audit events: {len(acted)} approved "
                f"actions and {len(blocked)} refused by the policy engine.")
        if withheld:
            text += (f" {len(withheld)} action(s) were withheld for human approval rather "
                     f"than refused.")
        if blocked or withheld:
            rules = sorted({r["rules_fired"] for r in [*blocked, *withheld]
                            if r["rules_fired"]})
            text += f" Rules that fired: {', '.join(rules)}."
        return Answer(intent=q.intent, query={}, text=text,
                      data={"events": len(rows), "approved": len(acted),
                            "rejected": len(blocked), "human_review": len(withheld)},
                      case_ids=[q.transaction_id])

    def _count_cases(self, q: Query) -> Answer:
        rows = self._filtered(q, limit=2000)
        bits = []
        if q.status:
            bits.append(f"**{q.status}**")
        if q.failure_code:
            bits.append(f"with cause **{q.failure_code.value.replace('_', ' ')}**")
        where = " ".join(bits) or "in total"
        total_at_risk = sum(r["amount_usd"] or 0 for r in rows)
        return Answer(intent=q.intent, query={},
                      text=f"**{len(rows):,}** cases {where}, carrying "
                           f"{_money(total_at_risk)} at risk.",
                      data={"count": len(rows), "amount_at_risk": round(total_at_risk, 2)},
                      case_ids=[r["transaction_id"] for r in rows[:10]])

    def _sum_revenue(self, q: Query) -> Answer:
        rows = self._filtered(q, limit=2000)
        recovered = sum(r["amount_recovered"] or 0 for r in rows)
        at_risk = sum(r["amount_usd"] or 0 for r in rows)
        won = sum(1 for r in rows if (r["amount_recovered"] or 0) > 0)
        scope = []
        if q.failure_code:
            scope.append(q.failure_code.value.replace("_", " "))
        if q.status:
            scope.append(q.status)
        label = " / ".join(scope) or "all cases"
        rate = recovered / at_risk if at_risk else 0.0
        return Answer(
            intent=q.intent, query={},
            text=(f"On {label}: **{_money(recovered)}** recovered out of "
                  f"{_money(at_risk)} at risk ({rate:.1%} by value), across {won} of "
                  f"{len(rows):,} cases."),
            data={"recovered": round(recovered, 2), "at_risk": round(at_risk, 2),
                  "cases": len(rows), "cases_won": won, "value_rate": round(rate, 4)},
            case_ids=[r["transaction_id"] for r in rows[:10]])

    def _top_cases(self, q: Query) -> Answer:
        rows = self._filtered(q, limit=q.limit, order_by=q.order_by)
        if not rows:
            return Answer(intent=q.intent, query={}, text="No cases match that.")
        lines = [
            f"{i}. `{r['transaction_id']}` - {_money(r['amount_usd'])} at risk, "
            f"{str(r['failure_code']).replace('_', ' ')}, **{r['status']}**, "
            f"recovered {_money(r['amount_recovered'])}"
            for i, r in enumerate(rows, 1)
        ]
        return Answer(intent=q.intent, query={},
                      text=f"Top {len(rows)} by {q.order_by.replace('_', ' ')}:\n\n"
                           + "\n".join(lines),
                      data={"rows": rows},
                      case_ids=[r["transaction_id"] for r in rows])

    def _arm_comparison(self, q: Query) -> Answer:
        agent = self.cases.get_run("recoverai") or {}
        base = self.cases.get_run("baseline") or {}
        ctrl = self.cases.get_run("control") or {}
        vc = (self.cases.get_run("vs_control") or {}).get("recoverai", {})
        if not agent:
            return Answer(intent=q.intent, query={}, text="No completed run to compare.")
        return Answer(
            intent=q.intent, query={},
            text=(f"Across the same {agent.get('cases', 0):,} cases - control "
                  f"{_money(ctrl.get('revenue_recovered'))}, baseline "
                  f"{_money(base.get('revenue_recovered'))}, RecoverAI "
                  f"**{_money(agent.get('revenue_recovered'))}**. "
                  f"Net of what the untouched control arm collected anyway, RecoverAI "
                  f"caused **{_money(vc.get('incremental_recovered_revenue'))}** "
                  f"({vc.get('share_of_revenue_that_is_causal', 0):.1%} of its gross), on "
                  f"{agent.get('total_retries', 0):,} retries against the baseline's "
                  f"{base.get('total_retries', 0):,}."),
            data={"control": ctrl.get("revenue_recovered"),
                  "baseline": base.get("revenue_recovered"),
                  "recoverai": agent.get("revenue_recovered"),
                  "caused": vc.get("incremental_recovered_revenue")})

    def _failure_breakdown(self, q: Query) -> Answer:
        agent = self.cases.get_run("recoverai") or {}
        key = "by_category" if q.group_by == "failure_category" else "by_failure_code"
        table = agent.get(key) or {}
        if not table:
            return Answer(intent=q.intent, query={}, text="No breakdown in this run.")
        rows = sorted(table.items(), key=lambda kv: -(kv[1].get("revenue_recovered") or 0))
        lines = [f"- **{k.replace('_', ' ')}** - {_money(v.get('revenue_recovered'))} "
                 f"recovered of {_money(v.get('revenue_at_risk'))} at risk "
                 f"({v.get('recovery_rate', 0):.1%} of {v.get('cases', 0)} cases)"
                 for k, v in rows[: q.limit]]
        return Answer(intent=q.intent, query={},
                      text=f"By {q.group_by.replace('_', ' ')}:\n\n" + "\n".join(lines),
                      data={"rows": dict(rows)})

    def _policy_activity(self, q: Query) -> Answer:
        if self.audit is None:
            return Answer(intent=q.intent, query={}, text="No audit store available.")
        verdicts = self.audit.decision_counts()
        rules = self.audit.rule_counts()
        top = list(rules.items())[:6]
        total = sum(verdicts.values())
        return Answer(
            intent=q.intent, query={},
            text=(f"The policy engine issued {total:,} verdicts: "
                  f"{verdicts.get('approve', 0):,} approved, "
                  f"{verdicts.get('modify', 0):,} modified, "
                  f"{verdicts.get('human_review', 0):,} withheld for human approval, "
                  f"**{verdicts.get('reject', 0):,} rejected**. Only the first two "
                  f"permit execution. Most-fired rules: "
                  + ", ".join(f"`{k}` ({v})" for k, v in top) + "."),
            data={"verdicts": verdicts, "rules": dict(top)})

    def _dlq_status(self, q: Query) -> Answer:
        if self.dlq is None:
            return Answer(intent=q.intent, query={},
                          text="No dead letter queue for this run.")
        s = self.dlq.stats()
        entries = self.dlq.entries(limit=q.limit)
        text = (f"**{s['quarantined']}** customer/channel pairs are quarantined after "
                f"{s['threshold']} consecutive hard bounces, out of {s['tracked_pairs']} "
                f"pairs that have bounced at least once ({s['total_failures']} failures "
                f"in total).")
        if entries:
            text += "\n\n" + "\n".join(
                f"- `{e['customer_id']}` on {e['channel']} - {e['failures']} failures"
                for e in entries)
        return Answer(intent=q.intent, query={}, text=text, data={"stats": s})

    # ------------------------------------------------------------------ helper
    def _filtered(self, q: Query, limit: int, order_by: str = "expected_recovery") -> list[dict]:
        return self.cases.queue(
            "recoverai", limit=limit,
            status=q.status or None,
            failure_code=q.failure_code.value if q.failure_code else None,
            order_by=order_by)
