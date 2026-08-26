"""Turn a `DataQuery` + `Result` into a sentence, and be honest about what was not asked.

Every figure here is read from `Result`, which came from SQL. Nothing is generated.

Two things this layer owes the reader beyond the number itself:

* **A name they recognise.** Column names are an implementation detail. "Average amount usd
  over status recovered" is a schema dump, not an answer.
* **Something to compare against.** A bare "147" tells nobody whether that is most of the
  run or a rounding error, so counts carry their share of the run and totals carry a
  per-case average.
"""
from __future__ import annotations

from backend.app.chat.dsl import Agg, DataQuery, Op, Result

MONEY_FIELDS = {"amount_usd", "amount_recovered", "amount_lost", "expected_recovery",
                "cost", "amount"}
PERCENT_FIELDS = {"recovery_probability", "risk_score"}

#: What each column is called in a sentence. Anything missing falls back to the column
#: name with underscores stripped, which is survivable but not the intent.
LABELS = {
    "amount_usd": "amount at risk", "amount_recovered": "amount recovered",
    "amount_lost": "amount still uncollected", "expected_recovery": "expected recovery",
    "recovery_probability": "recovery probability", "risk_score": "risk score",
    "recovery_hours": "hours to recovery", "retries": "retries",
    "contacts": "customer contacts", "cost": "action cost",
    "customer_id": "customer", "transaction_id": "transaction",
    "failure_code": "failure cause", "failure_category": "failure category",
    "root_cause": "root cause", "status": "outcome",
    "recommended_action": "recommended action", "stop_reason": "stop reason",
    "recovered": "recovered", "currency": "currency", "amount": "amount in local currency",
}

OP_WORDS = {Op.GT: "over", Op.GTE: "at least", Op.LT: "under", Op.LTE: "at most"}

VERBS = {Agg.SUM: "Total", Agg.AVG: "Average", Agg.MIN: "Lowest", Agg.MAX: "Highest"}


def label(field: str | None) -> str:
    if not field:
        return "cases"
    return LABELS.get(field, field.replace("_", " "))


def _fmt(field: str | None, v) -> str:
    """Format a figure the way its column should read."""
    if v is None:
        return "n/a"
    v = float(v)
    if field in MONEY_FIELDS:
        return f"${v:,.2f}"
    if field in PERCENT_FIELDS:
        return f"{v:.1%}"
    if v % 1:
        return f"{v:,.2f}"
    return f"{int(v):,}"


def _value(field: str, raw) -> str:
    """Render a filter or bucket value. Identifiers keep their underscores - they are keys
    people paste back into the search box, not prose."""
    if isinstance(raw, (int, float)) and field not in {"transaction_id", "customer_id"}:
        return _fmt(field, raw)
    text = str(raw)
    if text.startswith(("cust_", "txn_")):
        return f"`{text}`"
    return text.replace("_", " ").lower()


def _emph(field: str, raw) -> str:
    """Bold a bucket name, unless it is an identifier already set in code type - nesting
    the two markers renders the backticks literally."""
    v = _value(field, raw)
    return v if v.startswith("`") else f"**{v}**"


def _pct(part: float, whole: float) -> str:
    return "n/a" if not whole else f"{part / whole:.1%}"


def _conditions(q: DataQuery) -> str:
    bits = []
    for f in q.filters:
        name = label(f.field)
        val = _value(f.field, f.value)
        if f.op is Op.EQ:
            bits.append(f"{name} is {val}")
        elif f.op is Op.NE:
            bits.append(f"{name} is not {val}")
        elif f.op is Op.CONTAINS:
            bits.append(f"{name} contains {val}")
        else:
            bits.append(f"{name} is {OP_WORDS[f.op]} {val}")
    return " and ".join(bits)


def describe(q: DataQuery, r: Result | None = None) -> str:
    """The population the numbers actually describe - stated, so it can be checked."""
    conds = _conditions(q)
    if not conds:
        total = f"{r.grand_total:,} " if r and r.grand_total else "all "
        return f"all {total}cases in this run".replace("all all ", "all ")
    n = f"{r.total_matched:,} " if r else ""
    return f"the {n}cases where {conds}"


def render(q: DataQuery, r: Result) -> str:
    scope = describe(q, r)
    conds = _conditions(q)

    # ------------------------------------------------------------------ list of cases
    if q.agg is Agg.LIST:
        if not r.rows:
            return (f"No cases match {conds}. "
                    f"This run has {r.grand_total:,} cases in total - try widening the "
                    f"filter, or ask for a breakdown to see what is there.")
        lines = [
            f"{i}. `{x['transaction_id']}` - {_fmt('amount_usd', x['amount_usd'])} at risk, "
            f"{_value('failure_code', x['failure_code'])}, **{x['status']}**, "
            f"recovered {_fmt('amount_recovered', x['amount_recovered'] or 0)}"
            for i, x in enumerate(r.rows, 1)
        ]
        shown = len(r.rows)
        head = (f"Showing the top {shown} of {r.total_matched:,} "
                f"{'matching cases' if conds else 'cases in this run'}, "
                f"ranked by {label(q.order_by or 'expected_recovery')}")
        if conds:
            head += f" ({conds})"
        return head + ":\n\n" + "\n".join(lines)

    # ------------------------------------------------------------------ breakdown
    if q.group_by:
        if not r.rows:
            return f"No cases match {conds}, so there is nothing to break down."
        vals = [float(x["value"]) if x["value"] is not None else 0.0 for x in r.rows]
        # `r.scalar` is the aggregate over all buckets, including any beyond the LIMIT.
        total = r.scalar if r.scalar else sum(vals)
        share_ok = q.agg in (Agg.COUNT, Agg.SUM) and total > 0
        lines = []
        for x, v in zip(r.rows, vals):
            bucket = _emph(q.group_by, x["bucket"])
            if q.agg is Agg.COUNT:
                shown = f"{int(v):,} cases"          # the value *is* the case count
            elif q.agg is Agg.RATE:
                shown = f"{v:.1%} of {x['n']:,} cases"
            else:
                shown = f"{_fmt(q.field, v)} across {x['n']:,} cases"
            lines.append(f"- {bucket} - {shown}"
                         + (f" ({_pct(v, total)})" if share_ok else ""))

        measure = ("Number of cases" if q.agg is Agg.COUNT
                   else "Recovery rate" if q.agg is Agg.RATE
                   else f"{VERBS[q.agg]} {label(q.field)}")
        head = f"{measure} by {label(q.group_by)}, across {scope}."
        if q.descending and r.rows:
            top = _emph(q.group_by, r.rows[0]["bucket"])
            best = (f"{int(vals[0]):,} cases" if q.agg is Agg.COUNT
                    else f"{vals[0]:.1%}" if q.agg is Agg.RATE
                    else _fmt(q.field, vals[0]))
            head += f" The largest is {top} at {best}."
        if len(r.rows) >= q.limit:
            head += f" (Top {q.limit} shown.)"
        return head + "\n\n" + "\n".join(lines)

    # ------------------------------------------------------------------ single figure
    if q.agg is Agg.COUNT:
        n = int(r.scalar or 0)
        if not conds:
            return (f"This run contains **{n:,} cases** in total - every payment that "
                    f"failed and entered recovery.")
        if n == 0:
            return (f"**No cases** match {conds}, out of {r.grand_total:,} in this run.")
        return (f"**{n:,} cases** match {conds} - "
                f"{_pct(n, r.grand_total)} of the {r.grand_total:,} cases in this run.")

    if q.agg is Agg.RATE:
        if not r.total_matched:
            return f"No cases match {conds}, so there is no rate to report."
        share = r.scalar or 0.0
        won = round(share * r.total_matched)
        return (f"**{share:.1%}** of {scope} were recovered - "
                f"{won:,} of {r.total_matched:,}.")

    if r.scalar is None or not r.total_matched:
        return (f"No cases match {conds}, so {label(q.field)} cannot be computed. "
                f"This run has {r.grand_total:,} cases in total.")

    figure = _fmt(q.field, r.scalar)
    body = f"{VERBS[q.agg]} {label(q.field)} across {scope}: **{figure}**."
    if q.agg is Agg.SUM and r.total_matched > 1:
        body += f" That is {_fmt(q.field, r.scalar / r.total_matched)} per case."
    return body


def caveat(q: DataQuery) -> str:
    if not q.unresolved:
        return ""
    terms = "; ".join(q.unresolved)
    return (f"\n\n⚠ **Heads up:** I could not narrow the answer by {terms}. "
            f"The figures above cover {describe(q)} instead, so treat them as the wider "
            f"picture rather than the specific group you asked about.")
