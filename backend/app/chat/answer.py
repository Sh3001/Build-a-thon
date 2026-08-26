"""Turn a `DataQuery` + `Result` into a sentence, and be honest about what was not asked.

Every figure here is read from `Result`, which came from SQL. Nothing is generated.
"""
from __future__ import annotations

from backend.app.chat.dsl import Agg, DataQuery, Op, Result

MONEY_FIELDS = {"amount_usd", "amount_recovered", "expected_recovery", "cost", "amount"}
OP_WORDS = {Op.GT: "over", Op.GTE: "at least", Op.LT: "under", Op.LTE: "at most",
            Op.EQ: "", Op.NE: "not", Op.CONTAINS: "containing"}


def _fmt(field: str | None, v: float | None) -> str:
    if v is None:
        return "-"
    if field in MONEY_FIELDS:
        return f"${v:,.2f}"
    if field in {"recovery_probability", "risk_score"}:
        return f"{v:.1%}"
    return f"{v:,.2f}".rstrip("0").rstrip(".") if v % 1 else f"{int(v):,}"


def describe(q: DataQuery) -> str:
    """The population the numbers actually describe -- stated, so it can be checked."""
    if not q.filters:
        return "all cases"
    bits = []
    for f in q.filters:
        word = OP_WORDS[f.op]
        name = f.field.replace("_", " ")
        val = _fmt(f.field, f.value) if isinstance(f.value, (int, float)) else f.value
        bits.append(f"{name} {word} {val}".replace("  ", " ").strip()
                    if word else f"{name} {val}")
    return " and ".join(bits)


def render(q: DataQuery, r: Result) -> str:
    scope = describe(q)

    if q.agg is Agg.LIST:
        if not r.rows:
            return f"No cases match {scope}."
        lines = [f"{i}. `{x['transaction_id']}` - ${x['amount_usd']:,.2f} at risk, "
                 f"{str(x['failure_code']).replace('_', ' ')}, **{x['status']}**, "
                 f"recovered ${x['amount_recovered'] or 0:,.2f}"
                 for i, x in enumerate(r.rows, 1)]
        head = (f"{len(r.rows)} of {r.total_matched:,} cases matching {scope}"
                f", by {(q.order_by or 'expected recovery').replace('_', ' ')}:")
        return head + "\n\n" + "\n".join(lines)

    if q.group_by:
        if not r.rows:
            return f"No cases match {scope}."
        lines = []
        for x in r.rows:
            bucket = str(x["bucket"]).replace("_", " ")
            value = (f"{int(x['value']):,}" if q.agg is Agg.COUNT
                     else _fmt(q.field, x["value"]))
            lines.append(f"- **{bucket}** - {value} ({x['n']:,} cases)")
        label = "Cases" if q.agg is Agg.COUNT else f"{q.agg.value.title()} {(q.field or '').replace('_', ' ')}"
        return (f"{label} by {q.group_by.replace('_', ' ')}, over {scope} "
                f"({r.total_matched:,} cases):\n\n" + "\n".join(lines))

    if q.agg is Agg.COUNT:
        return f"**{int(r.scalar or 0):,}** cases match {scope}."

    verb = {Agg.SUM: "Total", Agg.AVG: "Average", Agg.MIN: "Minimum",
            Agg.MAX: "Maximum"}[q.agg]
    field = (q.field or "").replace("_", " ")
    return (f"{verb} {field} over {scope}: **{_fmt(q.field, r.scalar)}** "
            f"({r.total_matched:,} cases).")


def caveat(q: DataQuery) -> str:
    if not q.unresolved:
        return ""
    return ("\n\n⚠ I could not apply: " + "; ".join(q.unresolved)
            + ". The figures above therefore describe " + describe(q)
            + " - not the narrower group you asked about.")
