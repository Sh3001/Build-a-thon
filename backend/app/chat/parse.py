"""Natural language -> `DataQuery`, without dropping anything on the floor.

The contract that matters: a term the parser recognises as *meaningful but unbindable*
goes into `unresolved`, and the caller must surface it. The previous keyword router had no
such notion, so an unmatched filter simply vanished and the answer described a broader
population than the question asked about.

`parse` is deterministic. An LLM router may replace it, but it must produce the same
`DataQuery` type and is held to the same schema validation.
"""
from __future__ import annotations

import re

from backend.app.chat.dsl import FIELDS, NUMERIC, Agg, DataQuery, Filter, Op
from backend.app.models.enums import FailureCode

MONEY = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(k|m)?\b", re.I)

#: Words that name a column. Longest phrases first so "amount recovered" beats "amount".
FIELD_WORDS: list[tuple[str, str]] = [
    ("amount recovered", "amount_recovered"), ("recovered amount", "amount_recovered"),
    ("expected recovery", "expected_recovery"), ("recovery probability", "recovery_probability"),
    ("probability", "recovery_probability"), ("risk score", "risk_score"),
    ("recovery hours", "recovery_hours"), ("hours", "recovery_hours"),
    ("retries", "retries"), ("retry", "retries"), ("attempts", "retries"),
    ("contacts", "contacts"), ("touches", "contacts"),
    ("cost", "cost"), ("spend", "cost"),
    ("amount", "amount_usd"), ("value", "amount_usd"), ("size", "amount_usd"),
    ("recovered", "amount_recovered"),
    ("customer", "customer_id"), ("failure code", "failure_code"),
    ("failure cause", "failure_code"), ("cause", "failure_code"),
    ("category", "failure_category"), ("status", "status"),
    ("action", "recommended_action"), ("stop reason", "stop_reason"),
]

AGG_WORDS: list[tuple[str, Agg]] = [
    ("how many", Agg.COUNT), ("number of", Agg.COUNT), ("count", Agg.COUNT),
    ("average", Agg.AVG), ("avg", Agg.AVG), ("mean", Agg.AVG), ("typical", Agg.AVG),
    ("total", Agg.SUM), ("sum", Agg.SUM), ("how much", Agg.SUM),
    ("largest", Agg.MAX), ("biggest", Agg.MAX), ("highest", Agg.MAX), ("max", Agg.MAX),
    ("smallest", Agg.MIN), ("lowest", Agg.MIN), ("min", Agg.MIN),
]

STATUSES = ("recovered", "escalated", "stopped", "exhausted")

COMPARATORS: list[tuple[re.Pattern, Op]] = [
    (re.compile(r"\b(?:over|above|more than|greater than|exceeding|>)\s*\$?\s*[\d,]"), Op.GT),
    (re.compile(r"\b(?:at least|no less than|>=)\s*\$?\s*[\d,]"), Op.GTE),
    (re.compile(r"\b(?:under|below|less than|cheaper than|<)\s*\$?\s*[\d,]"), Op.LT),
    (re.compile(r"\b(?:at most|no more than|<=)\s*\$?\s*[\d,]"), Op.LTE),
]

#: Things people plausibly ask about that this table does not carry. Naming them beats
#: answering a narrower question as though it were the one asked.
NOT_STORED = {
    "whatsapp": "channel", "sms": "channel", "email": "channel", "in app": "channel",
    "channel": "channel",
    "enterprise": "customer segment", "smb": "customer segment",
    "consumer": "customer segment", "segment": "customer segment",
    "country": "country", "card": "payment method", "upi": "payment method",
    "netbanking": "payment method", "wallet": "payment method", "ach": "payment method",
    "payment method": "payment method", "tenure": "customer tenure",
}


def _number(text: str) -> float | None:
    m = MONEY.search(text)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    if m.group(2):
        v *= 1_000 if m.group(2).lower() == "k" else 1_000_000
    return v


def _field(text: str) -> str | None:
    for phrase, col in FIELD_WORDS:
        if phrase in text:
            return col
    return None


def parse(question: str) -> DataQuery:
    q = question.lower().strip()
    filters: list[Filter] = []
    unresolved: list[str] = []

    # ---------------------------------------------------------------- unstorable terms
    for term, what in NOT_STORED.items():
        if re.search(rf"\b{re.escape(term)}\b", q):
            unresolved.append(f"{term} ({what} is not stored on the case record)")

    # ---------------------------------------------------------------- filters
    for code in FailureCode:
        if code.value in q or code.value.replace("_", " ") in q:
            filters.append(Filter(field="failure_code", op=Op.EQ, value=code.value))
            break
    for status in STATUSES:
        if re.search(rf"\b{status}\b", q):
            filters.append(Filter(field="status", op=Op.EQ, value=status))
            break

    cust = re.search(r"\b(cust_[0-9a-z]+)\b", question, re.I)
    if cust:
        filters.append(Filter(field="customer_id", op=Op.EQ, value=cust.group(1)))
    txn = re.search(r"\b(txn_[0-9a-z]+)\b", question, re.I)
    if txn:
        filters.append(Filter(field="transaction_id", op=Op.EQ, value=txn.group(1)))

    # numeric comparison, e.g. "over $5,000", "more than 3 retries"
    for pattern, op in COMPARATORS:
        m = pattern.search(q)
        if not m:
            continue
        value = _number(q[m.start():])
        if value is None:
            continue
        tail = q[m.end():]
        col = _field(tail) or _field(q[: m.start()]) or "amount_usd"
        if col in NUMERIC:
            filters.append(Filter(field=col, op=op, value=value))
        break

    # ---------------------------------------------------------------- aggregate
    agg = Agg.COUNT
    for phrase, a in AGG_WORDS:
        if phrase in q:
            agg = a
            break

    field = None
    if agg in (Agg.SUM, Agg.AVG, Agg.MIN, Agg.MAX):
        field = _field(q) or "amount_recovered"
        if field not in NUMERIC:
            field = "amount_usd"

    # ---------------------------------------------------------------- shape
    group_by = None
    gm = re.search(r"\b(?:by|per|grouped by|breakdown by)\s+([a-z ]+)", q)
    if gm:
        cand = _field(gm.group(1).strip())
        if cand:
            group_by = cand
    if re.search(r"\bwhich\s+(customers?|causes?|codes?|categor)", q) and not group_by:
        group_by = "customer_id" if "customer" in q else "failure_code"
        if agg is Agg.COUNT and "most" in q:
            pass                     # COUNT per bucket is exactly right

    listy = any(w in q for w in ("show me", "list", "top", "which cases", "examples"))
    if listy and not group_by:
        agg = Agg.LIST
        field = None

    order_by = None
    if agg is Agg.LIST:
        order_by = _field(q) if _field(q) in NUMERIC else "expected_recovery"

    limit = 10
    lm = re.search(r"\b(?:top|first|show me)\s+(\d{1,2})\b", q)
    if lm:
        limit = max(1, min(50, int(lm.group(1))))

    return DataQuery(agg=agg, field=field, filters=filters, group_by=group_by,
                     order_by=order_by, limit=limit, unresolved=unresolved,
                     descending="lowest" not in q and "smallest" not in q
                                and "ascending" not in q)
