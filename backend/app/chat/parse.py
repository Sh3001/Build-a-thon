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

from backend.app.chat.dsl import DIMENSIONS, NUMERIC, Agg, DataQuery, Filter, Op
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
    ("failure category", "failure_category"), ("failure cause", "failure_code"),
    ("failure code", "failure_code"), ("failure reason", "failure_code"),
    ("root cause", "root_cause"),
    ("amount", "amount_usd"), ("value", "amount_usd"), ("size", "amount_usd"),
    ("recovered", "amount_recovered"),
    ("customer", "customer_id"), ("cause", "failure_code"), ("code", "failure_code"),
    ("category", "failure_category"), ("status", "status"), ("outcome", "status"),
    ("action", "recommended_action"), ("stop reason", "stop_reason"),
]

AGG_WORDS: list[tuple[str, Agg]] = [
    ("how many", Agg.COUNT), ("number of", Agg.COUNT), ("count", Agg.COUNT),
    ("average", Agg.AVG), ("avg", Agg.AVG), ("mean", Agg.AVG), ("typical", Agg.AVG),
    ("total", Agg.SUM), ("sum", Agg.SUM), ("how much", Agg.SUM),
    ("largest", Agg.MAX), ("biggest", Agg.MAX), ("highest", Agg.MAX), ("max", Agg.MAX),
    ("smallest", Agg.MIN), ("lowest", Agg.MIN), ("min", Agg.MIN),
]

#: "recover" is a verb far more often than a status on this data. "total recovered by
#: category" is a revenue question about ALL cases; filtering to already-won ones dropped
#: the RISK_COMPLIANCE bucket entirely and reported per-bucket case counts for the wrong
#: population. So the word only counts as a status when it actually qualifies the noun.
STATUS_PHRASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brecovered\s+cases?\b"), "recovered"),
    (re.compile(r"\bcases?\s+(?:that\s+(?:were\s+)?)?recovered\b"), "recovered"),
    (re.compile(r"\b(?:were|was|got|are)\s+recovered\b"), "recovered"),
    (re.compile(r"\bstatus\s+(?:is\s+)?recovered\b"), "recovered"),
    (re.compile(r"\bsuccessful\s+cases?\b"), "recovered"),
    (re.compile(r"\bcases?\s+we\s+won\b"), "recovered"),
]
#: Words that are unambiguously a status wherever they appear.
STATUS_WORDS = {"escalated": "escalated", "escalation": "escalated",
                "stopped": "stopped", "exhausted": "exhausted"}

#: Natural phrasings that do not match a `FailureCode` value literally. Without these the
#: parser ignored "fraud" outright and answered about all 1,842 cases instead of the 45.
FAILURE_SYNONYMS: list[tuple[str, FailureCode]] = [
    ("high risk", FailureCode.HIGH_RISK_TRANSACTION),
    ("suspected fraud", FailureCode.SUSPECTED_FRAUD), ("fraud", FailureCode.SUSPECTED_FRAUD),
    ("compliance hold", FailureCode.COMPLIANCE_HOLD), ("compliance", FailureCode.COMPLIANCE_HOLD),
    ("dead card", FailureCode.EXPIRED_CARD), ("expired", FailureCode.EXPIRED_CARD),
    ("no funds", FailureCode.INSUFFICIENT_FUNDS), ("nsf", FailureCode.INSUFFICIENT_FUNDS),
    ("insufficient", FailureCode.INSUFFICIENT_FUNDS),
    ("timed out", FailureCode.BANK_TIMEOUT), ("timeout", FailureCode.BANK_TIMEOUT),
    ("closed account", FailureCode.CLOSED_ACCOUNT),
    ("invalid account", FailureCode.INVALID_ACCOUNT),
    ("network", FailureCode.NETWORK_ERROR),
    ("processor", FailureCode.PROCESSOR_UNAVAILABLE),
]

COMPARATORS: list[tuple[re.Pattern, Op]] = [
    (re.compile(r"\b(?:over|above|more than|greater than|exceeding|>)\s*\$?\s*[\d,]"), Op.GT),
    (re.compile(r"\b(?:at least|no less than|>=)\s*\$?\s*[\d,]"), Op.GTE),
    (re.compile(r"\b(?:under|below|less than|cheaper than|<)\s*\$?\s*[\d,]"), Op.LT),
    (re.compile(r"\b(?:at most|no more than|<=)\s*\$?\s*[\d,]"), Op.LTE),
]

#: Money the run failed to collect, versus money that was ever exposed.
LOSS_RE = re.compile(r"\b(lost|lose|loses|losing|leak\w*|uncollected|unrecovered|"
                     r"missed|forfeit\w*|written off|write[- ]?off)\b")
RISK_RE = re.compile(r"\b(at risk|exposure|exposed|owed|outstanding)\b")
MONEY_RE = re.compile(r"\b(money|revenue|dollars?|cash|worth|amount|value|\$)")
RATE_RE = re.compile(r"\b(?:recovery|success|win|hit)\s+rate\b|\brates?\b|"
                     r"\bwhat\s+(?:share|percent|percentage|proportion|fraction)\b|"
                     r"\bhow\s+often\b|\bpercentage\s+of\b")
LIST_RE = re.compile(r"\b(show me|list|top \d|top |which cases|examples?|worst|give me)\b")

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


def _field(text: str, only: set[str] | None = None) -> str | None:
    """First column named in `text`. `only` restricts the search to a role - an aggregate
    needs a measure, a GROUP BY needs a dimension, and picking the wrong kind silently
    changed "total recovered by category" into a total of the wrong column."""
    for phrase, col in FIELD_WORDS:
        if phrase in text and (only is None or col in only):
            return col
    return None


def _failure_code(q: str) -> tuple[FailureCode | None, str]:
    """Returns the code and the phrase that matched it, so the caller can blank it out."""
    for code in FailureCode:
        for form in (code.value, code.value.replace("_", " ")):
            if form in q:
                return code, form
    for phrase, code in FAILURE_SYNONYMS:
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return code, phrase
    return None, ""


def parse(question: str) -> DataQuery:
    q = question.lower().strip()
    filters: list[Filter] = []
    unresolved: list[str] = []

    # ---------------------------------------------------------------- filters
    code, matched_phrase = _failure_code(q)
    if code:
        filters.append(Filter(field="failure_code", op=Op.EQ, value=code.value))

    status = None
    for pattern, s in STATUS_PHRASES:
        if pattern.search(q):
            status = s
            break
    if status is None:
        for word, s in STATUS_WORDS.items():
            if re.search(rf"\b{word}\b", q):
                status = s
                break
    if status:
        filters.append(Filter(field="status", op=Op.EQ, value=status))

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
        col = _field(q[m.end():]) or _field(q[: m.start()]) or "amount_usd"
        if col in NUMERIC:
            filters.append(Filter(field=col, op=op, value=value))
        break

    # ------------------------------------------------- terms this table cannot express
    # Scanned against the question with any matched failure phrase blanked out, so
    # "expired card" reports a card failure rather than warning about payment method.
    residual = q.replace(matched_phrase, " ") if matched_phrase else q
    for term, what in NOT_STORED.items():
        if re.search(rf"\b{re.escape(term)}\b", residual):
            unresolved.append(f"{term} ({what} is not recorded on a case)")

    # ---------------------------------------------------------------- what to measure
    measure = "amount_lost" if LOSS_RE.search(q) else "amount_usd" if RISK_RE.search(q) else None

    explicit_count = any(w in q for w in ("how many", "number of", "count of"))
    agg: Agg | None = None
    field: str | None = None

    if RATE_RE.search(q) and not explicit_count:
        agg, field = Agg.RATE, "recovered"
    else:
        for phrase, a in AGG_WORDS:
            if phrase in q:
                agg = a
                break

    # ---------------------------------------------------------------- shape
    group_by = None
    gm = re.search(r"\b(?:grouped by|broken down by|breakdown by|by|per|for each|across)"
                   r"\s+([a-z ]{2,30})", q)
    if gm:
        group_by = _field(gm.group(1).strip(), only=DIMENSIONS)
    if group_by is None:
        wm = re.search(r"\bwhich\s+([a-z ]{2,40})", q)
        if wm:
            group_by = _field(wm.group(1).strip(), only=DIMENSIONS)
    # A grouped question already filtered to one bucket answers itself; drop the filter
    # so "which cause loses most" compares causes instead of returning a single row.
    if group_by:
        filters = [f for f in filters if f.field != group_by]

    listy = bool(LIST_RE.search(q)) and not group_by
    if listy:
        agg, field = Agg.LIST, None
    elif agg is None:
        agg = Agg.SUM if (measure or MONEY_RE.search(q)) else Agg.COUNT

    if agg in (Agg.SUM, Agg.AVG, Agg.MIN, Agg.MAX):
        field = measure or _field(q, only=NUMERIC) or "amount_recovered"

    order_by = None
    if agg is Agg.LIST:
        order_by = measure or _field(q, only=NUMERIC) or "expected_recovery"

    limit = 10
    lm = re.search(r"\b(?:top|first|show me|worst|best)\s+(\d{1,2})\b", q)
    if lm:
        limit = max(1, min(50, int(lm.group(1))))
    elif group_by:
        limit = 15

    ascending = any(w in q for w in ("lowest", "smallest", "least", "ascending", "fewest"))
    return DataQuery(agg=agg, field=field, filters=filters, group_by=group_by,
                     order_by=order_by, limit=limit, unresolved=unresolved,
                     descending=not ascending)
