"""Map a natural-language question onto a typed `Query`.

Two routers, same output type:

`KeywordRouter`  - deterministic, instant, no model. Handles the phrasings people actually
                   use on this data. It is the default, not a degraded mode: on measured
                   accuracy the local models available here are worse than rules at every
                   task tried so far, and a question router is not obviously different.

`LLMRouter`      - wraps whichever planner is configured. Only ever produces a `Query`;
                   it cannot produce prose, numbers, or an intent outside the enum. On any
                   parse failure it hands back to the keyword router rather than guessing.

Neither router ever sees a figure or writes an answer. That belongs to `ChatEngine`.
"""
from __future__ import annotations

import json
import re

from backend.app.chat.query import Intent, Query
from backend.app.models.enums import FailureCode

TXN_RE = re.compile(r"\b(txn_[0-9a-z]+)\b", re.I)
NUM_RE = re.compile(r"\b(\d{1,2})\b")

#: Words that are unambiguously a *status*, whatever the surrounding sentence.
STATUS_WORDS = {
    "escalated": "escalated", "escalation": "escalated",
    "stopped": "stopped",
    "exhausted": "exhausted",
}

#: "recover" is a verb far more often than a status on this data -- "how much did we
#: recover from expired cards" is a revenue question about ALL such cases, not a filter to
#: the ones already won. Filtering there reported a 100% recovery rate, which is true of
#: the filtered set and badly misleading as an answer. So these only count as a status when
#: they actually qualify the noun.
STATUS_PHRASES = [
    (re.compile(r"\brecovered\s+cases?\b"), "recovered"),
    (re.compile(r"\bcases?\s+(?:that\s+(?:were\s+)?)?recovered\b"), "recovered"),
    (re.compile(r"\bstatus\s+(?:is\s+)?recovered\b"), "recovered"),
    (re.compile(r"\bsuccessful\s+cases?\b"), "recovered"),
    (re.compile(r"\bcases?\s+we\s+won\b"), "recovered"),
]

SYSTEM = """You convert a question about a payment-recovery dataset into a query object.
You never answer the question and never state a number: another component runs the query
and writes the answer.

Choose exactly one intent:
- case_detail        one specific transaction, when an id is given
- case_trace         the decision/audit chain for one transaction
- count_cases        "how many ..." questions
- sum_revenue        "how much ..." money questions
- top_cases          "top / largest / biggest N ..." questions
- arm_comparison     control vs baseline vs RecoverAI, or "did it work"
- failure_breakdown  totals grouped by failure cause or category
- policy_activity    rules fired, blocked or rejected actions, the policy engine
- dlq_status         bounced addresses, quarantined channels, the dead letter queue
- unknown            anything else

Fill only the parameters the question implies. Leave the rest unset."""


class KeywordRouter:
    """Deterministic intent matching. Order matters: the most specific patterns first."""

    name = "keywords"

    def route(self, question: str) -> Query:
        q = question.lower().strip()

        txn = TXN_RE.search(question)
        if txn:
            trace_words = ("trace", "audit", "chain", "why", "decision", "happened", "steps")
            intent = Intent.CASE_TRACE if any(w in q for w in trace_words) else Intent.CASE_DETAIL
            return Query(intent=intent, transaction_id=txn.group(1))

        if any(w in q for w in ("bounce", "quarantin", "dead letter", "dlq", "undeliverable")):
            return Query(intent=Intent.DLQ_STATUS, limit=self._limit(q, 10))

        if any(w in q for w in ("rule", "policy", "blocked", "rejected", "refused", "guardrail")):
            return Query(intent=Intent.POLICY_ACTIVITY)

        if any(w in q for w in ("baseline", "control", "compare", "comparison", "arm",
                                "did it work", "vs", "versus", "uplift", "caused")):
            return Query(intent=Intent.ARM_COMPARISON)

        code = self._failure_code(q)
        status = self._status(q)

        if any(w in q for w in ("breakdown", "by cause", "by failure", "by category",
                                "which cause", "per cause")):
            group = "failure_category" if "categor" in q else "failure_code"
            return Query(intent=Intent.FAILURE_BREAKDOWN, group_by=group,
                         limit=self._limit(q, 8))

        if any(w in q for w in ("top", "biggest", "largest", "highest", "worst", "show me")):
            order = "amount_usd" if any(w in q for w in ("biggest", "largest", "amount")) \
                else "amount_recovered" if "recovered" in q else "expected_recovery"
            return Query(intent=Intent.TOP_CASES, limit=self._limit(q, 5),
                         order_by=order, status=status, failure_code=code)

        # An explicit "how many" wins over money words in the same sentence: "how many
        # recovered cases" is a count, even though "recovered" also reads as revenue.
        if any(w in q for w in ("how many", "count of", "number of")):
            return Query(intent=Intent.COUNT_CASES, status=status, failure_code=code)

        if any(w in q for w in ("how much", "revenue", "money", "recovered", "value",
                                "at risk", "total")):
            return Query(intent=Intent.SUM_REVENUE, status=status, failure_code=code)

        if any(w in q for w in ("count", "cases")):
            return Query(intent=Intent.COUNT_CASES, status=status, failure_code=code)

        if code or status:
            return Query(intent=Intent.SUM_REVENUE, status=status, failure_code=code)
        return Query(intent=Intent.UNKNOWN)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _limit(q: str, default: int) -> int:
        m = NUM_RE.search(q)
        if not m:
            return default
        return max(1, min(25, int(m.group(1))))

    @staticmethod
    def _status(q: str) -> str | None:
        for pattern, status in STATUS_PHRASES:
            if pattern.search(q):
                return status
        for word, status in STATUS_WORDS.items():
            if word in q:
                return status
        return None

    @staticmethod
    def _failure_code(q: str) -> FailureCode | None:
        for code in FailureCode:
            if code.value.replace("_", " ") in q or code.value in q:
                return code
        # A few natural phrasings that do not match an enum value literally.
        for phrase, code in (("dead card", FailureCode.EXPIRED_CARD),
                             ("expired", FailureCode.EXPIRED_CARD),
                             ("no funds", FailureCode.INSUFFICIENT_FUNDS),
                             ("nsf", FailureCode.INSUFFICIENT_FUNDS),
                             ("fraud", FailureCode.SUSPECTED_FRAUD),
                             ("timeout", FailureCode.BANK_TIMEOUT)):
            if phrase in q:
                return code
        return None


class LLMRouter:
    """Uses a planner's transport to classify. Falls back to keywords on any failure."""

    name = "llm"

    def __init__(self, planner, fallback: KeywordRouter | None = None):
        self.planner = planner
        self.fallback = fallback or KeywordRouter()

    def route(self, question: str) -> Query:
        parsed = self._classify(question)
        if parsed is None:
            return self.fallback.route(question)
        return parsed

    def _classify(self, question: str) -> Query | None:
        p = self.planner
        if p is None or not getattr(p, "enabled", False):
            return None
        try:
            from backend.app.agents.ollama import OllamaPlanner
            if isinstance(p, OllamaPlanner):
                return self._via_ollama(p, question)
            return self._via_anthropic(p, question)
        except Exception:
            return None

    def _via_ollama(self, p, question: str) -> Query | None:
        import httpx
        try:
            r = httpx.post(f"{p.host}/api/chat", timeout=p.timeout, json={
                "model": p.fast_model, "stream": False,
                "format": Query.model_json_schema(),
                "options": {"temperature": 0, "num_predict": 300},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": question}],
            })
            r.raise_for_status()
            return Query.model_validate(json.loads(r.json()["message"]["content"]))
        except Exception:
            return None

    def _via_anthropic(self, p, question: str) -> Query | None:
        try:
            resp = p.client.messages.parse(
                model=p.fast_model, max_tokens=300, system=SYSTEM,
                messages=[{"role": "user", "content": question}],
                response_format=Query,
            )
            return getattr(resp, "parsed", None)
        except Exception:
            return None


def build_router(mode: str = "keywords"):
    """`keywords` is the default. `llm` adds a model in front, keywords behind it."""
    kw = KeywordRouter()
    if mode != "llm":
        return kw
    from backend.app.agents.llm import build_planner
    planner = build_planner("auto")
    if planner is None:
        try:
            from backend.app.agents.ollama import OllamaPlanner
            local = OllamaPlanner()
            planner = local if local.enabled else None
        except Exception:
            planner = None
    return LLMRouter(planner, kw) if planner is not None else kw
