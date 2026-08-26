"""Phase 6 -- the bounded LLM planner.

The model reasons; it never acts. Concretely:

* It returns **data** -- a `LLMDiagnosis` or an `LLMAction` -- parsed into a Pydantic
  schema. It is handed no tools and has no execution path.
* Its chosen action is a value from a closed enum. An invented action name fails schema
  validation here, and would be refused again by `R-ALLOWLIST` at the gate.
* Whatever it returns still goes through the policy engine like any other proposal.
* **Every failure degrades to the deterministic planner** -- missing key, API error,
  refusal, malformed output, unparseable action. Returning `None` means "use the rules",
  so a demo is never one timeout away from a blank screen.

Cost is metered from real token usage and charged to the case, so inference shows up in
the ROI figure instead of being quietly excluded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from backend.app.config import (
    ANTHROPIC_MODEL, ANTHROPIC_MODEL_FAST, LLM_ESCALATE_EV_USD,
)
from backend.app.agents.redaction import scrub
from backend.app.models.enums import Channel, FailureCode, InterventionType
from backend.app.models.schemas import AgentState, Diagnosis, ProposedAction, Transaction

#: USD per million tokens. Used only to meter spend into the ROI figure.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

SYSTEM = """You are the reasoning layer of an automated revenue-recovery system for failed \
payments. You do not execute anything: you return structured judgements that a \
deterministic policy engine then validates and may reject or rewrite.

Facts you must respect:
- expired_card and invalid_payment_method CANNOT be recovered by retrying. The instrument \
is dead; it must be replaced first.
- closed_account and invalid_account CANNOT be debited at all. Only a customer-initiated \
payment can work.
- suspected_fraud, high_risk_transaction and compliance_hold are NEVER automated. \
Escalate to a human.
- Transient faults (bank_timeout, network_error, processor_unavailable) decay quickly: \
retry early or not at all.
- insufficient_funds behaves in the opposite way: the account tends to refill around \
payday, so a retry roughly 5-6 days after the failure beats one at 24 hours.
- Every customer contact reduces the response rate of the next one. Contact is a budget.

Be decisive and brief. Prefer the action with the highest expected value net of contact \
cost, not the one that looks most active."""


class LLMDiagnosis(BaseModel):
    root_cause: FailureCode
    confidence: float = Field(ge=0.0, le=1.0)
    recoverable: bool
    rationale: str = Field(max_length=400)
    evidence: list[str] = Field(default_factory=list, max_length=6)


class LLMAction(BaseModel):
    action: InterventionType
    channel: Channel | None = None
    delay_hours: float = Field(default=0.0, ge=0.0, le=720.0)
    reason: str = Field(max_length=400)


@dataclass
class LLMPlanner:
    """Optional planner. `enabled` is False without an API key, and every method then
    returns None so the deterministic path runs unchanged."""
    api_key: str | None = None
    model: str = ANTHROPIC_MODEL
    fast_model: str = ANTHROPIC_MODEL_FAST
    client: Any = None
    calls: int = 0
    failures: int = 0
    cost_usd: float = 0.0
    _pending_cost: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self.client is None and self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception:
                self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ routing
    def _model_for(self, expected_recovery: float) -> str:
        """Spend the expensive model only where it can change the decision."""
        return self.model if expected_recovery >= LLM_ESCALATE_EV_USD else self.fast_model

    def _meter(self, model: str, usage: Any) -> None:
        cin, cout = PRICING.get(model, (1.0, 5.0))
        it = getattr(usage, "input_tokens", 0) or 0
        ot = getattr(usage, "output_tokens", 0) or 0
        spend = (it * cin + ot * cout) / 1_000_000
        self.cost_usd += spend
        self._pending_cost += spend

    def pop_cost(self) -> float:
        c, self._pending_cost = self._pending_cost, 0.0
        return round(c, 6)

    # ------------------------------------------------------------------ calls
    def _parse(self, model: str, schema: type[BaseModel], prompt: str) -> BaseModel | None:
        if not self.enabled:
            return None
        try:
            self.calls += 1
            resp = self.client.messages.parse(
                model=model, max_tokens=700, system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
            )
            self._meter(model, getattr(resp, "usage", None))
            parsed = getattr(resp, "parsed", None)
            if parsed is None:
                # Some responses carry text instead (e.g. a refusal). Try JSON, else give up.
                text = "".join(getattr(b, "text", "") for b in getattr(resp, "content", []))
                parsed = schema.model_validate(json.loads(text)) if text.strip().startswith("{") else None
            return parsed
        except (ValidationError, json.JSONDecodeError):
            self.failures += 1
            return None
        except Exception:
            # Any API error, refusal or timeout: fall back silently and keep working.
            self.failures += 1
            return None

    # ------------------------------------------------------------------ interface
    def diagnose(self, txn: Transaction | None) -> Diagnosis | None:
        if txn is None or not self.enabled:
            return None
        prompt = (
            "Diagnose the true root cause of this failed payment. The processor code is "
            "evidence, not truth: repeated failures on a customer who never pays are a "
            "dead relationship, not a temporary fault.\n\n"
            f"{json.dumps(self._case_view(txn), indent=2)}"
        )
        out = self._parse(self._model_for(txn.amount_usd), LLMDiagnosis, prompt)
        if not isinstance(out, LLMDiagnosis):
            return None
        from backend.app.models.enums import FailureCategory, category_of
        cat = category_of(out.root_cause)
        # Retry viability is DERIVED, never taken from the model: whether a dead
        # instrument can be re-presented is a fact about the rails, not a judgement.
        retry_viable = cat is not FailureCategory.RISK_COMPLIANCE and out.root_cause not in (
            FailureCode.CLOSED_ACCOUNT, FailureCode.INVALID_ACCOUNT,
            FailureCode.EXPIRED_CARD, FailureCode.INVALID_PAYMENT_METHOD,
        )
        return Diagnosis(
            root_cause=out.root_cause, category=cat,
            confidence=out.confidence, rationale=out.rationale, evidence=out.evidence,
            source="llm",
            # A model claiming a fraud hold is "recoverable" must not make it so.
            recoverable=out.recoverable and cat is not FailureCategory.RISK_COMPLIANCE,
            retry_viable=retry_viable,
        )

    def select(self, state: AgentState, dx: Diagnosis) -> ProposedAction | None:
        if not self.enabled or state.transaction is None:
            return None
        prompt = (
            "Choose the single next recovery action for this case. A deterministic policy "
            "engine will validate your choice and may reject or rewrite it.\n\n"
            f"case: {json.dumps(self._case_view(state.transaction), indent=2)}\n"
            f"diagnosis: {dx.root_cause.value} (confidence {dx.confidence:.2f}) -- {dx.rationale}\n"
            f"P(recovery)={state.recovery_probability:.3f}  "
            f"expected_recovery=${state.expected_recovery:,.2f}\n"
            f"retries so far: {state.attempt_count}   customer contacts so far: {state.contact_count}\n"
            f"actions already taken: {state.actions_taken or 'none'}\n"
            f"actions already refused by policy: {state.blocked_actions or 'none'}\n"
            f"hours since the original failure: {state.elapsed_hours:.0f}"
        )
        out = self._parse(self._model_for(state.expected_recovery), LLMAction, prompt)
        if not isinstance(out, LLMAction):
            return None
        return ProposedAction(action=out.action, channel=out.channel,
                              delay_hours=out.delay_hours, reason=out.reason, source="llm")

    @staticmethod
    def _case_view(txn: Transaction) -> dict:
        """The model-facing view. An allowlist: it names what goes out, so adding a field
        to Transaction never silently widens what reaches a third-party API. `scrub()` is
        a backstop against this list drifting, not the primary defence."""
        return scrub({
            "failure_code": txn.failure_code.value,
            "failure_category": txn.failure_category.value,
            "payment_method": txn.payment_method.value,
            "amount_usd": round(txn.amount_usd, 2),
            "customer_average_transaction_usd": round(txn.avg_transaction_value, 2),
            "failure_count": txn.failure_count,
            "days_since_failure": txn.days_since_failure,
            "customer_tenure_days": txn.customer_tenure,
            "previous_success_rate": round(txn.previous_success_rate, 3),
            "previous_recovery_count": txn.previous_recovery_count,
            "overdue_days": txn.overdue_days,
            "segment": txn.customer_segment.value,
            "preferred_channel": txn.preferred_channel.value,
        })


def build_planner(kind: str = "auto") -> LLMPlanner | None:
    """`rules` forces the deterministic path; `llm` requires an Anthropic key; `ollama`
    requires a local daemon; `auto` uses the hosted model when a key is present and the
    deterministic planner otherwise. `auto` never selects a local model on its own."""
    if kind == "rules":
        return None

    def _ollama():
        from backend.app.agents.ollama import OllamaPlanner
        return OllamaPlanner()

    if kind == "ollama":
        p = _ollama()
        if not p.enabled:
            raise RuntimeError(
                "planner 'ollama' requires a running Ollama daemon with a pulled model "
                "(`ollama serve` then `ollama pull llama3.2:3b`)")
        return p

    hosted = LLMPlanner()
    if kind == "llm":
        if not hosted.enabled:
            raise RuntimeError("planner 'llm' requires ANTHROPIC_API_KEY")
        return hosted
    # `auto` deliberately does NOT fall back to a local model. Benchmarked diagnosis
    # accuracy is 100% for the deterministic planner and 25-65% for the local models on
    # this machine, so silently preferring Ollama merely because a daemon happens to be
    # running would degrade every run on that machine with no signal. Local inference is
    # opt-in: `--planner ollama`.
    return hosted if hosted.enabled else None
