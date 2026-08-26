"""Phase 1 -- Pydantic schemas. These are the contract between every stage of the
pipeline and the API surface."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from backend.app.config import to_usd
from backend.app.models.enums import (
    ActionOutcome, CaseStatus, Channel, CustomerSegment, FailureCategory, FailureCode,
    InterventionType, PaymentMethod, PolicyDecision, category_of,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Transaction(BaseModel):
    """One failed payment, exactly as it arrives from the events feed. The columns
    mirror the dataset schema one-for-one."""
    model_config = ConfigDict(use_enum_values=False)

    customer_id: str
    transaction_id: str
    subscription_id: str | None = None
    invoice_id: str | None = None
    amount: float
    currency: str = "USD"
    payment_method: PaymentMethod
    failure_code: FailureCode
    failure_count: int = 1
    days_since_failure: float = 0.0
    customer_tenure: int = 0                 # days
    previous_success_rate: float = 0.0       # 0..1
    previous_payment_attempts: int = 0
    avg_transaction_value: float = 0.0
    days_since_last_payment: float = 0.0
    customer_segment: CustomerSegment = CustomerSegment.CONSUMER
    country: str = "US"
    preferred_channel: Channel = Channel.EMAIL
    subscription_age: int = 0                # days
    overdue_days: int = 0
    previous_recovery_count: int = 0
    failed_at: datetime = Field(default_factory=utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def amount_usd(self) -> float:
        return to_usd(self.amount, self.currency)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failure_category(self) -> FailureCategory:
        return category_of(self.failure_code)


class ScoreResult(BaseModel):
    """Output of the ML stage."""
    recovery_probability: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0, description="1 - recovery_probability")
    expected_recovery: float = Field(ge=0.0, description="amount_usd * recovery_probability")
    model_version: str = "unknown"


class Diagnosis(BaseModel):
    """Output of the root-cause stage."""
    root_cause: FailureCode
    category: FailureCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    source: Literal["rules", "llm"] = "rules"
    #: Can ANY automated path still collect? False only for risk/compliance, where we
    #: are forbidden to act at all. A closed account is still `recoverable`: the customer
    #: can pay by another method even though the account itself cannot be debited.
    recoverable: bool = True
    #: Can re-presenting the EXISTING instrument work? False for dead cards and
    #: unreachable accounts. Distinct from `recoverable` -- conflating the two makes the
    #: agent abandon cases a payment link could still collect.
    retry_viable: bool = True


class ProposedAction(BaseModel):
    """What the strategy agent *wants* to do. Never executed as-is."""
    action: InterventionType
    channel: Channel | None = None
    delay_hours: float = 0.0
    message: str | None = None
    reason: str = ""
    source: Literal["rules", "llm"] = "rules"


class PolicyResult(BaseModel):
    """The deterministic validator's verdict. `effective_action` is what may run."""
    decision: PolicyDecision
    effective_action: ProposedAction | None = None
    rules_fired: list[str] = Field(default_factory=list)
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision in (PolicyDecision.APPROVE, PolicyDecision.MODIFY)


class ActionResult(BaseModel):
    """What the executor observed after running an approved action."""
    action: InterventionType
    outcome: ActionOutcome
    amount_recovered: float = 0.0
    cost: float = 0.0
    detail: str = ""
    idempotency_key: str = ""
    replayed: bool = Field(default=False, description="True if a cached result was returned")
    at: datetime = Field(default_factory=utcnow)


class AuditEvent(BaseModel):
    """One append-only row. `seq`/`row_hash` are assigned by the store."""
    seq: int | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    transaction_id: str
    customer_id: str = ""
    agent_decision: str = ""
    reason: str = ""
    risk_score: float | None = None
    recovery_probability: float | None = None
    expected_recovery: float | None = None
    policy_result: str | None = None
    rules_fired: str = ""
    action: str | None = None
    action_result: str | None = None
    amount_recovered: float = 0.0
    cost: float = 0.0
    next_step: str = ""
    attempt_count: int = 0
    prev_hash: str | None = None
    row_hash: str | None = None


class AgentState(BaseModel):
    """The LangGraph state object, carried node to node."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transaction_id: str
    customer_id: str
    amount: float
    currency: str = "USD"
    failure_code: FailureCode
    transaction: Transaction | None = None

    risk_score: float = 0.0
    recovery_probability: float = 0.0
    expected_recovery: float = 0.0
    root_cause: FailureCode | None = None
    diagnosis: Diagnosis | None = None

    proposed_action: ProposedAction | None = None
    policy_result: PolicyResult | None = None
    action_result: ActionResult | None = None

    attempt_count: int = 0            # retry_payment attempts only
    contact_count: int = 0            # customer-facing touches
    step_count: int = 0               # graph iterations, hard-capped
    next_action_time: datetime | None = None
    elapsed_hours: float = 0.0

    actions_taken: list[str] = Field(default_factory=list)
    blocked_actions: list[str] = Field(default_factory=list)
    #: True when the case resolved on its own before any action of ours landed. The
    #: money still arrives, but this arm did not cause it.
    recovered_passively: bool = False
    hours_since_last_attempt: float = 1e9
    instrument_fixed: bool = False

    status: CaseStatus = CaseStatus.PENDING
    outcome: str | None = None
    amount_recovered: float = 0.0
    total_cost: float = 0.0
    audit_events: list[AuditEvent] = Field(default_factory=list)
    stop_reason: str = ""

    @property
    def amount_usd(self) -> float:
        return to_usd(self.amount, self.currency)

    @property
    def is_terminal(self) -> bool:
        from backend.app.models.enums import TERMINAL_STATUSES
        return self.status in TERMINAL_STATUSES


class RecoveryCase(BaseModel):
    """A row in the recovery queue / case API."""
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    amount_usd: float
    failure_code: FailureCode
    failure_category: FailureCategory
    root_cause: FailureCode | None = None
    recovery_probability: float = 0.0
    risk_score: float = 0.0
    expected_recovery: float = 0.0
    recommended_action: InterventionType | None = None
    status: CaseStatus = CaseStatus.PENDING
    attempt_count: int = 0
    amount_recovered: float = 0.0
    cost: float = 0.0
    recovery_hours: float | None = None
    stop_reason: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
