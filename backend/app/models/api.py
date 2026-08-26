"""Phase 8 -- API response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str = "ok"
    model_trained: bool
    model_version: str
    dataset_ready: bool
    experiment_ready: bool
    audit_rows: int
    audit_chain_valid: bool | None = None
    llm_enabled: bool
    #: Which storage engine this process is actually talking to: "sqlite" or "postgres".
    db_engine: str = "sqlite"


class OverviewCards(BaseModel):
    revenue_at_risk: float
    revenue_recovered: float
    recovery_rate: float
    incremental_recovery_vs_baseline: float
    recovery_uplift_pct: float | None = None
    #: 90% paired-bootstrap interval on the incremental figure. Surfaced next to the
    #: point estimate because a lognormal revenue total without one invites over-reading.
    incremental_ci_low: float | None = None
    incremental_ci_high: float | None = None
    incremental_ci_excludes_zero: bool | None = None
    cases_processed: int
    cases_escalated: int
    cases_stopped: int
    cases_recovered: int
    total_cost: float
    net_recovered: float
    avg_recovery_hours: float | None = None
    baseline_recovered: float
    baseline_recovery_rate: float
    unsafe_actions_prevented: int
    #: The no-touch counterfactual. `incremental_vs_control` is the causal figure --
    #: revenue that would NOT have arrived had we left every case alone.
    control_recovered: float | None = None
    control_recovery_rate: float | None = None
    incremental_vs_control: float | None = None
    share_of_revenue_that_is_causal: float | None = None
    control_ci_low: float | None = None
    control_ci_high: float | None = None
    model_version: str = ""
    planner: str = ""


class QueueRow(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    amount_usd: float
    failure_code: str
    failure_category: str
    root_cause: str | None = None
    recovery_probability: float | None = None
    risk_score: float | None = None
    expected_recovery: float | None = None
    recommended_action: str | None = None
    status: str
    amount_recovered: float = 0.0
    retries: int = 0
    contacts: int = 0
    actions: list[str] = Field(default_factory=list)
    stop_reason: str = ""


class CaseDetail(BaseModel):
    case: QueueRow
    baseline: dict[str, Any] | None = None
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    chain_valid: bool = True


class AuditPage(BaseModel):
    transaction_id: str | None = None
    rows: list[dict[str, Any]]
    total: int
    chain_valid: bool


class RevenueAtRisk(BaseModel):
    total_at_risk: float
    total_recovered: float
    total_outstanding: float
    by_failure_category: dict[str, Any]
    by_failure_code: dict[str, Any]
    top_cases: list[QueueRow]


class RunRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=5000)
    split: str = "test"
    persist: bool = True


class RunResponse(BaseModel):
    cases_processed: int
    summary: dict[str, Any]
    seconds: float


class LiveTrace(BaseModel):
    transaction_id: str
    status: str
    amount_recovered: float
    recovery_probability: float
    expected_recovery: float
    root_cause: str | None
    steps: list[dict[str, Any]]
    stop_reason: str
