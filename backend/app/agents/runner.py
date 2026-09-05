"""Phase 6 -- batch runner. Works a queue of cases and emits the same `CaseOutcome`
records the baseline produces, so the two are summarised by identical code.

The gateway is constructed per run but seeded identically, and its latents are derived
from (seed, transaction_id). Both strategies therefore face exactly the same customers
with exactly the same hidden behaviour -- the only difference is what each one does.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from backend.app.agents.graph import RecoveryAgent
from backend.app.config import SEED
from backend.app.database.operational import ActionLedger, DLQStore
from backend.app.ml.scorer import RecoveryScorer, get_scorer
from backend.app.models.enums import CaseStatus, FailureCategory, category_of
from backend.app.models.schemas import AgentState, AuditEvent
from backend.app.services.results import CaseOutcome, summarize
from backend.app.tools.executor import ActionExecutor
from simulation.payment_gateway import PaymentGateway

CONTACT_ACTIONS = {"send_reminder", "send_payment_link", "request_payment_method_update"}


def to_outcome(state: AgentState) -> CaseOutcome:
    cat = category_of(state.failure_code)
    risk_actions = sum(1 for a in state.actions_taken
                       if a != "escalate_case") if cat is FailureCategory.RISK_COMPLIANCE else 0
    return CaseOutcome(
        transaction_id=state.transaction_id,
        customer_id=state.customer_id,
        amount_usd=state.amount_usd,
        failure_code=state.failure_code,
        failure_category=cat,
        strategy="recoverai",
        recovered=state.status is CaseStatus.RECOVERED,
        amount_recovered=state.amount_recovered,
        recovery_hours=state.elapsed_hours if state.status is CaseStatus.RECOVERED else None,
        retries=state.attempt_count,
        contacts=state.contact_count,
        actions=list(state.actions_taken),
        cost=round(state.total_cost, 4),
        status=state.status.value,
        passive_recovery=state.recovered_passively,
        stop_reason=state.stop_reason,
        escalated=state.status is CaseStatus.ESCALATED,
        policy_blocked=len(state.blocked_actions),
        risk_actions=risk_actions,
        recovery_probability=state.recovery_probability,
        expected_recovery=state.expected_recovery,
    )


def run_agent_batch(
    transactions: Iterable[dict],
    gateway: PaymentGateway | None = None,
    scorer: RecoveryScorer | None = None,
    planner=None,
    on_audit: Callable[[AuditEvent], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    ledger: ActionLedger | None = None,
    dlq: DLQStore | None = None,
    reviews=None,
    optimizer=None,
    bus=None,
    opted_out: frozenset[str] = frozenset(),
    tenant_id: str = "default",
    run_id: str = "",
) -> tuple[list[CaseOutcome], dict, list[AgentState]]:
    txns = list(transactions)
    agent = RecoveryAgent(
        executor=ActionExecutor(gateway=gateway or PaymentGateway(seed=SEED),
                                ledger=ledger, dlq=dlq),
        scorer=scorer or get_scorer(),
        planner=planner,
        on_audit=on_audit,
        reviews=reviews,
        optimizer=optimizer,
        bus=bus,
        opted_out=opted_out,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    states: list[AgentState] = []
    for i, t in enumerate(txns, 1):
        states.append(agent.run(t))
        if progress and (i % 200 == 0 or i == len(txns)):
            progress(i, len(txns))

    outcomes = [to_outcome(s) for s in states]
    stats = summarize(outcomes, "recoverai")
    stats["bounces"] = len(agent.executor.bounces)
    stats["dlq"] = dlq.stats() if dlq is not None else {"quarantined": 0, "tracked_pairs": 0}
    return outcomes, stats, states
