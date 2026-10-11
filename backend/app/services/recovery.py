"""Persistent orchestration shared by the worker and approval jobs.

The current execution adapter is explicitly simulation-only. Source payments remain
read-only until a provider integration supplies execution and reconciliation.
"""
from __future__ import annotations

from backend.app.agents.graph import RecoveryAgent
from backend.app.agents.runner import to_outcome
from backend.app.audit.store import AuditStore
from backend.app.database.jobs import JobStore
from backend.app.database.operational import ActionLedger, DLQStore
from backend.app.database.review import ConsentStore, ReviewStore
from backend.app.database.store import CaseStore
from backend.app.models.schemas import AgentState, ProposedAction, Transaction
from backend.app.policies.engine import PolicyContext, validate
from backend.app.policies.version import policy_version
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.results import summarize
from backend.app.tools.executor import ActionExecutor

STAGES = ["load_transaction", "score_recovery", "diagnose_root_cause",
          "calculate_expected_recovery", "select_intervention", "validate_policy",
          "execute_action", "monitor_outcome"]


class RecoveryBlocked(RuntimeError):
    """A safe, operator-facing explanation for withholding execution."""


class Cancelled(RuntimeError):
    pass


class RecoveryService:
    def __init__(self, jobs: JobStore):
        self.jobs = jobs
        self.tenant = jobs.tenant_id
        # Stable across HTTP requests and job retries, scoped separately from experiments.
        self.run_id = "operational-v1"
        self.ledger = ActionLedger(self.run_id, conn=jobs.conn, tenant_id=self.tenant)
        self.dlq = DLQStore(self.run_id, conn=jobs.conn, tenant_id=self.tenant)
        self.reviews = ReviewStore(conn=jobs.conn, tenant_id=self.tenant)
        self.consent = ConsentStore(conn=jobs.conn, tenant_id=self.tenant)
        self.cases = CaseStore(conn=jobs.conn, tenant_id=self.tenant)
        self.audit = AuditStore(conn=jobs.conn, tenant_id=self.tenant)

    def agent(self, job_id: str = ""):
        return RecoveryAgent(executor=ActionExecutor(ledger=self.ledger, dlq=self.dlq),
                             tenant_id=self.tenant, run_id=job_id or self.run_id,
                             reviews=self.reviews)

    def check_cancelled(self, job_id):
        if self.jobs.get(job_id)["status"] == "cancelled":
            raise Cancelled("Cancelled between recovery steps")

    def run_case(self, txn: dict, job_id: str) -> AgentState:
        agent = self.agent(job_id)
        saved = self.jobs.checkpoint(txn["transaction_id"])
        if saved:
            state = AgentState.model_validate(saved["state"])
            stage = saved["next"]
            agent.executor.gateway.restore_case(state.transaction_id, saved.get("simulator", {}))
        else:
            existing = self.cases.get_case(txn["transaction_id"])
            if existing and existing["status"] not in {"pending", "in_progress"}:
                raise RecoveryBlocked("Case is already closed and has no resumable operational checkpoint")
            t = Transaction.model_validate(txn)
            state = AgentState(transaction_id=t.transaction_id, customer_id=t.customer_id,
                               amount=t.amount, currency=t.currency, failure_code=t.failure_code,
                               transaction=t, elapsed_hours=t.days_since_failure * 24)
            stage = STAGES[0]
        while stage != "done":
            self.check_cancelled(job_id)
            channel = (state.proposed_action.channel.value
                       if state.proposed_action and state.proposed_action.channel
                       else state.transaction.preferred_channel.value)
            agent.opted_out = frozenset({state.customer_id}) if self.consent.is_opted_out(
                state.customer_id, channel) else frozenset()
            # Revalidate after a restart immediately before any dispatch.
            if stage == "execute_action":
                state = state.model_copy(update=agent.validate_policy(state))
                if not state.policy_result.allowed:
                    stage = "monitor_outcome"
            update = getattr(agent, stage)(state)
            state = state.model_copy(update=update)
            if stage == "validate_policy":
                next_stage = "execute_action" if state.policy_result.allowed else "monitor_outcome"
            elif stage == "monitor_outcome":
                next_stage = "done" if state.is_terminal else "select_intervention"
            else:
                next_stage = STAGES[STAGES.index(stage) + 1]
            with self.jobs.conn.transaction():
                for rid in state.review_ids:
                    self.jobs.conn.execute(
                        "UPDATE human_review SET state_json = ? WHERE tenant_id = ? "
                        "AND review_id = ? AND state_json IS NULL",
                        (state.model_dump_json(), self.tenant, rid))
                self.jobs.save_checkpoint(state.transaction_id,
                    {"state": state.model_dump(mode="json"), "next": next_stage,
                     "simulator": agent.executor.gateway.snapshot_case(state.transaction_id)})
                if next_stage == "done":
                    self.persist(state)
            stage = next_stage
        return state

    def persist(self, state: AgentState):
        self.audit.append_many(state.audit_events)
        self.cases.save_cases([to_outcome(state)], "recoverai", {
            state.transaction_id: {"currency": state.currency, "amount": state.amount,
                                  "root_cause": state.root_cause.value if state.root_cause else None,
                                  "recommended_action": state.actions_taken[0]
                                  if state.actions_taken else None}})
        self.jobs.event(f"{self.tenant}:{state.transaction_id}:closed", "case.closed",
                        {"transaction_id": state.transaction_id, "status": state.status.value})

    def execute_review(self, review_id: str, job_id: str) -> dict:
        self.check_cancelled(job_id)
        job = self.jobs.get(job_id)
        if job["status"] == "succeeded":
            return job["result"]
        row = self.reviews.get(review_id)
        if not row or row["status"] != "approved" or not row["state_json"]:
            raise RecoveryBlocked("Approved review has no executable checkpoint")
        from backend.app.database.jobs import now
        if row["expires_at"] and row["expires_at"] <= now():
            raise RecoveryBlocked("Review approval expired before execution")
        if row["policy_version"] != policy_version():
            raise RecoveryBlocked("Policy changed; a new review is required")
        state = AgentState.model_validate_json(row["state_json"])
        current = self.cases.get_case(state.transaction_id)
        if not current or current["status"] != "escalated":
            raise RecoveryBlocked("Case is no longer awaiting review execution")
        action = ProposedAction.model_validate_json(row["proposed_action"])
        channel = action.channel.value if action.channel else state.transaction.preferred_channel.value
        ctx = PolicyContext(
            tenant_id=self.tenant, elapsed_hours=state.elapsed_hours,
            hours_since_last_attempt=state.hours_since_last_attempt,
            instrument_fixed=state.instrument_fixed,
            already_executed=frozenset(state.actions_taken), contact_channel=channel,
            customer_opted_out=self.consent.is_opted_out(state.customer_id, channel),
            quarantined_channels=frozenset({channel})
            if self.dlq.is_quarantined(state.customer_id, channel) else frozenset(),
            diagnosis_confidence=state.diagnosis.confidence if state.diagnosis else 0,
            consecutive_execution_failures=state.consecutive_execution_failures,
            support_requested=state.support_requested)
        from backend.app.policies.engine import REVIEW_RULES
        if row["rule_id"] not in REVIEW_RULES:
            raise RecoveryBlocked("This escalation requires manual handling")
        verdict = validate(state, action, ctx, approved_review_rule=row["rule_id"])
        if not verdict.allowed or verdict.effective_action != action:
            raise RecoveryBlocked("Current policy does not permit the approved action")
        executor = self.agent().executor
        checkpoint = self.jobs.checkpoint(state.transaction_id)
        executor.gateway.restore_case(state.transaction_id, (checkpoint or {}).get("simulator", {}))
        result = executor.execute(state, verdict, state.transaction.model_dump(mode="json"))
        with self.jobs.conn.transaction():
            # The closed parent case stays closed. This follow-up is a separately
            # recorded execution, with its own outcome and audit event.
            from backend.app.models.schemas import AuditEvent
            self.audit.append(AuditEvent(transaction_id=state.transaction_id,
                customer_id=state.customer_id, tenant_id=self.tenant,
                agent_decision="human_approved_execution", reason=row["resolution_note"],
                action=action.action.value, action_result=result.outcome.value,
                amount_recovered=result.amount_recovered, cost=result.cost,
                policy_result="approve", policy_version=policy_version(), agent_run_id=job_id))
            self.jobs.event(f"review:{review_id}", "review.executed",
                            {"review_id": review_id, "result": result.model_dump(mode="json")})
            self.jobs.finish(job_id, "succeeded", result.model_dump(mode="json"))
        return result.model_dump(mode="json")

    def run(self, job: dict):
        payload = job["payload"]
        if payload.get("review_id"):
            return self.execute_review(payload["review_id"], job["job_id"])
        txns = to_transactions(load_split(payload.get("split", "test")))
        if payload.get("transaction_id"):
            txns = [t for t in txns if t["transaction_id"] == payload["transaction_id"]]
            if not txns:
                raise RecoveryBlocked("Transaction not found")
        txns = txns[:payload.get("limit", 100)]
        if not txns:
            raise RecoveryBlocked("No transactions are available for this job")
        outcomes = []
        for txn in txns:
            state = self.run_case(txn, job["job_id"])
            outcomes.append(to_outcome(state))
            self.jobs.finish(job["job_id"], "running", {"completed": len(outcomes), "total": len(txns)})
        report = summarize(outcomes, "recoverai")
        completed = []
        for row in self.jobs.conn.execute(
                "SELECT state_json FROM recovery_checkpoints WHERE tenant_id = ?", (self.tenant,)):
            import json
            checkpoint = json.loads(row["state_json"])
            if checkpoint["next"] == "done":
                completed.append(to_outcome(AgentState.model_validate(checkpoint["state"])))
        with self.jobs.conn.transaction():
            self.cases.save_run("recoverai", summarize(completed, "recoverai"))
            self.cases.save_run("meta", {"run_id": self.run_id, "provenance": "simulated"})
            self.jobs.finish(job["job_id"], "succeeded",
                             {"cases_processed": len(outcomes), "summary": report})
        return report
