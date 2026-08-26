"""Phase 6 -- the LangGraph recovery workflow.

    START -> load_transaction -> score_recovery -> diagnose_root_cause
          -> calculate_expected_recovery -> select_intervention -> validate_policy
          -> execute_action -> monitor_outcome -> {success | retry | stop | escalate} -> END

`validate_policy` sits between proposal and execution and cannot be bypassed: the
executor's input is the policy engine's *output*, so there is no code path from a
proposal to a side effect that skips the gate.

The loop is bounded three ways -- the retry caps, `MAX_AGENT_STEPS`, and the recovery
horizon -- so it terminates even if a planner misbehaves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from backend.app.agents.diagnose import diagnose as rules_diagnose
from backend.app.agents.strategy import select_intervention as rules_select
from backend.app.config import MAX_AGENT_STEPS, RECOVERY_HORIZON_DAYS
from backend.app.ml.scorer import RecoveryScorer, get_scorer
from backend.app.models.enums import (
    CONTACT_ACTIONS, ActionOutcome, CaseStatus, FailureCategory, InterventionType,
    PolicyDecision, category_of,
)
from backend.app.models.schemas import (
    ActionResult, AgentState, AuditEvent, Diagnosis, PolicyResult, ProposedAction, Transaction,
)
from backend.app.policies.engine import PolicyContext, validate
from backend.app.tools.executor import ActionExecutor


def _txn_dict(state: AgentState) -> dict:
    """The view the mock rails need. Never includes any historical outcome label."""
    if state.transaction is None:
        return {"transaction_id": state.transaction_id, "customer_id": state.customer_id,
                "amount": state.amount, "amount_usd": state.amount_usd,
                "currency": state.currency, "failure_code": state.failure_code.value}
    d = state.transaction.model_dump(mode="json")
    d["amount_usd"] = state.transaction.amount_usd
    return d


@dataclass
class RecoveryAgent:
    """Compiled agent. One instance may run many cases; per-case state is never shared."""
    executor: ActionExecutor = field(default_factory=ActionExecutor)
    scorer: RecoveryScorer = field(default_factory=get_scorer)
    #: Optional LLM planner. Must expose .diagnose(txn) and .select(state, dx); either may
    #: return None to fall back to the deterministic path.
    planner: Any = None
    on_audit: Callable[[AuditEvent], None] | None = None
    _graph: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._graph = self._build()

    # ------------------------------------------------------------------ audit
    def _emit(self, state: AgentState, decision: str, reason: str, **kw) -> AuditEvent:
        ev = AuditEvent(
            transaction_id=state.transaction_id, customer_id=state.customer_id,
            agent_decision=decision, reason=reason,
            # `x or None` turns a legitimate 0.0 into NULL -- and 0.0 is exactly the
            # value a risk case carries, so the audit log lost real information.
            risk_score=round(state.risk_score, 6),
            recovery_probability=round(state.recovery_probability, 6),
            expected_recovery=round(state.expected_recovery, 2),
            attempt_count=state.attempt_count, **kw,
        )
        if self.on_audit:
            self.on_audit(ev)
        return ev

    # ------------------------------------------------------------------ nodes
    def load_transaction(self, state: AgentState) -> dict:
        ev = self._emit(state, "load_transaction",
                        f"case opened: {state.failure_code.value} on "
                        f"{state.currency} {state.amount:,.2f}",
                        next_step="score_recovery")
        return {"status": CaseStatus.IN_PROGRESS, "audit_events": [*state.audit_events, ev]}

    def score_recovery(self, state: AgentState) -> dict:
        score = self.scorer.score(_txn_dict(state))
        upd = {"recovery_probability": score.recovery_probability,
               "risk_score": score.risk_score,
               "expected_recovery": score.expected_recovery}
        tmp = state.model_copy(update=upd)
        ev = self._emit(tmp, "score_recovery",
                        f"P(recovery)={score.recovery_probability:.3f} via {score.model_version}",
                        next_step="diagnose_root_cause")
        return {**upd, "audit_events": [*state.audit_events, ev]}

    def diagnose_root_cause(self, state: AgentState) -> dict:
        dx: Diagnosis | None = None
        if self.planner is not None:
            dx = self.planner.diagnose(state.transaction)      # may return None
        if dx is None:
            dx = rules_diagnose(state.transaction)
        ev = self._emit(state, "diagnose_root_cause",
                        f"root cause {dx.root_cause.value} "
                        f"(confidence {dx.confidence:.2f}, via {dx.source}): {dx.rationale}",
                        next_step="calculate_expected_recovery")
        return {"diagnosis": dx, "root_cause": dx.root_cause,
                "audit_events": [*state.audit_events, ev]}

    def calculate_expected_recovery(self, state: AgentState) -> dict:
        """Expected recovery is recomputed here so it reflects the *diagnosed* cause, not
        just the reported code -- an overruled diagnosis should move the number."""
        ev_value = state.expected_recovery
        dx = state.diagnosis
        if dx is not None and dx.root_cause != state.failure_code:
            # The reported code was overruled; discount by the diagnosis confidence.
            ev_value = round(state.amount_usd * state.recovery_probability * dx.confidence, 2)
        if dx is not None and not dx.recoverable:
            # Only risk/compliance reaches zero. Zeroing an unreachable account here
            # made the value floor stop the case before the payment-link route was
            # ever considered, so those cases silently took no action at all.
            ev_value = 0.0
        ev = self._emit(state.model_copy(update={"expected_recovery": ev_value}),
                        "calculate_expected_recovery",
                        f"expected recovery ${ev_value:,.2f} "
                        f"= ${state.amount_usd:,.2f} x {state.recovery_probability:.3f}"
                        + ("" if ev_value == state.expected_recovery else " (diagnosis-adjusted)"),
                        next_step="select_intervention")
        return {"expected_recovery": ev_value, "audit_events": [*state.audit_events, ev]}

    def select_intervention(self, state: AgentState) -> dict:
        dx = state.diagnosis or rules_diagnose(state.transaction)
        action: ProposedAction | None = None
        if self.planner is not None:
            action = self.planner.select(state, dx)            # may return None
        if action is None:
            action = rules_select(state, dx)
        ev = self._emit(state, "select_intervention",
                        f"proposed {action.action.value} (via {action.source}): {action.reason}",
                        action=action.action.value, next_step="validate_policy")
        return {"proposed_action": action, "step_count": state.step_count + 1,
                "audit_events": [*state.audit_events, ev]}

    def validate_policy(self, state: AgentState) -> dict:
        assert state.proposed_action is not None
        dlq = getattr(self.executor, "dlq", None)
        channel = str(_txn_dict(state).get("preferred_channel", "email"))
        quarantined = frozenset(
            c for c in {channel}
            if dlq is not None and dlq.is_quarantined(state.customer_id, c)
        )
        ctx = PolicyContext(
            hours_since_last_attempt=state.hours_since_last_attempt,
            instrument_fixed=state.instrument_fixed,
            elapsed_hours=state.elapsed_hours,
            already_executed=frozenset(state.actions_taken),
            quarantined_channels=quarantined,
            contact_channel=channel,
        )
        result = validate(state, state.proposed_action, ctx)
        ev = self._emit(state, "validate_policy", result.reason,
                        policy_result=result.decision.value,
                        rules_fired=",".join(result.rules_fired),
                        action=state.proposed_action.action.value,
                        next_step="execute_action" if result.allowed else "monitor_outcome")
        upd: dict = {"policy_result": result, "audit_events": [*state.audit_events, ev]}
        if not result.allowed:
            # Record the block so the planner advances instead of re-proposing it.
            upd["blocked_actions"] = [*state.blocked_actions,
                                      state.proposed_action.action.value]
        return upd

    def execute_action(self, state: AgentState) -> dict:
        policy = state.policy_result
        assert policy is not None
        if not policy.allowed or policy.effective_action is None:
            return {}                                     # routed away; nothing to do

        action = policy.effective_action
        if action.action in (InterventionType.WAIT, InterventionType.STOP):
            ev = self._emit(state, "execute_action", f"no side effect: {action.action.value}",
                            action=action.action.value, next_step="monitor_outcome")
            return {"action_result": ActionResult(action=action.action,
                                                  outcome=ActionOutcome.PENDING,
                                                  detail=action.reason),
                    "audit_events": [*state.audit_events, ev]}

        txn = _txn_dict(state)
        elapsed = state.elapsed_hours + action.delay_hours

        # Before acting, check whether the case already resolved by itself. Skipping
        # this would credit the agent with money that arrived while it was waiting --
        # which is exactly the illusion the control arm exists to dispel.
        cured = self.executor.gateway.check_self_cure(txn, elapsed)
        if cured is not None:
            passive = ActionResult(action=action.action, outcome=ActionOutcome.SUCCESS,
                                   amount_recovered=round(cured.amount, 2),
                                   detail=cured.detail)
            ev = self._emit(state, "execute_action", cured.detail,
                            action=action.action.value, action_result="self_cured",
                            amount_recovered=passive.amount_recovered,
                            next_step="monitor_outcome")
            return {"action_result": passive, "elapsed_hours": elapsed,
                    "recovered_passively": True,
                    "audit_events": [*state.audit_events, ev]}

        result = self.executor.execute(state, policy, txn)
        upd: dict = {
            "action_result": result,
            "elapsed_hours": elapsed,
            "total_cost": round(state.total_cost + result.cost, 4),
            "actions_taken": [*state.actions_taken, action.action.value],
            "instrument_fixed": self.executor.gateway.instrument_fixed(state.transaction_id),
        }
        if action.action is InterventionType.RETRY_PAYMENT:
            upd["attempt_count"] = state.attempt_count + 1
            upd["hours_since_last_attempt"] = 0.0
        else:
            if action.action in CONTACT_ACTIONS:
                upd["contact_count"] = state.contact_count + 1
            upd["hours_since_last_attempt"] = state.hours_since_last_attempt + action.delay_hours

        ev = self._emit(state, "execute_action", result.detail,
                        action=action.action.value, action_result=result.outcome.value,
                        amount_recovered=result.amount_recovered, cost=result.cost,
                        policy_result=policy.decision.value,
                        rules_fired=",".join(policy.rules_fired),
                        next_step="monitor_outcome")
        upd["audit_events"] = [*state.audit_events, ev]
        return upd

    def monitor_outcome(self, state: AgentState) -> dict:
        """Decide whether the case is finished, and why. The only place a case closes."""
        res = state.action_result
        policy = state.policy_result
        upd: dict = {}

        if res is not None and res.amount_recovered > 0:
            upd.update(status=CaseStatus.RECOVERED, outcome="recovered",
                       amount_recovered=round(state.amount_recovered + res.amount_recovered, 2),
                       stop_reason=("self-cured before our action landed"
                                    if state.recovered_passively else "payment succeeded"))
        elif res is not None and res.action is InterventionType.ESCALATE_CASE:
            upd.update(status=CaseStatus.ESCALATED, outcome="escalated",
                       stop_reason="handed to human review")
        elif state.proposed_action is not None \
                and state.proposed_action.action is InterventionType.STOP:
            upd.update(status=CaseStatus.STOPPED, outcome="stopped",
                       stop_reason=state.proposed_action.reason or "no further action worthwhile")
        elif policy is not None and not policy.allowed:
            # A block is not automatically the end -- the planner gets one more chance to
            # find a different route, bounded by the step cap.
            if state.step_count >= MAX_AGENT_STEPS:
                upd.update(status=CaseStatus.EXHAUSTED, outcome="exhausted",
                           stop_reason="agent step ceiling reached")
        elif state.elapsed_hours > RECOVERY_HORIZON_DAYS * 24:
            upd.update(status=CaseStatus.EXHAUSTED, outcome="exhausted",
                       stop_reason=f"{RECOVERY_HORIZON_DAYS}-day horizon reached")
        elif state.step_count >= MAX_AGENT_STEPS:
            upd.update(status=CaseStatus.EXHAUSTED, outcome="exhausted",
                       stop_reason="agent step ceiling reached")

        # The clock is advanced solely by each action's `delay_hours` in execute_action.
        # Adding time here as well would double-count it and silently decay every case.

        tmp = state.model_copy(update=upd)
        ev = self._emit(tmp, "monitor_outcome",
                        upd.get("stop_reason", "case still open; re-planning"),
                        action_result=(res.outcome.value if res else None),
                        amount_recovered=res.amount_recovered if res else 0.0,
                        next_step="END" if tmp.is_terminal else "select_intervention")
        upd["audit_events"] = [*state.audit_events, ev]
        return upd

    # ------------------------------------------------------------------ routing
    @staticmethod
    def route(state: AgentState) -> str:
        return "end" if state.is_terminal else "retry"

    @staticmethod
    def gate(state: AgentState) -> str:
        """After validation: execute only if the policy engine allowed it."""
        pr = state.policy_result
        return "execute" if (pr is not None and pr.allowed) else "monitor"

    # ------------------------------------------------------------------ build/run
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("load_transaction", self.load_transaction)
        g.add_node("score_recovery", self.score_recovery)
        g.add_node("diagnose_root_cause", self.diagnose_root_cause)
        g.add_node("calculate_expected_recovery", self.calculate_expected_recovery)
        g.add_node("select_intervention", self.select_intervention)
        g.add_node("validate_policy", self.validate_policy)
        g.add_node("execute_action", self.execute_action)
        g.add_node("monitor_outcome", self.monitor_outcome)

        g.add_edge(START, "load_transaction")
        g.add_edge("load_transaction", "score_recovery")
        g.add_edge("score_recovery", "diagnose_root_cause")
        g.add_edge("diagnose_root_cause", "calculate_expected_recovery")
        g.add_edge("calculate_expected_recovery", "select_intervention")
        g.add_edge("select_intervention", "validate_policy")
        g.add_conditional_edges("validate_policy", self.gate,
                                {"execute": "execute_action", "monitor": "monitor_outcome"})
        g.add_edge("execute_action", "monitor_outcome")
        g.add_conditional_edges("monitor_outcome", self.route,
                                {"retry": "select_intervention", "end": END})
        return g.compile()

    def run(self, txn: dict) -> AgentState:
        """Work one case to a terminal state."""
        t = Transaction(**{k: v for k, v in txn.items()
                           if k in Transaction.model_fields})
        state = AgentState(
            transaction_id=t.transaction_id, customer_id=t.customer_id,
            amount=t.amount, currency=t.currency, failure_code=t.failure_code,
            transaction=t,
            # The clock starts at the original failure, not at "now", so cause-specific
            # decay and the horizon are measured from the right moment.
            elapsed_hours=float(t.days_since_failure) * 24.0,
        )
        # recursion_limit bounds the graph independently of our own step cap.
        final = self._graph.invoke(state, {"recursion_limit": MAX_AGENT_STEPS * 6 + 20})
        return AgentState(**final) if isinstance(final, dict) else final
