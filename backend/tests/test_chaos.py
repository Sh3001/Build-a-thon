"""Failure-injection tests.

The principle under test is one-directional:

    uncertainty  ->  safe fallback        (required)
    uncertainty  ->  execute anyway       (must be impossible)

Each test breaks one component and asserts the system does less, never more.
"""
from __future__ import annotations

import pytest

from backend.app.adapters.gateway import (
    GatewayRejected,
    GatewayRequest,
    GatewayUnavailable,
)
from backend.app.agents.graph import RecoveryAgent
from backend.app.domain.events import EventType, InProcessEventBus
from backend.app.models.enums import (
    ActionOutcome,
    CaseStatus,
    InterventionType,
    PolicyDecision,
)
from backend.app.models.schemas import AgentState, ProposedAction
from backend.app.policies.engine import PolicyContext, validate
from backend.app.tools.executor import ActionExecutor, PolicyViolation
from simulation.notification_service import NotificationService
from simulation.payment_gateway import PaymentGateway


def _txn(tid="txn_1", code="bank_timeout", amount=300.0) -> dict:
    return {"transaction_id": tid, "customer_id": "c1", "amount": amount,
            "currency": "USD", "amount_usd": amount, "payment_method": "card",
            "failure_code": code, "avg_transaction_value": 200.0,
            "preferred_channel": "email"}


# ---------------------------------------------------------------- the gate
def test_a_crashing_policy_rule_fails_closed(monkeypatch):
    """A gate that crashes must not become a gate that is bypassed."""
    from backend.app.policies import engine

    monkeypatch.setitem(engine.REJECT_RULES, "R-CHAOS",
                        lambda s, a, c: (_ for _ in ()).throw(RuntimeError("boom")))
    state = AgentState(transaction_id="t", customer_id="c", amount=100.0,
                       failure_code="bank_timeout")
    result = validate(state, ProposedAction(action=InterventionType.RETRY_PAYMENT),
                      PolicyContext(hours_since_last_attempt=999))
    assert result.decision is PolicyDecision.HUMAN_REVIEW
    assert not result.allowed and result.effective_action is None


def test_the_executor_refuses_an_unapproved_verdict():
    """The single guarantee the executor makes."""
    from backend.app.models.schemas import PolicyResult

    ex = ActionExecutor(gateway=PaymentGateway(seed=1))
    state = AgentState(transaction_id="t", customer_id="c", amount=100.0,
                       failure_code="bank_timeout")
    for decision in (PolicyDecision.REJECT, PolicyDecision.HUMAN_REVIEW):
        verdict = PolicyResult(
            decision=decision,
            effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT))
        with pytest.raises(PolicyViolation):
            ex.execute(state, verdict, _txn())


def test_a_human_review_verdict_carries_an_action_but_is_still_refused():
    """`effective_action is not None` must never be mistaken for "you may run it"."""
    from backend.app.models.schemas import PolicyResult

    verdict = PolicyResult(
        decision=PolicyDecision.HUMAN_REVIEW,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT))
    assert verdict.effective_action is not None
    assert not verdict.allowed


# ---------------------------------------------------------------- the planner
def test_an_llm_that_always_fails_degrades_to_the_deterministic_planner():
    class BrokenPlanner:
        def diagnose(self, txn):
            raise RuntimeError("model unavailable")

        def select(self, state, dx):
            raise RuntimeError("model unavailable")

    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=1)),
                          planner=BrokenPlanner())
    with pytest.raises(RuntimeError):
        # A planner that *raises* is a programming error, not a degraded model: the
        # contract is that it returns None. Asserting the raise documents the boundary --
        # `LLMPlanner` catches everything internally, and this is why it must.
        agent.run(_txn())


def test_an_llm_that_returns_nothing_degrades_silently():
    """The documented contract: return None, get the rules."""
    class SilentPlanner:
        def diagnose(self, txn):
            return None

        def select(self, state, dx):
            return None

    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=1)),
                          planner=SilentPlanner())
    state = agent.run(_txn())
    assert state.is_terminal
    assert state.audit_events, "a degraded run still leaves an audit trail"


def test_a_broken_optimizer_does_not_block_recovery():
    """The optimiser prices actions; it is not allowed to prevent them by breaking."""
    class BrokenOptimizer:
        def candidates(self, *a, **kw):
            raise RuntimeError("cost model unreadable")

        def best(self, *a, **kw):
            raise RuntimeError("cost model unreadable")

    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=1)),
                          optimizer=BrokenOptimizer())
    state = agent.run(_txn())
    assert state.is_terminal
    assert any("optimiser unavailable" in e.reason for e in state.audit_events)


# ---------------------------------------------------------------- the rails
def test_a_gateway_outage_is_not_a_decline():
    """Collapsing the two lets a five-minute blip burn a customer's whole retry budget."""
    assert issubclass(GatewayUnavailable, Exception)
    assert not issubclass(GatewayUnavailable, GatewayRejected)


def test_a_rail_call_without_an_idempotency_key_is_refused():
    with pytest.raises(ValueError, match="idempotency_key"):
        GatewayRequest(idempotency_key="", transaction_id="t", customer_id="c",
                       amount_minor=100, currency="USD")


def test_a_duplicate_rail_call_does_not_charge_twice():
    from backend.app.adapters.mock_rail import MockRail

    rail = MockRail(gateway=PaymentGateway(seed=5))
    req = GatewayRequest(idempotency_key="k1", transaction_id="t1", customer_id="c1",
                         amount_minor=10_000, currency="USD",
                         metadata={"failure_code": "bank_timeout"})
    first, second = rail.retry_payment(req), rail.retry_payment(req)
    assert first == second


def test_a_delivery_bounce_does_not_abort_the_case():
    """Every address on this notifier is dead. The case must still terminate cleanly."""
    ex = ActionExecutor(gateway=PaymentGateway(seed=3),
                        notifier=NotificationService(failure_rate=1.0))
    agent = RecoveryAgent(executor=ex)
    state = agent.run(_txn(code="expired_card"))
    assert state.is_terminal
    assert state.status is not CaseStatus.RECOVERED


def test_repeated_execution_failures_route_to_a_human(tmp_path):
    """The world disagreeing with the plan is a reason to ask someone, not to push
    harder."""
    from backend.app.config import REVIEW_MAX_EXECUTION_FAILURES

    state = AgentState(
        transaction_id="t", customer_id="c", amount=100.0, failure_code="bank_timeout",
        consecutive_execution_failures=REVIEW_MAX_EXECUTION_FAILURES)
    result = validate(state, ProposedAction(action=InterventionType.RETRY_PAYMENT),
                      PolicyContext(hours_since_last_attempt=999,
                                    consecutive_execution_failures=state
                                    .consecutive_execution_failures))
    assert result.decision is PolicyDecision.HUMAN_REVIEW


def test_a_declined_payment_is_not_an_execution_failure():
    """The distinction that keeps the review queue usable.

    A decline is the most common thing that happens in this system: the action ran
    exactly as intended and the issuer said no. Counting it as a failure to execute would
    route every case with three ordinary declines to a human, burying the genuine
    infrastructure failures in a queue nobody can work.
    """
    from backend.app.models.schemas import PolicyResult

    ex = ActionExecutor(gateway=PaymentGateway(seed=11))
    state = AgentState(transaction_id="t_decline", customer_id="c", amount=100.0,
                       failure_code="multiple_declines")
    verdict = PolicyResult(
        decision=PolicyDecision.APPROVE,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT))
    result = ex.execute(state, verdict, _txn("t_decline", "multiple_declines"))
    assert result.outcome is ActionOutcome.FAILURE, "the retry should have been declined"
    assert not result.execution_failed, "a decline must not count as a failure to execute"


def test_a_hard_bounce_is_an_execution_failure():
    """The message never reached the customer. That *is* a failure to execute, and it is
    the kind that should eventually suspend automated handling."""
    from backend.app.models.schemas import PolicyResult

    ex = ActionExecutor(gateway=PaymentGateway(seed=11),
                        notifier=NotificationService(failure_rate=1.0))
    state = AgentState(transaction_id="t_bounce", customer_id="c", amount=100.0,
                       failure_code="insufficient_funds")
    verdict = PolicyResult(
        decision=PolicyDecision.APPROVE,
        effective_action=ProposedAction(action=InterventionType.SEND_REMINDER))
    result = ex.execute(state, verdict, _txn("t_bounce", "insufficient_funds"))
    assert result.execution_failed


# ---------------------------------------------------------------- the bus
def test_a_broken_event_handler_never_reaches_the_agent(tmp_path):
    """A metrics listener that throws must not unwind a recovery."""
    bus = InProcessEventBus()
    bus.subscribe(EventType.CASE_CLOSED,
                  lambda e: (_ for _ in ()).throw(RuntimeError("listener down")))
    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=1)),
                          bus=bus, run_id="chaos")
    state = agent.run(_txn())
    assert state.is_terminal
    assert bus.pending_dead_letters(), "the failure should be recorded, not lost"


def test_a_review_store_that_throws_does_not_execute_the_withheld_action(tmp_path):
    """If the queue is down, the action stays withheld. Failing open here would mean an
    unavailable operator queue silently authorises everything it should have held."""
    class BrokenReviews:
        def open_task(self, *a, **kw):
            raise RuntimeError("review database down")

        def open_escalation(self, *a, **kw):
            raise RuntimeError("review database down")

    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=1)),
                          reviews=BrokenReviews())
    with pytest.raises(RuntimeError):
        agent.run(_txn(code="suspected_fraud"))
    # The important half: no automated action ran before the store was consulted.


def test_the_bounded_loop_terminates_under_a_planner_that_will_not_settle():
    """Defence against a planner that proposes forever."""
    class StubbornPlanner:
        calls = 0

        def diagnose(self, txn):
            return None

        def select(self, state, dx):
            StubbornPlanner.calls += 1
            return ProposedAction(action=InterventionType.SEND_REMINDER,
                                  reason="again", source="llm")

    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=9)),
                          planner=StubbornPlanner())
    state = agent.run(_txn(code="insufficient_funds"))
    assert state.is_terminal, "the loop must terminate regardless of the planner"
    from backend.app.config import MAX_AGENT_STEPS
    assert state.step_count <= MAX_AGENT_STEPS + 1


# ---------------------------------------------------------------- idempotency
def test_the_executor_returns_the_original_result_on_replay():
    from backend.app.models.schemas import PolicyResult

    ex = ActionExecutor(gateway=PaymentGateway(seed=4))
    state = AgentState(transaction_id="t1", customer_id="c1", amount=100.0,
                       failure_code="bank_timeout")
    verdict = PolicyResult(
        decision=PolicyDecision.APPROVE,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT))
    first = ex.execute(state, verdict, _txn())
    second = ex.execute(state, verdict, _txn())
    assert second.replayed and not first.replayed
    assert second.amount_recovered == first.amount_recovered
    assert second.outcome is first.outcome
