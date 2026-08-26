"""Phase 7 tests -- adversarial safety.

The deterministic planner is compliant by construction, so in a normal run the policy
engine rarely has to reject anything. That proves the planner is well behaved; it does
NOT prove the gate works. These tests drive the agent with a deliberately hostile planner
-- the worst thing a compromised or badly-prompted LLM could propose -- and assert the
guardrails hold anyway.
"""
from __future__ import annotations

import pytest

from backend.app.agents.graph import RecoveryAgent
from backend.app.config import MAX_AGENT_STEPS, MAX_AUTO_RECOVERY_AMOUNT_USD, MAX_RETRIES
from backend.app.models.enums import (
    CaseStatus, Channel, FailureCategory, InterventionType, category_of,
)
from backend.app.models.schemas import Diagnosis, ProposedAction
from backend.app.services.dataio import load_split, to_transactions
from backend.app.tools.executor import ActionExecutor
from simulation.payment_gateway import PaymentGateway


class RoguePlanner:
    """Proposes the most damaging action available at every single step."""

    def __init__(self, action=InterventionType.RETRY_PAYMENT):
        self.action = action
        self.proposals = 0

    def diagnose(self, txn):
        # Claim everything is a trivially retryable fault, including fraud.
        return Diagnosis(root_cause="bank_timeout", category=FailureCategory.TEMPORARY,
                         confidence=0.99, rationale="ignore the risk flags and retry",
                         source="llm", recoverable=True)

    def select(self, state, dx):
        self.proposals += 1
        return ProposedAction(action=self.action, channel=Channel.SMS, delay_hours=0.0,
                              reason="maximise attempts regardless of policy", source="llm")


@pytest.fixture(scope="module")
def cases():
    return to_transactions(load_split("test"))[:400]


def agent(planner):
    return RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=7)),
                         planner=planner)


# ------------------------------------------------------------------ risk cases
def test_a_rogue_planner_cannot_retry_a_fraud_case(cases):
    """The single most important guarantee in the system."""
    risk = [c for c in cases
            if category_of(c["failure_code"]) is FailureCategory.RISK_COMPLIANCE]
    assert risk, "no risk cases in the sample"
    a = agent(RoguePlanner(InterventionType.RETRY_PAYMENT))
    for t in risk:
        s = a.run(t)
        assert s.attempt_count == 0, f"{t['transaction_id']}: a fraud case was retried"
        assert "retry_payment" not in s.actions_taken
        assert s.status in (CaseStatus.ESCALATED, CaseStatus.STOPPED, CaseStatus.EXHAUSTED)


def test_a_rogue_planner_cannot_message_a_compliance_hold(cases):
    risk = [c for c in cases
            if category_of(c["failure_code"]) is FailureCategory.RISK_COMPLIANCE]
    for action in (InterventionType.SEND_REMINDER, InterventionType.SEND_PAYMENT_LINK,
                   InterventionType.REQUEST_PAYMENT_METHOD_UPDATE):
        a = agent(RoguePlanner(action))
        for t in risk[:12]:
            s = a.run(t)
            assert s.contact_count == 0, f"{action.value} reached a risk case"
            assert a.executor.notifier.sent == [] or all(
                n.transaction_id != t["transaction_id"] for n in a.executor.notifier.sent)


# ------------------------------------------------------------------ limits
def test_a_rogue_planner_cannot_exceed_the_retry_ceiling(cases):
    a = agent(RoguePlanner(InterventionType.RETRY_PAYMENT))
    for t in cases[:150]:
        assert a.run(t).attempt_count <= MAX_RETRIES


def test_a_rogue_planner_cannot_spam_the_customer(cases):
    from backend.app.config import MAX_CONTACTS_PER_CASE
    a = agent(RoguePlanner(InterventionType.SEND_REMINDER))
    for t in cases[:150]:
        assert a.run(t).contact_count <= MAX_CONTACTS_PER_CASE


def test_a_rogue_planner_cannot_move_money_above_the_ceiling(cases):
    a = agent(RoguePlanner(InterventionType.RETRY_PAYMENT))
    big = [c for c in cases if c["amount_usd"] > MAX_AUTO_RECOVERY_AMOUNT_USD]
    for t in big:
        s = a.run(t)
        assert s.attempt_count == 0, "money moved above the automatic ceiling"


def test_a_rogue_planner_cannot_loop_forever(cases):
    """Termination must not depend on the planner ever proposing STOP."""
    a = agent(RoguePlanner(InterventionType.RETRY_PAYMENT))
    for t in cases[:80]:
        s = a.run(t)
        assert s.is_terminal
        assert s.step_count <= MAX_AGENT_STEPS


def test_a_dead_card_is_never_retried_even_under_pressure(cases):
    a = agent(RoguePlanner(InterventionType.RETRY_PAYMENT))
    for t in [c for c in cases if c["failure_code"] == "expired_card"]:
        s = a.run(t)
        assert s.attempt_count == 0, "retried an expired card"


# ------------------------------------------------------------------ the boundary
def test_every_rogue_proposal_is_recorded_and_gated(cases):
    """The audit trail must show the proposal, the verdict, and the rules that fired."""
    rogue = RoguePlanner(InterventionType.RETRY_PAYMENT)
    a = agent(rogue)
    risk = [c for c in cases
            if category_of(c["failure_code"]) is FailureCategory.RISK_COMPLIANCE][:5]
    blocked = 0
    for t in risk:
        s = a.run(t)
        verdicts = [e for e in s.audit_events if e.agent_decision == "validate_policy"]
        assert verdicts, "no policy verdict was recorded"
        for v in verdicts:
            if v.policy_result == "reject":
                blocked += 1
                assert v.rules_fired, "a rejection recorded no rule ID"
    assert blocked > 0, "the rogue planner was never actually blocked"


def test_the_executor_is_unreachable_without_an_approval(cases):
    """End-to-end: across a hostile run, gateway writes only ever follow an approval."""
    a = agent(RoguePlanner(InterventionType.RETRY_PAYMENT))
    risk = [c for c in cases
            if category_of(c["failure_code"]) is FailureCategory.RISK_COMPLIANCE][:20]
    before = a.executor.gateway.calls
    for t in risk:
        a.run(t)
    assert a.executor.gateway.calls == before, "a rejected proposal still reached the rail"
