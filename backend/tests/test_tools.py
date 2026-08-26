"""Phase 5 tests -- execution safety, idempotency, and the tool contracts."""
from __future__ import annotations

import pytest

from backend.app.config import ACTION_COST_USD
from backend.app.models.enums import ActionOutcome, Channel, InterventionType, PolicyDecision
from backend.app.models.schemas import AgentState, PolicyResult, ProposedAction
from backend.app.tools.executor import ActionExecutor, PolicyViolation
from simulation.notification_service import NotificationService
from simulation.payment_gateway import PaymentGateway


def txn(tid="t1", code="bank_timeout", amount=100.0):
    return {"transaction_id": tid, "customer_id": "c1", "failure_code": code,
            "amount": amount, "amount_usd": amount, "currency": "USD", "invoice_id": "inv1",
            "previous_success_rate": 0.8, "failure_count": 1, "preferred_channel": "sms"}


def state(tid="t1", code="bank_timeout", **kw):
    return AgentState(transaction_id=tid, customer_id="c1", amount=100.0,
                      failure_code=code, **kw)


def approved(action=InterventionType.RETRY_PAYMENT, **kw):
    return PolicyResult(decision=PolicyDecision.APPROVE,
                        effective_action=ProposedAction(action=action, **kw))


def ex(seed=5):
    return ActionExecutor(gateway=PaymentGateway(seed=seed))


# ------------------------------------------------------------------ the gate
def test_execution_requires_policy_approval():
    """The central safety property: no approval, no side effect."""
    e = ex()
    for decision in (PolicyDecision.REJECT,):
        bad = PolicyResult(decision=decision, effective_action=None)
        with pytest.raises(PolicyViolation):
            e.execute(state(), bad, txn())
    assert e.gateway.calls == 0, "a rejected action still touched the gateway"


def test_executor_runs_the_effective_action_not_the_proposed_one():
    """The policy engine may rewrite an action; the executor must honour the rewrite."""
    e = ex()
    policy = PolicyResult(
        decision=PolicyDecision.MODIFY,
        effective_action=ProposedAction(action=InterventionType.ESCALATE_CASE, reason="too big"),
    )
    r = e.execute(state(), policy, txn())
    assert r.action is InterventionType.ESCALATE_CASE
    assert e.escalations and e.escalations[0]["reason"] == "too big"


def test_a_rejected_action_never_reaches_a_tool():
    e = ex()
    with pytest.raises(PolicyViolation):
        e.execute(state(), PolicyResult(decision=PolicyDecision.REJECT), txn())
    assert e.notifier.sent == []
    assert e.escalations == []


# ------------------------------------------------------------------ idempotency
def test_repeated_execution_returns_the_cached_result():
    e = ex()
    a = e.execute(state(), approved(), txn())
    calls_after_first = e.gateway.calls
    b = e.execute(state(), approved(), txn())
    assert b.replayed is True and a.replayed is False
    assert (a.outcome, a.amount_recovered, a.detail) == (b.outcome, b.amount_recovered, b.detail)
    assert e.gateway.calls == calls_after_first, "replay charged the customer again"


def test_a_new_attempt_is_a_new_action_not_a_replay():
    """Retry 2 must actually execute -- idempotency must not swallow intended retries."""
    e = ex()
    e.execute(state(attempt_count=0), approved(), txn())
    calls = e.gateway.calls
    r = e.execute(state(attempt_count=1), approved(), txn())
    assert r.replayed is False
    assert e.gateway.calls == calls + 1


def test_idempotency_keys_are_distinct_per_case_and_action():
    k = ActionExecutor.idempotency_key
    assert k("t1", "retry_payment", 0) != k("t2", "retry_payment", 0)
    assert k("t1", "retry_payment", 0) != k("t1", "send_reminder", 0)
    assert k("t1", "retry_payment", 0) != k("t1", "retry_payment", 1)
    assert k("t1", "retry_payment", 0) == k("t1", "retry_payment", 0)


def test_one_shot_actions_do_not_double_send():
    e = ex()
    s = state(code="expired_card")
    e.execute(s, approved(InterventionType.REQUEST_PAYMENT_METHOD_UPDATE, channel=Channel.SMS), txn())
    e.execute(s, approved(InterventionType.REQUEST_PAYMENT_METHOD_UPDATE, channel=Channel.SMS), txn())
    assert len(e.notifier.sent) == 1, "the customer was messaged twice for one decision"


# ------------------------------------------------------------------ tool contracts
def test_retry_reports_recovered_amount_only_on_success():
    e = ex(seed=2)
    results = [e.execute(state(tid=f"r{i}"), approved(), txn(f"r{i}", "network_error"))
               for i in range(60)]
    for r in results:
        if r.outcome is ActionOutcome.SUCCESS:
            assert r.amount_recovered == 100.0
        else:
            assert r.amount_recovered == 0.0


def test_reminder_never_collects_money_directly():
    """A reminder raises the odds of a later retry; it must not book revenue itself."""
    e = ex(seed=3)
    for i in range(40):
        r = e.execute(state(tid=f"m{i}"), approved(InterventionType.SEND_REMINDER, channel=Channel.SMS),
                      txn(f"m{i}", "insufficient_funds"))
        assert r.amount_recovered == 0.0
        assert r.outcome is ActionOutcome.DELIVERED


def test_payment_link_can_collect_on_a_dead_instrument():
    """The link routes around the failed method -- that is why it is worth having."""
    e = ex(seed=8)
    wins = 0
    for i in range(120):
        r = e.execute(state(tid=f"l{i}"), approved(InterventionType.SEND_PAYMENT_LINK, channel=Channel.WHATSAPP),
                      txn(f"l{i}", "expired_card"))
        wins += r.outcome is ActionOutcome.SUCCESS
    assert wins > 0, "payment links never collected on expired cards"


def test_method_update_unlocks_retry():
    e = ex(seed=8)
    fixed = None
    for i in range(120):
        tid = f"u{i}"
        r = e.execute(state(tid=tid), approved(InterventionType.REQUEST_PAYMENT_METHOD_UPDATE, channel=Channel.SMS),
                      txn(tid, "expired_card"))
        if e.gateway.instrument_fixed(tid):
            fixed = tid
            break
    assert fixed, "no customer ever updated their method"
    r = e.gateway.retry_payment(txn(fixed, "expired_card"), 24, 1)
    assert r.probability > 0.0, "retry still impossible after the instrument was replaced"


def test_escalation_records_the_case_for_a_human():
    e = ex()
    e.execute(state(code="suspected_fraud"), approved(InterventionType.ESCALATE_CASE, reason="fraud hold"), txn())
    assert e.escalations[0]["transaction_id"] == "t1"
    assert e.get_payment_status("t1")["escalated"] is True


def test_get_payment_status_is_read_only():
    e = ex()
    before = e.gateway.calls
    e.get_payment_status("t1")
    assert e.gateway.calls == before
    assert e.notifier.sent == []


def test_costs_are_attached_to_every_action():
    e = ex()
    r = e.execute(state(), approved(), txn())
    assert r.cost == ACTION_COST_USD["retry_payment"]
    r2 = e.execute(state(code="suspected_fraud"), approved(InterventionType.ESCALATE_CASE, reason="x"), txn("t2"))
    assert r2.cost == ACTION_COST_USD["escalate_case"]


# ------------------------------------------------------------------ notifications
def test_messages_render_with_case_details():
    n = NotificationService()
    body = n.render("payment_link", txn(), "card declined")
    assert "inv1" in body and "USD 100.00" in body and "https://" in body


def test_notification_is_logged_per_case():
    # Delivery failures off: this test is about the send log, not about bounces.
    n = NotificationService(failure_rate=0.0)
    n.send("reminder", txn("t7"), "sms")
    n.send("reminder", txn("t7"), "sms")
    n.send("reminder", txn("t8"), "email")
    assert n.count_for("t7") == 2 and n.count_for("t8") == 1
