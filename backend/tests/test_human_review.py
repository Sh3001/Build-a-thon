"""Human review: the queue, the override record, and the end-to-end path through the agent."""
from __future__ import annotations

from datetime import UTC

import pytest

from backend.app.agents.graph import RecoveryAgent
from backend.app.database.review import (
    NotAuthorised,
    ReviewClosed,
    ReviewStore,
)
from backend.app.models.enums import (
    CaseStatus,
    InterventionType,
    PolicyDecision,
    ReviewStatus,
    Role,
)
from backend.app.models.schemas import PolicyResult, ProposedAction
from backend.app.tools.executor import ActionExecutor
from simulation.payment_gateway import PaymentGateway


@pytest.fixture
def store(tmp_path):
    return ReviewStore(tmp_path / "review.db")


@pytest.fixture
def verdict():
    return PolicyResult(
        decision=PolicyDecision.HUMAN_REVIEW,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT,
                                        reason="high-value re-presentment"),
        rules_fired=["R-AMOUNT-CAP"],
        reason="R-AMOUNT-CAP: $9,000.00 exceeds the automatic recovery ceiling")


def test_only_a_review_verdict_opens_a_task(store):
    approve = PolicyResult(decision=PolicyDecision.APPROVE,
                           effective_action=ProposedAction(action=InterventionType.WAIT))
    with pytest.raises(ValueError):
        store.open_task("txn_1", approve)


def test_repeated_verdicts_on_one_case_do_not_flood_the_queue(store, verdict):
    """A case that re-plans hits the same rule again. One task, not one per iteration."""
    first = store.open_task("txn_1", verdict, amount_usd=9000.0)
    for _ in range(5):
        assert store.open_task("txn_1", verdict, amount_usd=9000.0) == first
    assert store.stats()["pending"] == 1


def test_a_read_only_role_cannot_resolve(store, verdict):
    rid = store.open_task("txn_1", verdict, amount_usd=9000.0)
    for role in (Role.ANALYST, Role.AUDITOR, Role.SYSTEM):
        with pytest.raises(NotAuthorised):
            store.approve(rid, "mallory", role, "let me through")
    assert store.get(rid)["status"] == ReviewStatus.PENDING.value


def test_an_override_without_a_reason_is_refused(store, verdict):
    """An unexplained override is indistinguishable from a mistake a year later."""
    rid = store.open_task("txn_1", verdict, amount_usd=9000.0)
    for blank in ("", "   "):
        with pytest.raises(ValueError):
            store.approve(rid, "alice", Role.OPERATOR, blank)


def test_a_resolved_task_cannot_be_re_decided(store, verdict):
    """Otherwise two operators can disagree and the last writer silently wins."""
    rid = store.open_task("txn_1", verdict, amount_usd=9000.0)
    store.approve(rid, "alice", Role.OPERATOR, "confirmed with the merchant")
    with pytest.raises(ReviewClosed):
        store.reject(rid, "bob", Role.OPERATOR, "I disagree")
    assert store.get(rid)["status"] == ReviewStatus.APPROVED.value


def test_approval_returns_a_verdict_the_executor_will_accept(store, verdict):
    """The override reaches execution through the same gate as everything else, not
    through a side door that skips it."""
    rid = store.open_task("txn_1", verdict, amount_usd=9000.0)
    result = store.approve(rid, "alice", Role.OPERATOR, "verified by phone")
    assert result.decision is PolicyDecision.APPROVE
    assert result.allowed
    assert result.effective_action.action is InterventionType.RETRY_PAYMENT
    assert "R-HUMAN-OVERRIDE" in result.rules_fired


def test_rejection_yields_nothing_executable(store, verdict):
    rid = store.open_task("txn_1", verdict, amount_usd=9000.0)
    result = store.reject(rid, "alice", Role.OPERATOR, "customer disputes the invoice")
    assert result.decision is PolicyDecision.REJECT
    assert result.effective_action is None and not result.allowed


def test_every_override_is_recorded_with_who_when_and_why(store, verdict):
    rid = store.open_task("txn_1", verdict, amount_usd=9000.0)
    store.approve(rid, "alice", Role.OPERATOR, "merchant confirmed the invoice by phone")
    row = store.overrides()[0]
    assert row["actor"] == "alice"
    assert row["actor_role"] == "operator"
    assert row["original_decision"] == "human_review"
    assert row["new_decision"] == "approve"
    assert "merchant confirmed" in row["reason"]
    assert row["at"]


def test_overdue_tasks_expire(store, verdict):
    """An unbounded queue is not a safety mechanism; it is where decisions go to be
    forgotten."""
    from datetime import datetime, timedelta
    store.open_task("txn_1", verdict, amount_usd=9000.0)
    later = datetime.now(UTC) + timedelta(hours=store.sla_hours + 1)
    assert store.expire_overdue(now=later) == 1
    assert store.stats()["pending"] == 0


def test_the_queue_is_ordered_by_value(store, verdict):
    """An operator's minute is the scarce resource."""
    for tid, amount in [("small", 50.0), ("huge", 90_000.0), ("mid", 5_000.0)]:
        store.open_task(tid, verdict, amount_usd=amount)
    assert [r["transaction_id"] for r in store.pending()] == ["huge", "mid", "small"]


# ---------------------------------------------------------------- end to end
def _txn(tid: str, code: str, amount: float) -> dict:
    return {"transaction_id": tid, "customer_id": "c1", "amount": amount,
            "currency": "USD", "amount_usd": amount, "payment_method": "card",
            "failure_code": code, "avg_transaction_value": 200.0,
            "preferred_channel": "email"}


def test_an_escalated_case_produces_a_review_task(tmp_path):
    """An escalation with no task attached is a case the agent thinks a human owns and
    no human has been told about."""
    store = ReviewStore(tmp_path / "e2e.db")
    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=1)),
                          reviews=store, run_id="test")
    state = agent.run(_txn("txn_fraud", "suspected_fraud", 400.0))

    assert state.status is CaseStatus.ESCALATED
    assert state.review_ids, "escalation left no trace in the operator queue"
    assert store.stats()["pending"] == 1
    assert store.get(state.review_ids[0])["transaction_id"] == "txn_fraud"


def test_review_is_optional_and_never_unblocks_a_case(tmp_path):
    """Without a store attached the verdict still blocks execution: review is how a
    withheld case gets *unblocked*, never how it gets blocked."""
    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=1)),
                          reviews=None)
    state = agent.run(_txn("txn_fraud", "suspected_fraud", 400.0))
    assert state.status is CaseStatus.ESCALATED
    assert not any(a == "retry_payment" for a in state.actions_taken)


def test_an_opted_out_customer_receives_no_contact(tmp_path):
    """The one limit no expected-profit calculation may outbid."""
    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=2)),
                          opted_out=frozenset({"c1"}))
    state = agent.run(_txn("txn_1", "expired_card", 300.0))
    contacts = {"send_reminder", "send_payment_link", "request_payment_method_update"}
    assert not (set(state.actions_taken) & contacts), \
        f"contacted an opted-out customer: {state.actions_taken}"
