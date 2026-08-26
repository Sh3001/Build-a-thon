"""Phase 5 -- the action executor: the only code in the system that causes side effects.

Two invariants make this safe:

1. **Nothing executes without a policy approval.** `execute()` takes a `PolicyResult` and
   runs `policy.effective_action` -- never the action originally proposed. Passing a
   rejected result raises. The agent cannot route around the gate, because the gate's
   output *is* the executor's input.
2. **Every action is idempotent.** A key derived from (transaction, action, attempt)
   caches the result, so a replayed step returns the original outcome instead of charging
   a customer twice. With an `ActionLedger` attached the cache is durable, so the
   guarantee survives a process restart instead of living only in this object.
3. **A dead address is not retried forever.** Hard bounces increment a per-(customer,
   channel) counter; once it crosses the threshold the pair is quarantined and the policy
   engine refuses further contact on it (`R-DLQ`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from backend.app.config import ACTION_COST_USD
from backend.app.database.operational import ActionLedger, DLQStore
from backend.app.models.enums import ActionOutcome, InterventionType
from backend.app.models.schemas import ActionResult, AgentState, PolicyResult
from simulation.notification_service import DeliveryFailure, NotificationService
from simulation.payment_gateway import PaymentGateway


class PolicyViolation(RuntimeError):
    """Raised when execution is attempted without a valid approval. Should be impossible."""


@dataclass
class ActionExecutor:
    gateway: PaymentGateway = field(default_factory=PaymentGateway)
    notifier: NotificationService = field(default_factory=NotificationService)
    #: idempotency key -> the result originally produced (in-process fast path)
    _seen: dict[str, ActionResult] = field(default_factory=dict, repr=False)
    #: Optional durable mirror of `_seen`. None keeps the original in-memory-only
    #: behaviour, which is what the unit tests want.
    ledger: ActionLedger | None = None
    #: Optional consecutive-failure tracker. None disables quarantining.
    dlq: DLQStore | None = None
    escalations: list[dict] = field(default_factory=list)
    bounces: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def idempotency_key(txn_id: str, action: str, nonce: int) -> str:
        return hashlib.sha256(f"{txn_id}:{action}:{nonce}".encode()).hexdigest()[:24]

    def _cost(self, action: InterventionType) -> float:
        return ACTION_COST_USD.get(action.value, 0.0)

    # ------------------------------------------------------------------ entry point
    def execute(self, state: AgentState, policy: PolicyResult, txn: dict) -> ActionResult:
        """Run the action the policy engine approved. Never the one that was proposed."""
        if not policy.allowed or policy.effective_action is None:
            raise PolicyViolation(
                f"refusing to execute: policy decision was {policy.decision.value}")

        action = policy.effective_action
        nonce = state.attempt_count if action.action is InterventionType.RETRY_PAYMENT \
            else state.contact_count
        key = self.idempotency_key(state.transaction_id, action.action.value, nonce)
        if key in self._seen:
            cached = self._seen[key]
            return cached.model_copy(update={"replayed": True})
        if self.ledger is not None:
            stored = self.ledger.recall(key)
            if stored is not None:
                cached = ActionResult(**stored)
                self._seen[key] = cached
                return cached.model_copy(update={"replayed": True})

        try:
            result = self._dispatch(state, policy, txn, key)
        except DeliveryFailure as exc:
            result = self._on_bounce(state, action.action, txn, exc, key)

        result.cost = self._cost(action.action)
        self._seen[key] = result
        if self.ledger is not None:
            self.ledger.remember(key, state.transaction_id, action.action.value,
                                 result.model_dump(mode="json"))
        return result

    def _on_bounce(self, state: AgentState, kind: InterventionType, txn: dict,
                   exc: DeliveryFailure, key: str) -> ActionResult:
        """A hard bounce costs money and delivers nothing. Recorded as a failure so the
        counter can accumulate; the quarantine decision belongs to the DLQ store."""
        channel = str(txn.get("preferred_channel", "email"))
        newly = False
        if self.dlq is not None:
            newly = self.dlq.record_failure(state.customer_id, channel, str(exc))
        self.bounces.append({"transaction_id": state.transaction_id,
                             "customer_id": state.customer_id, "channel": channel,
                             "action": kind.value, "quarantined": newly})
        detail = str(exc) + (" -- channel quarantined after repeated failures" if newly else "")
        return ActionResult(action=kind, outcome=ActionOutcome.FAILURE,
                            detail=detail, idempotency_key=key)

    def _dispatch(self, state: AgentState, policy: PolicyResult, txn: dict, key: str) -> ActionResult:
        action = policy.effective_action
        assert action is not None
        kind = action.action
        channel = (action.channel.value if action.channel else
                   str(txn.get("preferred_channel", "email")))
        hours = state.elapsed_hours + action.delay_hours

        if kind is InterventionType.RETRY_PAYMENT:
            return self.retry_payment(state, txn, hours, key)
        if kind is InterventionType.SEND_PAYMENT_LINK:
            return self.send_payment_link(state, txn, hours, channel, key)
        if kind is InterventionType.SEND_REMINDER:
            return self.send_reminder(state, txn, hours, channel, action.message, key)
        if kind is InterventionType.REQUEST_PAYMENT_METHOD_UPDATE:
            return self.request_payment_method_update(state, txn, channel, key)
        if kind is InterventionType.ESCALATE_CASE:
            return self.escalate_case(state, action.reason or "policy escalation", key)
        raise PolicyViolation(f"executor has no handler for {kind}")

    # ------------------------------------------------------------------ the tools
    def retry_payment(self, state: AgentState, txn: dict, hours: float, key: str = "") -> ActionResult:
        res = self.gateway.retry_payment(txn, hours_since_failure=hours,
                                         attempt=state.attempt_count + 1)
        return ActionResult(
            action=InterventionType.RETRY_PAYMENT,
            outcome=ActionOutcome.SUCCESS if res.success else ActionOutcome.FAILURE,
            amount_recovered=round(res.amount, 2), detail=res.detail, idempotency_key=key,
        )

    def _delivered(self, customer_id: str, channel: str) -> None:
        if self.dlq is not None:
            self.dlq.record_success(customer_id, channel)

    def send_payment_link(self, state: AgentState, txn: dict, hours: float,
                          channel: str, key: str = "") -> ActionResult:
        self.notifier.send("payment_link", txn, channel)
        self._delivered(state.customer_id, channel)
        res = self.gateway.customer_pays_via_link(txn, hours, channel, nonce=state.contact_count)
        return ActionResult(
            action=InterventionType.SEND_PAYMENT_LINK,
            outcome=ActionOutcome.SUCCESS if res.success else ActionOutcome.DELIVERED,
            amount_recovered=round(res.amount, 2),
            detail=f"link sent via {channel}; {res.detail}", idempotency_key=key,
        )

    def send_reminder(self, state: AgentState, txn: dict, hours: float, channel: str,
                      message: str | None = None, key: str = "") -> ActionResult:
        self.notifier.send("reminder", txn, channel, body=message)
        self._delivered(state.customer_id, channel)
        res = self.gateway.customer_responds_to_reminder(txn, hours, channel,
                                                         nonce=state.contact_count)
        return ActionResult(
            action=InterventionType.SEND_REMINDER,
            outcome=ActionOutcome.DELIVERED,
            detail=f"reminder sent via {channel}; {res.detail}", idempotency_key=key,
        )

    def request_payment_method_update(self, state: AgentState, txn: dict, channel: str,
                                      key: str = "") -> ActionResult:
        self.notifier.send("method_update", txn, channel)
        self._delivered(state.customer_id, channel)
        res = self.gateway.request_method_update(txn, channel, nonce=state.contact_count)
        return ActionResult(
            action=InterventionType.REQUEST_PAYMENT_METHOD_UPDATE,
            outcome=ActionOutcome.SUCCESS if res.success else ActionOutcome.DELIVERED,
            detail=f"update requested via {channel}; {res.detail}", idempotency_key=key,
        )

    def escalate_case(self, state: AgentState, reason: str, key: str = "") -> ActionResult:
        self.escalations.append({"transaction_id": state.transaction_id,
                                 "customer_id": state.customer_id,
                                 "amount_usd": state.amount_usd, "reason": reason})
        return ActionResult(
            action=InterventionType.ESCALATE_CASE, outcome=ActionOutcome.ESCALATED,
            detail=f"handed to human review: {reason}", idempotency_key=key,
        )

    def get_payment_status(self, transaction_id: str) -> dict:
        """Read-only status probe. Takes no policy approval because it changes nothing."""
        return {
            "transaction_id": transaction_id,
            "instrument_fixed": self.gateway.instrument_fixed(transaction_id),
            "contacts": self.gateway.contacts(transaction_id),
            "notifications_sent": self.notifier.count_for(transaction_id),
            "escalated": any(e["transaction_id"] == transaction_id for e in self.escalations),
        }
