"""Phase 5 -- mock notification service. Renders the customer-facing message and records
the send. Whether the customer *acts* is decided by the payment gateway, not here, so
message copy can never be mistaken for collection.

Delivery can fail. A hard bounce is a property of the (customer, channel) pair, not of
the individual send: the address is derived by hash, so a bad address bounces every time
rather than intermittently. That is what makes a consecutive-failure counter meaningful
-- an intermittent failure would never accumulate, and quarantining on it would be wrong.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.app.config import DELIVERY_FAILURE_RATE


@dataclass
class Notification:
    customer_id: str
    transaction_id: str
    channel: str
    kind: str
    body: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


TEMPLATES: dict[str, str] = {
    "reminder": ("Hi -- your payment of {amount} for invoice {invoice} did not go through "
                 "({reason}). You can retry any time; we will also try again automatically."),
    "payment_link": ("Hi -- your payment of {amount} for invoice {invoice} is still "
                     "outstanding. Pay securely here: {link}"),
    "method_update": ("Hi -- we could not charge your saved payment method for {amount} "
                      "({reason}). Please update your card details here: {link}"),
}


class DeliveryFailure(RuntimeError):
    """A hard bounce. Distinct from "delivered but ignored", which is not a failure."""


@dataclass
class NotificationService:
    sent: list[Notification] = field(default_factory=list)
    failed: list[Notification] = field(default_factory=list)
    failure_rate: float = DELIVERY_FAILURE_RATE

    def address_is_dead(self, customer_id: str, channel: str) -> bool:
        """Deterministic per (customer, channel) -- seeded by hash, never by a clock or a
        shared RNG, so a run replays identically and two arms see the same dead addresses."""
        if self.failure_rate <= 0:
            return False
        h = hashlib.sha256(f"{customer_id}:{channel}".encode()).hexdigest()
        return (int(h[:8], 16) % 10_000) < int(self.failure_rate * 10_000)

    def _link(self, txn_id: str) -> str:
        return f"https://pay.example.test/r/{txn_id[-8:]}"

    def render(self, kind: str, txn: dict, reason: str = "") -> str:
        amount = f"{txn.get('currency', 'USD')} {float(txn.get('amount', 0)):,.2f}"
        return TEMPLATES[kind].format(
            amount=amount,
            invoice=txn.get("invoice_id") or txn.get("transaction_id", ""),
            reason=reason or str(txn.get("failure_code", "payment declined")).replace("_", " "),
            link=self._link(txn.get("transaction_id", "")),
        )

    def send(self, kind: str, txn: dict, channel: str, reason: str = "",
             body: str | None = None) -> Notification:
        """Raises DeliveryFailure on a hard bounce. Callers must decide what that means;
        the notifier does not retry, because retrying a dead address is the exact
        behaviour the DLQ exists to stop."""
        customer_id = str(txn.get("customer_id", ""))
        n = Notification(
            customer_id=customer_id,
            transaction_id=str(txn.get("transaction_id", "")),
            channel=channel, kind=kind,
            body=body or self.render(kind, txn, reason),
        )
        if self.address_is_dead(customer_id, channel):
            self.failed.append(n)
            raise DeliveryFailure(f"hard bounce: {channel} address for {customer_id} is undeliverable")
        self.sent.append(n)
        return n

    def count_for(self, transaction_id: str) -> int:
        return sum(1 for n in self.sent if n.transaction_id == transaction_id)
