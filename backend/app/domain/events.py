"""The event vocabulary and an in-process bus.

Deliberately small. The system is a modular monolith and should stay one -- introducing
Kafka to move a message between two functions in the same process would be complexity
bought for appearance rather than for a problem. What the abstraction buys is real:

* a **seam**. `EventBus` is a protocol with an in-process implementation. A Redis or
  Kafka adapter can be written against it later without touching a publisher.
* **durability of the record**. Handlers may fail; a failed delivery goes to the DLQ
  with its attempt count rather than vanishing into a traceback.
* **ordering guarantees that are stated**. The in-process bus is synchronous and
  ordered per publish. That is a property worth writing down, because code that
  quietly depends on it will break under a bus that does not offer it.

Handlers are isolated: one raising does not stop the others, and does not unwind the
publisher. A notification handler that throws must not roll back a payment.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class EventType(str, Enum):
    PAYMENT_FAILED = "payment.failed"
    PAYMENT_RETRY_REQUESTED = "payment.retry_requested"
    PAYMENT_RETRY_SUCCEEDED = "payment.retry_succeeded"
    PAYMENT_RETRY_FAILED = "payment.retry_failed"
    PAYMENT_RECOVERED = "payment.recovered"
    PAYMENT_EXPIRED = "payment.expired"
    CUSTOMER_CONTACTED = "customer.contacted"
    CUSTOMER_OPTED_OUT = "customer.opted_out"
    DELIVERY_BOUNCED = "delivery.bounced"
    POLICY_DENIED = "policy.denied"
    HUMAN_REVIEW_REQUESTED = "human_review.requested"
    HUMAN_REVIEW_RESOLVED = "human_review.resolved"
    CASE_CLOSED = "case.closed"


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Event:
    """One thing that happened. Immutable: an event is a record, not a workspace."""
    type: EventType
    tenant_id: str = "default"
    transaction_id: str = ""
    customer_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    at: datetime = field(default_factory=_now)
    #: Set by a producer that can be replayed (a webhook). The bus deduplicates on it.
    dedupe_key: str | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "type": self.type.value,
            "tenant_id": self.tenant_id, "transaction_id": self.transaction_id,
            "customer_id": self.customer_id, "at": self.at.isoformat(),
            "dedupe_key": self.dedupe_key, "payload": self.payload,
        }


Handler = Callable[[Event], None]


@runtime_checkable
class EventBus(Protocol):
    """The seam. A Redis/Kafka adapter implements exactly this."""

    def subscribe(self, event_type: EventType, handler: Handler) -> None: ...

    def publish(self, event: Event) -> bool: ...


@dataclass
class DeadLetter:
    event: Event
    handler: str
    error: str
    attempts: int = 1
    first_failed_at: datetime = field(default_factory=_now)
    last_attempt_at: datetime = field(default_factory=_now)
    quarantined: bool = False
    quarantine_reason: str = ""
    resolved: bool = False


@dataclass
class InProcessEventBus:
    """Synchronous, ordered, in-process. The local-development default.

    Handlers run in subscription order and are isolated from one another. A handler that
    raises produces a dead letter and the publish continues -- the alternative, unwinding
    into the caller, would let a broken metrics listener abort a recovery.
    """
    max_attempts: int = 3
    _handlers: dict[EventType, list[Handler]] = field(default_factory=lambda: defaultdict(list))
    _seen: set[str] = field(default_factory=set)
    _log: list[Event] = field(default_factory=list)
    dead_letters: list[DeadLetter] = field(default_factory=list)
    published: int = 0
    duplicates_dropped: int = 0

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: Event) -> bool:
        """Returns False if the event was a duplicate and was dropped."""
        if event.dedupe_key is not None:
            if event.dedupe_key in self._seen:
                self.duplicates_dropped += 1
                return False
            self._seen.add(event.dedupe_key)

        self._log.append(event)
        self.published += 1
        for handler in self._handlers[event.type]:
            self._deliver(event, handler)
        return True

    def _deliver(self, event: Event, handler: Handler) -> None:
        name = getattr(handler, "__qualname__", repr(handler))
        try:
            handler(event)
        except Exception as exc:
            self.dead_letters.append(DeadLetter(
                event=event, handler=name, error=f"{type(exc).__name__}: {exc}"))

    # ------------------------------------------------------------------ DLQ ops
    def retry_dead_letters(self) -> tuple[int, int]:
        """Re-deliver every unresolved dead letter. Returns (recovered, still_failing).

        A letter that exceeds `max_attempts` is quarantined rather than retried forever:
        an event that fails deterministically will fail deterministically again, and the
        loop only burns budget while hiding the problem from an operator.
        """
        recovered = failing = 0
        for dl in self.dead_letters:
            if dl.resolved or dl.quarantined:
                continue
            handlers = [h for h in self._handlers[dl.event.type]
                        if getattr(h, "__qualname__", repr(h)) == dl.handler]
            dl.attempts += 1
            dl.last_attempt_at = _now()
            try:
                for h in handlers:
                    h(dl.event)
                dl.resolved = True
                recovered += 1
            except Exception as exc:
                dl.error = f"{type(exc).__name__}: {exc}"
                failing += 1
                if dl.attempts >= self.max_attempts:
                    dl.quarantined = True
                    dl.quarantine_reason = (
                        f"failed {dl.attempts} times; deterministic failure assumed")
        return recovered, failing

    def resolve(self, event_id: str) -> bool:
        """Operator marks a dead letter as handled out of band."""
        for dl in self.dead_letters:
            if dl.event.event_id == event_id and not dl.resolved:
                dl.resolved = True
                return True
        return False

    def pending_dead_letters(self) -> list[DeadLetter]:
        return [dl for dl in self.dead_letters if not dl.resolved]

    def history(self, event_type: EventType | None = None) -> list[Event]:
        return [e for e in self._log if event_type is None or e.type is event_type]

    def stats(self) -> dict:
        pending = self.pending_dead_letters()
        return {
            "published": self.published,
            "duplicates_dropped": self.duplicates_dropped,
            "handlers": {t.value: len(hs) for t, hs in self._handlers.items() if hs},
            "dead_letters": len(pending),
            "quarantined": sum(1 for dl in pending if dl.quarantined),
        }


def bus_from_env(url: str | None = None, **kw: Any) -> EventBus:
    """Factory. Only the in-process bus exists today; the signature is the extension
    point, and it fails loudly rather than silently downgrading, because a run that
    thinks it published to Redis and did not is worse than one that refuses to start."""
    if not url:
        return InProcessEventBus(**kw)
    raise NotImplementedError(
        f"no event-bus adapter for {url!r}. The in-process bus is the only "
        f"implementation; add an adapter satisfying the EventBus protocol.")


def publish_all(bus: EventBus, events: Iterable[Event]) -> int:
    return sum(1 for e in events if bus.publish(e))
