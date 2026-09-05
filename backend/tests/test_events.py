"""The event bus.

Handler isolation is the load-bearing property: a metrics listener that throws must not
unwind a recovery.
"""
from __future__ import annotations

import pytest

from backend.app.domain.events import (
    Event,
    EventType,
    InProcessEventBus,
    bus_from_env,
    publish_all,
)


def _boom(_event):
    raise RuntimeError("handler exploded")


def test_a_failing_handler_does_not_stop_the_others():
    bus = InProcessEventBus()
    seen = []
    bus.subscribe(EventType.PAYMENT_FAILED, _boom)
    bus.subscribe(EventType.PAYMENT_FAILED, seen.append)
    assert bus.publish(Event(EventType.PAYMENT_FAILED, transaction_id="t1"))
    assert len(seen) == 1, "a broken handler must not starve the ones after it"
    assert len(bus.pending_dead_letters()) == 1


def test_a_failing_handler_does_not_raise_into_the_publisher():
    """A notification failure must never roll back a payment."""
    bus = InProcessEventBus()
    bus.subscribe(EventType.PAYMENT_RECOVERED, _boom)
    bus.publish(Event(EventType.PAYMENT_RECOVERED))           # must not raise


def test_duplicates_are_dropped_on_the_dedupe_key():
    bus = InProcessEventBus()
    seen = []
    bus.subscribe(EventType.PAYMENT_RECOVERED, seen.append)
    assert bus.publish(Event(EventType.PAYMENT_RECOVERED, dedupe_key="evt_1"))
    assert not bus.publish(Event(EventType.PAYMENT_RECOVERED, dedupe_key="evt_1"))
    assert len(seen) == 1
    assert bus.stats()["duplicates_dropped"] == 1


def test_events_without_a_dedupe_key_are_never_deduplicated():
    """Two genuinely distinct failures on one case must both be delivered."""
    bus = InProcessEventBus()
    seen = []
    bus.subscribe(EventType.PAYMENT_RETRY_FAILED, seen.append)
    publish_all(bus, [Event(EventType.PAYMENT_RETRY_FAILED) for _ in range(3)])
    assert len(seen) == 3


def test_dead_letters_are_quarantined_rather_than_retried_forever():
    bus = InProcessEventBus(max_attempts=3)
    bus.subscribe(EventType.PAYMENT_FAILED, _boom)
    bus.publish(Event(EventType.PAYMENT_FAILED))
    for _ in range(5):
        bus.retry_dead_letters()
    dl = bus.pending_dead_letters()[0]
    assert dl.quarantined and dl.attempts <= 4, \
        "a deterministic failure must stop being retried"


def test_a_recovered_handler_resolves_its_dead_letter():
    bus = InProcessEventBus()
    state = {"fail": True}

    def flaky(_event):
        if state["fail"]:
            raise RuntimeError("transient")

    bus.subscribe(EventType.PAYMENT_FAILED, flaky)
    bus.publish(Event(EventType.PAYMENT_FAILED))
    state["fail"] = False
    recovered, failing = bus.retry_dead_letters()
    assert (recovered, failing) == (1, 0)
    assert not bus.pending_dead_letters()


def test_an_unknown_bus_url_refuses_rather_than_downgrading():
    """A run that thinks it published to Redis and did not is worse than one that
    refuses to start."""
    assert isinstance(bus_from_env(None), InProcessEventBus)
    with pytest.raises(NotImplementedError):
        bus_from_env("redis://localhost:6379")
