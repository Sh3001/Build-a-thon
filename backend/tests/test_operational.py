"""Durable idempotency, the DLQ counter, and the delivery-failure simulation.

These run on SQLite (temp files), so they need no server. The Postgres equivalents live
in test_postgres.py behind RECOVERAI_TEST_DB_URL.
"""
from __future__ import annotations

import pytest

from backend.app.database.operational import ActionLedger, DLQStore
from simulation.notification_service import DeliveryFailure, NotificationService


# ---------------------------------------------------------------- action ledger
def test_ledger_recalls_a_stored_result(tmp_path):
    led = ActionLedger("run-1", tmp_path / "a.db", url=None)
    assert led.recall("k1") is None
    led.remember("k1", "txn_1", "retry_payment", {"outcome": "success", "amount": 10})
    assert led.recall("k1") == {"outcome": "success", "amount": 10}
    led.close()


def test_ledger_survives_a_reopen(tmp_path):
    """The whole point: a restarted process must not re-charge a customer."""
    db = tmp_path / "a.db"
    led = ActionLedger("run-1", db, url=None)
    led.remember("k1", "txn_1", "retry_payment", {"outcome": "success"})
    led.close()

    reopened = ActionLedger("run-1", db, url=None)
    assert reopened.recall("k1") == {"outcome": "success"}
    reopened.close()


def test_ledger_is_scoped_by_run(tmp_path):
    """Without run scoping, a second experiment would replay the first one's results and
    recover nothing -- a silent no-op that would look like a catastrophic regression."""
    db = tmp_path / "a.db"
    first = ActionLedger("run-1", db, url=None)
    first.remember("k1", "txn_1", "retry_payment", {"outcome": "success"})
    first.close()

    second = ActionLedger("run-2", db, url=None)
    assert second.recall("k1") is None
    second.close()


def test_ledger_keeps_the_first_result_for_a_key(tmp_path):
    led = ActionLedger("run-1", tmp_path / "a.db", url=None)
    led.remember("k1", "txn_1", "retry_payment", {"outcome": "success"})
    led.remember("k1", "txn_1", "retry_payment", {"outcome": "failure"})
    assert led.recall("k1") == {"outcome": "success"}
    led.close()


# ---------------------------------------------------------------- DLQ
def test_dlq_quarantines_only_at_the_threshold(tmp_path):
    d = DLQStore("run-1", tmp_path / "d.db", url=None, threshold=3)
    assert d.record_failure("c1", "email", "bounce") is False
    assert d.is_quarantined("c1", "email") is False
    assert d.record_failure("c1", "email", "bounce") is False
    assert d.record_failure("c1", "email", "bounce") is True      # third trips it
    assert d.is_quarantined("c1", "email") is True
    d.close()


def test_dlq_reports_tipping_only_once(tmp_path):
    d = DLQStore("run-1", tmp_path / "d.db", url=None, threshold=2)
    d.record_failure("c1", "email", "bounce")
    assert d.record_failure("c1", "email", "bounce") is True
    assert d.record_failure("c1", "email", "bounce") is False     # already quarantined
    d.close()


def test_dlq_counter_resets_on_a_successful_delivery(tmp_path):
    """Consecutive, not cumulative: an outage that recovers must not quarantine anyone."""
    d = DLQStore("run-1", tmp_path / "d.db", url=None, threshold=3)
    d.record_failure("c1", "email", "bounce")
    d.record_failure("c1", "email", "bounce")
    d.record_success("c1", "email")
    assert d.record_failure("c1", "email", "bounce") is False
    assert d.is_quarantined("c1", "email") is False
    d.close()


def test_dlq_is_per_channel(tmp_path):
    d = DLQStore("run-1", tmp_path / "d.db", url=None, threshold=1)
    d.record_failure("c1", "email", "bounce")
    assert d.is_quarantined("c1", "email") is True
    assert d.is_quarantined("c1", "sms") is False
    d.close()


def test_dlq_release_puts_a_pair_back_in_service(tmp_path):
    d = DLQStore("run-1", tmp_path / "d.db", url=None, threshold=1)
    d.record_failure("c1", "email", "bounce")
    d.release("c1", "email")
    assert d.is_quarantined("c1", "email") is False
    d.close()


def test_dlq_entries_and_stats(tmp_path):
    d = DLQStore("run-1", tmp_path / "d.db", url=None, threshold=1)
    d.record_failure("c1", "email", "bounce")
    d.record_failure("c2", "sms", "bounce")
    assert len(d.entries()) == 2
    s = d.stats()
    assert s["quarantined"] == 2 and s["tracked_pairs"] == 2 and s["threshold"] == 1
    d.close()


# ---------------------------------------------------------------- delivery failures
def test_a_dead_address_is_dead_every_time():
    """Determinism is what makes a consecutive-failure counter meaningful. An address
    that failed intermittently would never accumulate to the threshold."""
    n = NotificationService(failure_rate=1.0)
    assert all(n.address_is_dead("c1", "email") for _ in range(10))


def test_failure_rate_zero_disables_bounces():
    n = NotificationService(failure_rate=0.0)
    assert not n.address_is_dead("c1", "email")


def test_send_raises_on_a_dead_address_and_records_it():
    n = NotificationService(failure_rate=1.0)
    txn = {"transaction_id": "t1", "customer_id": "c1", "amount": 10, "currency": "USD"}
    with pytest.raises(DeliveryFailure):
        n.send("reminder", txn, "email")
    assert len(n.failed) == 1 and n.sent == []


def test_bounces_do_not_count_as_sent():
    n = NotificationService(failure_rate=1.0)
    txn = {"transaction_id": "t1", "customer_id": "c1", "amount": 10, "currency": "USD"}
    with pytest.raises(DeliveryFailure):
        n.send("reminder", txn, "email")
    assert n.count_for("t1") == 0


def test_the_same_address_is_dead_across_service_instances():
    a, b = NotificationService(failure_rate=0.5), NotificationService(failure_rate=0.5)
    ids = [f"c{i}" for i in range(40)]
    assert [a.address_is_dead(c, "email") for c in ids] == \
           [b.address_is_dead(c, "email") for c in ids]
