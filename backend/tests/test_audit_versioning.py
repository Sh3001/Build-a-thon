"""The hash chain must survive a schema migration.

An append-only log whose hash covers a fixed column list cannot gain a column: every
pre-existing row would re-derive to a different hash and the chain would read as tampered.
These tests pin the resolution -- per-row field-list versioning -- and the tamper
detection it must not weaken.
"""
from __future__ import annotations

import pytest

from backend.app.audit.store import (
    CURRENT_HASH_VERSION,
    FIELD_VERSIONS,
    AuditStore,
)
from backend.app.models.schemas import AuditEvent


@pytest.fixture
def store(tmp_path):
    return AuditStore(tmp_path / "audit.db")


def _event(i: int) -> AuditEvent:
    return AuditEvent(transaction_id=f"txn_{i}", customer_id="c1",
                      agent_decision="validate_policy", reason=f"step {i}",
                      tenant_id="acme", model_version="xgb-v1",
                      policy_version="rules@abc+limits@def", agent_run_id="run_1",
                      input_hash=f"hash_{i}")


def test_a_fresh_chain_verifies(store):
    store.append_many(_event(i) for i in range(20))
    store.commit()
    assert store.verify() == (True, None)


def test_the_new_columns_are_covered_by_the_hash(store):
    """Adding a column that the chain does not cover would be security theatre."""
    store.append(_event(1))
    store.commit()
    store.conn.execute("UPDATE audit_log SET tenant_id = 'globex' WHERE seq = 1")
    store.conn.commit()
    ok, bad = store.verify()
    assert not ok and bad == 1


def test_tampering_with_an_old_field_is_still_detected(store):
    store.append_many(_event(i) for i in range(5))
    store.commit()
    store.conn.execute("UPDATE audit_log SET amount_recovered = '99999' WHERE seq = 3")
    store.conn.commit()
    ok, bad = store.verify()
    assert not ok and bad == 3


def test_a_deleted_row_breaks_the_chain(store):
    store.append_many(_event(i) for i in range(5))
    store.commit()
    store.conn.execute("DELETE FROM audit_log WHERE seq = 3")
    store.conn.commit()
    ok, bad = store.verify()
    assert not ok and bad == 4


def test_a_downgraded_hash_version_is_detected(store):
    """The version is inside the hashed payload, so an attacker cannot claim v1 to drop
    the columns v2 covers."""
    store.append(_event(1))
    store.commit()
    store.conn.execute("UPDATE audit_log SET hash_version = 1 WHERE seq = 1")
    store.conn.commit()
    assert store.verify()[0] is False


def test_an_unknown_hash_version_refuses_rather_than_passing(store):
    """"Cannot verify" must never be reported as "verified"."""
    store.append(_event(1))
    store.commit()
    store.conn.execute("UPDATE audit_log SET hash_version = 99 WHERE seq = 1")
    store.conn.commit()
    assert store.verify() == (False, 1)


def test_a_legacy_v1_row_still_verifies(store):
    """The whole point: a chain written before a column existed keeps verifying."""
    v1_fields = FIELD_VERSIONS[1]
    event = _event(1)
    payload = {f: (None if getattr(event, f, None) is None else str(getattr(event, f)))
               for f in v1_fields}
    prev = store.tail_hash()
    row_hash = AuditStore._hash(prev, payload)
    store.conn.execute(
        f"INSERT INTO audit_log (prev_hash, row_hash, hash_version, "
        f"{', '.join(v1_fields)}) VALUES (?, ?, NULL, "
        f"{', '.join('?' * len(v1_fields))})",
        [prev, row_hash, *[payload[f] for f in v1_fields]])
    store.conn.commit()
    store._tail = row_hash

    # A v2 row on top of the v1 row: a chain that spans the migration.
    store.append(_event(2))
    store.commit()
    assert store.verify() == (True, None), "the chain must span the schema change"


def test_the_field_versions_are_append_only():
    """Editing a version in place would invalidate every row written under it."""
    assert set(FIELD_VERSIONS) == {1, 2}
    assert FIELD_VERSIONS[1] == FIELD_VERSIONS[2][:len(FIELD_VERSIONS[1])]
    assert max(FIELD_VERSIONS) == CURRENT_HASH_VERSION


def test_the_store_offers_no_way_to_rewrite_history(store):
    """"Never overwrite an audit event" enforced by the type, not by convention."""
    public = {name for name in dir(store) if not name.startswith("_")}
    assert not (public & {"update", "delete", "edit", "remove", "truncate"})
