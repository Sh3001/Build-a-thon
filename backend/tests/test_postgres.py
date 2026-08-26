"""Postgres round-trip for both stores.

Skipped unless RECOVERAI_TEST_DB_URL points at a scratch database, so the default suite
stays offline and needs no server. Run it with:

    createdb recoverai_test
    RECOVERAI_TEST_DB_URL=postgresql://localhost/recoverai_test \
        .venv/bin/python -m pytest backend/tests/test_postgres.py -q

The tables are dropped at the start of every test, so the database is scratch space and
must not be one you care about.
"""
from __future__ import annotations

import os

import pytest

from backend.app.audit.store import AuditStore
from backend.app.database.db import POSTGRES, connect, qmark_to_pyformat
from backend.app.database.store import CaseStore

URL = os.environ.get("RECOVERAI_TEST_DB_URL")


@pytest.fixture()
def fresh():
    """Skips at the fixture, not the module, so the pure-translation tests below still
    run in the default offline suite."""
    if not URL:
        pytest.skip("RECOVERAI_TEST_DB_URL not set")
    c = connect(url=URL)
    # schema_migrations must go too: leave the ledger behind and the runner correctly
    # believes the (now dropped) tables are already there and skips recreating them.
    for t in ("audit_log", "cases", "runs", "schema_migrations"):
        c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    c.commit()
    c.close()
    yield URL


def _event(**kw):
    from backend.app.models.schemas import AuditEvent
    base = dict(timestamp="2026-01-01", transaction_id="txn_1",
                agent_decision="validate_policy", reason="r")
    return AuditEvent(**{**base, **kw})


def test_connect_reports_postgres(fresh):
    c = connect(url=fresh)
    assert c.dialect == POSTGRES
    c.close()


def test_appends_get_ascending_sequence_numbers(fresh):
    a = AuditStore(url=fresh)
    seqs = [a.append(_event(transaction_id=f"txn_{i}")) for i in range(1, 6)]
    a.commit()
    assert seqs == [1, 2, 3, 4, 5]
    assert a.count() == 5
    a.close()


def test_chain_verifies_and_catches_tampering(fresh):
    a = AuditStore(url=fresh)
    for i in range(1, 5):
        a.append(_event(transaction_id=f"txn_{i}", reason=f"step {i}"))
    a.commit()
    assert a.verify() == (True, None)

    a.conn.execute("UPDATE audit_log SET reason = 'tampered' WHERE seq = 2")
    a.conn.commit()
    ok, bad = a.verify()
    assert ok is False and bad == 2
    a.close()


def test_recent_filters(fresh):
    a = AuditStore(url=fresh)
    a.append(_event(policy_result="approve"))
    a.append(_event(policy_result="reject", rules_fired="R-MIN-VALUE"))
    a.append(_event(policy_result="reject", rules_fired="R-MAX-RETRIES"))
    a.commit()
    assert len(a.recent(policy_result="reject")) == 2
    assert len(a.recent(decision="validate_policy")) == 3
    assert a.decision_counts() == {"approve": 1, "reject": 2}
    assert a.rule_counts() == {"R-MIN-VALUE": 1, "R-MAX-RETRIES": 1}
    a.close()


def test_run_payload_upsert_overwrites(fresh):
    cs = CaseStore(url=fresh)
    cs.save_run("meta", {"seed": 1})
    cs.save_run("meta", {"seed": 2})
    assert cs.get_run("meta") == {"seed": 2}
    assert cs.list_runs() == ["meta"]
    cs.close()


def test_queue_sorts_both_directions(fresh):
    cs = CaseStore(url=fresh)
    rows = [("txn_a", "recoverai", 10.0), ("txn_b", "recoverai", 30.0),
            ("txn_c", "recoverai", 20.0)]
    for tid, strat, ev in rows:
        cs.conn.execute(
            "INSERT INTO cases (transaction_id, strategy, expected_recovery, status) "
            "VALUES (?, ?, ?, ?)", (tid, strat, ev, "recovered"))
    cs.conn.commit()
    desc = [r["expected_recovery"] for r in cs.queue(direction="desc")]
    asc = [r["expected_recovery"] for r in cs.queue(direction="asc")]
    assert desc == [30.0, 20.0, 10.0]
    assert asc == [10.0, 20.0, 30.0]
    cs.close()


def test_unknown_sort_column_is_ignored(fresh):
    cs = CaseStore(url=fresh)
    cs.queue(order_by="drop table cases")  # must not raise, must not interpolate
    cs.close()


@pytest.mark.parametrize("sql,expected", [
    ("SELECT * FROM t WHERE a = ?", "SELECT * FROM t WHERE a = %s"),
    ("SELECT '?' FROM t WHERE a = ?", "SELECT '?' FROM t WHERE a = %s"),
    ("SELECT * FROM t WHERE a LIKE '%x%'", "SELECT * FROM t WHERE a LIKE '%x%'"),
    ("SELECT 100 % 3", "SELECT 100 %% 3"),
])
def test_placeholder_translation(sql, expected):
    """Runs without a server -- pure string translation."""
    assert qmark_to_pyformat(sql) == expected


# ---------------------------------------------------------------- migrations
def test_migrations_apply_once_and_are_idempotent(fresh):
    from backend.app.database.migrations import migrate, pending
    c = connect(url=fresh)
    assert pending(c), "a fresh database should have pending migrations"
    first = migrate(c)
    assert first, "first run should apply something"
    assert pending(c) == []
    assert migrate(c) == [], "re-running must be a no-op"
    c.close()


def test_migration_ledger_records_versions(fresh):
    from backend.app.database.migrations import migrate
    c = connect(url=fresh)
    applied = migrate(c)
    recorded = {r["version"] for r in c.execute("SELECT version FROM schema_migrations")}
    assert recorded == set(applied)
    c.close()
