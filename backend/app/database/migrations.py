"""Versioned schema migrations for both engines.

`CREATE TABLE IF NOT EXISTS` is not a migration strategy: on a database that already has
the table it silently does nothing, so a changed column type or a new column never lands
and the app fails later with a missing-column error at runtime. That was survivable while
the database was a disposable local file; it stopped being survivable when Postgres became
an option and the database started outliving the code that made it.

Deliberately not Alembic. The project has no ORM and three tables; a list of numbered
steps and a `schema_migrations` ledger is the whole requirement, and it keeps the
dependency footprint where it is.

Adding a migration:

1. append `("00N_short_name", "<SQL>")` to MIGRATIONS -- never edit or renumber an
   existing entry, since applied versions are recorded by name
2. write SQL that runs on both dialects, or branch on `dialect` with a callable
3. `migrate()` runs on next start; it is idempotent and safe to call every boot
"""
from __future__ import annotations

from typing import Callable, Sequence

from backend.app.audit.store import schema_for as audit_schema
from backend.app.database.store import schema_for as case_schema

#: A step is (version, sql) where sql is either a literal string or dialect -> string.
Step = tuple[str, str | Callable[[str], str]]

MIGRATIONS: Sequence[Step] = [
    # The baseline is the schema as it stood when migrations were introduced, so an
    # existing database adopts the ledger without being rebuilt.
    ("001_baseline_cases_and_runs", case_schema),
    ("001_baseline_audit_log", audit_schema),
    ("002_action_ledger", """
CREATE TABLE IF NOT EXISTS action_ledger (
    idempotency_key TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    transaction_id  TEXT NOT NULL,
    action          TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (run_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_action_ledger_txn ON action_ledger(transaction_id);
"""),
    ("003_dlq", """
CREATE TABLE IF NOT EXISTS dlq (
    customer_id   TEXT NOT NULL,
    channel       TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    failures      INTEGER NOT NULL DEFAULT 0,
    quarantined   INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    first_seen_at TEXT,
    updated_at    TEXT,
    PRIMARY KEY (run_id, customer_id, channel)
);
CREATE INDEX IF NOT EXISTS idx_dlq_quarantined ON dlq(quarantined);
"""),
]

LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL
);
"""


def applied_versions(conn) -> set[str]:
    conn.executescript(LEDGER)
    conn.commit()
    return {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}


def migrate(conn) -> list[str]:
    """Apply every pending step in order. Returns the versions newly applied."""
    from datetime import datetime, timezone

    done = applied_versions(conn)
    fresh: list[str] = []
    for version, sql in MIGRATIONS:
        if version in done:
            continue
        body = sql(conn.dialect) if callable(sql) else sql
        conn.executescript(body)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        fresh.append(version)
    return fresh


def pending(conn) -> list[str]:
    done = applied_versions(conn)
    return [v for v, _ in MIGRATIONS if v not in done]
