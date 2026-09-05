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

from collections.abc import Callable, Sequence
from datetime import UTC

from backend.app.audit.store import schema_for as audit_schema
from backend.app.database.store import schema_for as case_schema

#: A step is (version, sql) where sql is either a literal string or dialect -> string.
Step = tuple[str, str | Callable[[str], str]]


def _tenant_migration(dialect: str) -> str:
    """Rebuild `cases` with tenant_id in the primary key, preserving existing rows.

    The copy-and-swap shape is the portable one: SQLite cannot alter a primary key at
    all, and doing it the same way on both engines means one code path to reason about
    rather than two. Existing rows adopt the `default` tenant, which is what they were.
    """
    from backend.app.database.store import CASE_COLUMNS
    from backend.app.database.store import POSTGRES as _PG
    real = "DOUBLE PRECISION" if dialect == _PG else "REAL"
    cols = ",\n    ".join(f"{c:<21}{t.replace('REAL', real)}" for c, t in CASE_COLUMNS)
    names = ", ".join(c for c, _ in CASE_COLUMNS)
    return f"""
CREATE TABLE IF NOT EXISTS cases_v2 (
    tenant_id             TEXT NOT NULL DEFAULT 'default',
    {cols},
    PRIMARY KEY (tenant_id, transaction_id, strategy)
);
INSERT INTO cases_v2 (tenant_id, {names}) SELECT 'default', {names} FROM cases;
DROP TABLE cases;
ALTER TABLE cases_v2 RENAME TO cases;
CREATE INDEX IF NOT EXISTS idx_cases_strategy ON cases(tenant_id, strategy);
CREATE INDEX IF NOT EXISTS idx_cases_ev ON cases(expected_recovery DESC);
ALTER TABLE action_ledger ADD COLUMN tenant_id TEXT;
ALTER TABLE dlq ADD COLUMN tenant_id TEXT;
"""

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
    # 004: reproducibility + tenancy columns on the audit log. Written as separate
    # ALTERs because SQLite takes exactly one ADD COLUMN per statement, and as
    # IF NOT EXISTS-free ALTERs because SQLite has no such clause -- the migration
    # ledger is what makes running this once safe, not the SQL.
    ("004_audit_provenance_columns", """
ALTER TABLE audit_log ADD COLUMN tenant_id TEXT;
ALTER TABLE audit_log ADD COLUMN model_version TEXT;
ALTER TABLE audit_log ADD COLUMN policy_version TEXT;
ALTER TABLE audit_log ADD COLUMN agent_run_id TEXT;
ALTER TABLE audit_log ADD COLUMN input_hash TEXT;
ALTER TABLE audit_log ADD COLUMN hash_version INTEGER;
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id);
"""),
    # 005: tenancy. `cases` is rebuilt rather than altered because tenant_id has to be
    # part of the primary key -- adding it as a plain column would leave the old
    # (transaction_id, strategy) key in place, so two tenants holding the same
    # transaction id would collide and one would silently overwrite the other. That is
    # the exact failure isolation exists to prevent, so a column-add here would be
    # isolation in name only. The other tables are keyed by run_id already and only need
    # the column for filtering and reporting.
    ("005_tenant_columns", _tenant_migration),
    # 006: human review. The queue an operator works, and the immutable record of every
    # override they make. Two tables rather than one because an override is evidence
    # about a person's decision and must not be updatable alongside task state.
    ("006_human_review", """
CREATE TABLE IF NOT EXISTS human_review (
    review_id        TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    run_id           TEXT,
    transaction_id   TEXT NOT NULL,
    customer_id      TEXT,
    status           TEXT NOT NULL,
    rule_id          TEXT,
    reason           TEXT,
    proposed_action  TEXT,
    amount_usd       REAL,
    expected_profit  REAL,
    model_version    TEXT,
    policy_version   TEXT,
    created_at       TEXT NOT NULL,
    expires_at       TEXT,
    resolved_at      TEXT,
    resolved_by      TEXT,
    resolution_note  TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON human_review(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_review_txn ON human_review(transaction_id);

CREATE TABLE IF NOT EXISTS decision_overrides (
    override_id       TEXT PRIMARY KEY,
    review_id         TEXT,
    tenant_id         TEXT NOT NULL DEFAULT 'default',
    transaction_id    TEXT NOT NULL,
    actor             TEXT NOT NULL,
    actor_role        TEXT NOT NULL,
    at                TEXT NOT NULL,
    original_decision TEXT NOT NULL,
    new_decision      TEXT NOT NULL,
    action            TEXT,
    reason            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_override_txn ON decision_overrides(transaction_id);
"""),
    # 007: the run/report table is tenant-scoped for the same reason `cases` is -- two
    # tenants both storing a report under the key "recoverai" must not overwrite one
    # another. Same copy-and-swap shape as 005.
    ("007_runs_tenant", """
CREATE TABLE IF NOT EXISTS runs_v2 (
    tenant_id  TEXT NOT NULL DEFAULT 'default',
    key        TEXT NOT NULL,
    created_at TEXT,
    payload    TEXT,
    PRIMARY KEY (tenant_id, key)
);
INSERT INTO runs_v2 (tenant_id, key, created_at, payload)
    SELECT 'default', key, created_at, payload FROM runs;
DROP TABLE runs;
ALTER TABLE runs_v2 RENAME TO runs;
"""),
    # 008: customer consent. Opt-out is enforced by a policy rule (R-OPT-OUT), and a
    # rule needs somewhere to read the fact from that outlives the process.
    ("008_customer_consent", """
CREATE TABLE IF NOT EXISTS customer_consent (
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    customer_id  TEXT NOT NULL,
    opted_out    INTEGER NOT NULL DEFAULT 0,
    channel      TEXT NOT NULL DEFAULT '*',
    reason       TEXT,
    source       TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (tenant_id, customer_id, channel)
);
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
    from datetime import datetime

    done = applied_versions(conn)
    fresh: list[str] = []
    for version, sql in MIGRATIONS:
        if version in done:
            continue
        body = sql(conn.dialect) if callable(sql) else sql
        conn.executescript(body)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        fresh.append(version)
    return fresh


def pending(conn) -> list[str]:
    done = applied_versions(conn)
    return [v for v, _ in MIGRATIONS if v not in done]
