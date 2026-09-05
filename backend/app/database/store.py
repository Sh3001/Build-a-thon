"""Phase 7 -- case and run persistence. The API reads a completed run from here rather
than recomputing it, so the dashboard and the evaluation always agree.

**Tenancy is enforced here, not by callers.** Every read and every write carries the
store's `tenant_id`, and there is no method that omits it. That is the shape that makes
isolation testable: a caller cannot forget a WHERE clause it never writes, and a new
query added to this class inherits the scoping from the constructor. The alternative --
passing a tenant into each call site -- fails the first time someone adds a method.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC
from pathlib import Path

from backend.app.config import DB_PATH, DB_URL
from backend.app.database.db import POSTGRES, Connection, connect
from backend.app.services.results import CaseOutcome

#: (column, sqlite type). REAL is float4 on Postgres, which would silently round the
#: money columns, so the float type is chosen per dialect.
CASE_COLUMNS: list[tuple[str, str]] = [
    ("transaction_id", "TEXT NOT NULL"), ("strategy", "TEXT NOT NULL"),
    ("customer_id", "TEXT"), ("amount_usd", "REAL"), ("currency", "TEXT"),
    ("amount", "REAL"), ("failure_code", "TEXT"), ("failure_category", "TEXT"),
    ("root_cause", "TEXT"), ("recovery_probability", "REAL"), ("risk_score", "REAL"),
    ("expected_recovery", "REAL"), ("recommended_action", "TEXT"), ("status", "TEXT"),
    ("recovered", "INTEGER"), ("amount_recovered", "REAL"), ("recovery_hours", "REAL"),
    ("retries", "INTEGER"), ("contacts", "INTEGER"), ("actions", "TEXT"),
    ("cost", "REAL"), ("stop_reason", "TEXT"), ("escalated", "INTEGER"),
    ("policy_blocked", "INTEGER"), ("risk_actions", "INTEGER"),
]
CASE_KEYS = ("transaction_id", "strategy")
CASE_NAMES = [c for c, _ in CASE_COLUMNS]


def schema_for(dialect: str) -> str:
    real = "DOUBLE PRECISION" if dialect == POSTGRES else "REAL"
    cols = ",\n    ".join(f"{c:<21}{t.replace('REAL', real)}" for c, t in CASE_COLUMNS)
    return f"""
CREATE TABLE IF NOT EXISTS cases (
    {cols},
    PRIMARY KEY (transaction_id, strategy)
);
CREATE INDEX IF NOT EXISTS idx_cases_strategy ON cases(strategy);
CREATE INDEX IF NOT EXISTS idx_cases_ev ON cases(expected_recovery DESC);

CREATE TABLE IF NOT EXISTS runs (
    key        TEXT PRIMARY KEY,
    created_at TEXT,
    payload    TEXT
);
"""


def upsert_sql(dialect: str, table: str, columns: list[str], keys: tuple[str, ...]) -> str:
    """SQLite has INSERT OR REPLACE; Postgres needs an explicit conflict target."""
    cols = ", ".join(columns)
    marks = ", ".join("?" * len(columns))
    if dialect != POSTGRES:
        return f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})"
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in keys)
    return (f"INSERT INTO {table} ({cols}) VALUES ({marks}) "
            f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates}")


def _migrate(conn) -> None:
    """Imported here rather than at module scope: migrations imports both stores to build
    the baseline step, so a top-level import would be circular."""
    from backend.app.database.migrations import migrate
    migrate(conn)


#: The tenant a single-tenant deployment writes under. Existing rows adopt it in
#: migration 005, so nothing about a local run changes.
DEFAULT_TENANT = "default"


class CaseStore:
    def __init__(self, path: Path | str = DB_PATH, conn: Connection | None = None,
                 url: str | None = DB_URL, tenant_id: str = DEFAULT_TENANT):
        self.conn = conn or connect(path, url)
        self.tenant_id = tenant_id
        _migrate(self.conn)

    # ------------------------------------------------------------------ write
    def save_cases(self, outcomes: Iterable[CaseOutcome], strategy: str,
                   extra: dict[str, dict] | None = None) -> int:
        extra = extra or {}
        rows = []
        for o in outcomes:
            e = extra.get(o.transaction_id, {})
            rows.append((
                o.transaction_id, strategy, o.customer_id, o.amount_usd,
                e.get("currency", "USD"), e.get("amount", o.amount_usd),
                o.failure_code.value, o.failure_category.value, e.get("root_cause"),
                o.recovery_probability, (1.0 - o.recovery_probability)
                if o.recovery_probability is not None else None,
                o.expected_recovery, e.get("recommended_action"), o.status,
                int(o.recovered), o.amount_recovered, o.recovery_hours, o.retries,
                o.contacts, json.dumps(o.actions), o.cost, o.stop_reason,
                int(o.escalated), o.policy_blocked, o.risk_actions,
            ))
        rows = [(self.tenant_id, *r) for r in rows]
        self.conn.executemany(
            upsert_sql(self.conn.dialect, "cases", ["tenant_id", *CASE_NAMES],
                       ("tenant_id", *CASE_KEYS)), rows)
        self.conn.commit()
        return len(rows)

    def save_run(self, key: str, payload: dict) -> None:
        from datetime import datetime
        self.conn.execute(
            upsert_sql(self.conn.dialect, "runs", ["tenant_id", "key", "created_at", "payload"],
                       ("tenant_id", "key")),
            (self.tenant_id, key, datetime.now(UTC).isoformat(),
             json.dumps(payload, default=str)))
        self.conn.commit()

    # ------------------------------------------------------------------ read
    def get_run(self, key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT payload FROM runs WHERE tenant_id = ? AND key = ?",
            (self.tenant_id, key)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_runs(self) -> list[str]:
        return [r["key"] for r in self.conn.execute(
            "SELECT key FROM runs WHERE tenant_id = ? ORDER BY created_at DESC",
            (self.tenant_id,))]

    def get_case(self, transaction_id: str, strategy: str = "recoverai") -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM cases WHERE tenant_id = ? AND transaction_id = ? AND strategy = ?",
            (self.tenant_id, transaction_id, strategy)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["actions"] = json.loads(d["actions"] or "[]")
        return d

    def queue(self, strategy: str = "recoverai", limit: int = 100, offset: int = 0,
              status: str | None = None, failure_code: str | None = None,
              order_by: str = "expected_recovery",
              direction: str = "desc") -> list[dict]:
        allowed = {"expected_recovery", "amount_usd", "recovery_probability",
                   "amount_recovered", "risk_score"}
        col = order_by if order_by in allowed else "expected_recovery"
        sql = "SELECT * FROM cases WHERE tenant_id = ? AND strategy = ?"
        args: list = [self.tenant_id, strategy]
        if status:
            sql += " AND status = ?"
            args.append(status)
        if failure_code:
            sql += " AND failure_code = ?"
            args.append(failure_code)
        # Both col and dir come from closed sets -- never interpolate caller input.
        dir_sql = "ASC" if str(direction).lower() == "asc" else "DESC"
        sql += f" ORDER BY {col} {dir_sql} LIMIT ? OFFSET ?"
        args += [limit, offset]
        out = []
        for r in self.conn.execute(sql, args):
            d = dict(r)
            d["actions"] = json.loads(d["actions"] or "[]")
            out.append(d)
        return out

    def count(self, strategy: str = "recoverai") -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM cases WHERE tenant_id = ? AND strategy = ?",
            (self.tenant_id, strategy)).fetchone()["c"])

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
