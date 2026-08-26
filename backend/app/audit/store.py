"""Phase 7 -- the append-only audit log.

Every decision is one row whose hash covers the row before it. Editing or deleting any
row breaks the chain from that point on, and `verify()` reports the first bad sequence
number. There is deliberately no update and no delete method: the store's API offers no
way to rewrite history, so "never overwrite previous audit events" is enforced by the
type, not by convention.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from backend.app.config import DB_PATH, DB_URL
from backend.app.database.db import POSTGRES, Connection, connect
from backend.app.models.schemas import AuditEvent

FIELDS = [
    "timestamp", "transaction_id", "customer_id", "agent_decision", "reason",
    "risk_score", "recovery_probability", "expected_recovery", "policy_result",
    "rules_fired", "action", "action_result", "amount_recovered", "cost", "next_step",
    "attempt_count",
]

def schema_for(dialect: str) -> str:
    """`seq` must be a gapless-ordered surrogate key on both engines -- the chain is
    verified in `seq` order, so the column type is not cosmetic."""
    seq = "BIGSERIAL PRIMARY KEY" if dialect == POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    return f"""
CREATE TABLE IF NOT EXISTS audit_log (
    seq       {seq},
    prev_hash TEXT NOT NULL,
    row_hash  TEXT NOT NULL,
    {', '.join(f'{f} TEXT' for f in FIELDS)}
);
CREATE INDEX IF NOT EXISTS idx_audit_txn ON audit_log(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_log(agent_decision);
"""

GENESIS = "0" * 64


def _migrate(conn) -> None:
    """Imported here rather than at module scope: migrations imports both stores to build
    the baseline step, so a top-level import would be circular."""
    from backend.app.database.migrations import migrate
    migrate(conn)


class AuditStore:
    """Hash-chained, append-only. Not thread-safe for concurrent writers by design --
    a single run owns the writer."""

    def __init__(self, path: Path | str = DB_PATH, conn: Connection | None = None,
                 url: str | None = DB_URL):
        self.conn = conn or connect(path, url)
        _migrate(self.conn)
        self._tail: str | None = None

    # ------------------------------------------------------------------ hashing
    @staticmethod
    def _hash(prev: str, payload: dict) -> str:
        blob = prev + json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def tail_hash(self) -> str:
        if self._tail is None:
            row = self.conn.execute(
                "SELECT row_hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            self._tail = row["row_hash"] if row else GENESIS
        return self._tail

    # ------------------------------------------------------------------ write
    @staticmethod
    def _normalise(event: AuditEvent) -> dict:
        """Hash exactly what is stored. Hashing the typed values while persisting their
        string forms would make every chain unverifiable on read-back."""
        out = {}
        for f in FIELDS:
            v = getattr(event, f, None)
            out[f] = None if v is None else str(v)
        return out

    def append(self, event: AuditEvent) -> int:
        payload = self._normalise(event)
        prev = self.tail_hash()
        row_hash = self._hash(prev, payload)
        seq = self.conn.execute_returning_id(
            f"INSERT INTO audit_log (prev_hash, row_hash, {', '.join(FIELDS)}) "
            f"VALUES (?, ?, {', '.join('?' * len(FIELDS))})",
            [prev, row_hash, *[payload[f] for f in FIELDS]],
            "seq",
        )
        self._tail = row_hash
        return seq

    def append_many(self, events: Iterable[AuditEvent]) -> int:
        n = 0
        for e in events:
            self.append(e)
            n += 1
        return n

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ read
    def verify(self) -> tuple[bool, int | None]:
        """Re-derive every hash. Returns (ok, first_bad_seq)."""
        prev = GENESIS
        for row in self.conn.execute("SELECT * FROM audit_log ORDER BY seq"):
            payload = {f: (row[f] if row[f] is not None else None) for f in FIELDS}
            expected = self._hash(prev, payload)
            if row["prev_hash"] != prev or row["row_hash"] != expected:
                return False, int(row["seq"])
            prev = row["row_hash"]
        return True, None

    def timeline(self, transaction_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM audit_log WHERE transaction_id = ? ORDER BY seq", (transaction_id,))
        return [dict(r) for r in rows]

    def recent(self, limit: int = 200, transaction_id: str | None = None,
               decision: str | None = None, policy_result: str | None = None) -> list[dict]:
        sql = "SELECT * FROM audit_log"
        where, args = [], []
        if transaction_id:
            where.append("transaction_id = ?")
            args.append(transaction_id)
        if decision:
            where.append("agent_decision = ?")
            args.append(decision)
        if policy_result:
            where.append("policy_result = ?")
            args.append(policy_result)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY seq DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM audit_log").fetchone()["c"])

    def decision_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT policy_result p, COUNT(*) c FROM audit_log "
            "WHERE policy_result IS NOT NULL GROUP BY policy_result")
        return {r["p"]: int(r["c"]) for r in rows}

    def rule_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.conn.execute(
                "SELECT rules_fired FROM audit_log WHERE rules_fired IS NOT NULL AND rules_fired != ''"):
            for rule in str(r["rules_fired"]).split(","):
                if rule:
                    out[rule] = out.get(rule, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
