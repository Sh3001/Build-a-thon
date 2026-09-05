"""Operational state that must outlive the process: the idempotency ledger and the DLQ.

Both are scoped by `run_id`. That scoping is load-bearing rather than cosmetic - without
it, a durable idempotency cache would make the *second* experiment run return the first
run's cached results and recover nothing, silently turning a re-run into a no-op. A run
id keeps replay-safety within a run while leaving runs independent.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from backend.app.config import DB_PATH, DB_URL, DLQ_FAILURE_THRESHOLD
from backend.app.database.db import Connection, connect


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _migrate(conn) -> None:
    from backend.app.database.migrations import migrate
    migrate(conn)


class ActionLedger:
    """Durable idempotency. `remember`/`recall` are keyed by (run_id, idempotency_key)."""

    def __init__(self, run_id: str, path=DB_PATH, conn: Connection | None = None,
                 url: str | None = DB_URL):
        self.run_id = run_id
        self.conn = conn or connect(path, url)
        _migrate(self.conn)

    def recall(self, key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT result_json FROM action_ledger WHERE run_id = ? AND idempotency_key = ?",
            (self.run_id, key)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def remember(self, key: str, transaction_id: str, action: str, result: dict) -> None:
        # A second write for the same key is the replay case, not an error: keep the
        # original result, since returning a *different* outcome for the same key would
        # defeat the point of the ledger.
        if self.recall(key) is not None:
            return
        self.conn.execute(
            "INSERT INTO action_ledger "
            "(idempotency_key, run_id, transaction_id, action, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, self.run_id, transaction_id, action, json.dumps(result, default=str), _now()))
        self.conn.commit()

    def count(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM action_ledger WHERE run_id = ?",
            (self.run_id,)).fetchone()["c"])

    def close(self) -> None:
        self.conn.close()


class DLQStore:
    """Consecutive-failure tracking per (customer, channel).

    A *successful* delivery resets the counter to zero - the point is to catch an address
    that is consistently dead, not one that failed once during an outage.
    """

    def __init__(self, run_id: str, path=DB_PATH, conn: Connection | None = None,
                 url: str | None = DB_URL, threshold: int = DLQ_FAILURE_THRESHOLD):
        self.run_id = run_id
        self.threshold = threshold
        self.conn = conn or connect(path, url)
        _migrate(self.conn)

    def _row(self, customer_id: str, channel: str) -> dict | None:
        return self.conn.execute(
            "SELECT * FROM dlq WHERE run_id = ? AND customer_id = ? AND channel = ?",
            (self.run_id, customer_id, channel)).fetchone()

    def is_quarantined(self, customer_id: str, channel: str) -> bool:
        row = self._row(customer_id, channel)
        return bool(row and row["quarantined"])

    def record_failure(self, customer_id: str, channel: str, error: str) -> bool:
        """Returns True if this failure tipped the pair into quarantine."""
        row = self._row(customer_id, channel)
        failures = int(row["failures"]) + 1 if row else 1
        quarantined = 1 if failures >= self.threshold else 0
        if row:
            self.conn.execute(
                "UPDATE dlq SET failures = ?, quarantined = ?, last_error = ?, updated_at = ? "
                "WHERE run_id = ? AND customer_id = ? AND channel = ?",
                (failures, quarantined, error, _now(), self.run_id, customer_id, channel))
        else:
            self.conn.execute(
                "INSERT INTO dlq (customer_id, channel, run_id, failures, quarantined, "
                "last_error, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (customer_id, channel, self.run_id, failures, quarantined, error,
                 _now(), _now()))
        self.conn.commit()
        return bool(quarantined) and (not row or not row["quarantined"])

    def record_success(self, customer_id: str, channel: str) -> None:
        row = self._row(customer_id, channel)
        if not row or not int(row["failures"]):
            return
        self.conn.execute(
            "UPDATE dlq SET failures = 0, quarantined = 0, updated_at = ? "
            "WHERE run_id = ? AND customer_id = ? AND channel = ?",
            (_now(), self.run_id, customer_id, channel))
        self.conn.commit()

    def release(self, customer_id: str, channel: str) -> None:
        """Manual review outcome: put the pair back in service."""
        self.conn.execute(
            "UPDATE dlq SET failures = 0, quarantined = 0, updated_at = ? "
            "WHERE run_id = ? AND customer_id = ? AND channel = ?",
            (_now(), self.run_id, customer_id, channel))
        self.conn.commit()

    def entries(self, quarantined_only: bool = True, limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM dlq WHERE run_id = ?"
        args: list = [self.run_id]
        if quarantined_only:
            sql += " AND quarantined = 1"
        sql += " ORDER BY failures DESC, customer_id LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def stats(self) -> dict:
        rows = [dict(r) for r in self.conn.execute(
            "SELECT quarantined, failures FROM dlq WHERE run_id = ?", (self.run_id,))]
        return {
            "tracked_pairs": len(rows),
            "quarantined": sum(1 for r in rows if r["quarantined"]),
            "total_failures": sum(int(r["failures"]) for r in rows),
            "threshold": self.threshold,
        }

    def close(self) -> None:
        self.conn.close()
