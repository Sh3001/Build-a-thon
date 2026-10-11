"""Durable jobs, checkpoints and transactional events in the existing state database."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from backend.app.database.db import connect
from backend.app.database.migrations import ensure_schema
from backend.app.database.store import upsert_sql


def now():
    return datetime.now(UTC).isoformat()


class JobConflict(ValueError):
    pass


class JobStore:
    def __init__(self, tenant_id: str, conn=None):
        self.tenant_id = tenant_id
        self.conn = conn or connect()
        ensure_schema(self.conn)

    def enqueue(self, payload: dict, request_key: str, *, request_id: str = "",
                actor: str = "") -> dict:
        encoded = json.dumps(payload, sort_keys=True)
        job_id = uuid.uuid4().hex
        stamp = now()
        self.conn.execute(
            "INSERT INTO recovery_jobs (job_id, tenant_id, request_key, payload, status, "
            "created_at, updated_at, request_id, requested_by) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?) "
            "ON CONFLICT (tenant_id, request_key) DO NOTHING",
            (job_id, self.tenant_id, request_key, encoded, stamp, stamp, request_id, actor))
        row = self.conn.execute(
            "SELECT * FROM recovery_jobs WHERE tenant_id = ? AND request_key = ?",
            (self.tenant_id, request_key)).fetchone()
        if row["payload"] != encoded:
            self.conn.rollback()
            raise JobConflict("Idempotency key was already used for a different request")
        self.conn.commit()
        return self.get(row["job_id"])

    def get(self, job_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM recovery_jobs WHERE tenant_id = ? AND job_id = ?",
            (self.tenant_id, job_id)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["payload"] = json.loads(out["payload"])
        out["result"] = json.loads(out["result"]) if out["result"] else None
        return out

    def cancel(self, job_id: str) -> dict | None:
        self.conn.execute(
            "UPDATE recovery_jobs SET status = 'cancelled', updated_at = ? "
            "WHERE tenant_id = ? AND job_id = ? AND status IN ('queued', 'running')",
            (now(), self.tenant_id, job_id))
        self.conn.commit()
        return self.get(job_id)

    def finish(self, job_id: str, status: str, result=None, error=None):
        self.conn.execute(
            "UPDATE recovery_jobs SET status = ?, result = ?, error = ?, updated_at = ? "
            "WHERE tenant_id = ? AND job_id = ? AND status != 'cancelled'",
            (status, json.dumps(result) if result is not None else None, error, now(),
             self.tenant_id, job_id))
        self.conn.commit()

    def checkpoint(self, transaction_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT state_json FROM recovery_checkpoints WHERE tenant_id = ? AND transaction_id = ?",
            (self.tenant_id, transaction_id)).fetchone()
        return json.loads(row["state_json"]) if row else None

    def save_checkpoint(self, transaction_id: str, payload: dict):
        self.conn.execute(upsert_sql(self.conn.dialect, "recovery_checkpoints",
                                    ["tenant_id", "transaction_id", "state_json"],
                                    ("tenant_id", "transaction_id")),
                          (self.tenant_id, transaction_id, json.dumps(payload)))
        self.conn.commit()

    def event(self, event_id: str, kind: str, payload: dict):
        self.conn.execute(
            "INSERT INTO recovery_outbox (event_id, tenant_id, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (event_id) DO NOTHING",
            (event_id, self.tenant_id, kind, json.dumps(payload), now()))
        self.conn.commit()

    def close(self):
        self.conn.close()
