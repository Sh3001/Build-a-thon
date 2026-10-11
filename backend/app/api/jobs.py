"""Submission is fast; a separate worker owns execution and persisted progress."""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from backend.app.api.deps import require
from backend.app.database.jobs import JobConflict, JobStore
from backend.app.models.api import RunRequest
from backend.app.security.auth import Principal
from backend.app.services import source

router = APIRouter(tags=["recovery jobs"])


def submit(payload, who, key):
    # Current executor is a simulator. An external source must never acquire simulated
    # recoveries, even if someone sets PAYMENT_RAIL to an unimplemented live adapter.
    if source.configured() or os.environ.get("RECOVERAI_PROFILE") == "production":
        raise HTTPException(409, "This deployment supports read-only external payments. "
                            "Execution requires an integrated payment provider.")
    if not key:
        key = uuid.uuid4().hex
    if len(key) > 128:
        raise HTTPException(422, "Idempotency-Key must be at most 128 characters")
    store = JobStore(who.tenant_id)
    try:
        from backend.app.observability.logging import current_trace_id
        return store.enqueue(payload, key, request_id=current_trace_id(), actor=who.subject)
    except JobConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        store.close()


@router.post("/api/recovery/run", status_code=202)
def run_batch(req: RunRequest, who: Principal = Depends(require("execute:action")),
              idempotency_key: str | None = Header(None)):
    return submit(req.model_dump(), who, idempotency_key)


@router.post("/api/recovery/run/{transaction_id}", status_code=202)
def run_one(transaction_id: str, who: Principal = Depends(require("execute:action")),
            idempotency_key: str | None = Header(None)):
    return submit({"transaction_id": transaction_id, "limit": 1, "split": "test"},
                  who, idempotency_key)


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, who: Principal = Depends(require("read:queue"))):
    store = JobStore(who.tenant_id)
    try:
        result = store.get(job_id)
        if not result:
            raise HTTPException(404, "job not found")
        return result
    finally:
        store.close()


@router.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str, who: Principal = Depends(require("execute:action"))):
    store = JobStore(who.tenant_id)
    try:
        result = store.cancel(job_id)
        if not result:
            raise HTTPException(404, "job not found")
        return result
    finally:
        store.close()


@router.get("/api/v1/operations")
def operations(who: Principal = Depends(require("read:metrics"))):
    from datetime import UTC, datetime, timedelta
    store = JobStore(who.tenant_id)
    try:
        rows = store.conn.execute(
            "SELECT status, COUNT(*) n, MIN(created_at) oldest FROM recovery_jobs "
            "WHERE tenant_id = ? GROUP BY status", (who.tenant_id,))
        jobs = {r["status"]: {"count": r["n"], "oldest": r["oldest"]} for r in rows}
        stamp = datetime.now(UTC)
        overdue = store.conn.execute(
            "SELECT COUNT(*) n FROM human_review WHERE tenant_id = ? "
            "AND status = 'pending' AND expires_at <= ?",
            (who.tenant_id, stamp.isoformat())).fetchone()["n"]
        stale = store.conn.execute(
            "SELECT COUNT(*) n FROM execution_claims WHERE tenant_id = ? "
            "AND (status = 'unknown' OR (status = 'in_progress' AND updated_at < ?))",
            (who.tenant_id, (stamp - timedelta(minutes=10)).isoformat())).fetchone()["n"]
        alerts = []
        if overdue:
            alerts.append({"kind": "review_sla", "count": overdue})
        if stale:
            alerts.append({"kind": "action_reconciliation", "count": stale})
        queued = jobs.get("queued")
        if queued and queued["oldest"] < (stamp - timedelta(minutes=5)).isoformat():
            alerts.append({"kind": "queue_delay", "count": queued["count"]})
        return {"jobs": jobs, "alerts": alerts, "execution_mode": "simulation",
                "rate_limit_scope": "per-process"}
    finally:
        store.close()


@router.get("/api/v1/events")
def events(limit: int = Query(100, ge=1, le=1000),
           who: Principal = Depends(require("read:audit"))):
    """Recent durable events for inspection; not a delivery acknowledgement API."""
    import json
    store = JobStore(who.tenant_id)
    try:
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in store.conn.execute(
            "SELECT * FROM recovery_outbox WHERE tenant_id = ? "
            "ORDER BY created_at DESC, event_id DESC LIMIT ?", (who.tenant_id, limit))]
    finally:
        store.close()
