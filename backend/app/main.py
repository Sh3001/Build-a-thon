"""HTTP entry point for external source browsing and tenant-scoped recovery records.

Execution is submitted through api.jobs and performed by a separate durable worker.
Startup does not provision databases or generate synthetic payments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.agents.llm import build_planner
from backend.app.api import ops as ops_router
from backend.app.api import review as review_router
from backend.app.api.deps import get_auth_policy, require
from backend.app.api.jobs import router as jobs_router
from backend.app.api.middleware import BodyLimitMiddleware
from backend.app.audit.store import AuditStore
from backend.app.config import DATA_PROCESSED, DB_PATH, DB_URL, ROOT
from backend.app.database.db import StorageUnavailable, engine_for, state_configured
from backend.app.database.operational import DLQStore
from backend.app.database.store import CaseStore
from backend.app.ml.scorer import get_scorer
from backend.app.models.api import (
    AuditPage,
    CaseDetail,
    Health,
    OverviewCards,
    QueueRow,
    RevenueAtRisk,
)
from backend.app.observability.logging import configure as configure_logging
from backend.app.observability.metrics import M, get_registry
from backend.app.policies.version import policy_version
from backend.app.profiles import validate_or_raise as validate_profile_or_raise
from backend.app.security.auth import Principal
from backend.app.services import source

#: Refuse a body larger than this before reading it. A request-size limit is the cheapest
#: denial-of-service defence there is, and it has to live in middleware because by the
#: time a handler runs, the body has already been read into memory.
MAX_REQUEST_BYTES = 1_048_576

app = FastAPI(title="RecoverAI", version="0.2.0",
              description="Autonomous AI revenue recovery agent")


@app.on_event("startup")
def _configure() -> None:
    """Install structured logging and resolve the auth policy.

    `AuthPolicy.from_env()` raises in the production profile when no signing secret is
    set, and letting that propagate here is the intent: a process that boots with
    authentication silently disabled is worse than one that refuses to boot.
    """
    configure_logging()
    # Refuse to boot a misconfigured production or staging environment. A process that
    # starts anyway looks healthy while being unsafe, which is strictly worse than a
    # crash loop somebody has to look at.
    validate_profile_or_raise()
    policy = get_auth_policy()
    get_registry().gauge("build_info", 1, "static build facts",
                         {"profile": policy.profile,
                          "auth_required": str(policy.required).lower(),
                          "policy_version": policy_version()})


@app.middleware("http")
async def _guard(request, call_next):
    """Body-size cap, security headers, and per-request metrics.

    Middleware rather than a dependency because all three have to apply to every route
    including the ones that do not declare dependencies -- and a size limit enforced in a
    handler has already lost.
    """
    import time as _time

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)

    from backend.app.observability.logging import new_trace_id
    request.state.trace_id = new_trace_id()
    started = _time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.trace_id
    elapsed = _time.perf_counter() - started

    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    registry = get_registry()
    registry.counter(M.API_REQUESTS, "API requests",
                     {"method": request.method, "path": path,
                      "status": str(response.status_code)})
    registry.observe(M.API_LATENCY, elapsed, "API request duration",
                     {"method": request.method, "path": path})

    # Conservative defaults. The dashboard is same-origin and loads no third-party
    # scripts, so a restrictive CSP costs nothing and closes the injection path that an
    # audit-log viewer rendering stored strings would otherwise open.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
    response.headers.setdefault("Cache-Control", response.headers.get(
        "Cache-Control", "no-store"))
    return response


@app.on_event("startup")
def _check_database() -> None:
    """Fail fast when RECOVERAI_DB_URL is set but unreachable.

    Without this the process starts happily, serves the dashboard shell, and then 500s on
    every API call -- which looks like an application bug rather than a database that is
    not running. The documented contract is "no silent fallback"; refusing to start is
    what makes that true instead of aspirational.
    """
    if not DB_URL:
        return
    from backend.app.database.db import connect
    try:
        conn = connect(url=DB_URL)
        conn.execute("SELECT 1").fetchone()
        conn.close()
    except Exception as exc:
        raise RuntimeError(
            "RECOVERAI_DB_URL is set but the database is unreachable.\n"
            "Start Postgres, or unset RECOVERAI_DB_URL to fall back to SQLite."
        ) from exc


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"], allow_headers=["*"],
)

app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)

app.include_router(review_router.router)
app.include_router(review_router.consent_router)
app.include_router(ops_router.router)

app.include_router(jobs_router)

FRONTEND_DIST = ROOT / "frontend" / "dist"


# ---------------------------------------------------------------- store access
def _audit(who: Principal) -> AuditStore:
    return AuditStore(DB_PATH, tenant_id=who.tenant_id)


def _cases(who: Principal, conn=None) -> CaseStore:
    return CaseStore(DB_PATH, conn=conn, tenant_id=who.tenant_id)


def _require_run(store: CaseStore, key: str) -> dict:
    run = store.get_run(key)
    if run is None:
        raise HTTPException(404, "No recovery results are available yet.")
    return run


def _row(d: dict) -> QueueRow:
    return QueueRow(
        transaction_id=d["transaction_id"], customer_id=d.get("customer_id") or "",
        amount=d.get("amount") or 0.0, currency=d.get("currency") or "USD",
        amount_usd=d.get("amount_usd") or 0.0, failure_code=d.get("failure_code") or "",
        failure_category=d.get("failure_category") or "", root_cause=d.get("root_cause"),
        recovery_probability=d.get("recovery_probability"), risk_score=d.get("risk_score"),
        expected_recovery=d.get("expected_recovery"),
        recommended_action=d.get("recommended_action"), status=d.get("status") or "pending",
        amount_recovered=d.get("amount_recovered") or 0.0, retries=d.get("retries") or 0,
        contacts=d.get("contacts") or 0, actions=d.get("actions") or [],
        stop_reason=d.get("stop_reason") or "",
        months_late=d.get("months_late"), snapshot_date=d.get("snapshot_date"),
        data_source=d.get("data_source"),
    )


@app.exception_handler(StorageUnavailable)
async def storage_unavailable(request, exc):
    return JSONResponse({"detail": str(exc), "code": "storage_not_configured"}, 503)


@app.exception_handler(source.SqlSourceError)
async def source_error(request, exc):
    message = ("UCI spreadsheet unavailable or invalid. Run python -m scripts.fetch_uci "
               "on the server and check RECOVERAI_UCI_PATH." if source.kind() == "uci" else
               "Payment source configuration or query is invalid.")
    return JSONResponse({"detail": message, "code": "source_unavailable"}, 503)


@app.get("/api/source/status")
def source_status():
    return source.status()


@app.get("/api/source/report")
def source_report(who: Principal = Depends(require("read:queue"))):
    rows, report = source.queue(who.tenant_id)
    from backend.app.services.model_monitoring import inspect
    return {**report, "model_monitoring": ({"status": "not_applicable"}
            if source.kind() == "uci" else inspect(rows))}


@app.get("/api/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/api/health/ready")
def readiness():
    from backend.app.database.db import connect
    from backend.app.database.migrations import ensure_schema
    if not source.configured() and not state_configured():
        return JSONResponse({"status": "setup_required"}, 503)
    try:
        if source.configured():
            source.queue(os.environ.get("RECOVERAI_SOURCE_TENANT", "default"))
        if state_configured():
            conn = connect()
            try:
                ensure_schema(conn)
                conn.execute("SELECT 1")
            finally:
                conn.close()
    except Exception:
        return JSONResponse({"status": "unavailable"}, 503)
    return {"status": "ready"}


# ---------------------------------------------------------------- endpoints
@app.get("/api/health", response_model=Health)
def health() -> Health:
    scorer = get_scorer()
    _auth = get_auth_policy()
    audit_rows, chain_ok, ready = 0, None, False
    return Health(
        model_trained=scorer.is_trained, model_version=scorer.model_version,
        dataset_ready=source.configured() or (state_configured() and (DATA_PROCESSED / "test.csv").exists()),
        experiment_ready=ready, audit_rows=audit_rows, audit_chain_valid=chain_ok,
        llm_enabled=build_planner("auto") is not None,
        db_engine=engine_for() if state_configured() else "unconfigured",
        # The security posture is part of health. "We thought auth was on" should be a
        # checkable statement, not a belief held until an incident.
        profile=_auth.profile,
        auth_required=_auth.required,
        policy_version=policy_version(),
    )


@app.get("/api/overview", response_model=OverviewCards)
def overview(who: Principal = Depends(require("read:metrics"))) -> OverviewCards:
    s = _cases(who)
    try:
        agent = _require_run(s, "recoverai")
        base = s.get_run("baseline") or {}
        comp = s.get_run("comparison") or {}
        meta = s.get_run("meta") or {}
        ctrl = s.get_run("control") or {}
        vsc = (s.get_run("vs_control") or {}).get("recoverai", {})
        vsc_ci = (vsc.get("bootstrap") or {}).get("incremental_revenue", {})
        return OverviewCards(
            revenue_at_risk=agent["revenue_at_risk"],
            revenue_recovered=agent["revenue_recovered"],
            recovery_rate=agent["recovery_rate"],
            incremental_recovery_vs_baseline=comp.get("incremental_recovered_revenue"),
            recovery_uplift_pct=comp.get("recovery_uplift_pct"),
            incremental_ci_low=(comp.get("bootstrap") or {}).get(
                "incremental_revenue", {}).get("p05"),
            incremental_ci_high=(comp.get("bootstrap") or {}).get(
                "incremental_revenue", {}).get("p95"),
            incremental_ci_excludes_zero=(comp.get("bootstrap") or {}).get("excludes_zero"),
            cases_processed=agent["cases"], cases_escalated=agent["cases_escalated"],
            cases_stopped=agent["cases_stopped"], cases_recovered=agent["cases_recovered"],
            total_cost=agent["total_cost"], net_recovered=agent["net_recovered"],
            avg_recovery_hours=agent["avg_recovery_hours"],
            baseline_recovered=base.get("revenue_recovered"),
            baseline_recovery_rate=base.get("recovery_rate"),
            unsafe_actions_prevented=(base["risk_actions_taken"] - agent["risk_actions_taken"])
            if "risk_actions_taken" in base else None,
            control_recovered=ctrl.get("revenue_recovered"),
            control_recovery_rate=ctrl.get("recovery_rate"),
            incremental_vs_control=vsc.get("incremental_recovered_revenue"),
            share_of_revenue_that_is_causal=vsc.get("share_of_revenue_that_is_causal"),
            control_ci_low=vsc_ci.get("p05"), control_ci_high=vsc_ci.get("p95"),
            model_version=meta.get("model_version", ""), planner=meta.get("planner", ""),
        )
    finally:
        s.conn.close()


@app.get("/api/revenue-at-risk", response_model=RevenueAtRisk)
def revenue_at_risk(top: int = Query(10, ge=1, le=100),
                    who: Principal = Depends(require("read:queue"))) -> RevenueAtRisk:
    s = _cases(who)
    try:
        agent = _require_run(s, "recoverai")
        rows = s.queue("recoverai", limit=top, order_by="expected_recovery")
        return RevenueAtRisk(
            total_at_risk=agent["revenue_at_risk"],
            total_recovered=agent["revenue_recovered"],
            total_outstanding=round(agent["revenue_at_risk"] - agent["revenue_recovered"], 2),
            by_failure_category=agent["by_category"], by_failure_code=agent["by_failure_code"],
            top_cases=[_row(r) for r in rows],
        )
    finally:
        s.conn.close()


@app.get("/api/recovery-queue", response_model=list[QueueRow])
def recovery_queue(
    limit: int = Query(100, ge=1, le=2000), offset: int = Query(0, ge=0),
    status: str | None = None, failure_code: str | None = None,
    sort: str = Query("expected_recovery"), strategy: str = "recoverai",
    direction: str = Query("desc", pattern="^(asc|desc)$"),
    who: Principal = Depends(require("read:queue")),
) -> list[QueueRow]:
    if source.configured():
        rows, _ = source.queue(who.tenant_id)
        rows = [r for r in rows if (not status or r["status"] == status)
                and (not failure_code or r["failure_code"] == failure_code)]
        col = sort if sort in {"amount_usd", "expected_recovery", "recovery_probability", "risk_score"} else "amount_usd"
        rows.sort(key=lambda r: (r.get(col) or 0, r["transaction_id"]), reverse=direction == "desc")
        return [_row(r) for r in rows[offset:offset + limit]]
    s = _cases(who)
    try:
        rows = s.queue(strategy, limit=limit, offset=offset, status=status,
                       failure_code=failure_code, order_by=sort, direction=direction)
        return [_row(r) for r in rows]
    finally:
        s.conn.close()


@app.get("/api/cases/{transaction_id}", response_model=CaseDetail)
def case_detail(transaction_id: str, who: Principal = Depends(require("read:queue"))) -> CaseDetail:
    if source.configured():
        rows, _ = source.queue(who.tenant_id)
        row = next((r for r in rows if r["transaction_id"] == transaction_id), None)
        if row is None:
            raise HTTPException(404, "unknown transaction")
        return CaseDetail(case=_row(row), chain_valid=True)
    a = _audit(who)
    try:
        s = _cases(who, a.conn)
        row = s.get_case(transaction_id, "recoverai")
        if row is None:
            raise HTTPException(404, f"unknown transaction {transaction_id}")
        base = s.get_case(transaction_id, "baseline")
        events = a.timeline(transaction_id)
        ok, _ = a.verify()
        return CaseDetail(case=_row(row), baseline=base,
                          audit_events=events if who.has("read:audit") else [], chain_valid=ok)
    finally:
        a.conn.close()


@app.get("/api/cases/{transaction_id}/audit", response_model=AuditPage)
def case_audit(transaction_id: str, who: Principal = Depends(require("read:audit"))) -> AuditPage:
    a = _audit(who)
    try:
        rows = a.timeline(transaction_id)
        ok, _ = a.verify()
        if not rows:
            raise HTTPException(404, f"no audit trail for {transaction_id}")
        return AuditPage(transaction_id=transaction_id, rows=rows, total=len(rows), chain_valid=ok)
    finally:
        a.conn.close()


@app.get("/api/audit", response_model=AuditPage)
def audit_log(limit: int = Query(200, ge=1, le=2000),
              decision: str | None = None,
              policy_result: str | None = Query(None, pattern="^(approve|modify|reject)$"),
              who: Principal = Depends(require("read:audit")),
              ) -> AuditPage:
    a = _audit(who)
    try:
        rows = a.recent(limit=limit, decision=decision, policy_result=policy_result)
        ok, _ = a.verify()
        total = a.count()
        return AuditPage(rows=rows, total=total, chain_valid=ok)
    finally:
        a.conn.close()


@app.get("/api/dlq")
def dlq_entries(quarantined_only: bool = True, limit: int = Query(500, ge=1, le=2000),
                who: Principal = Depends(require("read:review"))):
    """Quarantined (customer, channel) pairs awaiting review. Scoped to the active run's
    id, which lives in the run metadata -- a DLQ from a previous experiment is history,
    not a work queue."""
    s_ = _cases(who)
    try:
        meta = s_.get_run("meta") or {}
        run_id = meta.get("run_id")
        if not run_id:
            return JSONResponse({"run_id": None, "entries": [], "stats": {
                "tracked_pairs": 0, "quarantined": 0, "total_failures": 0, "threshold": 0}})
        d = DLQStore(run_id, DB_PATH, tenant_id=who.tenant_id)
        payload = {"run_id": run_id, "stats": d.stats(),
                   "entries": d.entries(quarantined_only=quarantined_only, limit=limit)}
        d.close()
        return JSONResponse(payload)
    finally:
        s_.conn.close()


@app.post("/api/dlq/release")
def dlq_release(customer_id: str, channel: str,
                who: Principal = Depends(require("write:dlq"))):
    """Human review outcome: put a quarantined pair back in service."""
    s_ = _cases(who)
    try:
        meta = s_.get_run("meta") or {}
        run_id = meta.get("run_id")
        if not run_id:
            raise HTTPException(404, "no active run")
        d = DLQStore(run_id, DB_PATH, tenant_id=who.tenant_id)
        d.release(customer_id, channel)
        ok = not d.is_quarantined(customer_id, channel)
        d.close()
        return JSONResponse({"released": ok, "customer_id": customer_id, "channel": channel})
    finally:
        s_.conn.close()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    #: "keywords" (default, deterministic) or "llm" to put a model in front of it.
    router: str = "keywords"


@app.post("/api/chat")
def chat(req: ChatRequest, who: Principal = Depends(require("read:queue"))) -> JSONResponse:
    """Answer a question about this run.

    The router only ever chooses an intent; every figure in the reply is computed by
    `ChatEngine` from the database. A model cannot state a number here, so it cannot
    state a wrong one.
    """
    from backend.app.chat import answer as render
    from backend.app.chat.dsl import run as run_query
    from backend.app.chat.parse import parse
    from backend.app.chat.query import ChatEngine
    from backend.app.chat.router import KeywordRouter

    a = _audit(who)
    try:
        s_ = _cases(who, a.conn)

        # Questions about the run as a whole (arm comparison, policy activity, the DLQ) are
        # not case-table queries, so they keep their dedicated handlers. Everything else goes
        # through the general query language.
        special = {"arm_comparison", "policy_activity", "dlq_status", "case_trace"}
        intent = KeywordRouter().route(req.question)
        if intent.intent.value in special:
            capability = "read:audit" if intent.intent.value in {"policy_activity", "case_trace"} else "read:metrics"
            if not who.has(capability):
                raise HTTPException(403, "missing capability")
            meta = s_.get_run("meta") or {}
            dlq = DLQStore(meta["run_id"], conn=a.conn, tenant_id=who.tenant_id) if meta.get("run_id") else None
            out = ChatEngine(s_, audit=a, dlq=dlq).run(intent, source="keywords")
            return JSONResponse(out.model_dump(mode="json"))

        q = parse(req.question)
        problems = q.validate_against_schema()
        if problems:
            return JSONResponse({
                "text": ("I understood the question but could not run it against the case "
                         "table: " + "; ".join(problems)
                         + ". Try naming a stored field - amount, amount recovered, retries, "
                           "failure cause, failure category, outcome or customer."),
                "intent": "invalid", "query": q.model_dump(mode="json"),
                "data": {}, "case_ids": [], "source": "parser"})

        result = run_query(a.conn, q, tenant_id=who.tenant_id)
        text = render.render(q, result) + render.caveat(q)
        ids = [r["transaction_id"] for r in result.rows if "transaction_id" in r][:10]
        return JSONResponse({
            "text": text, "intent": f"{q.agg.value}"
                                    + (f"({q.field})" if q.field else ""),
            "query": q.model_dump(mode="json", exclude_defaults=True),
            "data": {"scalar": result.scalar, "matched": result.total_matched,
                     "grand_total": result.grand_total},
            "sql": result.sql, "case_ids": ids,
            "unresolved": q.unresolved, "source": "query-language",
        })
    finally:
        a.conn.close()


@app.get("/api/metrics")
def metrics(who: Principal = Depends(require("read:metrics"))) -> JSONResponse:
    s = _cases(who)
    try:
        agent = _require_run(s, "recoverai")
        base = s.get_run("baseline") or {}
        comp = s.get_run("comparison") or {}
        meta = s.get_run("meta") or {}
        control = s.get_run("control") or {}
        vs_control = s.get_run("vs_control") or {}
        model_eval: dict[str, Any] = {}
        p = Path(DB_PATH).parent / "model_evaluation.json"
        if p.exists():
            import json
            model_eval = json.loads(p.read_text())
        return JSONResponse({"recoverai": agent, "baseline": base, "control": control,
                             "comparison": comp, "vs_control": vs_control,
                             "meta": meta, "model": model_eval})
    finally:
        s.conn.close()


@app.get("/api/baseline")
def baseline_metrics(who: Principal = Depends(require("read:metrics"))) -> JSONResponse:
    s = _cases(who)
    try:
        base = s.get_run("baseline") or {}
        return JSONResponse(base)
    finally:
        s.conn.close()


# ---------------------------------------------------------------- static
if FRONTEND_DIST.exists():
    # Asset filenames are content-hashed by Vite, so a new build always has a new name
    # and these can be cached hard.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        # index.html is the one file whose NAME never changes, so it must be revalidated
        # on every load. Served without Cache-Control, browsers fall back to heuristic
        # caching and keep serving an old index.html -- which points at the previous
        # bundle, making a rebuilt frontend invisible until someone hard-refreshes.
        return FileResponse(
            FRONTEND_DIST / "index.html",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )
