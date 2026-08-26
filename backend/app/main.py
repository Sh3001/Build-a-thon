"""Phase 8 -- the FastAPI backend.

Reads a completed experiment from SQLite. The dashboard therefore shows exactly the
numbers the evaluation produced -- there is no second implementation of the metrics.

`POST /api/recovery/run` and `/api/recovery/run/{id}` execute the agent live against the
mock rails, so a judge can watch a case being worked rather than only reading a snapshot.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.agents.graph import RecoveryAgent
from backend.app.agents.llm import build_planner
from backend.app.agents.runner import run_agent_batch, to_outcome
from backend.app.audit.store import AuditStore
from backend.app.config import DATA_PROCESSED, DB_PATH, DB_URL, ROOT, SEED
from backend.app.database.db import engine_for
from backend.app.database.operational import DLQStore
from backend.app.database.store import CaseStore
from backend.app.ml.scorer import get_scorer
from backend.app.models.api import (
    AuditPage, CaseDetail, Health, LiveTrace, OverviewCards, QueueRow, RevenueAtRisk,
    RunRequest, RunResponse,
)
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.results import compare, summarize
from backend.app.tools.executor import ActionExecutor
from simulation.payment_gateway import PaymentGateway

app = FastAPI(title="RecoverAI", version="0.1.0",
              description="Autonomous AI revenue recovery agent")

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
    except Exception as exc:                                  # noqa: BLE001
        raise RuntimeError(
            f"RECOVERAI_DB_URL is set but the database is unreachable: {exc}\n"
            f"Start Postgres, or unset RECOVERAI_DB_URL to fall back to SQLite."
        ) from exc


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIST = ROOT / "frontend" / "dist"


# ---------------------------------------------------------------- store access
def _audit() -> AuditStore:
    return AuditStore(DB_PATH)


def _cases(conn=None) -> CaseStore:
    return CaseStore(DB_PATH, conn=conn)


def _require_run(store: CaseStore, key: str) -> dict:
    run = store.get_run(key)
    if run is None:
        raise HTTPException(404, "no experiment found -- run `python scripts/run_experiment.py` first")
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
    )


# ---------------------------------------------------------------- endpoints
@app.get("/api/health", response_model=Health)
def health() -> Health:
    scorer = get_scorer()
    audit_rows, chain_ok, ready = 0, None, False
    # On Postgres there is no local file to stat, so presence of a URL is the signal.
    if DB_URL or DB_PATH.exists():
        a = _audit()
        audit_rows = a.count()
        chain_ok = a.verify()[0]
        ready = _cases(a.conn).get_run("meta") is not None
        a.close()
    return Health(
        model_trained=scorer.is_trained, model_version=scorer.model_version,
        dataset_ready=(DATA_PROCESSED / "test.csv").exists(),
        experiment_ready=ready, audit_rows=audit_rows, audit_chain_valid=chain_ok,
        llm_enabled=build_planner("auto") is not None,
        db_engine=engine_for(),
    )


@app.get("/api/overview", response_model=OverviewCards)
def overview() -> OverviewCards:
    s = _cases()
    agent = _require_run(s, "recoverai")
    base = _require_run(s, "baseline")
    comp = _require_run(s, "comparison")
    meta = s.get_run("meta") or {}
    ctrl = s.get_run("control") or {}
    vsc = (s.get_run("vs_control") or {}).get("recoverai", {})
    s.close()
    vsc_ci = (vsc.get("bootstrap") or {}).get("incremental_revenue", {})
    return OverviewCards(
        revenue_at_risk=agent["revenue_at_risk"],
        revenue_recovered=agent["revenue_recovered"],
        recovery_rate=agent["recovery_rate"],
        incremental_recovery_vs_baseline=comp["incremental_recovered_revenue"],
        recovery_uplift_pct=comp["recovery_uplift_pct"],
        incremental_ci_low=(comp.get("bootstrap") or {}).get(
            "incremental_revenue", {}).get("p05"),
        incremental_ci_high=(comp.get("bootstrap") or {}).get(
            "incremental_revenue", {}).get("p95"),
        incremental_ci_excludes_zero=(comp.get("bootstrap") or {}).get("excludes_zero"),
        cases_processed=agent["cases"], cases_escalated=agent["cases_escalated"],
        cases_stopped=agent["cases_stopped"], cases_recovered=agent["cases_recovered"],
        total_cost=agent["total_cost"], net_recovered=agent["net_recovered"],
        avg_recovery_hours=agent["avg_recovery_hours"],
        baseline_recovered=base["revenue_recovered"],
        baseline_recovery_rate=base["recovery_rate"],
        unsafe_actions_prevented=base["risk_actions_taken"] - agent["risk_actions_taken"],
        control_recovered=ctrl.get("revenue_recovered"),
        control_recovery_rate=ctrl.get("recovery_rate"),
        incremental_vs_control=vsc.get("incremental_recovered_revenue"),
        share_of_revenue_that_is_causal=vsc.get("share_of_revenue_that_is_causal"),
        control_ci_low=vsc_ci.get("p05"), control_ci_high=vsc_ci.get("p95"),
        model_version=meta.get("model_version", ""), planner=meta.get("planner", ""),
    )


@app.get("/api/revenue-at-risk", response_model=RevenueAtRisk)
def revenue_at_risk(top: int = Query(10, ge=1, le=100)) -> RevenueAtRisk:
    s = _cases()
    agent = _require_run(s, "recoverai")
    rows = s.queue("recoverai", limit=top, order_by="expected_recovery")
    s.close()
    return RevenueAtRisk(
        total_at_risk=agent["revenue_at_risk"],
        total_recovered=agent["revenue_recovered"],
        total_outstanding=round(agent["revenue_at_risk"] - agent["revenue_recovered"], 2),
        by_failure_category=agent["by_category"], by_failure_code=agent["by_failure_code"],
        top_cases=[_row(r) for r in rows],
    )


@app.get("/api/recovery-queue", response_model=list[QueueRow])
def recovery_queue(
    limit: int = Query(100, ge=1, le=2000), offset: int = Query(0, ge=0),
    status: str | None = None, failure_code: str | None = None,
    sort: str = Query("expected_recovery"), strategy: str = "recoverai",
    direction: str = Query("desc", pattern="^(asc|desc)$"),
) -> list[QueueRow]:
    s = _cases()
    rows = s.queue(strategy, limit=limit, offset=offset, status=status,
                   failure_code=failure_code, order_by=sort, direction=direction)
    s.close()
    return [_row(r) for r in rows]


@app.get("/api/cases/{transaction_id}", response_model=CaseDetail)
def case_detail(transaction_id: str) -> CaseDetail:
    a = _audit()
    s = _cases(a.conn)
    row = s.get_case(transaction_id, "recoverai")
    if row is None:
        a.close()
        raise HTTPException(404, f"unknown transaction {transaction_id}")
    base = s.get_case(transaction_id, "baseline")
    events = a.timeline(transaction_id)
    ok, _ = a.verify()
    a.close()
    return CaseDetail(case=_row(row), baseline=base, audit_events=events, chain_valid=ok)


@app.get("/api/cases/{transaction_id}/audit", response_model=AuditPage)
def case_audit(transaction_id: str) -> AuditPage:
    a = _audit()
    rows = a.timeline(transaction_id)
    ok, _ = a.verify()
    a.close()
    if not rows:
        raise HTTPException(404, f"no audit trail for {transaction_id}")
    return AuditPage(transaction_id=transaction_id, rows=rows, total=len(rows), chain_valid=ok)


@app.get("/api/audit", response_model=AuditPage)
def audit_log(limit: int = Query(200, ge=1, le=2000),
              decision: str | None = None,
              policy_result: str | None = Query(None, pattern="^(approve|modify|reject)$"),
              ) -> AuditPage:
    a = _audit()
    rows = a.recent(limit=limit, decision=decision, policy_result=policy_result)
    ok, _ = a.verify()
    total = a.count()
    a.close()
    return AuditPage(rows=rows, total=total, chain_valid=ok)


@app.get("/api/dlq")
def dlq_entries(quarantined_only: bool = True, limit: int = Query(500, ge=1, le=2000)):
    """Quarantined (customer, channel) pairs awaiting review. Scoped to the active run's
    id, which lives in the run metadata -- a DLQ from a previous experiment is history,
    not a work queue."""
    s_ = _cases()
    meta = s_.get_run("meta") or {}
    s_.close()
    run_id = meta.get("run_id")
    if not run_id:
        return JSONResponse({"run_id": None, "entries": [], "stats": {
            "tracked_pairs": 0, "quarantined": 0, "total_failures": 0, "threshold": 0}})
    d = DLQStore(run_id, DB_PATH)
    payload = {"run_id": run_id, "stats": d.stats(),
               "entries": d.entries(quarantined_only=quarantined_only, limit=limit)}
    d.close()
    return JSONResponse(payload)


@app.post("/api/dlq/release")
def dlq_release(customer_id: str, channel: str):
    """Human review outcome: put a quarantined pair back in service."""
    s_ = _cases()
    meta = s_.get_run("meta") or {}
    s_.close()
    run_id = meta.get("run_id")
    if not run_id:
        raise HTTPException(404, "no active run")
    d = DLQStore(run_id, DB_PATH)
    d.release(customer_id, channel)
    ok = not d.is_quarantined(customer_id, channel)
    d.close()
    return JSONResponse({"released": ok, "customer_id": customer_id, "channel": channel})


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    #: "keywords" (default, deterministic) or "llm" to put a model in front of it.
    router: str = "keywords"


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    """Answer a question about this run.

    The router only ever chooses an intent; every figure in the reply is computed by
    `ChatEngine` from the database. A model cannot state a number here, so it cannot
    state a wrong one.
    """
    from backend.app.chat import answer as render
    from backend.app.chat.dsl import Agg, run as run_query
    from backend.app.chat.parse import parse
    from backend.app.chat.query import ChatEngine
    from backend.app.chat.router import KeywordRouter

    a = _audit()
    s_ = _cases(a.conn)

    # Questions about the run as a whole (arm comparison, policy activity, the DLQ) are
    # not case-table queries, so they keep their dedicated handlers. Everything else goes
    # through the general query language.
    special = {"arm_comparison", "policy_activity", "dlq_status", "case_trace"}
    intent = KeywordRouter().route(req.question)
    if intent.intent.value in special:
        meta = s_.get_run("meta") or {}
        dlq = DLQStore(meta["run_id"], conn=a.conn) if meta.get("run_id") else None
        out = ChatEngine(s_, audit=a, dlq=dlq).run(intent, source="keywords")
        a.close()
        return JSONResponse(out.model_dump(mode="json"))

    q = parse(req.question)
    problems = q.validate_against_schema()
    if problems:
        a.close()
        return JSONResponse({
            "text": ("I understood the question but could not run it against the case "
                     "table: " + "; ".join(problems)
                     + ". Try naming a stored field - amount, amount recovered, retries, "
                       "failure cause, failure category, outcome or customer."),
            "intent": "invalid", "query": q.model_dump(mode="json"),
            "data": {}, "case_ids": [], "source": "parser"})

    result = run_query(a.conn, q)
    text = render.render(q, result) + render.caveat(q)
    ids = [r["transaction_id"] for r in result.rows if "transaction_id" in r][:10]
    a.close()
    return JSONResponse({
        "text": text, "intent": f"{q.agg.value}"
                                + (f"({q.field})" if q.field else ""),
        "query": q.model_dump(mode="json", exclude_defaults=True),
        "data": {"scalar": result.scalar, "matched": result.total_matched,
                 "grand_total": result.grand_total},
        "sql": result.sql, "case_ids": ids,
        "unresolved": q.unresolved, "source": "query-language",
    })


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    s = _cases()
    agent = _require_run(s, "recoverai")
    base = _require_run(s, "baseline")
    comp = _require_run(s, "comparison")
    meta = s.get_run("meta") or {}
    control = s.get_run("control") or {}
    vs_control = s.get_run("vs_control") or {}
    s.close()
    model_eval: dict[str, Any] = {}
    p = Path(DB_PATH).parent / "model_evaluation.json"
    if p.exists():
        import json
        model_eval = json.loads(p.read_text())
    return JSONResponse({"recoverai": agent, "baseline": base, "control": control,
                         "comparison": comp, "vs_control": vs_control,
                         "meta": meta, "model": model_eval})


@app.get("/api/baseline")
def baseline_metrics() -> JSONResponse:
    s = _cases()
    base = _require_run(s, "baseline")
    s.close()
    return JSONResponse(base)


@app.post("/api/recovery/run", response_model=RunResponse)
def run_recovery(req: RunRequest) -> RunResponse:
    """Work a slice of the queue live against the mock rails."""
    try:
        txns = to_transactions(load_split(req.split))[: req.limit]
    except FileNotFoundError:
        raise HTTPException(404, f"split {req.split!r} not found -- generate the dataset first")

    t0 = time.time()
    a = _audit()
    outcomes, report, states = run_agent_batch(
        txns, PaymentGateway(seed=SEED), scorer=get_scorer(),
        planner=build_planner("auto"), on_audit=a.append,
    )
    a.commit()
    if req.persist:
        extra = {s_.transaction_id: {
            "currency": s_.currency, "amount": s_.amount,
            "root_cause": s_.root_cause.value if s_.root_cause else None,
            "recommended_action": s_.actions_taken[0] if s_.actions_taken else None,
        } for s_ in states}
        store = _cases(a.conn)
        store.save_cases(outcomes, "recoverai", extra)
        store.save_run("recoverai", report)
    a.close()
    return RunResponse(cases_processed=len(outcomes), summary=report,
                       seconds=round(time.time() - t0, 2))


@app.post("/api/recovery/run/{transaction_id}", response_model=LiveTrace)
def run_one(transaction_id: str, split: str = "test") -> LiveTrace:
    """Work a single case live and return its full decision trace."""
    txns = {t["transaction_id"]: t for t in to_transactions(load_split(split))}
    txn = txns.get(transaction_id)
    if txn is None:
        raise HTTPException(404, f"unknown transaction {transaction_id}")

    a = _audit()
    agent = RecoveryAgent(executor=ActionExecutor(gateway=PaymentGateway(seed=SEED)),
                          scorer=get_scorer(), planner=build_planner("auto"),
                          on_audit=a.append)
    state = agent.run(txn)
    a.commit()
    store = _cases(a.conn)
    store.save_cases([to_outcome(state)], "recoverai", {
        transaction_id: {"currency": state.currency, "amount": state.amount,
                         "root_cause": state.root_cause.value if state.root_cause else None,
                         "recommended_action": state.actions_taken[0] if state.actions_taken else None}})
    a.close()

    return LiveTrace(
        transaction_id=state.transaction_id, status=state.status.value,
        amount_recovered=state.amount_recovered,
        recovery_probability=state.recovery_probability,
        expected_recovery=state.expected_recovery,
        root_cause=state.root_cause.value if state.root_cause else None,
        steps=[e.model_dump(mode="json") for e in state.audit_events],
        stop_reason=state.stop_reason,
    )


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
