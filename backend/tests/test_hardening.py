"""Regression tests for request isolation and durable execution failure boundaries."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from backend.app.api import deps
from backend.app.api.middleware import BodyLimitMiddleware
from backend.app.audit.store import AuditStore
from backend.app.database.db import StorageUnavailable, connect
from backend.app.database.jobs import JobStore
from backend.app.database.operational import ActionLedger
from backend.app.database.store import CaseStore
from backend.app.main import app
from backend.app.models.enums import FailureCategory, FailureCode, Role
from backend.app.models.schemas import AuditEvent
from backend.app.security.auth import AuthPolicy
from backend.app.services.results import CaseOutcome


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "state.db", url=None)
    yield conn
    conn.close()


@pytest.fixture
def client(db, monkeypatch):
    import backend.app.database.jobs as jobs
    import backend.app.main as main
    # Requests own distinct connections, as they do in the deployed application.
    path = db.raw.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setattr(jobs, "connect", lambda: connect(path, url=None))
    monkeypatch.setattr(main, "DB_PATH", path)
    monkeypatch.setattr(deps, "DB_PATH", path)
    policy = AuthPolicy.from_env({"RECOVERAI_PROFILE": "testing"})
    for tenant in ("acme", "globex"):
        policy.add_key(tenant, "secret", tenant, [Role.ANALYST, Role.AUDITOR, Role.SYSTEM, Role.OPERATOR])
    original = deps.get_auth_policy()
    deps.set_auth_policy(policy)
    yield TestClient(app)
    deps.set_auth_policy(original)
    deps.get_rate_limiter().reset()


def headers(tenant="acme"):
    return {"X-API-Key": f"{tenant}.secret"}


def outcome(tid="shared"):
    return CaseOutcome(transaction_id=tid, customer_id="cust", amount_usd=100,
                       failure_code=FailureCode.BANK_TIMEOUT,
                       failure_category=FailureCategory.TEMPORARY, strategy="recoverai")


def test_default_connection_never_creates_local_storage(monkeypatch):
    for key in ("RECOVERAI_DB", "RECOVERAI_DB_URL", "RECOVERAI_ALLOW_SQLITE"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(StorageUnavailable):
        connect(url=None)


def test_dashboard_and_chat_are_scoped(client, db):
    acme = CaseStore(conn=db, tenant_id="acme")
    globex = CaseStore(conn=db, tenant_id="globex")
    acme.save_cases([outcome("a")], "recoverai")
    globex.save_cases([outcome("b"), outcome("c")], "recoverai")
    assert [r["transaction_id"] for r in client.get("/api/recovery-queue", headers=headers()).json()] == ["a"]
    assert client.get("/api/cases/b", headers=headers()).status_code == 404
    response = client.post("/api/chat", json={"question": "how many cases"}, headers=headers())
    assert response.status_code == 200
    assert response.json()["data"]["scalar"] == 1
    assert client.get("/api/recovery-queue", headers={"X-API-Key": "bad.key"}).status_code == 401


def test_audit_does_not_leak_other_tenants(client, db):
    audit = AuditStore(conn=db)
    for tenant in ("acme", "globex"):
        audit.append(AuditEvent(transaction_id="shared", customer_id="cust", tenant_id=tenant,
                                agent_decision="test", reason=tenant))
    audit.commit()
    rows = client.get("/api/cases/shared/audit", headers=headers()).json()["rows"]
    assert len(rows) == 1 and rows[0]["tenant_id"] == "acme"


def test_job_submission_deduplicates_and_isolates(client):
    h = {**headers(), "Idempotency-Key": "request-one"}
    first = client.post("/api/recovery/run", json={"limit": 2}, headers=h)
    assert first.status_code == 202
    second = client.post("/api/recovery/run", json={"limit": 2}, headers=h)
    assert second.json()["job_id"] == first.json()["job_id"]
    assert client.post("/api/recovery/run", json={"limit": 3}, headers=h).status_code == 409
    job_id = first.json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}", headers=headers("globex")).status_code == 404
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=headers()).json()["status"] == "cancelled"


def test_atomic_claim_has_exactly_one_winner(db):
    path = db.raw.execute("PRAGMA database_list").fetchone()[2]
    ledgers = [ActionLedger("run", path, tenant_id="acme") for _ in range(8)]
    try:
        with ThreadPoolExecutor(8) as pool:
            won = list(pool.map(lambda ledger: ledger.claim("key", "txn"), ledgers))
        assert sum(won) == 1
        assert ActionLedger("run", conn=db, tenant_id="globex").claim("key", "txn")
    finally:
        for ledger in ledgers:
            ledger.close()


def test_uncertain_action_is_not_automatically_reclaimed(db):
    ledger = ActionLedger("run", conn=db)
    assert ledger.claim("key", "txn")
    ledger.uncertain("key")
    assert not ledger.claim("key", "txn")
    assert ledger.recall("key") is None


def test_transaction_rolls_back_job_and_event_together(db):
    jobs = JobStore("acme", conn=db)
    with pytest.raises(RuntimeError), db.transaction():
        jobs.enqueue({"limit": 1}, "request")
        jobs.event("evt", "test", {})
        raise RuntimeError("injected failure")
    assert db.execute("SELECT COUNT(*) n FROM recovery_jobs").fetchone()["n"] == 0
    assert db.execute("SELECT COUNT(*) n FROM recovery_outbox").fetchone()["n"] == 0


def test_chunked_body_limit_without_content_length():
    called = []
    sent = []
    async def downstream(scope, receive, send):
        called.append(True)
    messages = iter([{"type": "http.request", "body": b"abcd", "more_body": True},
                     {"type": "http.request", "body": b"efgh", "more_body": False}])
    async def receive():
        return next(messages)
    async def send(message):
        sent.append(message)
    asyncio.run(BodyLimitMiddleware(downstream, max_bytes=6)(
        {"type": "http", "headers": []}, receive, send))
    assert not called
    assert sent[0]["status"] == 413


def test_external_source_never_opens_state_db(client, monkeypatch):
    from backend.app.services.dataio import SqlSource
    monkeypatch.setenv("RECOVERAI_SOURCE_URL", "postgresql://unused")
    monkeypatch.setenv("RECOVERAI_SOURCE_QUERY", "SELECT * FROM failed_payments")
    monkeypatch.setenv("RECOVERAI_SOURCE_TENANT", "acme")
    result = SqlSource("postgresql://unused", "SELECT 1").build([
        {"transaction_id": "external", "customer_id": "c", "amount": 100,
         "payment_method": "card", "failure_code": "bank_timeout", "email": "private@example.com"}])
    monkeypatch.setattr(SqlSource, "fetch", lambda self: result)
    response = client.get("/api/recovery-queue", headers=headers())
    assert response.status_code == 200
    assert response.json()[0]["transaction_id"] == "external"
    assert "private@example.com" not in response.text
    assert client.get("/api/recovery-queue", headers=headers("globex")).status_code == 503
    assert client.post("/api/recovery/run", json={"limit": 1}, headers=headers()).status_code == 409


def test_resumable_case_and_audit_are_persisted_once(db, monkeypatch):
    from backend.app.services.recovery import RecoveryService
    jobs = JobStore("acme", conn=db)
    job = jobs.enqueue({"limit": 1}, "request")
    service = RecoveryService(jobs)
    txn = {"transaction_id": "txn_1", "customer_id": "cust", "amount": 100,
           "payment_method": "card", "failure_code": "bank_timeout"}
    state = service.run_case(txn, job["job_id"])
    count = service.audit.count()
    again = RecoveryService(jobs).run_case(txn, job["job_id"])
    assert again.status == state.status
    assert service.audit.count() == count
    assert service.audit.verify()[0]
    assert jobs.checkpoint("txn_1")["next"] == "done"


def test_checkpoint_resume_after_crash(db, monkeypatch):
    from backend.app.services.recovery import RecoveryService
    jobs = JobStore("acme", conn=db)
    job = jobs.enqueue({"limit": 1}, "request")
    service = RecoveryService(jobs)
    txn = {"transaction_id": "txn_resume", "customer_id": "cust", "amount": 100,
           "payment_method": "card", "failure_code": "bank_timeout"}
    save = jobs.save_checkpoint
    crashed = False
    def crash_after_checkpoint(tid, payload):
        nonlocal crashed
        save(tid, payload)
        if payload["next"] == "select_intervention" and not crashed:
            crashed = True
            raise RuntimeError("injected crash")
    monkeypatch.setattr(jobs, "save_checkpoint", crash_after_checkpoint)
    with pytest.raises(RuntimeError):
        service.run_case(txn, job["job_id"])
    monkeypatch.setattr(jobs, "save_checkpoint", save)
    final = RecoveryService(jobs).run_case(txn, job["job_id"])
    assert final.is_terminal
    assert service.audit.verify()[0]


def test_review_approval_queues_execution_atomically(client, db):
    from backend.app.database.review import ReviewStore
    from backend.app.models.enums import InterventionType, PolicyDecision
    from backend.app.models.schemas import AgentState, PolicyResult, ProposedAction, Transaction
    from backend.app.policies.version import policy_version
    txn = Transaction(transaction_id="review-case", customer_id="cust", amount=9000,
                      payment_method="card", failure_code="bank_timeout")
    state = AgentState(transaction_id=txn.transaction_id, customer_id=txn.customer_id,
                       amount=txn.amount, failure_code=txn.failure_code, transaction=txn)
    reviews = ReviewStore(conn=db, tenant_id="acme")
    rid = reviews.open_task(txn.transaction_id, PolicyResult(
        decision=PolicyDecision.HUMAN_REVIEW,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT),
        rules_fired=["R-AMOUNT-CAP"], reason="over cap"), policy_version=policy_version())
    db.execute("UPDATE human_review SET state_json = ? WHERE review_id = ?",
               (state.model_dump_json(), rid))
    db.commit()
    response = client.post(f"/api/v1/reviews/{rid}/approve", headers=headers(),
                           json={"reason": "confirmed with customer"})
    assert response.status_code == 200 and response.json()["job_id"]
    job = client.get(f"/api/jobs/{response.json()['job_id']}", headers=headers()).json()
    assert job["payload"]["review_id"] == rid
    assert client.post(f"/api/v1/reviews/{rid}/approve", headers=headers(),
                       json={"reason": "second approval"}).status_code == 409
    assert len(reviews.overrides()) == 1


def test_review_override_cannot_bypass_an_additional_review_rule():
    from backend.app.models.enums import InterventionType
    from backend.app.models.schemas import AgentState, ProposedAction
    from backend.app.policies.engine import PolicyContext, validate
    state = AgentState(transaction_id="t", customer_id="c", amount=9000,
                       failure_code="bank_timeout", expected_recovery=5000)
    action = ProposedAction(action=InterventionType.RETRY_PAYMENT)
    result = validate(state, action, PolicyContext(diagnosis_confidence=0.1),
                      approved_review_rule="R-AMOUNT-CAP")
    assert not result.allowed


def test_executor_crash_requires_reconciliation_before_retry(db, monkeypatch):
    from backend.app.tools.executor import ActionExecutor, ActionPending
    from backend.tests.test_tools import approved, state, txn
    ledger = ActionLedger("run", conn=db)
    executor = ActionExecutor(ledger=ledger)
    calls = []
    def crash(*args):
        calls.append(True)
        raise RuntimeError("provider outcome unknown")
    monkeypatch.setattr(executor, "_dispatch", crash)
    with pytest.raises(RuntimeError, match="unknown"):
        executor.execute(state(), approved(), txn())
    with pytest.raises(ActionPending):
        executor.execute(state(), approved(), txn())
    assert len(calls) == 1


def test_readonly_profile_requires_authentication():
    with pytest.raises(RuntimeError):
        AuthPolicy.from_env({"RECOVERAI_PROFILE": "readonly"})
    policy = AuthPolicy.from_env({"RECOVERAI_PROFILE": "readonly",
                                  "RECOVERAI_JWT_SECRET": "x" * 32})
    assert policy.required


def test_worker_completes_job_and_dashboard_without_experiment(client, db, monkeypatch):
    import pandas as pd

    import backend.app.services.recovery as recovery
    from scripts.worker import process_next
    monkeypatch.setattr(recovery, "load_split", lambda name: pd.DataFrame([
        {"transaction_id": "worker_1", "customer_id": "c", "amount": 100,
         "payment_method": "card", "failure_code": "bank_timeout"}]))
    response = client.post("/api/recovery/run", headers=headers(), json={"limit": 1})
    assert response.status_code == 202
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert process_next(db)
    job = client.get(f"/api/jobs/{response.json()['job_id']}", headers=headers()).json()
    assert job["status"] == "succeeded"
    overview = client.get("/api/overview", headers=headers())
    assert overview.status_code == 200
    assert overview.json()["cases_processed"] == 1
    assert overview.json()["incremental_recovery_vs_baseline"] is None


def approval_fixture(db, action_name="retry_payment", rule="R-AMOUNT-CAP"):
    from backend.app.agents.diagnose import diagnose
    from backend.app.models.schemas import AgentState, PolicyResult, ProposedAction, Transaction
    from backend.app.policies.version import policy_version
    from backend.app.services.recovery import RecoveryService
    jobs = JobStore("acme", conn=db)
    service = RecoveryService(jobs)
    txn = Transaction(transaction_id="large", customer_id="cust", amount=9000,
                      payment_method="card", failure_code="bank_timeout")
    state = AgentState(transaction_id=txn.transaction_id, customer_id=txn.customer_id,
                       amount=txn.amount, failure_code=txn.failure_code, transaction=txn,
                       diagnosis=diagnose(txn), expected_recovery=6300, recovery_probability=0.7)
    action = ProposedAction(action=action_name, channel="email" if action_name != "retry_payment" else None)
    verdict = PolicyResult(decision="human_review", effective_action=action,
                           rules_fired=[rule], reason="requires approval")
    rid = service.reviews.open_task(txn.transaction_id, verdict, policy_version=policy_version())
    db.execute("UPDATE human_review SET state_json = ? WHERE review_id = ?", (state.model_dump_json(), rid))
    service.cases.save_cases([outcome("large").model_copy(update={"status": "escalated"})], "recoverai")
    service.reviews.approve(rid, "operator", Role.OPERATOR, "confirmed with customer")
    followup = jobs.enqueue({"review_id": rid}, f"review:{rid}")
    return service, rid, followup["job_id"]


def test_approved_case_executes_once(db):
    service, rid, job_id = approval_fixture(db)
    first = service.execute_review(rid, job_id)
    count, audits = service.ledger.count(), service.audit.count()
    second = service.execute_review(rid, job_id)
    assert second == first
    assert service.ledger.count() == count
    assert service.audit.count() == audits


def test_new_consent_blocks_an_already_approved_contact(db):
    from backend.app.services.recovery import RecoveryBlocked
    service, rid, job_id = approval_fixture(db, "send_reminder", "R-REVIEW-LOW-CONFIDENCE")
    service.consent.opt_out("cust")
    with pytest.raises(RecoveryBlocked, match="Current policy"):
        service.execute_review(rid, job_id)
    assert service.ledger.count() == 0


def test_explicit_generation_creates_requested_directories(tmp_path, monkeypatch):
    from scripts import generate_dataset
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    monkeypatch.setattr(generate_dataset, "DATA_PROCESSED", processed)
    monkeypatch.setattr("sys.argv", ["generate_dataset", "--n", "100", "--out", str(raw)])
    assert generate_dataset.main() == 0
    assert (raw / "payment_recovery.csv").exists()
    assert (processed / "test.csv").exists()
