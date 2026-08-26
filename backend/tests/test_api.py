"""Phase 8 tests -- API contracts against the real experiment database."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.config import DB_PATH
from backend.app.main import app

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="no experiment database -- run scripts/run_experiment.py first",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["audit_chain_valid"] is True, "the audit chain is broken"


def test_overview_cards(client):
    d = client.get("/api/overview").json()
    for k in ("revenue_at_risk", "revenue_recovered", "recovery_rate",
              "incremental_recovery_vs_baseline", "cases_processed",
              "cases_escalated", "cases_stopped"):
        assert k in d, f"overview card {k} missing"
    assert d["revenue_recovered"] <= d["revenue_at_risk"]
    assert 0.0 <= d["recovery_rate"] <= 1.0


def test_overview_matches_the_metrics_endpoint(client):
    """The dashboard and the evaluation must not disagree."""
    ov = client.get("/api/overview").json()
    m = client.get("/api/metrics").json()
    assert ov["revenue_recovered"] == m["recoverai"]["revenue_recovered"]
    assert ov["incremental_recovery_vs_baseline"] == m["comparison"]["incremental_recovered_revenue"]
    assert ov["baseline_recovered"] == m["baseline"]["revenue_recovered"]


def test_recovery_queue_is_sorted_by_expected_recovery(client):
    rows = client.get("/api/recovery-queue?limit=50").json()
    assert rows
    ev = [r["expected_recovery"] for r in rows]
    assert ev == sorted(ev, reverse=True), "queue is not ranked by expected recovery"


def test_recovery_queue_supports_sorting_and_filtering(client):
    rows = client.get("/api/recovery-queue?limit=20&sort=amount_usd").json()
    amounts = [r["amount_usd"] for r in rows]
    assert amounts == sorted(amounts, reverse=True)

    rec = client.get("/api/recovery-queue?limit=20&status=recovered").json()
    assert all(r["status"] == "recovered" for r in rec)

    fc = client.get("/api/recovery-queue?limit=20&failure_code=expired_card").json()
    assert all(r["failure_code"] == "expired_card" for r in fc)


def test_queue_pagination_does_not_repeat_rows(client):
    a = client.get("/api/recovery-queue?limit=10&offset=0").json()
    b = client.get("/api/recovery-queue?limit=10&offset=10").json()
    assert not ({r["transaction_id"] for r in a} & {r["transaction_id"] for r in b})


def test_case_detail_and_audit_trail(client):
    tid = client.get("/api/recovery-queue?limit=1").json()[0]["transaction_id"]
    d = client.get(f"/api/cases/{tid}").json()
    assert d["case"]["transaction_id"] == tid
    assert d["chain_valid"] is True
    assert len(d["audit_events"]) >= 5

    a = client.get(f"/api/cases/{tid}/audit").json()
    assert a["transaction_id"] == tid and a["total"] == len(a["rows"])
    stages = {r["agent_decision"] for r in a["rows"]}
    for expected in ("load_transaction", "score_recovery", "diagnose_root_cause",
                     "select_intervention", "validate_policy", "monitor_outcome"):
        assert expected in stages, f"{expected} missing from the trace"


def test_audit_rows_are_in_workflow_order(client):
    tid = client.get("/api/recovery-queue?limit=1").json()[0]["transaction_id"]
    rows = client.get(f"/api/cases/{tid}/audit").json()["rows"]
    assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)
    assert rows[0]["agent_decision"] == "load_transaction"


def test_unknown_case_returns_404(client):
    assert client.get("/api/cases/does_not_exist").status_code == 404
    assert client.get("/api/cases/does_not_exist/audit").status_code == 404


def test_revenue_at_risk_breakdown(client):
    d = client.get("/api/revenue-at-risk?top=5").json()
    assert d["total_at_risk"] > 0
    assert len(d["top_cases"]) == 5
    assert set(d["by_failure_category"]) <= {"TEMPORARY", "CUSTOMER_ACTION",
                                             "PERSISTENT", "RISK_COMPLIANCE"}
    assert d["total_outstanding"] == pytest.approx(
        d["total_at_risk"] - d["total_recovered"], abs=0.01)


def test_metrics_and_baseline_endpoints(client):
    m = client.get("/api/metrics").json()
    for k in ("recoverai", "baseline", "comparison", "meta"):
        assert k in m
    b = client.get("/api/baseline").json()
    assert b["strategy"] == "baseline"
    assert b["risk_actions_taken"] > 0, "baseline should show unsafe actions to contrast against"


def test_agent_beats_baseline_through_the_api(client):
    """The headline claim, asserted on what the API actually serves."""
    m = client.get("/api/metrics").json()
    assert m["comparison"]["incremental_recovered_revenue"] > 0
    assert m["recoverai"]["risk_actions_taken"] == 0
    assert m["recoverai"]["total_retries"] < m["baseline"]["total_retries"]


def test_run_one_case_live(client):
    tid = client.get("/api/recovery-queue?limit=1").json()[0]["transaction_id"]
    r = client.post(f"/api/recovery/run/{tid}")
    assert r.status_code == 200
    d = r.json()
    assert d["transaction_id"] == tid
    assert d["status"] in ("recovered", "escalated", "stopped", "exhausted")
    assert len(d["steps"]) >= 5
    assert d["stop_reason"]


def test_run_batch_live(client):
    r = client.post("/api/recovery/run", json={"limit": 25, "persist": False})
    assert r.status_code == 200
    d = r.json()
    assert d["cases_processed"] == 25
    assert d["summary"]["risk_actions_taken"] == 0


def test_run_rejects_an_out_of_range_limit(client):
    assert client.post("/api/recovery/run", json={"limit": 99999}).status_code == 422
    assert client.post("/api/recovery/run", json={"limit": 0}).status_code == 422
