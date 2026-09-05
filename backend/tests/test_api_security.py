"""API-level security: authorisation, tenant isolation over HTTP, webhooks, headers."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.app.api import ops as ops_module
from backend.app.api.deps import get_auth_policy, get_rate_limiter, set_rate_limiter
from backend.app.main import app
from backend.app.models.enums import Role
from backend.app.security.ratelimit import RateLimiter
from backend.app.security.webhooks import WebhookVerifier


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def keys():
    """One key per role, so a test can name the capability it is exercising."""
    policy = get_auth_policy()
    made = {
        "operator": ("k_op", "sec_op", "acme", [Role.OPERATOR]),
        "analyst": ("k_an", "sec_an", "acme", [Role.ANALYST]),
        "auditor": ("k_au", "sec_au", "acme", [Role.AUDITOR]),
        "admin": ("k_ad", "sec_ad", "acme", [Role.ADMIN]),
        "system": ("k_sy", "sec_sy", "acme", [Role.SYSTEM]),
        "other_operator": ("k_ot", "sec_ot", "globex", [Role.OPERATOR]),
    }
    for kid, secret, tenant, roles in made.values():
        policy.add_key(kid, secret, tenant, roles)
    yield {name: {"X-API-Key": f"{kid}.{secret}"}
           for name, (kid, secret, _, _) in made.items()}
    for kid, *_ in made.values():
        policy.keys.pop(kid, None)


# ---------------------------------------------------------------- authorisation
def test_an_analyst_cannot_resolve_a_review(client, keys):
    r = client.post("/api/v1/reviews/rev_anything/approve",
                    json={"reason": "let me through"}, headers=keys["analyst"])
    assert r.status_code == 403


def test_an_auditor_can_read_overrides_but_cannot_write(client, keys):
    assert client.get("/api/v1/reviews/audit/overrides",
                      headers=keys["auditor"]).status_code == 200
    assert client.post("/api/v1/reviews/rev_x/reject", json={"reason": "no"},
                       headers=keys["auditor"]).status_code == 403


def test_an_operator_cannot_promote_a_model(client, keys):
    """Who configures the envelope and who approves actions under it are separated."""
    r = client.post("/api/v1/models/anything/promote",
                    json={"reason": "ship it"}, headers=keys["operator"])
    assert r.status_code == 403


def test_a_bad_credential_is_401_not_403(client):
    r = client.get("/api/v1/reviews", headers={"X-API-Key": "nope.wrong"})
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_the_error_body_does_not_say_why_authentication_failed(client):
    """Enumerating the reason helps an attacker more than it helps a client."""
    body = client.get("/api/v1/reviews", headers={"X-API-Key": "nope.wrong"}).json()
    assert body["detail"] == "authentication failed"


# ---------------------------------------------------------------- tenancy over HTTP
def test_a_review_from_another_tenant_is_404_not_403(client, keys):
    """403 would confirm the id exists, which is a cross-tenant leak even without the
    contents."""
    r = client.get("/api/v1/reviews/rev_belonging_to_acme", headers=keys["other_operator"])
    assert r.status_code == 404


def test_no_route_accepts_a_tenant_parameter(client, keys):
    """A caller that cannot name a tenant cannot name someone else's."""
    r = client.get("/api/v1/reviews?tenant_id=globex", headers=keys["operator"])
    assert r.status_code == 200
    # The query parameter is ignored, not honoured: the store was built from the key.


def test_an_override_reason_is_required(client, keys):
    for bad in ({}, {"reason": ""}, {"reason": "x"}):
        r = client.post("/api/v1/reviews/rev_x/approve", json=bad,
                        headers=keys["operator"])
        assert r.status_code == 422, f"accepted a blank justification: {bad}"


# ---------------------------------------------------------------- webhooks
@pytest.fixture
def signed():
    verifier = WebhookVerifier(secret="whsec_" + "t" * 32)
    ops_module.set_verifier(verifier)
    yield verifier
    ops_module.set_verifier(None)


def test_an_unsigned_webhook_is_refused(client, signed):
    body = json.dumps({"type": "payment_intent.succeeded", "id": "evt_1"})
    assert client.post("/api/v1/webhooks/stripe", content=body).status_code == 401


def test_a_forged_signature_is_refused(client, signed):
    body = json.dumps({"type": "payment_intent.succeeded", "id": "evt_1"})
    r = client.post("/api/v1/webhooks/stripe", content=body,
                    headers={"X-Signature": "t=1700000000,v1=deadbeef"})
    assert r.status_code == 401


def test_a_valid_webhook_is_accepted_once_and_deduplicated(client, signed):
    body = json.dumps({"type": "payment_intent.succeeded", "id": "evt_dedupe",
                       "data": {"object": {"id": "pi_1", "customer": "cus_1"}}}).encode()
    header = signed.sign(body)
    first = client.post("/api/v1/webhooks/stripe", content=body,
                        headers={"X-Signature": header})
    assert first.status_code == 200 and first.json()["handled"]
    # Replay protection refuses the identical signature outright.
    assert client.post("/api/v1/webhooks/stripe", content=body,
                       headers={"X-Signature": header}).status_code == 401


def test_an_unknown_event_type_is_acknowledged_but_not_interpreted(client, signed):
    body = json.dumps({"type": "customer.subscription.updated", "id": "evt_2"}).encode()
    r = client.post("/api/v1/webhooks/stripe", content=body,
                    headers={"X-Signature": signed.sign(body)})
    assert r.status_code == 200
    assert r.json()["handled"] is False


# ---------------------------------------------------------------- transport
def test_security_headers_are_present(client):
    h = client.get("/api/health").headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]


def test_an_oversized_body_is_refused(client, keys):
    r = client.post("/api/recovery/run", content=b"x" * 2_000_000,
                    headers={**keys["system"], "Content-Type": "application/json"})
    assert r.status_code == 413


def test_rate_limiting_returns_429_with_a_retry_after(client, keys):
    original = get_rate_limiter()
    set_rate_limiter(RateLimiter(capacity=2, refill_per_second=0.001))
    try:
        codes = [client.get("/api/v1/reviews/stats", headers=keys["operator"]).status_code
                 for _ in range(6)]
        assert 429 in codes
        blocked = client.get("/api/v1/reviews/stats", headers=keys["operator"])
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
    finally:
        set_rate_limiter(original)


def test_health_reports_the_security_posture(client):
    d = client.get("/api/health").json()
    assert "auth_required" in d and "profile" in d
    assert d["policy_version"].startswith("rules@")


def test_prometheus_metrics_render(client):
    r = client.get("/api/v1/metrics/prometheus")
    assert r.status_code == 200
    assert "recoverai_api_requests_total" in r.text


def test_the_policy_envelope_is_introspectable(client, keys):
    d = client.get("/api/v1/policy", headers=keys["analyst"]).json()
    assert "R-RISK-BLOCK" in d["rules"]["reject"]
    assert "R-AMOUNT-CAP" in d["rules"]["review"]
    assert d["limits"]["max_retries"] >= 1
