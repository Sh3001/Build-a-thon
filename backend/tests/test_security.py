"""Security tests.

Each one names a concrete attack rather than a feature. A test called
`test_jwt_works` proves nothing an attacker cares about.
"""
from __future__ import annotations

import base64
import json
import time

import pytest

from backend.app.security.auth import (
    ANONYMOUS,
    AuthError,
    AuthPolicy,
    ForbiddenError,
)
from backend.app.security.ratelimit import RateLimiter
from backend.app.security.tokens import ApiKey, TokenError, decode, encode
from backend.app.security.webhooks import WebhookError, WebhookVerifier, map_event

SECRET = "k" * 40


@pytest.fixture
def prod():
    return AuthPolicy.from_env({
        "RECOVERAI_PROFILE": "production",
        "RECOVERAI_JWT_SECRET": SECRET,
        "RECOVERAI_API_KEYS": "kid1:s3cret:acme:operator,kid2:s3cret2:globex:analyst",
    })


# ---------------------------------------------------------------- tokens
def test_alg_none_is_refused():
    """The classic JWT break: a verifier that trusts the token to say how to verify it."""
    token = encode({"sub": "a", "tenant": "t", "roles": ["admin"]}, SECRET)
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    forged = f"{header}.{token.split('.')[1]}."
    with pytest.raises(TokenError, match="algorithm"):
        decode(forged, SECRET)


def test_a_tampered_payload_is_refused():
    token = encode({"sub": "a", "tenant": "acme", "roles": ["analyst"]}, SECRET)
    h, body, sig = token.split(".")
    tampered = base64.urlsafe_b64encode(
        json.dumps({"sub": "a", "tenant": "acme", "roles": ["admin"],
                    "exp": int(time.time()) + 3600}).encode()).rstrip(b"=").decode()
    with pytest.raises(TokenError):
        decode(f"{h}.{tampered}.{sig}", SECRET)


def test_an_expired_token_is_refused():
    token = encode({"sub": "a", "tenant": "t", "roles": ["analyst"]}, SECRET, ttl_seconds=-1)
    with pytest.raises(TokenError):
        decode(token, SECRET)


def test_a_short_signing_secret_is_refused_at_issue_time():
    """A short HMAC secret is brute-forceable offline from one captured token."""
    with pytest.raises(TokenError):
        encode({"sub": "a"}, "tooshort")


def test_api_key_hashing_is_slow_and_salted():
    a = ApiKey.hash_secret("s3cret", "salt-a")
    b = ApiKey.hash_secret("s3cret", "salt-b")
    assert a != b, "a shared salt would make one rainbow table cover every deployment"
    assert len(a) == 64


# ---------------------------------------------------------------- authentication
def test_production_refuses_to_start_without_a_signing_secret():
    """A process that silently downgrades to open access is worse than one that will
    not boot."""
    with pytest.raises(RuntimeError, match="Refusing to start"):
        AuthPolicy.from_env({"RECOVERAI_PROFILE": "production"})


def test_no_credential_is_refused_in_production(prod):
    with pytest.raises(AuthError):
        prod.authenticate()


def test_presenting_two_credentials_is_refused(prod):
    """Either a client bug or a probe for which credential the server prefers."""
    token = encode({"sub": "a", "tenant": "acme", "roles": ["analyst"]}, SECRET)
    with pytest.raises(AuthError):
        prod.authenticate(authorization=f"Bearer {token}", api_key="kid1.s3cret")


def test_an_unknown_role_grants_nothing_rather_than_everything(prod):
    token = encode({"sub": "a", "tenant": "acme", "roles": ["superuser"]}, SECRET)
    with pytest.raises(AuthError, match="no recognised role"):
        prod.authenticate(authorization=f"Bearer {token}")


def test_a_token_without_a_tenant_is_refused(prod):
    token = encode({"sub": "a", "roles": ["analyst"]}, SECRET)
    with pytest.raises(AuthError, match="tenant"):
        prod.authenticate(authorization=f"Bearer {token}")


def test_the_anonymous_principal_can_never_execute():
    """Even wide open in development, an unauthenticated caller cannot move money."""
    assert ANONYMOUS.has("read:metrics")
    assert not ANONYMOUS.has("execute:action")
    assert not ANONYMOUS.has("write:review")
    with pytest.raises(ForbiddenError):
        ANONYMOUS.require("execute:action")


def test_roles_are_capabilities_not_a_hierarchy(prod):
    """ADMIN is not "operator plus more": separating who configures the envelope from who
    approves actions under it is what makes the audit log meaningful."""
    analyst = prod.authenticate(api_key="kid2.s3cret2")
    assert analyst.has("read:metrics")
    assert not analyst.has("write:review")
    assert not analyst.has("execute:action")


# ---------------------------------------------------------------- tenancy
def test_the_tenant_comes_from_the_credential_not_the_request(prod):
    """The whole of tenant isolation. A caller cannot name a tenant, so a caller cannot
    name someone else's."""
    acme = prod.authenticate(api_key="kid1.s3cret")
    globex = prod.authenticate(api_key="kid2.s3cret2")
    assert acme.tenant_id == "acme"
    assert globex.tenant_id == "globex"
    assert acme.tenant_id != globex.tenant_id


# ---------------------------------------------------------------- webhooks
@pytest.fixture
def verifier():
    return WebhookVerifier(secret="whsec_" + "a" * 32)


def _body(**kw):
    return json.dumps({"type": "payment_intent.succeeded", "id": "evt_1", **kw}).encode()


def test_a_valid_webhook_is_accepted(verifier):
    body = _body()
    assert verifier.verify(body, verifier.sign(body))["id"] == "evt_1"


def test_a_replayed_webhook_is_refused(verifier):
    """A captured signature must not be usable five times to trigger five actions."""
    body = _body()
    header = verifier.sign(body)
    verifier.verify(body, header)
    with pytest.raises(WebhookError):
        verifier.verify(body, header)
    assert verifier.stats()["replays_blocked"] == 1


def test_a_stale_signature_is_refused(verifier):
    body = _body()
    with pytest.raises(WebhookError):
        verifier.verify(body, verifier.sign(body, int(time.time()) - 3600))


def test_a_signature_without_a_timestamp_is_refused(verifier):
    """Replay protection is not optional, so an unfreshenable signature is not accepted."""
    body = _body()
    import hashlib
    import hmac
    bare = hmac.new(verifier.secret.encode(), body, hashlib.sha256).hexdigest()
    with pytest.raises(WebhookError):
        verifier.verify(body, bare)


def test_whitespace_reformatting_does_not_pass(verifier):
    """The signature covers raw bytes, not the parsed document: `{"a":1}` and
    `{ "a" : 1 }` parse identically and must not share a signature."""
    body = _body()
    header = verifier.sign(body)
    reformatted = json.dumps(json.loads(body), indent=2).encode()
    with pytest.raises(WebhookError):
        verifier.verify(reformatted, header)


def test_an_oversized_body_is_refused_before_hashing(verifier):
    """HMAC over an attacker-chosen 500MB body is a DoS that costs one request."""
    verifier.max_body_bytes = 100
    with pytest.raises(WebhookError, match="exceeds"):
        verifier.verify(b"x" * 1000, "t=1,v1=deadbeef")


def test_an_unconfigured_secret_refuses_everything():
    """Better to reject every webhook than to trust unverified payment events."""
    with pytest.raises(WebhookError, match="no webhook secret"):
        WebhookVerifier(secret="").verify(b"{}", "t=1,v1=x")


def test_unknown_provider_events_map_to_nothing():
    """A webhook we do not understand must not be interpreted as one we do."""
    assert map_event("payment_intent.succeeded") == "payment.recovered"
    assert map_event("customer.subscription.trial_will_end") is None


# ---------------------------------------------------------------- rate limiting
def test_the_bucket_refills_over_time():
    rl = RateLimiter(capacity=3, refill_per_second=1.0)
    assert [rl.check("k", now=0.0)[0] for _ in range(4)] == [True, True, True, False]
    assert rl.check("k", now=2.0)[0], "two seconds should refill two tokens"


def test_buckets_are_per_key():
    rl = RateLimiter(capacity=1, refill_per_second=0.0)
    assert rl.check("a")[0] and rl.check("b")[0]
    assert not rl.check("a")[0]


def test_tracked_keys_are_bounded():
    """An unbounded dict keyed on attacker-controlled input is memory exhaustion."""
    rl = RateLimiter(capacity=5, max_keys=50)
    for i in range(500):
        rl.check(f"key_{i}")
    assert rl.stats()["tracked_keys"] <= 50


def test_a_blocked_request_reports_a_retry_after():
    rl = RateLimiter(capacity=1, refill_per_second=2.0)
    rl.check("k", now=0.0)
    ok, retry_after = rl.check("k", now=0.0)
    assert not ok and 0 < retry_after <= 1.0
