"""Minimal HS256 JWT.

Written rather than pulled in because the requirement is fifty lines and the dependency
is not: `PyJWT` and `python-jose` have both shipped algorithm-confusion CVEs, and the
attack surface of a token library is almost entirely in the parts this system does not
use (RSA, JWKS fetching, `alg` negotiation).

The two decisions that matter are both refusals:

* **`alg` is not negotiated.** The header is checked against the one algorithm this
  module implements. The classic JWT vulnerability is a verifier that trusts the token
  to say how it should be verified -- `alg: none`, or an RS256 public key replayed as an
  HMAC secret. A verifier that accepts one algorithm cannot be confused about which.
* **Comparison is constant-time.** `hmac.compare_digest`, not `==`.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

ALGORITHM = "HS256"


class TokenError(ValueError):
    """Any failure to produce a trustworthy claim set. Deliberately one exception type:
    distinguishing "expired" from "bad signature" in an error returned to a caller tells
    an attacker which half of the token to keep working on."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(message: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode(), message, hashlib.sha256).digest()


def encode(claims: dict, secret: str, ttl_seconds: int = 3600) -> str:
    if not secret or len(secret) < 32:
        raise TokenError(
            "signing secret must be at least 32 characters; a short HMAC secret is "
            "brute-forceable offline from a single captured token")
    now = int(time.time())
    payload = {**claims, "iat": now, "exp": now + ttl_seconds}
    header = _b64url_encode(json.dumps({"alg": ALGORITHM, "typ": "JWT"},
                                       separators=(",", ":")).encode())
    body = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    message = f"{header}.{body}".encode()
    return f"{header}.{body}.{_b64url_encode(_sign(message, secret))}"


def decode(token: str, secret: str, leeway_seconds: int = 0) -> dict:
    """Verify and return the claims. Raises `TokenError` on anything less than perfect."""
    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("malformed token")
    header_b64, body_b64, sig_b64 = parts

    try:
        header = json.loads(_b64url_decode(header_b64))
    except Exception as exc:
        raise TokenError("malformed token header") from exc

    # The token does not get to choose. This single line is the difference between a
    # verifier and a suggestion box.
    if header.get("alg") != ALGORITHM:
        raise TokenError(f"unsupported algorithm; only {ALGORITHM} is accepted")

    expected = _sign(f"{header_b64}.{body_b64}".encode(), secret)
    try:
        given = _b64url_decode(sig_b64)
    except Exception as exc:
        raise TokenError("malformed signature") from exc
    if not hmac.compare_digest(expected, given):
        raise TokenError("signature verification failed")

    try:
        claims = json.loads(_b64url_decode(body_b64))
    except Exception as exc:
        raise TokenError("malformed token body") from exc

    now = int(time.time())
    if "exp" not in claims:
        raise TokenError("token has no expiry; a non-expiring bearer token is a password")
    if now > int(claims["exp"]) + leeway_seconds:
        raise TokenError("token rejected")
    if "iat" in claims and int(claims["iat"]) > now + max(leeway_seconds, 60):
        raise TokenError("token rejected")
    return claims


@dataclass(frozen=True)
class ApiKey:
    """An API key, stored as a hash. The plaintext exists exactly once, at issue time.

    `key_id` is a public prefix so a key can be revoked, rate-limited and attributed in
    logs without the secret ever being written down anywhere.
    """
    key_id: str
    secret_hash: str
    tenant_id: str
    roles: tuple[str, ...]
    label: str = ""
    active: bool = True

    @staticmethod
    def hash_secret(secret: str, salt: str) -> str:
        # PBKDF2 rather than a bare SHA-256: an API key is a credential, and a leaked
        # table of single-round hashes is a leaked table of credentials.
        return hashlib.pbkdf2_hmac("sha256", secret.encode(), salt.encode(), 120_000).hex()

    def verify(self, secret: str, salt: str) -> bool:
        return self.active and hmac.compare_digest(
            self.secret_hash, self.hash_secret(secret, salt))
