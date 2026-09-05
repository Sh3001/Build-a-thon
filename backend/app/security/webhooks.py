"""Verifying inbound payment events.

A webhook endpoint is an unauthenticated write path into the system's understanding of
what has been paid. Treating an unverified `payment.succeeded` as true lets anyone who
can reach the port mark invoices settled.

Four checks, in this order, and the order matters:

1. **Signature.** HMAC over `timestamp.body`, constant-time compared. Signing the raw
   body and not the parsed JSON, because `{"a":1}` and `{ "a" : 1 }` parse identically
   and hash differently -- verifying the parse lets an attacker who controls whitespace
   present a different document to the verifier than to the handler.
2. **Freshness.** Outside the tolerance window, refuse. Without it a signature captured
   once is valid forever.
3. **Replay.** A signature seen before is refused even inside the window, so a capture
   cannot be re-sent five times in five minutes to trigger five refunds.
4. **Parse.** Only then is the body interpreted -- untrusted bytes are not JSON-decoded
   until something has vouched for them.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

#: Providers send `t=<unix>,v1=<hex>`; a bare hex signature is also accepted.
DEFAULT_TOLERANCE_SECONDS = 300
#: Bound on the replay cache. An unbounded set is a memory leak with a security label.
DEFAULT_REPLAY_CAPACITY = 50_000
#: Refuse oversized bodies before hashing them: HMAC over an attacker-chosen 500MB body
#: is a denial of service that costs the attacker one request.
MAX_BODY_BYTES = 1_048_576


class WebhookError(ValueError):
    """Verification failed. One type on purpose -- telling a caller *which* check failed
    tells an attacker which one to work on next."""


@dataclass
class WebhookVerifier:
    secret: str
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS
    max_body_bytes: int = MAX_BODY_BYTES
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    _seen: OrderedDict[str, float] = field(default_factory=OrderedDict, repr=False)
    rejected: int = 0
    accepted: int = 0
    replays_blocked: int = 0

    # ------------------------------------------------------------------ signing
    def sign(self, body: bytes, timestamp: int | None = None) -> str:
        """Produce a header this verifier accepts. Used by tests and by the sender side
        of a loopback integration -- never by the receiving path."""
        ts = int(timestamp if timestamp is not None else time.time())
        mac = hmac.new(self.secret.encode(), f"{ts}.".encode() + body,
                       hashlib.sha256).hexdigest()
        return f"t={ts},v1={mac}"

    # ------------------------------------------------------------------ verifying
    def verify(self, body: bytes, header: str | None,
               now: float | None = None) -> dict[str, Any]:
        if not self.secret:
            raise WebhookError(
                "no webhook secret configured; refusing to accept unverified payment "
                "events rather than trusting them")
        if not header:
            raise WebhookError("missing signature header")
        if len(body) > self.max_body_bytes:
            self.rejected += 1
            raise WebhookError(
                f"body exceeds {self.max_body_bytes} bytes; refused before hashing")

        ts, signatures = self._parse_header(header)
        clock = now if now is not None else time.time()

        if ts is None:
            self.rejected += 1
            raise WebhookError("signature verification failed")
        if abs(clock - ts) > self.tolerance_seconds:
            self.rejected += 1
            raise WebhookError("signature verification failed")

        expected = hmac.new(self.secret.encode(), f"{ts}.".encode() + body,
                            hashlib.sha256).hexdigest()
        # compare_digest against every candidate, and do not short-circuit on the first
        # match -- the loop's timing should not depend on which signature matched.
        matched = False
        for candidate in signatures:
            if hmac.compare_digest(expected, candidate):
                matched = True
        if not matched:
            self.rejected += 1
            raise WebhookError("signature verification failed")

        if self._is_replay(expected, clock):
            self.replays_blocked += 1
            raise WebhookError("signature verification failed")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            self.rejected += 1
            raise WebhookError("payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            self.rejected += 1
            raise WebhookError("payload must be a JSON object")

        self.accepted += 1
        return payload

    def _parse_header(self, header: str) -> tuple[int | None, list[str]]:
        ts: int | None = None
        sigs: list[str] = []
        if "=" not in header:
            # A bare signature with no timestamp cannot be checked for freshness, so it
            # is not accepted. Replay protection is not optional.
            return None, []
        for part in header.split(","):
            key, _, value = part.strip().partition("=")
            if key == "t":
                try:
                    ts = int(value)
                except ValueError:
                    return None, []
            elif key.startswith("v"):
                sigs.append(value.strip())
        return ts, sigs

    def _is_replay(self, signature: str, now: float) -> bool:
        cutoff = now - self.tolerance_seconds
        while self._seen and next(iter(self._seen.values())) < cutoff:
            self._seen.popitem(last=False)
        if signature in self._seen:
            return True
        self._seen[signature] = now
        while len(self._seen) > self.replay_capacity:
            self._seen.popitem(last=False)
        return False

    def stats(self) -> dict:
        return {"accepted": self.accepted, "rejected": self.rejected,
                "replays_blocked": self.replays_blocked,
                "replay_cache": len(self._seen),
                "tolerance_seconds": self.tolerance_seconds}


#: Provider event names mapped to ours. Unknown types are ignored rather than guessed --
#: a webhook we do not understand must not be interpreted as one we do.
EVENT_MAP: dict[str, str] = {
    "payment_intent.succeeded": "payment.recovered",
    "payment_intent.payment_failed": "payment.failed",
    "charge.succeeded": "payment.recovered",
    "charge.failed": "payment.failed",
    "payment.captured": "payment.recovered",
    "payment.failed": "payment.failed",
    "invoice.payment_succeeded": "payment.recovered",
    "invoice.payment_failed": "payment.failed",
}


def map_event(provider_type: str) -> str | None:
    return EVENT_MAP.get(provider_type)
