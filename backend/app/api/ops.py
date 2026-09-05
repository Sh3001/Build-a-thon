"""Operational endpoints: metrics, webhooks, model registry, policy introspection.

The webhook route is the one that deserves scrutiny. It is the only unauthenticated write
path in the system, because payment providers cannot present an API key -- so its entire
security rests on signature verification, which is why the body is read as raw bytes and
verified before anything parses it.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.app.api.deps import require
from backend.app.domain.events import Event, EventType, InProcessEventBus
from backend.app.ml.registry import ModelRegistry, ModelStatus, RegistryError
from backend.app.observability.metrics import M, get_registry
from backend.app.policies.version import describe as describe_policy
from backend.app.security.auth import Principal
from backend.app.security.webhooks import WebhookError, WebhookVerifier, map_event

router = APIRouter(prefix="/api/v1", tags=["ops"])

#: One bus per process. Handlers are attached at startup in `main`.
_bus = InProcessEventBus()
_verifier: WebhookVerifier | None = None


def get_bus() -> InProcessEventBus:
    return _bus


def get_verifier() -> WebhookVerifier:
    global _verifier
    if _verifier is None:
        _verifier = WebhookVerifier(secret=os.environ.get("RECOVERAI_WEBHOOK_SECRET", ""))
    return _verifier


def set_verifier(v: WebhookVerifier) -> None:
    global _verifier
    _verifier = v


# ---------------------------------------------------------------- metrics
@router.get("/metrics/prometheus", response_class=Response)
def prometheus() -> Response:
    """Prometheus text exposition.

    Unauthenticated on purpose, and safe because of what it contains: counters, latencies
    and rates. No customer identifiers, no amounts per case, no tenant-specific series
    beyond a tenant label an operator already knows. A scrape endpoint behind auth is a
    scrape endpoint nobody configures.
    """
    return Response(content=get_registry().render(),
                    media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/metrics/snapshot")
def metrics_snapshot(who: Principal = Depends(require("read:metrics"))) -> dict:
    """The same series as JSON, for the dashboard, which does not speak Prometheus."""
    return get_registry().snapshot()


# ---------------------------------------------------------------- policy
@router.get("/policy")
def policy(who: Principal = Depends(require("read:metrics"))) -> dict:
    """The safety envelope as it currently stands: every rule ID by tier, every
    configured limit, and the derived version hash that audit rows are stamped with.

    Readable rather than only inspectable in source, because "what were the limits when
    this decision was taken" is the first question anyone asks about a stored verdict.
    """
    return describe_policy()


# ---------------------------------------------------------------- models
class PromoteRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
    force: bool = False


@router.get("/models")
def models(who: Principal = Depends(require("read:models"))) -> dict:
    return ModelRegistry().describe()


@router.post("/models/{version}/promote")
def promote(version: str, body: PromoteRequest,
            who: Principal = Depends(require("write:models"))) -> dict:
    """Promote a model to PRODUCTION. ADMIN only, gated on quality, and a forced
    promotion records what it overrode."""
    try:
        record = ModelRegistry().promote(version, actor=who.subject, force=body.force,
                                         reason=body.reason)
    except RegistryError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"version": record.version, "status": record.status,
            "promoted_by": record.promoted_by, "promoted_at": record.promoted_at,
            "notes": record.notes}


@router.post("/models/{version}/retire")
def retire(version: str, body: PromoteRequest,
           who: Principal = Depends(require("write:models"))) -> dict:
    try:
        record = ModelRegistry().transition(version, ModelStatus.RETIRED,
                                            actor=who.subject, reason=body.reason)
    except RegistryError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"version": record.version, "status": record.status,
            "retired_at": record.retired_at}


# ---------------------------------------------------------------- webhooks
@router.post("/webhooks/{provider}")
async def webhook(provider: str, request: Request,
                  x_signature: str | None = Header(default=None, alias="X-Signature"),
                  ) -> dict:
    """Inbound payment events.

    Raw bytes in, verified, then parsed -- in that order. Verifying the *parsed* document
    would let an attacker who controls whitespace present one document to the verifier and
    a different one to the handler, because `{"a":1}` and `{ "a" : 1 }` parse identically
    and hash differently.

    A rejected webhook returns 401 with no detail. Telling a caller which check failed
    tells an attacker which half of the forgery to keep working on.
    """
    registry = get_registry()
    body = await request.body()
    try:
        payload = get_verifier().verify(body, x_signature)
    except WebhookError as exc:
        registry.counter(M.WEBHOOK_REJECTED, "rejected webhooks", {"provider": provider})
        raise HTTPException(401, "webhook verification failed") from exc

    provider_type = str(payload.get("type", ""))
    mapped = map_event(provider_type)
    if mapped is None:
        # An event we do not understand is acknowledged and ignored, not guessed at.
        # Returning an error would make the provider retry something we will never
        # handle; interpreting it as a type we do know would be worse.
        return {"received": True, "handled": False,
                "reason": f"no mapping for provider event {provider_type!r}"}

    data = payload.get("data") or {}
    obj = data.get("object") if isinstance(data, dict) else {}
    obj = obj if isinstance(obj, dict) else {}

    event = Event(
        type=EventType(mapped),
        transaction_id=str(obj.get("id") or payload.get("id") or ""),
        customer_id=str(obj.get("customer") or ""),
        payload={"provider": provider, "provider_event": provider_type,
                 "amount_minor": obj.get("amount"), "currency": obj.get("currency")},
        # Provider event ids are the natural idempotency key: a provider that retries a
        # delivery sends the same id, and the bus drops the duplicate.
        dedupe_key=f"{provider}:{payload.get('id') or ''}",
    )
    published = get_bus().publish(event)
    return {"received": True, "handled": True, "event": mapped,
            "duplicate": not published}


# ---------------------------------------------------------------- dead letters
@router.get("/dlq/events")
def dead_letters(who: Principal = Depends(require("read:metrics"))) -> dict:
    """Events whose handler failed. Distinct from the delivery DLQ at `/api/dlq`, which
    tracks bounced customer addresses -- two different failures, two different queues, and
    conflating them would put an unreachable email next to a crashed handler."""
    bus = get_bus()
    return {
        "stats": bus.stats(),
        "entries": [{
            "event_id": dl.event.event_id, "type": dl.event.type.value,
            "transaction_id": dl.event.transaction_id, "handler": dl.handler,
            "error": dl.error, "attempts": dl.attempts,
            "quarantined": dl.quarantined, "quarantine_reason": dl.quarantine_reason,
            "first_failed_at": dl.first_failed_at.isoformat(),
            "last_attempt_at": dl.last_attempt_at.isoformat(),
        } for dl in bus.pending_dead_letters()],
    }


@router.post("/dlq/events/retry")
def retry_dead_letters(who: Principal = Depends(require("write:dlq"))) -> dict:
    recovered, failing = get_bus().retry_dead_letters()
    return {"recovered": recovered, "still_failing": failing,
            "stats": get_bus().stats()}


@router.post("/dlq/events/{event_id}/resolve")
def resolve_dead_letter(event_id: str,
                        who: Principal = Depends(require("write:dlq"))) -> dict:
    if not get_bus().resolve(event_id):
        raise HTTPException(404, "no unresolved dead letter with that event id")
    return {"event_id": event_id, "resolved": True, "resolved_by": who.subject}
