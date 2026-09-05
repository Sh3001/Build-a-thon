"""Human-review endpoints.

The queue an operator works, and the two verbs that resolve a task. Three properties are
enforced here rather than trusted to the client:

* **Authorisation is per capability.** Reading the queue needs `read:review`; resolving a
  task needs `write:review`, which an ANALYST does not have. The frontend hides the
  buttons; this is what makes hiding them irrelevant.
* **A reason is mandatory.** An override with no justification is indistinguishable from
  a mistake when someone reads the log a year later, so the schema requires one and the
  store requires one again.
* **Approval does not execute.** It records an approval and returns the `PolicyResult`
  that would permit the action. Execution still goes through the executor, which still
  requires a permitting verdict -- so an approved action reaches the rails through the
  same single gate as everything else.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.api.deps import require, tenant_consent, tenant_reviews
from backend.app.database.review import NotAuthorised, ReviewClosed
from backend.app.models.enums import Role
from backend.app.observability.metrics import M, get_registry
from backend.app.security.auth import Principal

router = APIRouter(prefix="/api/v1/reviews", tags=["human review"])


class ResolveRequest(BaseModel):
    #: Not optional and not defaulted. A blank justification is refused by the store too;
    #: requiring it in two places is deliberate, because this is the field that makes an
    #: override auditable.
    reason: str = Field(min_length=3, max_length=1000)


class ReviewRow(BaseModel):
    review_id: str
    transaction_id: str
    customer_id: str | None = None
    status: str
    rule_id: str | None = None
    reason: str | None = None
    amount_usd: float | None = None
    expected_profit: float | None = None
    proposed_action: str | None = None
    model_version: str | None = None
    policy_version: str | None = None
    created_at: str
    expires_at: str | None = None
    resolved_at: str | None = None
    resolved_by: str | None = None


def _row(d: dict) -> ReviewRow:
    return ReviewRow(**{k: d.get(k) for k in ReviewRow.model_fields})


def _first_role(who: Principal) -> Role:
    """The role recorded against an override. Preferring OPERATOR when a principal holds
    both means the audit shows the capacity in which they acted, not an arbitrary pick
    from a set."""
    for candidate in (Role.OPERATOR, Role.ADMIN):
        if candidate in who.roles:
            return candidate
    return sorted(who.roles, key=lambda r: r.value)[0]


@router.get("", response_model=list[ReviewRow])
def pending(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
            who: Principal = Depends(require("read:review"))) -> list[ReviewRow]:
    """The queue, highest value first -- an operator's minute is the scarce resource."""
    store = tenant_reviews(who)
    try:
        store.expire_overdue()
        return [_row(r) for r in store.pending(limit=limit, offset=offset)]
    finally:
        store.close()


@router.get("/stats")
def stats(who: Principal = Depends(require("read:review"))) -> dict:
    store = tenant_reviews(who)
    try:
        out = store.stats()
        get_registry().gauge(M.HUMAN_REVIEW_PENDING, out["pending"],
                             "human-review tasks awaiting an operator",
                             {"tenant": who.tenant_id})
        return out
    finally:
        store.close()


@router.get("/{review_id}", response_model=ReviewRow)
def detail(review_id: str,
           who: Principal = Depends(require("read:review"))) -> ReviewRow:
    store = tenant_reviews(who)
    try:
        row = store.get(review_id)
        if row is None:
            # 404 for another tenant's task as well as for a missing one. Distinguishing
            # them would confirm the id exists, which is a cross-tenant information leak
            # even without returning the contents.
            raise HTTPException(404, "no such review task")
        return _row(row)
    finally:
        store.close()


@router.post("/{review_id}/approve")
def approve(review_id: str, body: ResolveRequest,
            who: Principal = Depends(require("write:review"))) -> dict:
    """Authorise the withheld action. Records the override; does not execute."""
    store = tenant_reviews(who)
    try:
        result = store.approve(review_id, who.subject, _first_role(who), body.reason)
    except KeyError as exc:
        raise HTTPException(404, "no such review task") from exc
    except ReviewClosed as exc:
        raise HTTPException(409, str(exc)) from exc
    except NotAuthorised as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        store.close()

    get_registry().counter(M.HUMAN_REVIEWS, "human review resolutions",
                           {"outcome": "approved", "tenant": who.tenant_id})
    return {
        "review_id": review_id, "decision": result.decision.value,
        "rules_fired": result.rules_fired, "reason": result.reason,
        "approved_action": result.effective_action.model_dump(mode="json")
                           if result.effective_action else None,
        "note": ("recorded as approved. The action still runs through the executor, "
                 "which requires this verdict -- approval authorises, it does not "
                 "execute."),
    }


@router.post("/{review_id}/reject")
def reject(review_id: str, body: ResolveRequest,
           who: Principal = Depends(require("write:review"))) -> dict:
    store = tenant_reviews(who)
    try:
        result = store.reject(review_id, who.subject, _first_role(who), body.reason)
    except KeyError as exc:
        raise HTTPException(404, "no such review task") from exc
    except ReviewClosed as exc:
        raise HTTPException(409, str(exc)) from exc
    except NotAuthorised as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        store.close()

    get_registry().counter(M.HUMAN_REVIEWS, labels={"outcome": "rejected",
                                                    "tenant": who.tenant_id})
    return {"review_id": review_id, "decision": result.decision.value,
            "reason": result.reason}


@router.get("/audit/overrides")
def overrides(limit: int = Query(200, ge=1, le=1000),
              transaction_id: str | None = None,
              who: Principal = Depends(require("read:overrides"))) -> dict:
    """Every human override, append-only. Readable by AUDITOR, which is the point of
    having an auditor role that cannot approve anything."""
    store = tenant_reviews(who)
    try:
        return {"overrides": store.overrides(limit=limit, transaction_id=transaction_id)}
    finally:
        store.close()


consent_router = APIRouter(prefix="/api/v1/consent", tags=["consent"])


class ConsentRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128)
    channel: str = Field(default="*", max_length=32)
    reason: str = Field(default="", max_length=500)


@consent_router.post("/opt-out")
def opt_out(body: ConsentRequest,
            who: Principal = Depends(require("write:review"))) -> dict:
    """Withdraw consent. Takes effect on the next policy evaluation via `R-OPT-OUT`.

    `channel="*"` is a blanket opt-out and beats any per-channel record: someone who said
    "stop contacting me" has not agreed to be reached on a channel they did not name.
    """
    store = tenant_consent(who)
    try:
        store.opt_out(body.customer_id, body.channel, body.reason, source=who.subject)
        return {"customer_id": body.customer_id, "channel": body.channel,
                "opted_out": True}
    finally:
        store.close()


@consent_router.post("/opt-in")
def opt_in(body: ConsentRequest,
           who: Principal = Depends(require("write:review"))) -> dict:
    store = tenant_consent(who)
    try:
        store.opt_in(body.customer_id, body.channel)
        return {"customer_id": body.customer_id, "channel": body.channel,
                "opted_out": store.is_opted_out(body.customer_id, body.channel)}
    finally:
        store.close()
