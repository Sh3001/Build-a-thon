"""Request-scoped dependencies: who is calling, what they may do, and which tenant's
data they see.

The last of those is the one worth reading carefully. `tenant_store()` builds every store
from `principal.tenant_id`, and the principal comes from a verified credential. No route
accepts a tenant parameter, so there is no code path by which a caller can name someone
else's tenant -- which makes isolation a property of the wiring rather than a rule each
handler has to remember.
"""
from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Header, HTTPException, Request

from backend.app.config import DB_PATH
from backend.app.database.review import ConsentStore, ReviewStore
from backend.app.database.store import CaseStore
from backend.app.observability.logging import new_trace_id
from backend.app.observability.metrics import M, get_registry
from backend.app.security.auth import (
    AuthError,
    AuthPolicy,
    ForbiddenError,
    Principal,
)
from backend.app.security.ratelimit import RateLimiter

#: One policy and one limiter per process, installed at startup so tests can swap them.
_policy: AuthPolicy | None = None
_limiter = RateLimiter()


def set_auth_policy(policy: AuthPolicy) -> None:
    global _policy
    _policy = policy


def get_auth_policy() -> AuthPolicy:
    global _policy
    if _policy is None:
        _policy = AuthPolicy.from_env()
    return _policy


def get_rate_limiter() -> RateLimiter:
    return _limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    global _limiter
    _limiter = limiter


def principal(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    """Authenticate, then rate-limit.

    In that order, deliberately. Limiting before authenticating means an attacker's
    unauthenticated flood consumes the bucket a legitimate key would have used; limiting
    after means each caller's budget is their own. The trade-off is that authentication
    work happens for unauthenticated requests, which is why the auth path is cheap and
    the body-size limit is enforced by middleware before either.
    """
    registry = get_registry()
    new_trace_id()
    policy = get_auth_policy()
    try:
        who = policy.authenticate(authorization=authorization, api_key=x_api_key)
    except AuthError as exc:
        registry.counter(M.AUTH_FAILURES, "failed authentications",
                         {"reason": type(exc).__name__})
        # No WWW-Authenticate challenge detail beyond the scheme: enumerating why a
        # credential failed helps an attacker more than it helps a client.
        raise HTTPException(401, "authentication failed",
                            headers={"WWW-Authenticate": "Bearer"}) from exc

    key = who.key_id or who.subject
    if who.is_anonymous:
        key = f"ip:{request.client.host if request.client else 'unknown'}"
    allowed, retry_after = get_rate_limiter().check(key)
    if not allowed:
        registry.counter(M.RATE_LIMITED, "rate-limited requests", {"principal": key[:32]})
        # A non-refilling bucket has no finite retry-after; advertise a minute rather
        # than an infinity a client cannot parse.
        seconds = 60 if retry_after == float("inf") else int(retry_after) + 1
        raise HTTPException(429, "rate limit exceeded",
                            headers={"Retry-After": str(seconds)})

    request.state.principal = who
    return who


def require(capability: str) -> Callable[[Principal], Principal]:
    """Dependency factory: `Depends(require("write:review"))`.

    A capability string rather than a role, so a route says what it needs rather than
    which job titles happen to have it today. Adding a role then means editing one table
    instead of auditing every route.
    """
    def dep(who: Principal = Depends(principal)) -> Principal:
        try:
            who.require(capability)
        except ForbiddenError as exc:
            raise HTTPException(403, str(exc)) from exc
        return who
    return dep


# ---------------------------------------------------------------- tenant-scoped stores
def tenant_cases(who: Principal) -> CaseStore:
    return CaseStore(DB_PATH, tenant_id=who.tenant_id)


def tenant_reviews(who: Principal, conn=None) -> ReviewStore:
    return ReviewStore(DB_PATH, conn=conn, tenant_id=who.tenant_id)


def tenant_consent(who: Principal, conn=None) -> ConsentStore:
    return ConsentStore(DB_PATH, conn=conn, tenant_id=who.tenant_id)
