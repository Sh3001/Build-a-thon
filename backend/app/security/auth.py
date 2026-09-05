"""Who is calling, and what they may do.

The API had no authentication at all: every endpoint, including the ones that execute
recovery actions against the rails, was open to anyone who could reach the port. That is
defensible for a local demo and indefensible the moment the process is reachable from
anywhere else, so the fix has to be one that cannot be forgotten -- which means the
*default* has to be secure and the demo has to be the special case, not the other way
round.

Hence `AuthPolicy.from_env()`:

* in the `production` profile, authentication is mandatory and the process refuses to
  start without a signing secret;
* in `development`/`simulation`, an anonymous principal is granted so `docker compose up`
  still shows a dashboard with no setup;
* and the health endpoint reports which of those is in force, so "we thought auth was on"
  is a checkable statement rather than a belief.

Roles are checked here, in the backend, on every protected route. The frontend hiding a
button is a courtesy to the operator and never a control.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Iterable
from dataclasses import dataclass, field

from backend.app.models.enums import Role
from backend.app.security.tokens import ApiKey, TokenError, decode

#: What each role may do. Read as a capability list, not a hierarchy -- ADMIN is not
#: "OPERATOR plus more", because the person who configures the safety envelope should not
#: also be the person who approves individual actions under it. Separating them is what
#: makes the audit log meaningful when both appear in it.
CAPABILITIES: dict[Role, frozenset[str]] = {
    Role.ADMIN: frozenset({
        "read:queue", "read:metrics", "read:audit", "read:review", "write:policy",
        "write:review", "write:dlq", "write:keys", "read:models", "write:models",
    }),
    Role.OPERATOR: frozenset({
        "read:queue", "read:metrics", "read:review", "write:review", "write:dlq",
        "read:models",
    }),
    Role.ANALYST: frozenset({
        "read:queue", "read:metrics", "read:experiments", "read:models",
    }),
    Role.AUDITOR: frozenset({
        "read:audit", "read:review", "read:metrics", "read:overrides",
    }),
    Role.SYSTEM: frozenset({
        "execute:action", "read:queue", "write:events",
    }),
}


class AuthError(PermissionError):
    """Authentication failed. Mapped to 401 at the API boundary."""


class ForbiddenError(PermissionError):
    """Authenticated, but not permitted. Mapped to 403."""


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Every store the request touches is scoped to
    `tenant_id`, and the value comes from the credential -- never from a query parameter,
    a header, or a request body. That is the whole of tenant isolation: if the tenant is
    not caller-supplied, a caller cannot supply another tenant's."""
    subject: str
    tenant_id: str
    roles: frozenset[Role]
    method: str = "anonymous"
    key_id: str | None = None

    def has(self, capability: str) -> bool:
        return any(capability in CAPABILITIES.get(r, frozenset()) for r in self.roles)

    def require(self, capability: str) -> None:
        if not self.has(capability):
            raise ForbiddenError(
                f"{self.subject} has roles {sorted(r.value for r in self.roles)}, none of "
                f"which grants {capability!r}")

    @property
    def is_anonymous(self) -> bool:
        return self.method == "anonymous"

    def describe(self) -> dict:
        return {"subject": self.subject, "tenant_id": self.tenant_id,
                "roles": sorted(r.value for r in self.roles), "method": self.method}


#: The principal used when authentication is disabled. Read-only on purpose: even in a
#: wide-open development profile, an unauthenticated caller must not be able to move
#: money. Executing an action needs a real credential in every profile.
ANONYMOUS = Principal(
    subject="anonymous", tenant_id="default",
    roles=frozenset({Role.ANALYST, Role.AUDITOR}), method="anonymous")


@dataclass
class AuthPolicy:
    """Configuration and verification. One object so the API has one thing to consult."""
    required: bool = True
    secret: str = ""
    #: Salt for API-key hashing. Separate from the JWT secret so rotating one does not
    #: invalidate the other.
    key_salt: str = ""
    keys: dict[str, ApiKey] = field(default_factory=dict)
    profile: str = "production"

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AuthPolicy:
        e = env if env is not None else dict(os.environ)
        profile = e.get("RECOVERAI_PROFILE", "development").lower()
        # Secure by default where it matters: production and staging require auth unless
        # someone explicitly turns it off, and turning it off is a thing they have to
        # type. Development is open unless someone turns it on.
        configured = e.get("RECOVERAI_AUTH_REQUIRED")
        required: bool = (profile in ("production", "staging")) if configured is None \
            else configured.lower() in ("1", "true", "yes", "on")

        secret = e.get("RECOVERAI_JWT_SECRET", "")
        if required and len(secret) < 32:
            raise RuntimeError(
                f"profile {profile!r} requires authentication but RECOVERAI_JWT_SECRET is "
                f"unset or shorter than 32 characters. Refusing to start: a process that "
                f"silently downgrades to open access is worse than one that will not boot.")

        policy = cls(required=required, secret=secret,
                     key_salt=e.get("RECOVERAI_KEY_SALT", "recoverai-default-salt"),
                     profile=profile)
        policy._load_keys_from_env(e)
        return policy

    def _load_keys_from_env(self, e: dict[str, str]) -> None:
        """`RECOVERAI_API_KEYS = keyid:secret:tenant:role1|role2, ...`

        Environment configuration for a small deployment. A real one would read a table;
        the shape is the same, and the point of doing it here is that the plaintext never
        touches the database -- only the PBKDF2 hash is retained.
        """
        raw = e.get("RECOVERAI_API_KEYS", "").strip()
        if not raw:
            return
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) != 4:
                raise ValueError(
                    f"malformed RECOVERAI_API_KEYS entry {entry!r}; expected "
                    f"keyid:secret:tenant:role1|role2")
            key_id, secret, tenant, roles = parts
            self.add_key(key_id, secret, tenant,
                         [Role(r.strip()) for r in roles.split("|") if r.strip()])

    def add_key(self, key_id: str, secret: str, tenant_id: str,
                roles: Iterable[Role], label: str = "") -> ApiKey:
        key = ApiKey(key_id=key_id,
                     secret_hash=ApiKey.hash_secret(secret, self.key_salt),
                     tenant_id=tenant_id,
                     roles=tuple(r.value for r in roles), label=label)
        self.keys[key_id] = key
        return key

    @staticmethod
    def mint_secret() -> str:
        return secrets.token_urlsafe(32)

    # ------------------------------------------------------------------ verification
    def authenticate(self, authorization: str | None = None,
                     api_key: str | None = None) -> Principal:
        """Resolve a credential to a principal, or raise.

        Order is bearer-then-key, and presenting both is refused rather than resolved:
        two credentials on one request is either a client bug or an attempt to find out
        which one the server prefers, and neither deserves a best guess.
        """
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()

        if bearer and api_key:
            raise AuthError("present exactly one credential, not both")

        if bearer:
            return self._from_jwt(bearer)
        if api_key:
            return self._from_api_key(api_key)

        if self.required:
            raise AuthError("authentication required")
        return ANONYMOUS

    def _from_jwt(self, token: str) -> Principal:
        if not self.secret:
            raise AuthError("bearer tokens are not accepted: no signing secret configured")
        try:
            claims = decode(token, self.secret)
        except TokenError as exc:
            raise AuthError(str(exc)) from exc
        roles = self._roles(claims.get("roles", []))
        tenant = str(claims.get("tenant") or "").strip()
        if not tenant:
            raise AuthError("token carries no tenant claim")
        return Principal(subject=str(claims.get("sub", "unknown")), tenant_id=tenant,
                         roles=roles, method="jwt")

    def _from_api_key(self, presented: str) -> Principal:
        key_id, _, secret = presented.partition(".")
        record = self.keys.get(key_id)
        if record is None or not secret:
            # Do the hash anyway. Returning early on an unknown key id turns the endpoint
            # into an oracle for which key ids exist, measurable over the wire.
            ApiKey.hash_secret(secret or "x", self.key_salt)
            raise AuthError("invalid API key")
        if not record.verify(secret, self.key_salt):
            raise AuthError("invalid API key")
        return Principal(subject=f"key:{key_id}", tenant_id=record.tenant_id,
                         roles=self._roles(record.roles), method="api_key", key_id=key_id)

    @staticmethod
    def _roles(raw: Iterable[str]) -> frozenset[Role]:
        out = set()
        for r in raw:
            try:
                out.add(Role(str(r)))
            except ValueError:
                # An unrecognised role grants nothing rather than everything. Silently
                # ignoring it is right; treating it as a wildcard would be catastrophic.
                continue
        if not out:
            raise AuthError("credential carries no recognised role")
        return frozenset(out)

    def describe(self) -> dict:
        return {
            "profile": self.profile,
            "auth_required": self.required,
            "methods": ["jwt", "api_key"] if self.secret or self.keys else ["api_key"],
            "api_keys_configured": len(self.keys),
            "anonymous_roles": sorted(r.value for r in ANONYMOUS.roles)
                               if not self.required else [],
        }


def fingerprint(secret: str) -> str:
    """A stable, non-reversible label for a secret, for logs and health output. Never log
    the secret; logging whether it *changed* is legitimate and sometimes necessary."""
    return hashlib.sha256(secret.encode()).hexdigest()[:8] if secret else "unset"
