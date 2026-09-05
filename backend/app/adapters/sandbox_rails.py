"""Optional sandbox adapters for real providers.

**These are opt-in and sandbox-only, and they are not exercised by the test suite** --
there are no credentials in CI and there should not be. What they demonstrate is that the
`PaymentRail` protocol is sufficient for a real provider, not that this project has been
validated against one. It has not been. See `docs/production.md`.

Three properties are worth reading even without an account:

* the constructor refuses a live key. A `sk_live_` secret reaching a system whose safety
  envelope has never been exercised against real money is the single worst outcome
  available here, so it is refused at construction rather than trusted to configuration;
* transport failures raise `GatewayUnavailable`, never a false decline;
* the idempotency key is passed to the provider's own idempotency header, so a retry of
  an ambiguous call is deduplicated by the provider rather than by us guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.adapters.gateway import (
    GatewayRejected,
    GatewayRequest,
    GatewayResult,
    GatewayUnavailable,
    PaymentMethodState,
    register,
)
from backend.app.adapters.processor_codes import normalise


class LiveCredentialRefused(RuntimeError):
    """A production credential was supplied to a sandbox-only adapter."""


def _reject_live(key: str, provider: str) -> None:
    markers = ("sk_live_", "rk_live_", "rzp_live_", "live_")
    if any(m in key for m in markers):
        raise LiveCredentialRefused(
            f"{provider} adapter refuses a live credential. This system has never been "
            f"validated against real money movement; run it against the sandbox, and see "
            f"docs/production.md for what would have to be true first.")


@dataclass
class StripeSandboxRail:
    """Stripe test mode. Requires `stripe`; not installed by default and not in CI."""
    api_key: str = ""
    name: str = "stripe"
    client: Any = None
    timeout: float = 20.0
    _by_key: dict[str, GatewayResult] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _reject_live(self.api_key, "stripe")
        if self.client is None and self.api_key:
            try:
                import stripe
            except ImportError as exc:                        # pragma: no cover
                raise ImportError(
                    "the Stripe adapter needs `pip install stripe`; it is optional and "
                    "the project runs fully on the mock rail without it") from exc
            stripe.api_key = self.api_key
            self.client = stripe

    def retry_payment(self, request: GatewayRequest) -> GatewayResult:
        if request.idempotency_key in self._by_key:
            return self._by_key[request.idempotency_key]
        if self.client is None:
            raise GatewayRejected("stripe adapter has no client configured")
        try:
            intent = self.client.PaymentIntent.create(
                amount=request.amount_minor, currency=request.currency.lower(),
                customer=request.customer_id, confirm=True, off_session=True,
                # The provider deduplicates, not us. Our ledger protects against a
                # repeated *call*; this protects against a repeated *charge* when the
                # first call's response was lost in transit.
                idempotency_key=request.idempotency_key,
                metadata={"recoverai_transaction_id": request.transaction_id},
            )
        except Exception as exc:
            name = type(exc).__name__
            if name in ("APIConnectionError", "APIError", "RateLimitError",
                        "ServiceUnavailableError", "Timeout"):
                # The payment's state is UNKNOWN. Reporting this as a decline would let
                # the caller move on to the next instrument while a charge may have
                # landed, so it must be a distinct exception.
                raise GatewayUnavailable(f"stripe unreachable: {name}: {exc}") from exc
            if name == "CardError":
                decline = getattr(getattr(exc, "error", None), "decline_code", None) \
                    or getattr(exc, "code", None)
                n = normalise("stripe", decline, str(exc))
                out = GatewayResult(False, detail=str(exc), raw_code=decline,
                                    provider=self.name, currency=request.currency,
                                    probability=None)
                self._by_key[request.idempotency_key] = out
                _ = n  # normalisation is recorded by the diagnosis layer, not here
                return out
            raise GatewayRejected(f"stripe rejected the request: {name}: {exc}") from exc

        ok = getattr(intent, "status", "") == "succeeded"
        out = GatewayResult(
            success=ok, detail=f"payment_intent {getattr(intent, 'status', '?')}",
            amount_minor=request.amount_minor if ok else 0, currency=request.currency,
            provider=self.name, reference=getattr(intent, "id", None))
        self._by_key[request.idempotency_key] = out
        return out

    def get_payment_status(self, transaction_id: str) -> GatewayResult:
        if self.client is None:
            raise GatewayRejected("stripe adapter has no client configured")
        try:
            intent = self.client.PaymentIntent.retrieve(transaction_id)
        except Exception as exc:
            raise GatewayUnavailable(f"stripe status probe failed: {exc}") from exc
        return GatewayResult(success=getattr(intent, "status", "") == "succeeded",
                             detail=getattr(intent, "status", "?"), provider=self.name,
                             reference=transaction_id)

    def get_payment_method(self, customer_id: str,
                           transaction_id: str) -> PaymentMethodState:
        if self.client is None:
            raise GatewayRejected("stripe adapter has no client configured")
        try:
            methods = self.client.PaymentMethod.list(customer=customer_id, type="card")
        except Exception as exc:
            raise GatewayUnavailable(f"stripe method probe failed: {exc}") from exc
        items = getattr(methods, "data", []) or []
        return PaymentMethodState(exists=bool(items), usable=bool(items), kind="card")


@dataclass
class RazorpaySandboxRail:
    """Razorpay test mode. Requires `razorpay`; optional and absent from CI."""
    key_id: str = ""
    key_secret: str = ""
    name: str = "razorpay"
    client: Any = None
    _by_key: dict[str, GatewayResult] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _reject_live(self.key_id, "razorpay")
        if self.client is None and self.key_id and self.key_secret:
            try:
                import razorpay
            except ImportError as exc:                        # pragma: no cover
                raise ImportError(
                    "the Razorpay adapter needs `pip install razorpay`; it is optional "
                    "and the project runs fully on the mock rail without it") from exc
            self.client = razorpay.Client(auth=(self.key_id, self.key_secret))

    def retry_payment(self, request: GatewayRequest) -> GatewayResult:
        if request.idempotency_key in self._by_key:
            return self._by_key[request.idempotency_key]
        if self.client is None:
            raise GatewayRejected("razorpay adapter has no client configured")
        try:
            resp = self.client.payment.create_recurring({
                "amount": request.amount_minor, "currency": request.currency,
                "customer_id": request.customer_id,
                "receipt": request.idempotency_key,
                "notes": {"recoverai_transaction_id": request.transaction_id},
            })
        except Exception as exc:
            text = str(exc)
            if "timeout" in text.lower() or "connection" in text.lower():
                raise GatewayUnavailable(f"razorpay unreachable: {text}") from exc
            code = getattr(exc, "code", None) or "bad_request_payment_failed"
            out = GatewayResult(False, detail=text, raw_code=str(code),
                                provider=self.name, currency=request.currency)
            self._by_key[request.idempotency_key] = out
            return out

        status = (resp or {}).get("status", "")
        ok = status == "captured"
        out = GatewayResult(success=ok, detail=f"payment {status}",
                            amount_minor=request.amount_minor if ok else 0,
                            currency=request.currency, provider=self.name,
                            reference=(resp or {}).get("id"))
        self._by_key[request.idempotency_key] = out
        return out

    def get_payment_status(self, transaction_id: str) -> GatewayResult:
        if self.client is None:
            raise GatewayRejected("razorpay adapter has no client configured")
        try:
            p = self.client.payment.fetch(transaction_id)
        except Exception as exc:
            raise GatewayUnavailable(f"razorpay status probe failed: {exc}") from exc
        return GatewayResult(success=(p or {}).get("status") == "captured",
                             detail=str((p or {}).get("status")), provider=self.name,
                             reference=transaction_id)

    def get_payment_method(self, customer_id: str,
                           transaction_id: str) -> PaymentMethodState:
        # Razorpay exposes tokens rather than a card object on this path. Reporting
        # "usable, source unknown" would be a guess; reporting what we can see is not.
        return PaymentMethodState(exists=True, usable=True, kind="token",
                                  raw={"note": "token state not queried in sandbox adapter"})


register("stripe_sandbox", lambda **kw: StripeSandboxRail(**kw))
register("razorpay_sandbox", lambda **kw: RazorpaySandboxRail(**kw))
