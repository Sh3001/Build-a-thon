"""The payment-rail interface.

The executor previously imported `simulation.payment_gateway.PaymentGateway` directly. It
worked, and it meant the business layer had a hard dependency on the simulator: adding a
real provider would have required editing the executor, which is the file with the
strictest safety invariants and the one that should change least.

So the rail is a Protocol. Three implementations are anticipated and one exists:

    MockGateway            the seeded simulator -- the default, and the only one needed
                           to run this project end to end with no account anywhere
    StripeSandboxGateway   optional, opt-in, sandbox only
    RazorpaySandboxGateway optional, opt-in, sandbox only

Two rules that the interface enforces rather than documents:

1. **Every call carries an idempotency key.** Not optional, not defaulted -- a rail
   method that can be invoked without one is a rail method that will eventually charge
   someone twice. `GatewayRequest` has no default for it.

2. **A failure to reach the provider is not a decline.** `GatewayUnavailable` is a
   distinct exception from a `GatewayResult(success=False)`, because the correct response
   differs completely: a decline means stop trying this instrument, an outage means try
   again later. Collapsing them is how a five-minute processor blip burns a customer's
   entire retry budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable


class GatewayError(RuntimeError):
    """Base class for rail failures."""


class GatewayUnavailable(GatewayError):
    """The provider could not be reached, or returned a server error.

    Distinct from a decline on purpose. The state of the payment is *unknown*, so the
    only safe reading is "do not assume it did not happen" -- which is why the caller
    must re-present under the same idempotency key rather than as a new attempt.
    """


class GatewayRejected(GatewayError):
    """The provider refused the request as malformed or unauthorised. Never retryable."""


@dataclass(frozen=True)
class GatewayRequest:
    """One call to a rail. `idempotency_key` has no default, deliberately."""
    idempotency_key: str
    transaction_id: str
    customer_id: str
    amount_minor: int
    currency: str
    payment_method: str = "card"
    #: Hours since the original failure. Timing changes success odds on every real rail,
    #: not only in the simulator, so it is part of the interface.
    hours_since_failure: float = 0.0
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            raise ValueError(
                "idempotency_key is required: a rail call without one can charge twice")


@dataclass(frozen=True)
class GatewayResult:
    """What the rail said. `raw_code` is the provider's own vocabulary, unnormalised --
    normalisation is `adapters.processor_codes`' job, and keeping the raw string means a
    mapping gap is visible in the audit log rather than silently coerced."""
    success: bool
    detail: str = ""
    amount_minor: int = 0
    currency: str = "USD"
    raw_code: str | None = None
    provider: str = "mock"
    #: Provider-side reference, for reconciliation.
    reference: str | None = None
    probability: float | None = None
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class PaymentMethodState:
    """What the rail knows about the stored instrument right now."""
    exists: bool
    usable: bool
    kind: str = "card"
    expired: bool = False
    updated_since_failure: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PaymentRail(Protocol):
    """What the executor may ask of a payment provider. Nothing more.

    Note what is absent: no refund, no capture, no customer mutation, no arbitrary
    request. The interface is the second boundary after the policy engine -- an adapter
    cannot expose a capability the business layer has no way to call.
    """

    name: str

    def retry_payment(self, request: GatewayRequest) -> GatewayResult:
        """Re-present the existing instrument under `request.idempotency_key`."""
        ...

    def get_payment_status(self, transaction_id: str) -> GatewayResult:
        """Read-only. Takes no policy approval because it changes nothing."""
        ...

    def get_payment_method(self, customer_id: str,
                           transaction_id: str) -> PaymentMethodState:
        """Read-only view of the stored instrument."""
        ...


# ---------------------------------------------------------------- registry
_RAILS: dict[str, Any] = {}


def register(name: str, factory) -> None:
    _RAILS[name] = factory


def available() -> list[str]:
    return sorted(_RAILS)


def build_rail(name: str = "mock", **kw) -> PaymentRail:
    """Construct a rail by name.

    Unknown names raise rather than falling back to the mock. A deployment that believes
    it is talking to Stripe and is in fact talking to a simulator is the worst possible
    failure of this abstraction, and a silent default is exactly how it would happen.
    """
    if name not in _RAILS:
        raise KeyError(
            f"unknown payment rail {name!r}; available: {available()}. "
            f"There is deliberately no fallback -- a run that thinks it is live and is "
            f"not would report simulated recoveries as real ones.")
    rail = _RAILS[name](**kw)
    if not isinstance(rail, PaymentRail):
        raise TypeError(f"{name} does not satisfy the PaymentRail protocol")
    return rail
