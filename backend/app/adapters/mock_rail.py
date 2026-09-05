"""The simulator, presented as a `PaymentRail`.

A thin adapter, and thin is the point: the simulator keeps its own richer interface
(latents, self-cure, contact fatigue) which the evaluation needs, while the executor sees
only the four methods every rail has. Nothing about the simulator's extra surface leaks
through the protocol, so code written against `PaymentRail` cannot accidentally depend on
being run against a simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.adapters.gateway import (
    GatewayRequest,
    GatewayResult,
    PaymentMethodState,
    register,
)
from simulation.payment_gateway import PaymentGateway


@dataclass
class MockRail:
    """Wraps a seeded `PaymentGateway`. Deterministic, offline, free."""
    gateway: PaymentGateway = field(default_factory=PaymentGateway)
    name: str = "mock"
    #: Idempotency at the rail boundary, mirroring what a real provider gives you for an
    #: `Idempotency-Key` header. The executor has its own ledger; this is defence in
    #: depth, and it is what makes the *adapter* testable for double-charging on its own.
    _by_key: dict[str, GatewayResult] = field(default_factory=dict, repr=False)

    def retry_payment(self, request: GatewayRequest) -> GatewayResult:
        if request.idempotency_key in self._by_key:
            return self._by_key[request.idempotency_key]

        txn = {
            "transaction_id": request.transaction_id,
            "customer_id": request.customer_id,
            "failure_code": request.metadata.get("failure_code", "temporary_decline"),
            "amount_usd": request.amount_minor / 100.0,
            "failure_count": request.metadata.get("failure_count", 1),
            "days_since_failure": request.hours_since_failure / 24.0,
            "previous_success_rate": request.metadata.get("previous_success_rate", 0.5),
        }
        res = self.gateway.retry_payment(txn, request.hours_since_failure, request.attempt)
        out = GatewayResult(
            success=res.success, detail=res.detail,
            amount_minor=int(round(res.amount * 100)) if res.success else 0,
            currency=request.currency,
            raw_code=None if res.success else txn["failure_code"],
            provider=self.name, probability=res.probability,
            reference=f"mock_{request.idempotency_key[:12]}",
        )
        self._by_key[request.idempotency_key] = out
        return out

    def get_payment_status(self, transaction_id: str) -> GatewayResult:
        fixed = self.gateway.instrument_fixed(transaction_id)
        return GatewayResult(
            success=False, provider=self.name,
            detail=f"instrument_fixed={fixed}; contacts={self.gateway.contacts(transaction_id)}",
            reference=transaction_id)

    def get_payment_method(self, customer_id: str,
                           transaction_id: str) -> PaymentMethodState:
        fixed = self.gateway.instrument_fixed(transaction_id)
        return PaymentMethodState(
            exists=True, usable=fixed, updated_since_failure=fixed,
            raw={"source": "simulation", "transaction_id": transaction_id})


register("mock", lambda **kw: MockRail(**kw))
