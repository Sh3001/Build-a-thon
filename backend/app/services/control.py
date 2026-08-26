"""The no-touch control arm.

Answers the only question that turns "money recovered" into "money we caused": how much
would have arrived with no intervention at all? Without it, an agent that works cases the
customer was going to pay anyway looks indistinguishable from one that recovers money.

The control takes zero actions, spends nothing, and contacts nobody. Its recoveries come
purely from `PaymentGateway.self_cure_hour`, which is drawn from the same per-transaction
RNG stream every other arm sees -- so the arms are a true counterfactual over one
population, not three different populations.
"""
from __future__ import annotations

from backend.app.config import RECOVERY_HORIZON_DAYS
from backend.app.models.enums import FailureCode, category_of
from backend.app.services.results import CaseOutcome, summarize
from simulation.payment_gateway import PaymentGateway


def run_case(txn: dict, gateway: PaymentGateway,
             horizon_days: int = RECOVERY_HORIZON_DAYS) -> CaseOutcome:
    code = FailureCode(txn["failure_code"])
    out = CaseOutcome(
        transaction_id=txn["transaction_id"],
        customer_id=txn.get("customer_id", ""),
        amount_usd=float(txn["amount_usd"]),
        failure_code=code,
        failure_category=category_of(code),
        strategy="control",
        status="stopped",
        stop_reason="never touched (control arm)",
    )
    cure = gateway.self_cure_hour(txn, horizon_hours=horizon_days * 24)
    if cure is not None:
        out.recovered = True
        out.passive_recovery = True
        out.amount_recovered = out.amount_usd
        out.recovery_hours = cure
        out.status = "recovered"
        out.stop_reason = "self-cured with no intervention"
    return out


def run_control(transactions: list[dict], gateway: PaymentGateway | None = None,
                horizon_days: int = RECOVERY_HORIZON_DAYS) -> tuple[list[CaseOutcome], dict]:
    gw = gateway or PaymentGateway()
    outcomes = [run_case(t, gw, horizon_days) for t in transactions]
    return outcomes, summarize(outcomes, "control")
