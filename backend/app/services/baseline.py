"""Phase 2 -- the baseline strategy, deliberately dumb.

    retry every eligible failed payment after 24 hours, stop after 3 retries

This is what an untuned dunning system does. It has no diagnosis, so it re-presents dead
cards, hammers closed accounts, and retries fraud holds. Those wasted and unsafe attempts
are counted (`policy_blocked` is always 0 here -- nothing validates it) so the comparison
against RecoverAI can separate *intelligence* from *safety*.

Interpretation note: "eligible" is read literally as every failed payment, which is the
naive behaviour being argued against. `risk_actions` records how many attempts it made on
RISK_COMPLIANCE cases; RecoverAI must hold that number at zero.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.app.config import (
    ACTION_COST_USD,
    MAX_RETRIES,
    MIN_RETRY_INTERVAL_HOURS,
    RECOVERY_HORIZON_DAYS,
)
from backend.app.models.enums import FailureCategory, FailureCode, category_of
from backend.app.services.results import CaseOutcome, summarize
from simulation.payment_gateway import PaymentGateway


@dataclass
class BaselineConfig:
    retry_interval_hours: float = MIN_RETRY_INTERVAL_HOURS
    max_retries: int = MAX_RETRIES
    horizon_days: int = RECOVERY_HORIZON_DAYS


def run_case(txn: dict, gateway: PaymentGateway, cfg: BaselineConfig) -> CaseOutcome:
    code = FailureCode(txn["failure_code"])
    cat = category_of(code)
    out = CaseOutcome(
        transaction_id=txn["transaction_id"],
        customer_id=txn.get("customer_id", ""),
        amount_usd=float(txn["amount_usd"]),
        failure_code=code,
        failure_category=cat,
        strategy="baseline",
        status="exhausted",
        stop_reason="retry limit reached",
    )

    horizon_hours = cfg.horizon_days * 24
    # The clock starts at the ORIGINAL failure, not at the moment we happen to pick the
    # case up. Starting both strategies at zero would hand the baseline an artificially
    # fresh success probability and make the comparison meaningless.
    hours = float(txn.get("days_since_failure", 0.0)) * 24.0
    for attempt in range(1, cfg.max_retries + 1):
        hours += cfg.retry_interval_hours
        if hours > horizon_hours:
            out.stop_reason = "horizon reached"
            break

        cured = gateway.check_self_cure(txn, hours)
        if cured is not None:
            out.recovered = True
            out.passive_recovery = True
            out.amount_recovered = out.amount_usd
            out.recovery_hours = hours
            out.status = "recovered"
            out.stop_reason = "self-cured without intervention"
            break

        res = gateway.retry_payment(txn, hours_since_failure=hours, attempt=attempt)
        out.retries += 1
        out.actions.append("retry_payment")
        out.cost += ACTION_COST_USD["retry_payment"]
        if cat is FailureCategory.RISK_COMPLIANCE:
            out.risk_actions += 1

        if res.success:
            out.recovered = True
            out.amount_recovered = out.amount_usd
            out.recovery_hours = hours
            out.status = "recovered"
            out.stop_reason = "payment succeeded"
            break

    out.cost = round(out.cost, 4)
    return out


def run_baseline(transactions: list[dict], gateway: PaymentGateway | None = None,
                 cfg: BaselineConfig | None = None) -> tuple[list[CaseOutcome], dict]:
    gw = gateway or PaymentGateway()
    cfg = cfg or BaselineConfig()
    outcomes = [run_case(t, gw, cfg) for t in transactions]
    return outcomes, summarize(outcomes, "baseline")
