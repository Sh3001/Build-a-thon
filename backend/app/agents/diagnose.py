"""Phase 6 -- the root cause engine.

A processor error code is a *claim*, not a diagnosis. A card that reports
`insufficient_funds` for the fifth time on a customer who has never once paid is not a
liquidity problem, it is a dead relationship, and retrying it as though funds will appear
is how dunning systems waste their contact budget.

This engine therefore reconciles the reported code against behavioural signals and may
overrule it. `root_cause` is what the agent acts on; `failure_code` is only evidence.
"""
from __future__ import annotations

from backend.app.models.enums import FailureCategory, FailureCode, category_of
from backend.app.models.schemas import Diagnosis, Transaction

#: Chronic-failure thresholds. Above these the reported code stops being believable.
CHRONIC_FAILURES = 4
CHRONIC_SUCCESS_RATE = 0.25


def diagnose(txn: Transaction) -> Diagnosis:
    """Deterministic diagnosis. Always available, never calls out."""
    code = txn.failure_code
    cat = category_of(code)
    evidence: list[str] = [f"processor code {code.value}"]
    confidence = 0.90
    root = code
    notes: list[str] = []

    ratio = txn.amount_usd / max(txn.avg_transaction_value, 1.0) if txn.avg_transaction_value else 1.0

    # --- reconcile the reported code against behaviour ------------------------
    chronic = (txn.failure_count >= CHRONIC_FAILURES
               and txn.previous_success_rate < CHRONIC_SUCCESS_RATE)

    if chronic and cat in (FailureCategory.TEMPORARY, FailureCategory.CUSTOMER_ACTION):
        # Repeated "transient" failures on a customer who never pays are not transient.
        root = FailureCode.MULTIPLE_DECLINES
        confidence = 0.72
        evidence.append(f"{txn.failure_count} consecutive failures")
        evidence.append(f"historic success rate {txn.previous_success_rate:.0%}")
        notes.append("reported code overruled: chronic failure pattern, not a transient fault")

    elif code is FailureCode.INSUFFICIENT_FUNDS and ratio > 3.0:
        # Not broke -- billed far more than usual. An instalment or a link works better
        # than hammering the same debit.
        confidence = 0.78
        evidence.append(f"amount is {ratio:.1f}x this customer's average")
        notes.append("affordability shock rather than a persistently empty account")

    elif code is FailureCode.TEMPORARY_DECLINE and txn.failure_count >= 3:
        root = FailureCode.MULTIPLE_DECLINES
        confidence = 0.70
        evidence.append(f"{txn.failure_count} declines in a row")
        notes.append("issuer is declining consistently; treat as persistent")

    elif cat is FailureCategory.TEMPORARY and txn.days_since_failure > 14:
        # The retry window for a transient fault has closed.
        confidence = 0.65
        evidence.append(f"{txn.days_since_failure:.0f} days since failure")
        notes.append("transient window has closed; retry value is largely spent")

    if txn.previous_recovery_count > 0:
        evidence.append(f"recovered {txn.previous_recovery_count}x before")
        confidence = min(0.95, confidence + 0.03)

    root_cat = category_of(root)
    # Only risk/compliance is truly unrecoverable, because there we may not act at all.
    # A closed account cannot be *debited*, but the customer can still pay by another
    # means -- so it stays recoverable while losing retry viability.
    recoverable = root_cat is not FailureCategory.RISK_COMPLIANCE
    retry_viable = recoverable and root not in (
        FailureCode.CLOSED_ACCOUNT, FailureCode.INVALID_ACCOUNT,
        FailureCode.EXPIRED_CARD, FailureCode.INVALID_PAYMENT_METHOD,
    )

    if root_cat is FailureCategory.RISK_COMPLIANCE:
        confidence = 0.97
        notes.append("risk/compliance: not an automation decision")

    rationale = "; ".join(notes) or f"{code.value} taken at face value; signals are consistent"
    return Diagnosis(root_cause=root, category=root_cat, confidence=round(confidence, 3),
                     rationale=rationale, evidence=evidence, source="rules",
                     recoverable=recoverable, retry_viable=retry_viable)
