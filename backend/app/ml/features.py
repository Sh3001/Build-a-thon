"""Phase 3 -- feature engineering.

The categorical vocabulary is derived from the enums rather than from whatever happened
to appear in the training file. A category the model never saw therefore produces an
all-zero block instead of a shifted feature matrix, and training and serving cannot drift
apart as the data changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.config import FX_TO_USD
from backend.app.models.enums import (
    Channel, CustomerSegment, FailureCategory, FailureCode, PaymentMethod, category_of,
)

NUMERIC = [
    "amount_usd", "failure_count", "days_since_failure", "customer_tenure",
    "previous_success_rate", "previous_payment_attempts", "avg_transaction_value_usd",
    "days_since_last_payment", "subscription_age", "overdue_days",
    "previous_recovery_count",
]

DERIVED = [
    "amount_ratio",          # this bill vs the customer's norm -- affordability
    "log_amount",
    "attempts_per_year",     # billing frequency
    "recovery_history_rate", # how often past dunning worked for this customer
    "days_x_temporary",      # the cause x timing interactions, made explicit
    "days_x_funds",
    "overdue_ratio",
]

CATEGORICALS: dict[str, list[str]] = {
    "payment_method": [e.value for e in PaymentMethod],
    "failure_code": [e.value for e in FailureCode],
    "failure_category": [e.value for e in FailureCategory],
    "customer_segment": [e.value for e in CustomerSegment],
    "preferred_channel": [e.value for e in Channel],
    "country": ["US", "IN", "GB", "DE", "SG", "AU"],
}


def feature_names() -> list[str]:
    names = list(NUMERIC) + list(DERIVED)
    for col, vocab in CATEGORICALS.items():
        names += [f"{col}={v}" for v in vocab]
    return names


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Frame -> numeric matrix with stable column order."""
    d = df.copy()

    if "amount_usd" not in d.columns:
        fx = d["currency"].map(FX_TO_USD).fillna(1.0)
        d["amount_usd"] = d["amount"] * fx
    if "avg_transaction_value_usd" not in d.columns:
        fx = d["currency"].map(FX_TO_USD).fillna(1.0)
        d["avg_transaction_value_usd"] = d["avg_transaction_value"] * fx
    if "failure_category" not in d.columns:
        d["failure_category"] = d["failure_code"].map(lambda c: category_of(c).value)

    out = pd.DataFrame(index=d.index)
    for col in NUMERIC:
        out[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    atv = np.maximum(out["avg_transaction_value_usd"].to_numpy(), 1.0)
    out["amount_ratio"] = np.clip(out["amount_usd"].to_numpy() / atv, 0.0, 50.0)
    out["log_amount"] = np.log1p(out["amount_usd"].to_numpy())
    out["attempts_per_year"] = out["previous_payment_attempts"] / np.maximum(
        out["customer_tenure"].to_numpy() / 365.0, 0.08)
    out["recovery_history_rate"] = out["previous_recovery_count"] / np.maximum(
        out["previous_payment_attempts"].to_numpy(), 1.0)

    days = out["days_since_failure"].to_numpy()
    is_temp = (d["failure_category"] == FailureCategory.TEMPORARY.value).to_numpy().astype(float)
    is_funds = (d["failure_code"] == FailureCode.INSUFFICIENT_FUNDS.value).to_numpy().astype(float)
    out["days_x_temporary"] = days * is_temp
    out["days_x_funds"] = days * is_funds
    out["overdue_ratio"] = out["overdue_days"] / np.maximum(days, 0.5)

    for col, vocab in CATEGORICALS.items():
        vals = d[col].astype(str).to_numpy()
        for v in vocab:
            out[f"{col}={v}"] = (vals == v).astype(np.int8)

    return out[feature_names()]


def build_features_from_transaction(txn: dict) -> pd.DataFrame:
    """Single-row serving path. Same code path as training, by construction."""
    return build_features(pd.DataFrame([txn]))
