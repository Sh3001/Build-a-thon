"""Phase 1 -- synthetic dataset generation.

Records are drawn from an explicit generative model: customer attributes first, then a
transaction, then a failure conditioned on the payment method, and finally the label
sampled from a logistic model over the features.

Two properties make this worth training on rather than a lookup table:

1. **Timing interacts with cause.** TEMPORARY failures decay fast -- retry now or lose it.
   `insufficient_funds` does the opposite: recovery probability *rises* for about a week
   (the customer gets paid) before decaying. A model that ignores the interaction cannot
   express both, which is exactly the timing decision the agent has to make.
2. **Nothing is deterministic.** The label is a Bernoulli draw with logit noise, so a
   perfect model still cannot exceed the Bayes rate. Reported AUC is therefore honest.

Ground truth (`_p_recover`) is used only to *sample* the label. It is never written to the
dataset and no downstream component can read it.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import DATA_PROCESSED, DATA_RAW, FX_TO_USD, N_RECORDS, SEED
from backend.app.models.enums import FailureCategory, FailureCode, category_of

COUNTRIES = ["US", "IN", "GB", "DE", "SG", "AU"]
COUNTRY_W = [0.34, 0.30, 0.13, 0.11, 0.07, 0.05]
COUNTRY_CCY = {"US": "USD", "IN": "INR", "GB": "GBP", "DE": "EUR", "SG": "SGD", "AU": "AUD"}

METHODS_BY_COUNTRY = {
    "US": (["card", "ach", "wallet"], [0.72, 0.22, 0.06]),
    "IN": (["upi", "upi_autopay", "card", "netbanking"], [0.34, 0.28, 0.28, 0.10]),
    "GB": (["card", "ach", "wallet"], [0.78, 0.16, 0.06]),
    "DE": (["card", "ach", "wallet"], [0.55, 0.39, 0.06]),
    "SG": (["card", "wallet", "netbanking"], [0.70, 0.20, 0.10]),
    "AU": (["card", "ach", "wallet"], [0.76, 0.18, 0.06]),
}

SEGMENTS = ["consumer", "smb", "enterprise"]
SEGMENT_W = [0.70, 0.22, 0.08]
#: (mu, sigma) of lognormal average transaction value in USD.
SEGMENT_ATV = {"consumer": (3.4, 0.55), "smb": (5.1, 0.62), "enterprise": (6.9, 0.75)}
CHANNELS = ["email", "sms", "whatsapp", "in_app"]

#: Failure-code mix conditioned on payment method. Card expiry cannot happen on UPI.
CODE_MIX: dict[str, dict[str, float]] = {
    "card": {
        "insufficient_funds": 0.22, "expired_card": 0.14, "temporary_decline": 0.11,
        "bank_timeout": 0.09, "multiple_declines": 0.09, "network_error": 0.06,
        "payment_limit_exceeded": 0.06, "invalid_payment_method": 0.05,
        "processor_unavailable": 0.05, "high_risk_transaction": 0.04,
        "suspected_fraud": 0.035, "invalid_account": 0.02, "closed_account": 0.015,
        "compliance_hold": 0.01,
    },
    "upi": {
        "insufficient_funds": 0.26, "bank_timeout": 0.19, "temporary_decline": 0.13,
        "network_error": 0.11, "processor_unavailable": 0.09,
        "payment_limit_exceeded": 0.08, "invalid_payment_method": 0.05,
        "multiple_declines": 0.04, "invalid_account": 0.02, "high_risk_transaction": 0.02,
        "suspected_fraud": 0.007, "closed_account": 0.002, "compliance_hold": 0.001,
    },
    "upi_autopay": {
        "insufficient_funds": 0.34, "bank_timeout": 0.15, "temporary_decline": 0.11,
        "processor_unavailable": 0.09, "network_error": 0.08,
        "invalid_payment_method": 0.07, "payment_limit_exceeded": 0.05,
        "multiple_declines": 0.04, "invalid_account": 0.03, "closed_account": 0.02,
        "high_risk_transaction": 0.015, "suspected_fraud": 0.01, "compliance_hold": 0.005,
    },
    "netbanking": {
        "bank_timeout": 0.25, "insufficient_funds": 0.20, "network_error": 0.14,
        "processor_unavailable": 0.12, "temporary_decline": 0.10,
        "payment_limit_exceeded": 0.06, "multiple_declines": 0.05,
        "invalid_account": 0.03, "invalid_payment_method": 0.02,
        "high_risk_transaction": 0.02, "closed_account": 0.005,
        "suspected_fraud": 0.003, "compliance_hold": 0.002,
    },
    "ach": {
        "insufficient_funds": 0.30, "invalid_account": 0.12, "bank_timeout": 0.11,
        "closed_account": 0.09, "temporary_decline": 0.09, "processor_unavailable": 0.08,
        "network_error": 0.07, "multiple_declines": 0.06, "compliance_hold": 0.03,
        "payment_limit_exceeded": 0.03, "invalid_payment_method": 0.01,
        "high_risk_transaction": 0.008, "suspected_fraud": 0.002,
    },
    "wallet": {
        "insufficient_funds": 0.33, "temporary_decline": 0.14, "network_error": 0.12,
        "bank_timeout": 0.11, "payment_limit_exceeded": 0.10,
        "processor_unavailable": 0.07, "invalid_payment_method": 0.05,
        "multiple_declines": 0.04, "high_risk_transaction": 0.02,
        "suspected_fraud": 0.012, "invalid_account": 0.005, "closed_account": 0.002,
        "compliance_hold": 0.001,
    },
}

#: Baseline log-odds of recovery within the horizon, by failure code.
BASE_LOGIT: dict[str, float] = {
    "bank_timeout": 0.15, "network_error": 0.30, "processor_unavailable": 0.05,
    "temporary_decline": -0.40,
    "insufficient_funds": -1.15, "expired_card": -1.55,
    "invalid_payment_method": -1.80, "payment_limit_exceeded": -1.05,
    "multiple_declines": -2.65, "invalid_account": -3.75, "closed_account": -4.15,
    "suspected_fraud": -4.00, "high_risk_transaction": -3.35, "compliance_hold": -3.60,
}

SEGMENT_BONUS = {"consumer": 0.0, "smb": 0.22, "enterprise": 0.46}


def _weighted(rng: np.random.Generator, n: int, choices: list[str], w: list[float]) -> np.ndarray:
    p = np.asarray(w, dtype=float)
    return rng.choice(choices, size=n, p=p / p.sum())


def _timing_term(code: np.ndarray, days: np.ndarray) -> np.ndarray:
    """The cause x timing interaction. This is the structure the agent must learn to
    exploit, and the reason a linear-in-days model underfits."""
    out = np.zeros(len(code), dtype=float)
    cat = np.array([category_of(c).value for c in code])

    temp = cat == FailureCategory.TEMPORARY.value
    out[temp] = -0.185 * days[temp]                       # the window closes fast

    funds = code == "insufficient_funds"
    d = days[funds]
    out[funds] = np.where(d <= 7.0, 0.26 * d, 1.82 - 0.20 * (d - 7.0))   # payday effect

    action = np.isin(code, ["expired_card", "invalid_payment_method", "payment_limit_exceeded"])
    out[action] = -0.025 * days[action]                   # waits on the customer, not the clock

    slow = np.isin(cat, [FailureCategory.PERSISTENT.value, FailureCategory.RISK_COMPLIANCE.value])
    out[slow] = -0.05 * days[slow]
    return out


def _p_recover(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Ground-truth recovery probability. Never persisted."""
    code = df["failure_code"].to_numpy()
    logit = np.array([BASE_LOGIT[c] for c in code], dtype=float)

    logit += 2.45 * (df["previous_success_rate"].to_numpy() - 0.55)
    logit += 0.30 * np.log1p(df["customer_tenure"].to_numpy() / 180.0)
    logit -= 0.55 * (df["failure_count"].to_numpy() - 1)
    logit += 0.22 * np.minimum(df["previous_recovery_count"].to_numpy(), 4)
    logit -= 0.012 * df["overdue_days"].to_numpy()
    logit += np.array([SEGMENT_BONUS[s] for s in df["customer_segment"]], dtype=float)

    # Paying a bill much larger than this customer's norm is harder.
    ratio = df["amount_usd"].to_numpy() / np.maximum(df["avg_transaction_value_usd"].to_numpy(), 1.0)
    logit -= 0.42 * np.log10(np.clip(ratio, 0.05, 40.0))

    # ...but enterprises absorb large invoices far better than consumers do.
    ent = (df["customer_segment"] == "enterprise").to_numpy()
    logit[ent] += 0.30 * np.log10(np.clip(ratio[ent], 0.05, 40.0))

    logit += _timing_term(code, df["days_since_failure"].to_numpy())
    logit += rng.normal(0.0, 0.70, len(df))               # irreducible noise
    return 1.0 / (1.0 + np.exp(-logit))


def generate(n: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Customers repeat across records, so a customer-level split is meaningful.
    n_cust = max(1, int(n / 1.7))
    cust_ids = np.array([f"cust_{i:06d}" for i in range(n_cust)])
    c_country = _weighted(rng, n_cust, COUNTRIES, COUNTRY_W)
    c_segment = _weighted(rng, n_cust, SEGMENTS, SEGMENT_W)
    c_tenure = rng.integers(5, 2200, n_cust)
    c_channel = _weighted(rng, n_cust, CHANNELS, [0.45, 0.25, 0.18, 0.12])
    # Longer-tenured customers have a better payment history, with spread.
    c_alpha = 2.0 + 5.0 * (c_tenure / 2200.0)
    c_psr = np.clip(rng.beta(c_alpha, 2.2), 0.02, 0.995)
    c_atv_usd = np.array([rng.lognormal(*SEGMENT_ATV[s]) for s in c_segment])

    idx = rng.integers(0, n_cust, n)                      # which customer each row belongs to
    country = c_country[idx]
    segment = c_segment[idx]
    tenure = c_tenure[idx]
    psr = c_psr[idx]
    atv_usd = c_atv_usd[idx]

    method = np.empty(n, dtype=object)
    for ctry in COUNTRIES:
        m = country == ctry
        opts, w = METHODS_BY_COUNTRY[ctry]
        method[m] = _weighted(rng, int(m.sum()), opts, w)

    code = np.empty(n, dtype=object)
    for meth, mix in CODE_MIX.items():
        m = method == meth
        if m.sum():
            code[m] = _weighted(rng, int(m.sum()), list(mix), list(mix.values()))

    amount_usd = np.round(atv_usd * rng.lognormal(0.0, 0.42, n), 2)
    currency = np.array([COUNTRY_CCY[c] for c in country])
    fx = np.array([FX_TO_USD[c] for c in currency])
    amount = np.round(amount_usd / fx, 2)

    cat = np.array([category_of(c).value for c in code])
    # Persistent failures have been failing repeatedly by definition.
    fail_count = 1 + rng.poisson(np.where(cat == "PERSISTENT", 2.1, 0.55), n)
    fail_count = np.clip(fail_count, 1, 8)
    days_since_failure = np.round(np.clip(rng.exponential(4.5, n), 0.0, 30.0), 2)
    overdue = np.clip((days_since_failure + rng.exponential(6.0, n)).astype(int), 0, 120)
    attempts = 1 + rng.poisson(np.maximum(tenure / 90.0, 0.5)).astype(int)
    prev_recov = rng.binomial(np.minimum(attempts, 40), 0.16 * psr)
    sub_age = np.minimum(tenure, rng.integers(0, 1500, n))
    days_last_pay = np.round(np.clip(rng.exponential(28.0, n), 0.0, 400.0), 1)

    df = pd.DataFrame({
        "customer_id": cust_ids[idx],
        "transaction_id": [f"txn_{i:07d}" for i in range(n)],
        "subscription_id": [f"sub_{i:06d}" for i in idx],
        "invoice_id": [f"inv_{i:07d}" for i in range(n)],
        "amount": amount,
        "currency": currency,
        "payment_method": method,
        "failure_code": code,
        "failure_count": fail_count,
        "days_since_failure": days_since_failure,
        "customer_tenure": tenure,
        "previous_success_rate": np.round(psr, 4),
        "previous_payment_attempts": attempts,
        "avg_transaction_value": np.round(atv_usd / fx, 2),
        "days_since_last_payment": days_last_pay,
        "customer_segment": segment,
        "country": country,
        "preferred_channel": c_channel[idx],
        "subscription_age": sub_age,
        "overdue_days": overdue,
        "previous_recovery_count": prev_recov,
    })
    # USD views are derived, not stored -- used for the label and for revenue maths.
    df["amount_usd"] = amount_usd
    df["avg_transaction_value_usd"] = np.round(atv_usd, 2)

    p = _p_recover(df, rng)
    df["recovered"] = (rng.random(n) < p).astype(int)

    # Time to cash, conditioned on cause. NaN where the case never recovered.
    base_days = np.select(
        [cat == "TEMPORARY", code == "insufficient_funds", cat == "CUSTOMER_ACTION",
         cat == "PERSISTENT"],
        [rng.exponential(0.8, n), 3.0 + rng.exponential(3.5, n),
         2.0 + rng.exponential(4.0, n), 5.0 + rng.exponential(6.0, n)],
        default=7.0 + rng.exponential(8.0, n),
    )
    rec_days = np.round(np.clip(base_days, 0.05, 30.0), 2)
    df["recovery_days"] = np.where(df["recovered"] == 1, rec_days, np.nan)
    return df


def split(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Split on customer_id, not on rows: the same customer must never appear in both
    train and test, or the model scores its own customers and the metric is inflated."""
    def bucket(cid: str) -> float:
        h = hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    b = df["customer_id"].map(bucket)
    return {
        "train": df[b < 0.70].reset_index(drop=True),
        "val": df[(b >= 0.70) & (b < 0.85)].reset_index(drop=True),
        "test": df[b >= 0.85].reset_index(drop=True),
    }


RAW_COLUMNS = [
    "customer_id", "transaction_id", "subscription_id", "invoice_id", "amount", "currency",
    "payment_method", "failure_code", "failure_count", "days_since_failure",
    "customer_tenure", "previous_success_rate", "previous_payment_attempts",
    "avg_transaction_value", "days_since_last_payment", "customer_segment", "country",
    "preferred_channel", "subscription_age", "overdue_days", "previous_recovery_count",
    "recovered", "recovery_days",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the RecoverAI dataset.")
    ap.add_argument("--n", type=int, default=N_RECORDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=DATA_RAW)
    a = ap.parse_args()

    df = generate(a.n, a.seed)
    raw = a.out / "payment_recovery.csv"
    df[RAW_COLUMNS].to_csv(raw, index=False)

    parts = split(df, a.seed)
    for name, part in parts.items():
        part.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    print(f"  generated  {len(df):,} records -> {raw}")
    print(f"  recovered  {df['recovered'].mean():.1%} overall")
    for name, part in parts.items():
        print(f"  {name:<6}    {len(part):>6,} rows   recovered {part['recovered'].mean():.1%}   "
              f"customers {part['customer_id'].nunique():,}")
    overlap = set(parts["train"]["customer_id"]) & set(parts["test"]["customer_id"])
    print(f"  train/test customer overlap: {len(overlap)}")
    print("\n  recovery rate by failure category")
    for c in sorted(df["failure_code"].map(lambda x: category_of(x).value).unique()):
        m = df["failure_code"].map(lambda x: category_of(x).value) == c
        print(f"    {c:<16} n={m.sum():>6,}  recovered {df.loc[m,'recovered'].mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
