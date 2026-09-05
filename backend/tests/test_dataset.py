"""Phase 1 tests -- schema conformance, split hygiene, and the structural properties the
rest of the system depends on."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.app.models.enums import FAILURE_CATEGORY, FailureCategory, FailureCode, category_of
from scripts.generate_dataset import RAW_COLUMNS, generate, split


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate(n=6000, seed=4242)


def test_taxonomy_is_total():
    """Every failure code has a category. A missing entry would crash the policy engine."""
    assert set(FAILURE_CATEGORY) == set(FailureCode)
    assert set(FAILURE_CATEGORY.values()) == set(FailureCategory)


def test_required_columns_present(df):
    for col in RAW_COLUMNS:
        assert col in df.columns, f"missing required column {col}"


def test_row_count_and_label(df):
    assert len(df) == 6000
    assert set(df["recovered"].unique()) <= {0, 1}
    assert 0.20 < df["recovered"].mean() < 0.60, "label balance outside a plausible range"


def test_deterministic_for_a_seed():
    a, b = generate(n=800, seed=7), generate(n=800, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a, b = generate(n=800, seed=7), generate(n=800, seed=8)
    assert not a["recovered"].equals(b["recovered"])


def test_no_ground_truth_leaks_into_raw_columns(df):
    """`_p_recover` must not be persisted -- otherwise the model trains on the answer."""
    for banned in ("p_recover", "true_p", "logit", "latent"):
        assert not any(banned in c for c in RAW_COLUMNS)


def test_amounts_and_rates_are_sane(df):
    assert (df["amount"] > 0).all()
    assert df["previous_success_rate"].between(0, 1).all()
    assert (df["failure_count"] >= 1).all()
    assert df["days_since_failure"].between(0, 30).all()
    assert (df["subscription_age"] <= df["customer_tenure"]).all()


def test_recovery_days_only_when_recovered(df):
    assert df.loc[df["recovered"] == 0, "recovery_days"].isna().all()
    assert df.loc[df["recovered"] == 1, "recovery_days"].notna().all()
    assert df.loc[df["recovered"] == 1, "recovery_days"].between(0, 30).all()


def test_payment_method_constrains_failure_code(df):
    """A UPI mandate cannot fail with an expired card. Impossible combinations would
    teach the model a correlation that does not exist in production."""
    upi = df[df["payment_method"].isin(["upi", "upi_autopay", "netbanking"])]
    assert "expired_card" not in set(upi["failure_code"])


def test_split_is_customer_disjoint(df):
    parts = split(df, seed=4242)
    train, val, test = (set(parts[k]["customer_id"]) for k in ("train", "val", "test"))
    assert not (train & test), "customer leakage between train and test"
    assert not (train & val)
    assert not (val & test)
    assert sum(len(parts[k]) for k in parts) == len(df)


def test_split_proportions_are_roughly_right(df):
    parts = split(df, seed=4242)
    frac = {k: len(v) / len(df) for k, v in parts.items()}
    assert 0.63 < frac["train"] < 0.77
    assert 0.10 < frac["val"] < 0.20
    assert 0.10 < frac["test"] < 0.20


def test_category_ordering_is_realistic(df):
    """Temporary failures must recover far more often than persistent or risk ones. If
    this inverts, every downstream policy is being validated against nonsense."""
    cat = df["failure_code"].map(lambda c: category_of(c).value)
    rate = df.groupby(cat)["recovered"].mean()
    assert rate["TEMPORARY"] > rate["PERSISTENT"] + 0.20
    assert rate["TEMPORARY"] > rate["RISK_COMPLIANCE"] + 0.20
    assert rate["PERSISTENT"] < 0.20
    assert rate["RISK_COMPLIANCE"] < 0.20


def test_cause_timing_interaction_exists(df):
    """The core learnable structure: waiting *helps* insufficient_funds (payday) and
    *hurts* temporary failures. Without this the ML model adds nothing over a lookup."""
    funds = df[df["failure_code"] == "insufficient_funds"]
    early = funds[funds["days_since_failure"] <= 2]["recovered"].mean()
    later = funds[funds["days_since_failure"].between(5, 9)]["recovered"].mean()
    assert later > early + 0.05, f"payday effect missing: {early:.3f} -> {later:.3f}"

    temp = df[df["failure_code"].isin(["bank_timeout", "network_error", "processor_unavailable"])]
    t_early = temp[temp["days_since_failure"] <= 2]["recovered"].mean()
    t_late = temp[temp["days_since_failure"] >= 8]["recovered"].mean()
    assert t_early > t_late + 0.15, f"decay missing: {t_early:.3f} -> {t_late:.3f}"


def test_label_is_stochastic_not_a_lookup(df):
    """Identical failure codes must produce both outcomes, or a rule table would be a
    perfect model and the reported AUC would be meaningless."""
    for code in ("insufficient_funds", "bank_timeout", "expired_card"):
        sub = df[df["failure_code"] == code]["recovered"]
        assert 0 < sub.mean() < 1, f"{code} is deterministic"
