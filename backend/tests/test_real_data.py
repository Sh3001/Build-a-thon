"""Tests for the REAL-data track (UCI delinquency events).

These skip cleanly when the source file has not been downloaded, so the suite still runs
offline. Download it with:

    curl -L -o data/external/uci_credit.xls \\
      "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

The claim these protect is the one that needs no intervention assumption: that ranking a
real queue by expected recovery beats the obvious alternatives on real money.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.config import DATA_PROCESSED, MODEL_DIR, RUN_DIR

SOURCE = Path("data/external/uci_credit.xls")
REAL_DIR = DATA_PROCESSED / "real"

needs_source = pytest.mark.skipif(
    not SOURCE.exists(), reason="UCI source not downloaded (see module docstring)")
needs_splits = pytest.mark.skipif(
    not (REAL_DIR / "test.csv").exists(),
    reason="real splits not built -- run scripts/build_real_dataset.py")
needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "real" / "model.joblib").exists(),
    reason="real model not trained -- run ml/train_real.py")


@pytest.fixture(scope="module")
def source():
    return pd.read_excel(SOURCE, header=1)


@pytest.fixture(scope="module")
def events():
    return pd.read_csv(REAL_DIR / "events.csv")


@pytest.fixture(scope="module")
def real_test():
    return pd.read_csv(REAL_DIR / "test.csv")


# ------------------------------------------------------------------ extraction
@needs_source
def test_source_shape_is_what_we_think_it_is(source):
    """Guards against a silently different file appearing at the same path."""
    assert len(source) == 30000
    for col in ("PAY_0", "PAY_2", "PAY_6", "BILL_AMT1", "PAY_AMT1", "LIMIT_BAL"):
        assert col in source.columns


@needs_source
def test_extraction_matches_the_source_definition(source):
    """Re-derive one transition by hand and check the pipeline agrees.

    A row is an event iff the customer was >=1 month late; it is a recovery iff they
    were current (<=0) the following month. If this drifts, every real number moves.
    """
    from scripts.build_real_dataset import extract_events
    ev = extract_events(source)

    # PAY_6 (April) -> PAY_5 (May), the first transition
    m = (source["PAY_6"] >= 1) & (source["BILL_AMT6"] > 0)
    expected_n = int(m.sum())
    expected_cures = int((source.loc[m, "PAY_5"] <= 0).sum())

    got = ev[ev["month_index"] == 0]
    assert len(got) == expected_n
    assert int(got["recovered"].sum()) == expected_cures


@needs_splits
def test_events_carry_real_money(events):
    assert (events["amount_twd"] > 0).all()
    assert events["amount_usd"].sum() > 1_000_000, "implausibly small real exposure"
    assert (events["amount_usd"] <= events["amount_twd"]).all(), "FX applied backwards"


@needs_splits
def test_outcome_is_binary_and_plausible(events):
    assert set(events["recovered"].unique()) <= {0, 1}
    assert 0.10 < events["recovered"].mean() < 0.45, "real cure rate outside a sane range"


@needs_splits
def test_the_real_decay_curve_holds(events):
    """The empirical finding the project leans on: cure probability falls sharply with
    delinquency depth. If this inverts, the source or the extraction is wrong."""
    g = events.groupby("months_late")["recovered"].agg(["size", "mean"])
    g = g[g["size"] >= 50]
    shallow = g.loc[g.index.min(), "mean"]
    deep = g.loc[g.index.max(), "mean"]
    assert shallow > deep + 0.10, f"decay missing: {shallow:.3f} -> {deep:.3f}"


# ------------------------------------------------------------------ split hygiene
@needs_splits
def test_real_splits_are_customer_disjoint():
    """One customer contributes up to five events, so a row-wise split would leak."""
    parts = {n: pd.read_csv(REAL_DIR / f"{n}.csv") for n in ("train", "val", "test")}
    ids = {n: set(p["customer_id"]) for n, p in parts.items()}
    assert not (ids["train"] & ids["test"])
    assert not (ids["train"] & ids["val"])
    assert not (ids["val"] & ids["test"])


@needs_splits
def test_no_label_leakage_into_features():
    """`recovered` is next month's state. Nothing derived from it may enter the model."""
    from ml.train_real import FEATURES
    banned = {"recovered", "cured", "outcome", "next"}
    for f in FEATURES:
        assert not any(b in f.lower() for b in banned), f"suspicious feature {f}"


@needs_splits
def test_features_available_at_decision_time(real_test):
    """Prior history must only look backwards."""
    assert (real_test["prior_late_months"] <= real_test["prior_observations"]).all()
    assert (real_test["prior_observations"] == real_test["month_index"]).all()


# ------------------------------------------------------------------ model
@needs_model
@needs_splits
def test_real_model_beats_chance_on_held_out_data(real_test):
    from sklearn.metrics import roc_auc_score
    import joblib
    b = joblib.load(MODEL_DIR / "real" / "model.joblib")
    X = real_test[b["features"]].astype(float).fillna(0.0)
    raw = (b["xgb"].predict_proba(X)[:, 1] if b["kind"] == "xgboost"
           else b["lr"].predict_proba(b["scaler"].transform(X))[:, 1])
    auc = roc_auc_score(real_test["recovered"].to_numpy(), b["calibrator"].predict(raw))
    assert auc > 0.75, f"real held-out ROC-AUC collapsed to {auc:.3f}"


@needs_model
@needs_splits
def test_real_predictions_are_calibrated(real_test):
    import joblib
    b = joblib.load(MODEL_DIR / "real" / "model.joblib")
    X = real_test[b["features"]].astype(float).fillna(0.0)
    raw = (b["xgb"].predict_proba(X)[:, 1] if b["kind"] == "xgboost"
           else b["lr"].predict_proba(b["scaler"].transform(X))[:, 1])
    p = b["calibrator"].predict(raw)
    actual = real_test["recovered"].mean()
    assert abs(p.mean() - actual) < 0.08, f"predicted {p.mean():.3f} vs actual {actual:.3f}"


# ------------------------------------------------------------------ the headline claim
@needs_model
@needs_splits
def test_ranking_by_expected_recovery_beats_the_alternatives(real_test):
    """The assumption-free claim the pitch leads with. Real amounts, real outcomes,
    no intervention model. If this stops holding, the pitch is wrong."""
    import joblib
    b = joblib.load(MODEL_DIR / "real" / "model.joblib")
    d = real_test.copy()
    X = d[b["features"]].astype(float).fillna(0.0)
    raw = (b["xgb"].predict_proba(X)[:, 1] if b["kind"] == "xgboost"
           else b["lr"].predict_proba(b["scaler"].transform(X))[:, 1])
    d["p"] = b["calibrator"].predict(raw)
    d["ev"] = d["amount_usd"] * d["p"]

    recoverable = float((d["amount_usd"] * d["recovered"]).sum())
    K = 250

    def captured(order):
        top = d.loc[order[:K]]
        return float((top["amount_usd"] * top["recovered"]).sum()) / recoverable

    model = captured(d.sort_values("ev", ascending=False).index.to_numpy())
    amount = captured(d.sort_values("amount_usd", ascending=False).index.to_numpy())
    rng = np.random.default_rng(7)
    rand = float(np.mean([captured(rng.permutation(d.index.to_numpy())) for _ in range(50)]))

    assert model > amount, f"model {model:.3f} lost to amount-ranking {amount:.3f}"
    assert model > rand * 3, f"model {model:.3f} barely beat random {rand:.3f}"
    assert model > 0.40, f"top-{K} captured only {model:.1%} of recoverable revenue"


# ------------------------------------------------------------------ the experiment
@needs_model
@needs_splits
def test_status_quo_arm_is_not_simulated(real_test):
    """The control must be the observed label, exactly. Any drift here would turn a
    measured counterfactual back into a modelled one."""
    from scripts.run_real_experiment import run_status_quo
    outs, rep = run_status_quo(real_test)
    assert len(outs) == len(real_test)
    assert rep["cases_recovered"] == int(real_test["recovered"].sum())
    assert all(o.passive_recovery for o in outs if o.recovered), \
        "status-quo recoveries must never be credited to us"
    assert rep["total_cost"] == 0.0
    assert rep["total_actions"] == 0


def test_contact_fatigue_reduces_the_marginal_contact():
    """Without diminishing returns, contacting everyone wins by construction -- which is
    exactly the bias this parameter exists to remove."""
    from scripts.run_real_experiment import apply_plan
    rng = np.random.default_rng(0)
    one, _ = apply_plan(0.2, ["send_reminder"], 1.0, 1.0, np.random.default_rng(0))
    two, _ = apply_plan(0.2, ["send_reminder", "send_reminder"], 1.0, 1.0,
                        np.random.default_rng(0))
    three, _ = apply_plan(0.2, ["send_reminder"] * 3, 1.0, 1.0, np.random.default_rng(0))
    assert two > one, "a second contact should still add something"
    assert (two - one) > (three - two), "marginal contact value must diminish"


def test_deep_delinquency_damping_blocks_uplift():
    """The real data says 7-months-late cures 0% of the time. No modelled intervention
    may contradict the measurement."""
    from scripts.run_real_experiment import DEPTH_DAMPING, apply_plan
    assert DEPTH_DAMPING[7] == 0.0
    p, _ = apply_plan(0.05, ["send_payment_link", "send_reminder"],
                      DEPTH_DAMPING[7], 2.0, np.random.default_rng(0))
    assert p == pytest.approx(0.05, abs=1e-9), "uplift applied to a hopeless case"


@needs_model
@needs_splits
def test_experiment_is_reproducible(real_test):
    from scripts.run_real_experiment import load_model, run_agent
    d = real_test.copy()
    d["p_base"] = load_model()(d)
    a = run_agent(d, np.random.default_rng(1), 1.0)[1]
    b = run_agent(d, np.random.default_rng(1), 1.0)[1]
    assert a["revenue_recovered"] == b["revenue_recovered"]
    assert a["cases_recovered"] == b["cases_recovered"]


@needs_model
@needs_splits
def test_agent_respects_guardrails_on_real_cases(real_test):
    """Escalation and the value floor must behave the same on real data as on synthetic."""
    from scripts.run_real_experiment import (
        ESCALATE_MONTHS_LATE, MAX_AUTO_USD, load_model, run_agent,
    )
    d = real_test.copy()
    d["p_base"] = load_model()(d)
    outs, _ = run_agent(d, np.random.default_rng(1), 1.0)
    by_id = {o.transaction_id: o for o in outs}
    for _, r in d.iterrows():
        o = by_id[r["transaction_id"]]
        if r["months_late"] >= ESCALATE_MONTHS_LATE or r["amount_usd"] > MAX_AUTO_USD:
            assert o.escalated, f"{r['transaction_id']} should have gone to a human"
            assert o.actions == ["escalate_case"]
