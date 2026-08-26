"""Phase 3 tests -- feature stability, scoring contract, ranking, and calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.features import build_features, build_features_from_transaction, feature_names
from backend.app.ml.scorer import RecoveryScorer, get_scorer
from backend.app.services.dataio import load_split, to_transactions


@pytest.fixture(scope="module")
def test_df():
    return load_split("test")


# ----------------------------------------------------------------- features
def test_feature_matrix_is_stable_and_complete(test_df):
    X = build_features(test_df)
    assert list(X.columns) == feature_names()
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_unseen_category_does_not_shift_columns(test_df):
    """A new country must produce an all-zero block, not a misaligned matrix."""
    row = test_df.head(1).copy()
    row["country"] = "ZZ"
    X = build_features(row)
    assert list(X.columns) == feature_names()
    assert X[[c for c in X.columns if c.startswith("country=")]].to_numpy().sum() == 0


def test_single_row_path_matches_batch_path(test_df):
    """Serving and training must go through identical code, or the model sees different
    inputs in production than it was trained on."""
    txns = to_transactions(test_df.head(5))
    batch = build_features(test_df.head(5)).reset_index(drop=True)
    for i, t in enumerate(txns):
        single = build_features_from_transaction(t).reset_index(drop=True)
        pd.testing.assert_frame_equal(single, batch.iloc[[i]].reset_index(drop=True),
                                      check_dtype=False)


# ----------------------------------------------------------------- scoring
def test_scorer_contract(test_df):
    s = get_scorer()
    txn = to_transactions(test_df.head(1))[0]
    r = s.score(txn)
    assert 0.0 <= r.recovery_probability <= 1.0
    assert r.risk_score == pytest.approx(1.0 - r.recovery_probability, abs=1e-6)
    assert r.expected_recovery == pytest.approx(
        txn["amount_usd"] * r.recovery_probability, rel=1e-3)


def test_expected_recovery_never_exceeds_amount(test_df):
    s = get_scorer()
    scored = s.score_frame(test_df.drop(columns=["recovered", "recovery_days"]))
    assert (scored["expected_recovery"] <= scored["amount_usd"] + 0.01).all()
    assert (scored["expected_recovery"] >= 0).all()


def test_scoring_is_deterministic(test_df):
    s = get_scorer()
    d = test_df.drop(columns=["recovered", "recovery_days"]).head(200)
    a = s.score_frame(d)["recovery_probability"].to_numpy()
    b = s.score_frame(d)["recovery_probability"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_fallback_when_no_model_is_flagged(tmp_path):
    """A missing artefact must be visible, never silently mistaken for a prediction."""
    s = RecoveryScorer(model_dir=tmp_path)
    assert not s.is_trained
    assert s.model_version == "heuristic-fallback"
    p = s.predict_proba(pd.DataFrame([{"failure_code": "closed_account", "amount": 10,
                                       "currency": "USD"}]))
    assert 0.0 < float(p[0]) < 0.3


# ----------------------------------------------------------------- quality
def test_model_beats_chance_on_held_out_data(test_df):
    from sklearn.metrics import roc_auc_score
    s = get_scorer()
    if not s.is_trained:
        pytest.skip("model not trained")
    p = s.predict_proba(test_df.drop(columns=["recovered", "recovery_days"]))
    auc = roc_auc_score(test_df["recovered"].to_numpy(), p)
    assert auc > 0.70, f"held-out ROC-AUC collapsed to {auc:.3f}"


def test_ranking_orders_causes_correctly(test_df):
    """Persistent and risk failures must score below temporary ones, or the recovery
    queue will put effort where it cannot pay off."""
    from backend.app.models.enums import category_of
    s = get_scorer()
    if not s.is_trained:
        pytest.skip("model not trained")
    d = test_df.drop(columns=["recovered", "recovery_days"])
    scored = s.score_frame(d)
    cat = scored["failure_code"].map(lambda c: category_of(c).value)
    mean_p = scored.groupby(cat)["recovery_probability"].mean()
    assert mean_p["TEMPORARY"] > mean_p["PERSISTENT"] + 0.15
    assert mean_p["TEMPORARY"] > mean_p["RISK_COMPLIANCE"] + 0.15


def test_probabilities_are_roughly_calibrated(test_df):
    """expected_recovery = amount x p is only meaningful if p means what it says."""
    s = get_scorer()
    if not s.is_trained:
        pytest.skip("model not trained")
    d = test_df.drop(columns=["recovered", "recovery_days"])
    p = s.predict_proba(d)
    actual = test_df["recovered"].to_numpy().mean()
    assert abs(p.mean() - actual) < 0.10, f"mean predicted {p.mean():.3f} vs actual {actual:.3f}"


def test_ranking_by_expected_recovery_beats_random(test_df):
    """The core prioritisation claim: working the top of the queue captures
    disproportionate revenue."""
    s = get_scorer()
    if not s.is_trained:
        pytest.skip("model not trained")
    d = test_df.drop(columns=["recovered", "recovery_days"])
    scored = s.score_frame(d)
    scored["recovered"] = test_df["recovered"].to_numpy()
    ranked = scored.sort_values("expected_recovery", ascending=False)
    total = (ranked["amount_usd"] * ranked["recovered"]).sum()
    top10 = ranked.head(len(ranked) // 10)
    captured = (top10["amount_usd"] * top10["recovered"]).sum()
    assert captured / total > 0.35, f"top decile captured only {captured/total:.1%}"
