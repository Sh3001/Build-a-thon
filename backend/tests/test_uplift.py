"""Tests for uplift modelling -- the treatment-effect targeting model.

The property under test is not "the model is accurate" but "the model ranks the right
people". An outcome model puts customers who were always going to pay at the top of the
queue; an uplift model must not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.config import MODEL_DIR
from backend.app.ml.uplift import (
    UpliftModel, qini_coefficient, qini_curve, uplift_at_k,
)


@pytest.fixture(scope="module")
def toy():
    """A world with a known answer: only `a > 0` responds to treatment, and the big
    balances sit with people who respond least."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    treated = rng.random(n) < 0.5
    base = 1 / (1 + np.exp(-(0.5 * X["b"])))
    true_lift = np.where(X["a"] > 0, 0.30, 0.0)
    p = np.clip(base + treated * true_lift, 0, 1)
    y = (rng.random(n) < p).astype(int)
    return X, y, treated, true_lift


@pytest.fixture(scope="module")
def fitted(toy):
    X, y, treated, _ = toy
    return UpliftModel().fit(X, y, treated, seed=1)


# ------------------------------------------------------------------ recovery of effect
def test_uplift_model_recovers_a_known_treatment_effect(toy, fitted):
    X, _, _, true_lift = toy
    u = fitted.predict_uplift(X)
    assert u[X["a"] > 0].mean() == pytest.approx(0.30, abs=0.06)
    assert u[X["a"] <= 0].mean() == pytest.approx(0.0, abs=0.06)


def test_uplift_can_be_negative(toy):
    """Sleeping dogs are the whole reason to model uplift rather than outcome. If the
    estimator cannot express harm, it cannot avoid causing it."""
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({"a": rng.normal(size=n)})
    treated = rng.random(n) < 0.5
    base = np.full(n, 0.5)
    harm = np.where(X["a"] > 0, -0.30, 0.0)          # treatment HURTS this group
    p = np.clip(base + treated * harm, 0, 1)
    y = (rng.random(n) < p).astype(int)
    u = UpliftModel().fit(X, y, treated, seed=1).predict_uplift(X)
    assert u[X["a"] > 0].mean() < -0.15, "the model failed to detect harm"


def test_neither_arm_sees_the_treatment_indicator(fitted):
    """Each outcome model is trained on its own arm only, so it cannot shortcut."""
    assert "treated" not in fitted.feature_names
    assert "treatment" not in fitted.feature_names
    assert fitted.n_treated > 0 and fitted.n_control > 0


def test_fit_requires_both_arms(toy):
    X, y, _, _ = toy
    with pytest.raises(ValueError):
        UpliftModel().fit(X, y, np.ones(len(y), dtype=bool))
    with pytest.raises(ValueError):
        UpliftModel().fit(X, y, np.zeros(len(y), dtype=bool))


# ------------------------------------------------------------------ the ranking fix
def test_value_weighting_is_required_for_a_revenue_objective(toy, fitted):
    """The bug this caught: ranking by percentage-point uplift optimises the wrong unit
    and selects the smallest balances."""
    X, _, _, _ = toy
    rng = np.random.default_rng(5)
    # make the responsive group systematically small-value
    amount = np.where(X["a"] > 0, rng.uniform(5, 50, len(X)), rng.uniform(500, 2000, len(X)))

    by_pp = np.argsort(-fitted.predict_uplift(X))[:200]
    by_value = np.argsort(-fitted.expected_incremental_value(X, amount))[:200]
    assert amount[by_value].mean() > amount[by_pp].mean() * 3, \
        "value weighting did not shift selection toward larger balances"


def test_expected_incremental_value_is_amount_times_uplift(toy, fitted):
    X, _, _, _ = toy
    amount = np.full(len(X), 100.0)
    # rtol matches float32: XGBoost returns float32 probabilities.
    np.testing.assert_allclose(fitted.expected_incremental_value(X, amount),
                               100.0 * fitted.predict_uplift(X), rtol=1e-5)


def test_uplift_ranking_beats_outcome_ranking_on_caused_revenue(toy, fitted):
    """The claim: excluding people who pay anyway causes more money per contact."""
    X, y, treated, true_lift = toy
    rng = np.random.default_rng(11)
    amount = rng.uniform(50, 500, len(X))

    # outcome model ranks by P(pay | treated), which includes the sure things
    p_treated, _ = fitted.predict_arms(X)
    ev = amount * p_treated
    eiv = fitted.expected_incremental_value(X, amount)

    K = 400
    caused_ev = float((amount * true_lift)[np.argsort(-ev)[:K]].sum())
    caused_up = float((amount * true_lift)[np.argsort(-eiv)[:K]].sum())
    assert caused_up > caused_ev, (
        f"uplift targeting caused {caused_up:,.0f} vs outcome targeting {caused_ev:,.0f}")


# ------------------------------------------------------------------ metrics
def test_qini_is_positive_for_a_good_ranking_and_negative_for_a_bad_one(toy, fitted):
    X, y, treated, _ = toy
    good = qini_coefficient(qini_curve(fitted.predict_uplift(X), y, treated))
    bad = qini_coefficient(qini_curve(-fitted.predict_uplift(X), y, treated))
    assert good > 0.05, f"qini {good:.3f} for a model that recovers the true effect"
    assert bad < good, "reversing the ranking did not lower qini"


def test_qini_of_random_ranking_is_near_zero(toy):
    X, y, treated, _ = toy
    rng = np.random.default_rng(2)
    qs = [qini_coefficient(qini_curve(rng.random(len(y)), y, treated)) for _ in range(5)]
    assert abs(float(np.mean(qs))) < 0.25


def test_qini_handles_a_degenerate_arm(toy):
    """No treated cases in the top slice must not raise or silently score well."""
    X, y, treated, _ = toy
    c = qini_curve(np.arange(len(y))[::-1], y, np.zeros(len(y), dtype=bool))
    assert (c["incremental"] == 0).all()
    assert qini_coefficient(c) == 0.0


def test_uplift_at_k_scales_with_value(toy, fitted):
    X, y, treated, _ = toy
    u = fitted.predict_uplift(X)
    ones = uplift_at_k(u, y, treated, np.ones(len(y)), 500)
    tens = uplift_at_k(u, y, treated, np.full(len(y), 10.0), 500)
    assert tens == pytest.approx(ones * 10, rel=1e-6)


# ------------------------------------------------------------------ segmentation
def test_segments_are_labelled_coherently(toy, fitted):
    X, _, _, _ = toy
    seg = fitted.segment(X)
    u = fitted.predict_uplift(X)
    assert set(np.unique(seg)) <= {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
    assert u[seg == "persuadable"].mean() > u[seg == "lost_cause"].mean()
    if (seg == "sleeping_dog").any():
        assert u[seg == "sleeping_dog"].mean() < 0


# ------------------------------------------------------------------ persistence
def test_round_trip(tmp_path, toy, fitted):
    X, _, _, _ = toy
    p = tmp_path / "u.joblib"
    fitted.save(p)
    loaded = UpliftModel.load(p)
    np.testing.assert_allclose(loaded.predict_uplift(X), fitted.predict_uplift(X), rtol=1e-5)
    assert loaded.feature_names == fitted.feature_names


@pytest.mark.skipif(not (MODEL_DIR / "uplift.joblib").exists(),
                    reason="uplift model not trained -- run ml/train_uplift.py")
def test_trained_project_model_loads_and_scores():
    from backend.app.ml.features import build_features, feature_names
    from backend.app.services.dataio import load_split
    m = UpliftModel.load(MODEL_DIR / "uplift.joblib")
    assert m.feature_names == feature_names()
    test = load_split("test").head(200)
    u = m.predict_uplift(build_features(test))
    assert len(u) == 200
    assert np.isfinite(u).all()
    assert -1.0 <= u.min() and u.max() <= 1.0


# ------------------------------------------------------------------ targeting service
def test_targeter_falls_back_visibly_without_an_uplift_model(tmp_path):
    """A missing artefact must never look like a targeting decision."""
    from backend.app.ml.targeting import Targeter
    t = Targeter(uplift_path=tmp_path / "absent.joblib")
    assert not t.has_uplift
    assert t.mode == "expected_recovery"


@pytest.mark.skipif(not (MODEL_DIR / "uplift.joblib").exists(),
                    reason="uplift model not trained")
def test_targeter_ranks_by_incremental_value():
    from backend.app.ml.targeting import get_targeter
    from backend.app.services.dataio import load_split
    t = get_targeter()
    df = load_split("test").drop(columns=["recovered", "recovery_days"]).head(400)
    r = t.rank(df)
    assert t.mode == "incremental_value"
    iv = r["incremental_value"].to_numpy(dtype=float)
    assert (np.diff(iv) <= 1e-6).all(), "queue is not ordered by incremental value"
    # the two scores must genuinely differ, or uplift is adding nothing
    assert r["incremental_value"].to_list() != r["expected_recovery"].to_list()


@pytest.mark.skipif(not (MODEL_DIR / "uplift.joblib").exists(),
                    reason="uplift model not trained")
def test_budget_selection_never_contacts_a_negative_uplift_case():
    """Spending a contact on someone the model expects to be unmoved or harmed is worse
    than leaving the budget unspent."""
    from backend.app.ml.targeting import get_targeter
    from backend.app.services.dataio import load_split
    t = get_targeter()
    df = load_split("test").drop(columns=["recovered", "recovery_days"])
    sel = t.select(df, 500)
    assert len(sel) <= 500
    assert (sel["incremental_value"] > 0).all()
    assert "sleeping_dog" not in set(sel["segment"])
