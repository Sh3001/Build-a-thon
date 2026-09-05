"""Model lifecycle and drift detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.drift import (
    DriftMonitor,
    DriftSeverity,
    expected_calibration_error,
    population_stability_index,
    reliability_diagram,
)
from backend.app.ml.registry import (
    ModelRegistry,
    ModelStatus,
    RegistryError,
)


@pytest.fixture
def registry(tmp_path):
    art = tmp_path / "artefacts"
    art.mkdir()
    (art / "metadata.json").write_text("{}")
    (art / "calibrator.joblib").write_bytes(b"stub")
    return ModelRegistry(tmp_path), art


# ---------------------------------------------------------------- registry
def test_a_version_is_immutable(registry):
    reg, art = registry
    v = reg.register("recovery", art, metrics={"roc_auc": 0.8, "brier": 0.1})
    with pytest.raises(RegistryError, match="immutable"):
        reg.register("recovery", art, version=v.version)


def test_promotion_is_gated_on_quality(registry):
    reg, art = registry
    bad = reg.register("recovery", art, metrics={"roc_auc": 0.51, "brier": 0.40})
    reg.transition(bad.version, ModelStatus.STAGING)
    with pytest.raises(RegistryError, match="failed the promotion gate"):
        reg.promote(bad.version)
    assert reg.production() is None


def test_a_forced_promotion_records_what_it_overrode(registry):
    reg, art = registry
    bad = reg.register("recovery", art, metrics={"roc_auc": 0.51, "brier": 0.40})
    reg.transition(bad.version, ModelStatus.STAGING)
    rec = reg.promote(bad.version, actor="alice", force=True, reason="incident rollback")
    assert "FORCED PROMOTION" in rec.notes
    assert "incident rollback" in rec.notes


def test_a_material_regression_against_the_incumbent_is_refused(registry):
    reg, art = registry
    good = reg.register("recovery", art, metrics={"roc_auc": 0.85, "brier": 0.10})
    reg.transition(good.version, ModelStatus.STAGING)
    reg.promote(good.version)
    worse = reg.register("recovery", art, metrics={"roc_auc": 0.70, "brier": 0.12})
    reg.transition(worse.version, ModelStatus.STAGING)
    with pytest.raises(RegistryError, match="regresses"):
        reg.promote(worse.version)


def test_a_small_regression_is_allowed(registry):
    """A model may trade a little ranking for better calibration."""
    reg, art = registry
    a = reg.register("recovery", art, metrics={"roc_auc": 0.800, "brier": 0.14})
    reg.transition(a.version, ModelStatus.STAGING)
    reg.promote(a.version)
    b = reg.register("recovery", art, metrics={"roc_auc": 0.795, "brier": 0.09})
    reg.transition(b.version, ModelStatus.STAGING)
    reg.promote(b.version)
    assert reg.production().version == b.version


def test_promotion_retires_the_incumbent(registry):
    reg, art = registry
    a = reg.register("recovery", art, metrics={"roc_auc": 0.80, "brier": 0.10})
    reg.transition(a.version, ModelStatus.STAGING)
    reg.promote(a.version)
    b = reg.register("recovery", art, metrics={"roc_auc": 0.82, "brier": 0.10})
    reg.transition(b.version, ModelStatus.STAGING)
    reg.promote(b.version)
    assert reg.get(a.version).status == ModelStatus.RETIRED.value
    assert len(reg.list(ModelStatus.PRODUCTION)) == 1


def test_illegal_lifecycle_transitions_are_refused(registry):
    reg, art = registry
    v = reg.register("recovery", art, metrics={"roc_auc": 0.80, "brier": 0.10})
    with pytest.raises(RegistryError, match="not a legal transition"):
        reg.transition(v.version, ModelStatus.PRODUCTION)    # skipping STAGING


def test_a_retired_model_can_be_rolled_back_to_staging(registry):
    """Rollback is a real operation; a registry that cannot roll back gets worked around."""
    reg, art = registry
    v = reg.register("recovery", art, metrics={"roc_auc": 0.80, "brier": 0.10})
    reg.transition(v.version, ModelStatus.STAGING)
    reg.transition(v.version, ModelStatus.RETIRED)
    assert reg.transition(v.version, ModelStatus.STAGING).status == ModelStatus.STAGING.value


def test_loading_production_refuses_to_serve_an_unapproved_model(registry):
    """A production process that quietly serves whatever is newest has a registry for
    decoration."""
    reg, art = registry
    v = reg.register("recovery", art, metrics={"roc_auc": 0.90, "brier": 0.08})
    reg.transition(v.version, ModelStatus.STAGING)
    assert reg.load_production(strict=True) is None
    assert reg.load_production(strict=False)[0].version == v.version


def test_the_feature_schema_hash_is_order_sensitive():
    """A reordered feature matrix is a different matrix, and a model served against one
    produces plausible nonsense rather than an error."""
    assert ModelRegistry.feature_schema_hash(["a", "b"]) \
        != ModelRegistry.feature_schema_hash(["b", "a"])


# ---------------------------------------------------------------- drift
def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000).tolist()
    b = rng.normal(0, 1, 5000).tolist()
    assert population_stability_index(a, b) < 0.1


def test_psi_fires_on_a_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000).tolist()
    b = rng.normal(2.5, 1, 5000).tolist()
    assert population_stability_index(a, b) > 0.25


def test_bin_edges_come_from_the_baseline_not_the_pooled_data():
    """Pooling lets the current sample move the edges and hide its own shift -- the
    classic way PSI is computed wrong."""
    base = list(np.linspace(0, 1, 1000))
    shifted = list(np.linspace(10, 11, 1000))
    assert population_stability_index(base, shifted) > 1.0


def test_a_missing_feature_is_an_alert_not_a_pass():
    """The model is being served a different schema than it was trained on."""
    monitor = DriftMonitor(feature_baseline={"amount": [1.0] * 100,
                                             "tenure": [2.0] * 100})
    results = monitor.check_data_drift(pd.DataFrame({"amount": [1.0] * 50}))
    missing = next(r for r in results if r.metric == "feature.tenure")
    assert missing.severity is DriftSeverity.ALERT
    assert "missing" in missing.detail


def test_unchecked_is_not_reported_as_passing():
    """"We did not look" and "we looked and it was fine" must not render identically."""
    monitor = DriftMonitor(feature_baseline={"amount": [1.0] * 100})
    report = monitor.report(features=pd.DataFrame({"amount": [1.0] * 50}))
    assert "performance" in report["not_checked"]
    assert "business" in report["not_checked"]
    assert report["severity"] == "none"


def test_performance_degradation_fires():
    monitor = DriftMonitor(performance_baseline={"roc_auc": 0.85, "brier": 0.10})
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 2000)
    noise = rng.random(2000)                    # a model that has stopped ranking
    results = {r.metric: r for r in monitor.check_performance(y, noise)}
    assert results["performance.roc_auc"].severity is DriftSeverity.ALERT


def test_business_drift_fires_on_a_fall_not_a_rise():
    monitor = DriftMonitor(business_baseline={"recovery_rate": 0.44})
    fell = {r.metric: r for r in monitor.check_business({"recovery_rate": 0.30})}
    rose = {r.metric: r for r in monitor.check_business({"recovery_rate": 0.60})}
    assert fell["business.recovery_rate"].severity is DriftSeverity.ALERT
    assert rose["business.recovery_rate"].severity is DriftSeverity.NONE


def test_calibration_error_is_low_for_a_calibrated_predictor():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_calibration_error_is_high_for_an_overconfident_predictor():
    rng = np.random.default_rng(3)
    p = np.clip(rng.uniform(0.05, 0.95, 20000) * 1.6, 0, 1)
    y = (rng.random(20000) < p / 1.6).astype(int)
    assert expected_calibration_error(y, p) > 0.10


def test_the_reliability_diagram_covers_every_bin():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 1000)
    y = (rng.random(1000) < p).astype(int)
    rows = reliability_diagram(y, p, bins=10)
    assert len(rows) == 10
    assert sum(r["n"] for r in rows) == 1000
