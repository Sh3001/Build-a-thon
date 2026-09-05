"""Drift monitoring.

A calibrated model is calibrated on the distribution it was trained on. Nothing in the
system noticed when that distribution moved, which means the failure mode was: the model
keeps returning confident probabilities, the probabilities stop meaning anything, and the
expected-value ranking quietly degrades with no alarm. A model that is wrong loudly is
recoverable; one that is wrong silently is not.

Four kinds of drift, because they fail at different times and need different responses:

**Data drift** -- the inputs moved. Detected immediately, no labels needed. Population
Stability Index per feature.

**Prediction drift** -- the output distribution moved. Also label-free, and it catches
things feature-level PSI misses: a shift in the *joint* distribution can leave every
marginal looking normal.

**Performance drift** -- the model got worse. The one that actually matters, and the one
you cannot measure until outcomes arrive, which here is up to thirty days later.

**Business drift** -- recovery rate or profit per intervention moved. Slowest, noisiest,
and the only one whose alarm anybody outside the team will care about.

PSI thresholds are the conventional 0.1 / 0.25. They are conventional, not derived, and
they are configuration for that reason.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class DriftSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    ALERT = "alert"


#: Conventional PSI bands. Widely used, not derived from anything, and configurable for
#: exactly that reason -- a threshold nobody can justify should at least be visible.
PSI_WARNING = 0.10
PSI_ALERT = 0.25


@dataclass(frozen=True)
class DriftResult:
    metric: str
    value: float
    severity: DriftSeverity
    baseline: float | None = None
    detail: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def firing(self) -> bool:
        return self.severity is not DriftSeverity.NONE

    def to_dict(self) -> dict:
        return {"metric": self.metric, "value": round(self.value, 6),
                "severity": self.severity.value, "baseline": self.baseline,
                "detail": self.detail, "at": self.at.isoformat()}


def population_stability_index(baseline: Sequence[float], current: Sequence[float],
                               bins: int = 10, epsilon: float = 1e-6) -> float:
    """PSI between two samples.

    Bin edges come from the *baseline* quantiles, not from the pooled data. Pooling would
    let the current sample move the edges and hide its own shift -- the classic way PSI is
    computed wrong.
    """
    b = sorted(float(x) for x in baseline)
    c = [float(x) for x in current]
    if len(b) < bins or not c:
        return 0.0

    edges = [b[min(int(i * len(b) / bins), len(b) - 1)] for i in range(1, bins)]
    edges = sorted(set(edges))
    if not edges:
        return 0.0

    def histogram(xs: Sequence[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for x in xs:
            lo, hi = 0, len(edges)
            while lo < hi:
                mid = (lo + hi) // 2
                if x <= edges[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            counts[lo] += 1
        n = max(len(xs), 1)
        return [k / n for k in counts]

    hb, hc = histogram(b), histogram(c)
    psi = 0.0
    for pb, pc in zip(hb, hc):
        pb, pc = max(pb, epsilon), max(pc, epsilon)
        psi += (pc - pb) * math.log(pc / pb)
    return float(psi)


def severity_for(psi: float, warn: float = PSI_WARNING,
                 alert: float = PSI_ALERT) -> DriftSeverity:
    if psi >= alert:
        return DriftSeverity.ALERT
    if psi >= warn:
        return DriftSeverity.WARNING
    return DriftSeverity.NONE


@dataclass
class DriftMonitor:
    """Holds a training-time baseline and compares live batches against it."""

    #: feature name -> the training sample. Held as raw values so the bin edges can be
    #: recomputed; storing pre-binned histograms would fix the binning at baseline time
    #: and make a later change of `bins` silently incomparable.
    feature_baseline: dict[str, list[float]] = field(default_factory=dict)
    prediction_baseline: list[float] = field(default_factory=list)
    #: Label-dependent baselines, populated once outcomes exist.
    performance_baseline: dict[str, float] = field(default_factory=dict)
    business_baseline: dict[str, float] = field(default_factory=dict)
    psi_warning: float = PSI_WARNING
    psi_alert: float = PSI_ALERT
    #: Relative degradation that fires a warning / alert on a performance metric.
    performance_warning: float = 0.05
    performance_alert: float = 0.15
    model_version: str = ""

    @classmethod
    def from_training(cls, features, predictions: Sequence[float],
                      model_version: str = "", max_rows: int = 20_000,
                      **kw) -> DriftMonitor:
        """Snapshot a training set. Subsampled: the baseline is a distribution, and
        20,000 rows describes one as well as two million while keeping the artefact small
        enough to store beside the model."""
        import numpy as np
        cols = {}
        n = len(features)
        idx = None
        if n > max_rows:
            idx = np.random.default_rng(0).choice(n, size=max_rows, replace=False)
        for col in getattr(features, "columns", []):
            series = features[col].to_numpy(dtype=float, na_value=0.0)
            cols[str(col)] = (series[idx] if idx is not None else series).tolist()
        preds = list(predictions)
        if idx is not None and len(preds) == n:
            preds = [preds[i] for i in idx]
        return cls(feature_baseline=cols, prediction_baseline=[float(p) for p in preds],
                   model_version=model_version, **kw)

    # ------------------------------------------------------------------ checks
    def check_data_drift(self, features, top_n: int = 10) -> list[DriftResult]:
        """PSI per feature against the training baseline, worst first."""
        results: list[DriftResult] = []
        for col, base in self.feature_baseline.items():
            if col not in getattr(features, "columns", []):
                results.append(DriftResult(
                    f"feature.{col}", float("inf"), DriftSeverity.ALERT,
                    detail="feature is missing from the live batch: the model is being "
                           "served a different schema than it was trained on"))
                continue
            current = features[col].to_numpy(dtype=float, na_value=0.0).tolist()
            psi = population_stability_index(base, current)
            results.append(DriftResult(
                f"feature.{col}", psi, severity_for(psi, self.psi_warning, self.psi_alert),
                detail=f"PSI against training baseline over {len(current)} rows"))
        results.sort(key=lambda r: -r.value)
        return results[:top_n]

    def check_prediction_drift(self, predictions: Sequence[float]) -> DriftResult:
        psi = population_stability_index(self.prediction_baseline, list(predictions))
        return DriftResult(
            "prediction.psi", psi, severity_for(psi, self.psi_warning, self.psi_alert),
            detail=f"P(recovery) distribution over {len(predictions)} predictions")

    def check_performance(self, y_true: Sequence[int],
                          y_pred: Sequence[float]) -> list[DriftResult]:
        """Discrimination and calibration against their training-time values.

        Only computable once outcomes are known -- up to a full recovery horizon after
        the decision. That lag is the reason the label-free checks above exist rather
        than being redundant with this one.
        """
        current = performance_metrics(y_true, y_pred)
        out: list[DriftResult] = []
        for metric, value in current.items():
            base = self.performance_baseline.get(metric)
            if base is None:
                out.append(DriftResult(f"performance.{metric}", value,
                                       DriftSeverity.NONE, detail="no baseline recorded"))
                continue
            # AUC and PR-AUC are better when higher; Brier and log-loss when lower.
            higher_is_better = metric in ("roc_auc", "pr_auc")
            degradation = (base - value) / abs(base) if higher_is_better \
                else (value - base) / abs(base) if base else 0.0
            severity = (DriftSeverity.ALERT if degradation >= self.performance_alert
                        else DriftSeverity.WARNING if degradation >= self.performance_warning
                        else DriftSeverity.NONE)
            out.append(DriftResult(
                f"performance.{metric}", value, severity, baseline=base,
                detail=f"{degradation:+.1%} against the training baseline "
                       f"({'lower' if higher_is_better else 'higher'} is worse)"))
        return out

    def check_business(self, current: dict[str, float]) -> list[DriftResult]:
        out: list[DriftResult] = []
        for metric, value in current.items():
            base = self.business_baseline.get(metric)
            if base is None or base == 0:
                out.append(DriftResult(f"business.{metric}", value, DriftSeverity.NONE,
                                       detail="no baseline recorded"))
                continue
            change = (value - base) / abs(base)
            severity = (DriftSeverity.ALERT if change <= -self.performance_alert
                        else DriftSeverity.WARNING if change <= -self.performance_warning
                        else DriftSeverity.NONE)
            out.append(DriftResult(f"business.{metric}", value, severity, baseline=base,
                                   detail=f"{change:+.1%} against the recorded baseline"))
        return out

    def report(self, features=None, predictions: Sequence[float] | None = None,
               y_true: Sequence[int] | None = None,
               business: dict[str, float] | None = None) -> dict:
        """Everything that can be checked with what is available. Whatever is absent is
        reported as not-checked rather than as passing -- 'we did not look' and 'we looked
        and it was fine' must not render identically."""
        results: list[DriftResult] = []
        checked: list[str] = []
        if features is not None:
            results += self.check_data_drift(features)
            checked.append("data")
        if predictions is not None:
            results.append(self.check_prediction_drift(predictions))
            checked.append("prediction")
        if y_true is not None and predictions is not None:
            results += self.check_performance(y_true, predictions)
            checked.append("performance")
        if business is not None:
            results += self.check_business(business)
            checked.append("business")

        firing = [r for r in results if r.firing]
        worst = DriftSeverity.NONE
        for r in results:
            if r.severity is DriftSeverity.ALERT:
                worst = DriftSeverity.ALERT
                break
            if r.severity is DriftSeverity.WARNING:
                worst = DriftSeverity.WARNING
        return {
            "model_version": self.model_version,
            "checked": checked,
            "not_checked": [k for k in ("data", "prediction", "performance", "business")
                            if k not in checked],
            "severity": worst.value,
            "firing": [r.to_dict() for r in firing],
            "all": [r.to_dict() for r in results],
            "thresholds": {"psi_warning": self.psi_warning, "psi_alert": self.psi_alert,
                           "performance_warning": self.performance_warning,
                           "performance_alert": self.performance_alert},
        }

    def to_dict(self) -> dict:
        return {"model_version": self.model_version,
                "feature_baseline": self.feature_baseline,
                "prediction_baseline": self.prediction_baseline,
                "performance_baseline": self.performance_baseline,
                "business_baseline": self.business_baseline}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DriftMonitor:
        return cls(feature_baseline={k: list(v) for k, v in
                                     (raw.get("feature_baseline") or {}).items()},
                   prediction_baseline=list(raw.get("prediction_baseline") or []),
                   performance_baseline=dict(raw.get("performance_baseline") or {}),
                   business_baseline=dict(raw.get("business_baseline") or {}),
                   model_version=raw.get("model_version", ""))


def performance_metrics(y_true: Sequence[int], y_pred: Sequence[float]) -> dict[str, float]:
    """ROC-AUC, PR-AUC, Brier and log loss. Ranking and calibration reported together,
    because a model can improve at one while getting worse at the other and reporting
    only one of them hides exactly that."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    y = [int(v) for v in y_true]
    p = [min(max(float(v), 1e-6), 1 - 1e-6) for v in y_pred]
    if len(set(y)) < 2:
        # A single-class batch makes AUC undefined. Returning 0.5 would look like a
        # useless model rather than an unmeasurable one.
        return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1])}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def expected_calibration_error(y_true: Sequence[int], y_pred: Sequence[float],
                               bins: int = 10) -> float:
    """ECE: mean |predicted - observed| across probability bins, weighted by bin size.

    Reported alongside Brier because they answer different questions. Brier mixes
    calibration and discrimination into one number; ECE isolates calibration, which is the
    property the expected-value ranking actually depends on.
    """
    y = [int(v) for v in y_true]
    p = [float(v) for v in y_pred]
    if not y:
        return 0.0
    total, n = 0.0, len(y)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        members = [(yi, pi) for yi, pi in zip(y, p)
                   if (pi > lo or (i == 0 and pi >= lo)) and pi <= hi]
        if not members:
            continue
        observed = sum(m[0] for m in members) / len(members)
        predicted = sum(m[1] for m in members) / len(members)
        total += len(members) / n * abs(observed - predicted)
    return float(total)


def reliability_diagram(y_true: Sequence[int], y_pred: Sequence[float],
                        bins: int = 10) -> list[dict]:
    """Bin-by-bin predicted vs observed, for plotting. Returned as data rather than a
    picture so the dashboard and a notebook can render the same numbers."""
    y, p = [int(v) for v in y_true], [float(v) for v in y_pred]
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        members = [(yi, pi) for yi, pi in zip(y, p)
                   if (pi > lo or (i == 0 and pi >= lo)) and pi <= hi]
        out.append({
            "bin": i, "low": round(lo, 3), "high": round(hi, 3), "n": len(members),
            "predicted": round(sum(m[1] for m in members) / len(members), 6) if members else None,
            "observed": round(sum(m[0] for m in members) / len(members), 6) if members else None,
        })
    return out
