"""Uplift modelling -- rank by treatment effect, not by outcome probability.

The recovery model answers "who will pay?". For deciding *who to contact* that is the
wrong question, and measurably so: on real data, 44.8% of the top-250 cases ranked by
expected recovery paid on their own. Roughly half the contact budget goes to people who
needed nothing.

The right target is the conditional average treatment effect:

    uplift(x) = P(recover | treated, x) - P(recover | untreated, x)

which sorts the population into four groups, only one of which is worth spending on:

    persuadables  pay only if contacted        <- the entire value of a dunning programme
    sure things   pay either way               <- contacting them is pure cost
    lost causes   pay neither way              <- contacting them is pure cost
    sleeping dogs pay UNLESS contacted         <- contacting them LOSES money

The last group is real in collections: an ill-timed dunning message can provoke a dispute
or a cancellation from someone who would otherwise have quietly paid. An outcome model
cannot distinguish any of these; it ranks sure things at the very top.

Implemented as a **T-learner**: two independent outcome models, one per arm, differenced.
Chosen over a single-model (S-learner) approach because treatment effects here are small
relative to the outcome signal, and an S-learner tends to let the strong outcome features
dominate and shrink the treatment term toward zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class UpliftModel:
    """Two outcome models, differenced. `fit` needs a randomised treatment assignment."""
    treated_model: Any = None
    control_model: Any = None
    feature_names: list[str] = field(default_factory=list)
    n_treated: int = 0
    n_control: int = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray, treated: np.ndarray,
            seed: int = 20260822, **kw) -> "UpliftModel":
        """`treated` is a boolean mask of who received the intervention.

        Each model only ever sees its own arm, so neither can learn the treatment
        indicator as a shortcut.
        """
        from xgboost import XGBClassifier

        t, c = np.asarray(treated).astype(bool), ~np.asarray(treated).astype(bool)
        if t.sum() < 50 or c.sum() < 50:
            raise ValueError(f"need both arms populated (treated={t.sum()}, control={c.sum()})")

        params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
                      colsample_bytree=0.85, min_child_weight=10, reg_lambda=2.0,
                      eval_metric="logloss", random_state=seed, n_jobs=4,
                      tree_method="hist", **kw)
        self.treated_model = XGBClassifier(**params).fit(X[t], y[t])
        self.control_model = XGBClassifier(**params).fit(X[c], y[c])
        self.feature_names = list(X.columns)
        self.n_treated, self.n_control = int(t.sum()), int(c.sum())
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Per-case treatment effect. May be negative -- that is the point."""
        p_t = self.treated_model.predict_proba(X[self.feature_names])[:, 1]
        p_c = self.control_model.predict_proba(X[self.feature_names])[:, 1]
        return p_t - p_c

    def expected_incremental_value(self, X: pd.DataFrame,
                                   amount: np.ndarray) -> np.ndarray:
        """**The score to rank a recovery queue by.**

            amount x uplift(x)  =  expected revenue this contact CAUSES

        Ranking by uplift alone optimises percentage points and quietly selects the
        smallest balances -- on this data it picks cases averaging $206 against $917 for
        expected-value ranking, and loses badly on revenue despite finding more
        persuadable customers. The objective is money, so the score must carry the money.

        Contrast with the outcome model's `amount x P(recovery)`, which ranks a customer
        who was always going to pay at the very top. This subtracts them out.
        """
        return np.asarray(amount, dtype=float) * self.predict_uplift(X)

    def predict_arms(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (self.treated_model.predict_proba(X[self.feature_names])[:, 1],
                self.control_model.predict_proba(X[self.feature_names])[:, 1])

    def segment(self, X: pd.DataFrame, threshold: float = 0.02) -> np.ndarray:
        """Label each case persuadable / sure_thing / lost_cause / sleeping_dog."""
        u = self.predict_uplift(X)
        p_t, p_c = self.predict_arms(X)
        out = np.full(len(X), "lost_cause", dtype=object)
        out[u > threshold] = "persuadable"
        out[(u <= threshold) & (p_c > 0.5)] = "sure_thing"
        out[u < -threshold] = "sleeping_dog"
        return out

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump({"treated": self.treated_model, "control": self.control_model,
                     "features": self.feature_names,
                     "n_treated": self.n_treated, "n_control": self.n_control}, path)

    @classmethod
    def load(cls, path: Path) -> "UpliftModel":
        import joblib
        b = joblib.load(path)
        return cls(treated_model=b["treated"], control_model=b["control"],
                   feature_names=b["features"], n_treated=b["n_treated"],
                   n_control=b["n_control"])


# ---------------------------------------------------------------- evaluation
def qini_curve(score: np.ndarray, y: np.ndarray, treated: np.ndarray,
               value: np.ndarray | None = None, n_points: int = 50) -> pd.DataFrame:
    """Qini curve: incremental outcome from targeting the top-ranked fraction.

    At each cut-off, the incremental gain is the treated arm's result minus the control
    arm's result rescaled to the same population size -- so an arm being larger cannot
    flatter it. `value` weights each case (use the amount to get incremental revenue
    rather than incremental cases).
    """
    v = np.ones(len(y), dtype=float) if value is None else np.asarray(value, dtype=float)
    y, t = np.asarray(y, dtype=float), np.asarray(treated).astype(bool)
    order = np.argsort(-np.asarray(score, dtype=float))
    y, t, v = y[order], t[order], v[order]

    rows = []
    for frac in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(int(round(frac * len(y))), 1)
        yt, yc = y[:k][t[:k]], y[:k][~t[:k]]
        vt, vc = v[:k][t[:k]], v[:k][~t[:k]]
        nt, nc = len(yt), len(yc)
        if nt == 0 or nc == 0:
            rows.append({"fraction": frac, "n": k, "incremental": 0.0})
            continue
        # treated total minus control total scaled to the treated arm's size
        gain = float((yt * vt).sum() - (yc * vc).sum() * (nt / nc))
        rows.append({"fraction": frac, "n": k, "incremental": gain})
    return pd.DataFrame(rows)


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the Qini curve and the random-targeting diagonal, normalised.

    0 means no better than random; higher is better; negative means the ranking is
    actively worse than random.
    """
    x = curve["fraction"].to_numpy()
    y = curve["incremental"].to_numpy()
    total = y[-1]
    if total == 0:
        return 0.0
    random_line = total * x
    return float(np.trapezoid(y - random_line, x) / abs(total))


def uplift_at_k(score: np.ndarray, y: np.ndarray, treated: np.ndarray,
                value: np.ndarray, k: int) -> float:
    """Incremental value captured by treating only the top k by `score`."""
    order = np.argsort(-np.asarray(score, dtype=float))[:k]
    y, t = np.asarray(y, dtype=float)[order], np.asarray(treated).astype(bool)[order]
    v = np.asarray(value, dtype=float)[order]
    nt, nc = int(t.sum()), int((~t).sum())
    if nt == 0 or nc == 0:
        return 0.0
    return float((y[t] * v[t]).sum() - (y[~t] * v[~t]).sum() * (nt / nc))
