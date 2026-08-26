"""Phase 3 -- serving. One scorer used by the agent, the API and the evaluation, so the
number on the dashboard is the number the model produced.

If no trained model is on disk the scorer degrades to a transparent per-cause prior and
says so via `model_version`. A missing artefact must never look like a prediction.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import MODEL_DIR, to_usd
from backend.app.ml.features import build_features
from backend.app.models.enums import FailureCategory, category_of
from backend.app.models.schemas import ScoreResult

#: Fallback only. Rough per-category priors, used when no model has been trained yet.
FALLBACK_PRIOR: dict[str, float] = {
    FailureCategory.TEMPORARY.value: 0.50,
    FailureCategory.CUSTOMER_ACTION.value: 0.40,
    FailureCategory.PERSISTENT.value: 0.07,
    FailureCategory.RISK_COMPLIANCE.value: 0.06,
}


class RecoveryScorer:
    """Thread-safe lazy loader around the XGBoost model and its calibrator."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        self._model = None
        self._scaler = None
        self._kind: str | None = None
        self._calibrator = None
        self._meta: dict = {}
        self._loaded = False

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Load whichever algorithm won selection at training time."""
        with self._lock:
            if self._loaded:
                return
            import joblib
            xgb_path = self.model_dir / "model.json"
            lin_path = self.model_dir / "linear.joblib"

            if xgb_path.exists():
                from xgboost import XGBClassifier
                m = XGBClassifier()
                m.load_model(xgb_path)
                self._model, self._kind = m, "xgboost"
            elif lin_path.exists():
                bundle = joblib.load(lin_path)
                self._model, self._scaler = bundle["model"], bundle["scaler"]
                self._kind = "logistic"

            if self._model is not None:
                cp = self.model_dir / "calibrator.joblib"
                if cp.exists():
                    self._calibrator = joblib.load(cp)
                mep = self.model_dir / "metadata.json"
                if mep.exists():
                    self._meta = json.loads(mep.read_text())
            self._loaded = True

    @property
    def is_trained(self) -> bool:
        self.load()
        return self._model is not None

    @property
    def model_version(self) -> str:
        self.load()
        if self._model is None:
            return "heuristic-fallback"
        return self._meta.get("model_version", f"{self._kind}-recovery")

    @property
    def metadata(self) -> dict:
        self.load()
        return dict(self._meta)

    # ------------------------------------------------------------------ scoring
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated P(recovery) for a frame of transactions."""
        self.load()
        if self._model is None:
            cats = df["failure_code"].map(lambda c: category_of(c).value)
            return cats.map(FALLBACK_PRIOR).fillna(0.25).to_numpy(dtype=float)
        X = build_features(df)
        if self._kind == "logistic":
            X = self._scaler.transform(X)
        raw = self._model.predict_proba(X)[:, 1]
        if self._calibrator is not None:
            raw = self._calibrator.predict(raw)
        return np.clip(raw, 0.0005, 0.9995)

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds recovery_probability / risk_score / expected_recovery columns."""
        out = df.copy()
        p = self.predict_proba(df)
        out["recovery_probability"] = np.round(p, 6)
        out["risk_score"] = np.round(1.0 - p, 6)
        if "amount_usd" not in out.columns:
            out["amount_usd"] = [to_usd(a, c) for a, c in zip(out["amount"], out["currency"])]
        out["expected_recovery"] = np.round(out["amount_usd"].to_numpy() * p, 2)
        return out

    def score(self, txn: dict) -> ScoreResult:
        """Single-transaction serving path."""
        p = float(self.predict_proba(pd.DataFrame([txn]))[0])
        amount_usd = float(txn.get("amount_usd") or to_usd(txn["amount"], txn.get("currency", "USD")))
        return ScoreResult(
            recovery_probability=round(p, 6),
            risk_score=round(1.0 - p, 6),
            expected_recovery=round(amount_usd * p, 2),
            model_version=self.model_version,
        )


#: Process-wide singleton -- loading XGBoost per request would dominate latency.
_default = RecoveryScorer()


def get_scorer() -> RecoveryScorer:
    return _default
