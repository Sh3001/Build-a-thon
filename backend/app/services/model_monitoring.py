"""Optional drift checks against an explicitly supplied training baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path

from backend.app.ml.drift import DriftMonitor
from backend.app.ml.features import build_features
from backend.app.ml.scorer import get_scorer


def inspect(rows: list[dict]) -> dict:
    path = os.environ.get("RECOVERAI_DRIFT_BASELINE_FILE")
    if not path:
        return {"status": "not_configured", "checked": [],
                "not_checked": ["data", "prediction", "performance", "business"]}
    if len(rows) < 10:
        return {"status": "insufficient_data", "checked": []}
    try:
        import pandas as pd
        baseline = DriftMonitor.from_dict(json.loads(Path(path).read_text()))
        frame = pd.DataFrame(rows)
        scorer = get_scorer()
        predictions = None
        if (scorer.is_trained and baseline.prediction_baseline
                and baseline.model_version == scorer.model_version):
            predictions = scorer.predict_proba(frame)
        features = build_features(frame) if baseline.feature_baseline else None
        report = baseline.report(features=features, predictions=predictions)
        return {"status": "measured" if report["checked"] else "not_checked", **report}
    except Exception:
        return {"status": "unavailable", "checked": [],
                "message": "The configured model monitoring baseline could not be evaluated."}
