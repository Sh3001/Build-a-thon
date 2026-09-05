"""The assumption-free real-data result.

If a collections team can only work K cases, does ranking by expected recovery beat the
obvious alternatives? This uses ONLY real amounts and real observed outcomes -- there is
no intervention model anywhere in this file, so nothing here depends on an assumed
effect size. It is the part of the system that real data can fully validate.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from backend.app.config import MODEL_DIR, RUN_DIR
from ml.train_real import REAL_DIR

KS = [100, 250, 500, 1000]


def main() -> int:
    b = joblib.load(MODEL_DIR / "real" / "model.joblib")
    test = pd.read_csv(REAL_DIR / "test.csv")
    X = test[b["features"]].astype(float).fillna(0.0)
    raw = (b["xgb"].predict_proba(X)[:, 1] if b["kind"] == "xgboost"
           else b["lr"].predict_proba(b["scaler"].transform(X))[:, 1])
    test["p"] = np.clip(b["calibrator"].predict(raw), 1e-4, 1 - 1e-4)
    test["ev"] = test["amount_usd"] * test["p"]

    recoverable = float((test["amount_usd"] * test["recovered"]).sum())

    def captured(order: np.ndarray, k: int) -> float:
        top = test.loc[order[:k]]
        return float((top["amount_usd"] * top["recovered"]).sum()) / recoverable

    rng = np.random.default_rng(7)
    strategies = {
        "expected_recovery_model": test.sort_values("ev", ascending=False).index.to_numpy(),
        "biggest_amount_first": test.sort_values("amount_usd", ascending=False).index.to_numpy(),
        "freshest_delinquency_first": test.sort_values("months_late").index.to_numpy(),
    }
    results = {name: {f"top_{k}": round(captured(o, k), 4) for k in KS}
               for name, o in strategies.items()}
    results["random"] = {
        f"top_{k}": round(float(np.mean([captured(rng.permutation(test.index.to_numpy()), k)
                                         for _ in range(200)])), 4) for k in KS}

    print(f"  real held-out: {len(test):,} cases · ${test['amount_usd'].sum():,.0f} at risk · "
          f"${recoverable:,.0f} genuinely recoverable")
    print("  share of recoverable revenue captured by working only the top K:\n")
    print(f"  {'strategy':<30}" + "".join(f"{'top ' + str(k):>11}" for k in KS))
    for name, row in results.items():
        print(f"  {name:<30}" + "".join(f"{row[f'top_{k}']:>10.1%} " for k in KS))

    m = results["expected_recovery_model"]["top_250"]
    a = results["biggest_amount_first"]["top_250"]
    r = results["random"]["top_250"]
    print(f"\n  at K=250 ({250/len(test):.1%} of the queue): model {m:.1%} vs "
          f"amount-ranking {a:.1%} ({m/a:.2f}x) vs random {r:.1%} ({m/r:.2f}x)")

    out = {"source": "UCI default of credit card clients (real)",
           "n_cases": len(test), "recoverable_usd": round(recoverable, 2),
           "assumptions": "none -- real amounts and real observed outcomes only",
           "results": results}
    (RUN_DIR / "real_prioritization.json").write_text(json.dumps(out, indent=2))
    print(f"\n  saved -> {RUN_DIR}/real_prioritization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
