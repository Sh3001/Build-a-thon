"""Phase 3 -- held-out evaluation. This is the first and only time the test split is read.

Reports classification quality, calibration, and Recovery@K: if you can only work the top
K cases, how much of the recoverable revenue does ranking by expected value actually
capture? That is the metric that maps to a collections team's real constraint.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from backend.app.config import MODEL_DIR, RUN_DIR
from backend.app.ml.scorer import get_scorer
from backend.app.models.enums import category_of
from backend.app.services.dataio import load_split


def recovery_at_k(df: pd.DataFrame, ks: list[int]) -> list[dict]:
    """Rank by expected recovery; measure the recoverable revenue actually captured."""
    d = df.sort_values("expected_recovery", ascending=False).reset_index(drop=True)
    total_recoverable = float((d["amount_usd"] * d["recovered"]).sum())
    total_cases = int(d["recovered"].sum())
    rows = []
    for k in ks:
        if k > len(d):
            continue
        top = d.head(k)
        captured = float((top["amount_usd"] * top["recovered"]).sum())
        rows.append({
            "k": k,
            "share_of_queue": round(k / len(d), 4),
            "cases_recovered_in_topk": int(top["recovered"].sum()),
            "precision_at_k": round(float(top["recovered"].mean()), 4),
            "revenue_captured": round(captured, 2),
            "share_of_recoverable_revenue": round(captured / total_recoverable, 4)
                                            if total_recoverable else 0.0,
            "lift_vs_random": round((captured / total_recoverable) / (k / len(d)), 3)
                              if total_recoverable else None,
        })
    return rows


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() < 5:
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "predicted": round(float(p[m].mean()), 4),
                    "actual": round(float(y[m].mean()), 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate on the held-out test split.")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    df = load_split(a.split)
    y = df["recovered"].to_numpy()
    scorer = get_scorer()
    if not scorer.is_trained:
        print("  !! no trained model found -- run `python ml/train.py` first")
        return 1

    scored = scorer.score_frame(df.drop(columns=["recovered", "recovery_days"]))
    scored["recovered"] = y
    p = scored["recovery_probability"].to_numpy()

    # Threshold chosen on the *validation* split, never on test.
    val = load_split("val")
    pv = get_scorer().predict_proba(val.drop(columns=["recovered", "recovery_days"]))
    prec, rec, thr = precision_recall_curve(val["recovered"].to_numpy(), pv)
    f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best_thr = float(thr[int(np.argmax(f1s[:-1]))])

    yhat = (p >= best_thr).astype(int)
    metrics = {
        "split": a.split,
        "n": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
        "model_version": scorer.model_version,
        "threshold": round(best_thr, 4),
        "threshold_selected_on": "validation",
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "precision_at_0.5": round(float(precision_score(y, (p >= 0.5).astype(int), zero_division=0)), 4),
        "recall_at_0.5": round(float(recall_score(y, (p >= 0.5).astype(int), zero_division=0)), 4),
        "f1_at_0.5": round(float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)), 4),
    }
    metrics["recovery_at_k"] = recovery_at_k(scored, [50, 100, 250, 500, 1000])
    metrics["calibration"] = calibration_table(y, p)
    cats = scored["failure_code"].map(lambda x: category_of(x).value)
    metrics["by_category"] = {}
    for c in ("TEMPORARY", "CUSTOMER_ACTION", "PERSISTENT", "RISK_COMPLIANCE"):
        m = (cats == c).to_numpy()
        if m.sum():
            metrics["by_category"][c] = {
                "n": int(m.sum()),
                "actual_rate": round(float(y[m].mean()), 4),
                "predicted_rate": round(float(p[m].mean()), 4),
            }

    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    metrics["validation_reference"] = meta.get("validation", {})

    out = RUN_DIR / "model_evaluation.json"
    out.write_text(json.dumps(metrics, indent=2))

    print(f"  held-out {a.split}: {metrics['n']:,} rows, {metrics['positive_rate']:.1%} recovered\n")
    print(f"  ROC-AUC   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    {metrics['pr_auc']:.4f}")
    print(f"  Precision {metrics['precision']:.4f}   Recall {metrics['recall']:.4f}   "
          f"F1 {metrics['f1']:.4f}   (threshold {metrics['threshold']:.3f}, picked on validation)")
    print(f"  Brier     {metrics['brier']:.4f}")
    print("\n  Recovery@K -- rank by expected recovery")
    print(f"    {'K':>6} {'queue':>7} {'prec@K':>8} {'revenue':>12} {'of recoverable':>15} {'lift':>7}")
    for r in metrics["recovery_at_k"]:
        print(f"    {r['k']:>6} {r['share_of_queue']:>6.1%} {r['precision_at_k']:>8.3f} "
              f"${r['revenue_captured']:>11,.0f} {r['share_of_recoverable_revenue']:>14.1%} "
              f"{r['lift_vs_random']:>7.2f}x")
    print("\n  calibration (predicted vs actual, by decile)")
    for c in metrics["calibration"]:
        print(f"    n={c['n']:>4}  predicted {c['predicted']:.3f}   actual {c['actual']:.3f}")
    print(f"\n  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
