"""Train and evaluate the recovery model on REAL data (UCI delinquency events).

Same discipline as the synthetic track: customer-disjoint splits, two candidate
algorithms selected on validation, isotonic calibration fitted on a disjoint half, and
the test split read exactly once at the end.

Every number this prints comes from real customers and real outcomes. No simulation is
involved anywhere in this file.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.app.config import DATA_PROCESSED, MODEL_DIR, RUN_DIR, SEED

REAL_DIR = DATA_PROCESSED / "real"
OUT_DIR = MODEL_DIR / "real"

FEATURES = [
    "months_late",          # real delinquency depth -- the dominant signal
    "amount_usd",
    "credit_limit_twd",
    "utilisation",          # balance / limit
    "payment_ratio",        # what fraction of the bill they paid
    "paid_anything",
    "prior_late_months",
    "prior_late_rate",
    "prior_observations",
    "age",
    "education",
    "marriage",
    "month_index",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(REAL_DIR / f"{name}.csv")


def X(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype(float).fillna(0.0)


def recovery_at_k(df: pd.DataFrame, p: np.ndarray, ks: list[int]) -> list[dict]:
    d = df.copy()
    d["p"] = p
    d["ev"] = d["amount_usd"] * p
    d = d.sort_values("ev", ascending=False).reset_index(drop=True)
    total = float((d["amount_usd"] * d["recovered"]).sum())
    rows = []
    for k in ks:
        if k > len(d):
            continue
        top = d.head(k)
        cap = float((top["amount_usd"] * top["recovered"]).sum())
        rows.append({"k": k, "share_of_queue": round(k / len(d), 4),
                     "precision_at_k": round(float(top["recovered"].mean()), 4),
                     "revenue_captured": round(cap, 2),
                     "share_of_recoverable": round(cap / total, 4) if total else 0.0,
                     "lift_vs_random": round((cap / total) / (k / len(d)), 3) if total else None})
    return rows


def main() -> int:
    train, val, test = load("train"), load("val"), load("test")
    ytr, yv, yte = (d["recovered"].to_numpy() for d in (train, val, test))

    rng = np.random.default_rng(SEED)
    mask = rng.random(len(val)) < 0.5
    val_es, val_cal = val[mask], val[~mask]

    print(f"  train {len(train):,}   val/early-stop {len(val_es):,}   "
          f"val/calibration {len(val_cal):,}   test {len(test):,}")
    print(f"  real cure rate: train {ytr.mean():.1%}   test {yte.mean():.1%}\n")

    xgb = XGBClassifier(n_estimators=800, max_depth=4, learning_rate=0.04,
                        subsample=0.85, colsample_bytree=0.85, min_child_weight=8,
                        reg_lambda=2.0, eval_metric="logloss", early_stopping_rounds=60,
                        random_state=SEED, n_jobs=4, tree_method="hist")
    xgb.fit(X(train), ytr, eval_set=[(X(val_es), val_es["recovered"].to_numpy())], verbose=False)

    scaler = StandardScaler().fit(X(train))
    lr = LogisticRegression(max_iter=3000).fit(scaler.transform(X(train)), ytr)

    ycal = val_cal["recovered"].to_numpy()
    p_xgb = xgb.predict_proba(X(val_cal))[:, 1]
    p_lr = lr.predict_proba(scaler.transform(X(val_cal)))[:, 1]
    auc_xgb, auc_lr = roc_auc_score(ycal, p_xgb), roc_auc_score(ycal, p_lr)
    print(f"  validation ROC-AUC   xgboost {auc_xgb:.4f}   logistic {auc_lr:.4f}")

    chosen = "xgboost" if auc_xgb >= auc_lr else "logistic"
    print(f"  selected on validation: {chosen}\n")

    raw_cal = p_xgb if chosen == "xgboost" else p_lr
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, ycal)

    def predict(d: pd.DataFrame) -> np.ndarray:
        raw = (xgb.predict_proba(X(d))[:, 1] if chosen == "xgboost"
               else lr.predict_proba(scaler.transform(X(d)))[:, 1])
        return np.clip(calib.predict(raw), 0.0005, 0.9995)

    # threshold picked on validation, never on test
    pv = predict(val)
    prec, rec, thr = precision_recall_curve(yv, pv)
    f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best_thr = float(thr[int(np.argmax(f1s[:-1]))])

    pte = predict(test)
    yhat = (pte >= best_thr).astype(int)
    metrics = {
        "source": "UCI default of credit card clients (real)",
        "n_test": int(len(test)), "test_cure_rate": round(float(yte.mean()), 4),
        "algorithm": chosen,
        "candidates": {"xgboost": float(auc_xgb), "logistic": float(auc_lr)},
        "threshold": round(best_thr, 4), "threshold_selected_on": "validation",
        "roc_auc": round(float(roc_auc_score(yte, pte)), 4),
        "pr_auc": round(float(average_precision_score(yte, pte)), 4),
        "precision": round(float(precision_score(yte, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, yhat, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(yte, pte)), 4),
        "recovery_at_k": recovery_at_k(test, pte, [50, 100, 250, 500, 1000]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": chosen, "xgb": xgb, "lr": lr, "scaler": scaler,
                 "calibrator": calib, "features": FEATURES}, OUT_DIR / "model.joblib")
    (RUN_DIR / "real_model_evaluation.json").write_text(json.dumps(metrics, indent=2))

    print("  --- HELD-OUT TEST (real customers, real outcomes) ---")
    print(f"  ROC-AUC   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    {metrics['pr_auc']:.4f}")
    print(f"  Precision {metrics['precision']:.4f}   Recall {metrics['recall']:.4f}   "
          f"F1 {metrics['f1']:.4f}")
    print(f"  Brier     {metrics['brier']:.4f}")
    print("\n  Recovery@K (rank by expected recovery)")
    print(f"    {'K':>6} {'queue':>7} {'prec@K':>8} {'revenue':>12} {'of recoverable':>15} {'lift':>7}")
    for r in metrics["recovery_at_k"]:
        print(f"    {r['k']:>6} {r['share_of_queue']:>6.1%} {r['precision_at_k']:>8.3f} "
              f"${r['revenue_captured']:>11,.0f} {r['share_of_recoverable']:>14.1%} "
              f"{r['lift_vs_random']:>7.2f}x")
    imp = sorted(zip(FEATURES, (xgb.feature_importances_ if chosen == "xgboost"
                                else np.abs(lr.coef_[0]))), key=lambda x: -x[1])
    print("\n  top features")
    for f, v in imp[:8]:
        print(f"    {f:<22} {v:.4f}")
    print(f"\n  saved -> {OUT_DIR}/model.joblib and {RUN_DIR}/real_model_evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
