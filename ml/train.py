"""Phase 3 -- train the recovery-probability model.

Two details that matter more than the algorithm choice:

* **Calibration.** The agent ranks by `expected_recovery = amount x P(recovery)`, so a
  probability that is merely well-ordered is not enough -- it has to mean what it says.
  An isotonic calibrator is fitted after training and the Brier score is reported.
* **Disjoint validation halves.** Early stopping and calibration use different halves of
  the validation split, so the calibrator is not fitted on rounds selected using the same
  rows. The test split is never touched here.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.app.config import MODEL_DIR, SEED
from backend.app.ml.features import build_features, feature_names
from backend.app.services.dataio import load_split

MODEL_VERSION = "xgb-recovery-v1"


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the RecoverAI recovery model.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    a = ap.parse_args()

    train, val = load_split("train"), load_split("val")
    Xtr, ytr = build_features(train), train["recovered"].to_numpy()

    # Disjoint halves: one selects the number of rounds, the other fits the calibrator.
    rng = np.random.default_rng(a.seed)
    mask = rng.random(len(val)) < 0.5
    val_es, val_cal = val[mask], val[~mask]
    Xes, yes = build_features(val_es), val_es["recovered"].to_numpy()
    Xcal, ycal = build_features(val_cal), val_cal["recovered"].to_numpy()

    print(f"  train {len(train):,}   val(early-stop) {len(val_es):,}   val(calibration) {len(val_cal):,}")
    print(f"  positives: train {ytr.mean():.1%}")

    clf = XGBClassifier(
        n_estimators=a.n_estimators, max_depth=a.max_depth, learning_rate=a.learning_rate,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=6,
        reg_lambda=1.5, objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=60, random_state=a.seed, n_jobs=4, tree_method="hist",
    )
    clf.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
    print(f"  best iteration: {clf.best_iteration}  (of {a.n_estimators})")

    raw_cal = clf.predict_proba(Xcal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_cal, ycal)

    def report(name, X, y):
        raw = clf.predict_proba(X)[:, 1]
        cal = calibrator.predict(raw)
        print(f"  {name:<18} ROC-AUC {roc_auc_score(y, raw):.4f}   PR-AUC {average_precision_score(y, raw):.4f}"
              f"   Brier raw {brier_score_loss(y, raw):.4f} -> cal {brier_score_loss(y, cal):.4f}")
        return roc_auc_score(y, raw), average_precision_score(y, raw), brier_score_loss(y, cal)

    print("\n  --- validation (test split untouched) ---")
    report("val/early-stop", Xes, yes)
    v_auc, v_ap, v_brier = report("val/calibration", Xcal, ycal)

    # Does the gradient-boosted model actually earn its place over a linear one? This is
    # a selection, not a formality: whichever wins on validation is the model that ships.
    # Asserting "we used XGBoost" while a logistic regression scores higher would be a
    # claim the numbers do not support.
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(scaler.transform(Xtr), ytr)
    lr_raw_cal = lr.predict_proba(scaler.transform(Xcal))[:, 1]
    lr_auc = roc_auc_score(ycal, lr_raw_cal)
    print(f"  {'logistic baseline':<18} ROC-AUC {lr_auc:.4f}   (xgboost margin: {v_auc - lr_auc:+.4f})")

    chosen = "xgboost" if v_auc >= lr_auc else "logistic"
    print(f"\n  selected on validation: \033[1m{chosen}\033[0m")
    if chosen == "logistic":
        # Re-fit the calibrator against the winning model's own scores.
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(lr_raw_cal, ycal)
        v_auc, v_ap = lr_auc, average_precision_score(ycal, lr_raw_cal)
        v_brier = brier_score_loss(ycal, calibrator.predict(lr_raw_cal))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in ("model.json", "linear.joblib"):
        (MODEL_DIR / stale).unlink(missing_ok=True)
    if chosen == "xgboost":
        clf.save_model(MODEL_DIR / "model.json")
    else:
        joblib.dump({"model": lr, "scaler": scaler, "features": feature_names()},
                    MODEL_DIR / "linear.joblib")
    joblib.dump(calibrator, MODEL_DIR / "calibrator.joblib")

    # Importances must come from the model that actually shipped. For the linear model
    # that is |coefficient| on standardised features, which is directly comparable.
    if chosen == "xgboost":
        weights = clf.feature_importances_
    else:
        weights = np.abs(lr.coef_[0])
    imp = sorted(zip(feature_names(), weights), key=lambda x: -x[1])
    meta = {
        "model_version": f"{chosen}-recovery-v1",
        "algorithm": chosen,
        "algorithm_selected_on": "validation ROC-AUC",
        "candidates": {"xgboost": float(roc_auc_score(ycal, clf.predict_proba(Xcal)[:, 1])),
                       "logistic": float(lr_auc)},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": a.seed,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "best_iteration": int(clf.best_iteration),
        "features": feature_names(),
        "params": {"n_estimators": a.n_estimators, "max_depth": a.max_depth,
                   "learning_rate": a.learning_rate},
        "validation": {"roc_auc": float(v_auc), "pr_auc": float(v_ap), "brier": float(v_brier),
                       "logistic_roc_auc": float(lr_auc)},
        "top_features": [{"feature": f, "importance": float(v)} for f, v in imp[:15]],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  top features ({chosen})")
    for f, v in imp[:10]:
        print(f"    {f:<32} {v:.4f}")
    artefact = "model.json" if chosen == "xgboost" else "linear.joblib"
    print(f"\n  saved -> {MODEL_DIR}/{artefact}, calibrator.joblib, metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
