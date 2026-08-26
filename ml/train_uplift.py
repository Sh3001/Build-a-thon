"""Train and evaluate the uplift model.

Training an uplift model requires observing outcomes under BOTH treatment and no
treatment. Real data cannot supply that -- no public dataset records interventions -- so
this trains inside the simulated environment, which is a genuine randomised experiment:
the control arm and the agent arm are run over one population with shared latents.

The experimental design is deliberately realistic. Each training case is randomly assigned
to exactly one arm and only that arm's outcome is observed, exactly as in a live holdout.
The counterfactual is used only at *evaluation* time, to measure how well the model
recovered an effect it was never shown.

Headline question: given a fixed contact budget, does targeting by predicted uplift cause
more recovery than targeting by expected value?
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.agents.runner import run_agent_batch
from backend.app.config import MODEL_DIR, RUN_DIR, SEED
from backend.app.ml.features import build_features
from backend.app.ml.scorer import get_scorer
from backend.app.ml.uplift import (
    UpliftModel, qini_coefficient, qini_curve, uplift_at_k,
)
from backend.app.services.control import run_control
from backend.app.services.dataio import load_split, to_transactions
from simulation.payment_gateway import PaymentGateway

OUT = MODEL_DIR / "uplift.joblib"


def observe(txns: list[dict], treated: np.ndarray, seed: int) -> np.ndarray:
    """Run each case through its assigned arm only. Returns the observed outcome."""
    y = np.zeros(len(txns), dtype=int)
    t_idx = [i for i, t in enumerate(treated) if t]
    c_idx = [i for i, t in enumerate(treated) if not t]

    if t_idx:
        outs, _, _ = run_agent_batch([txns[i] for i in t_idx],
                                     PaymentGateway(seed=seed), scorer=get_scorer())
        for i, o in zip(t_idx, outs):
            y[i] = int(o.recovered)
    if c_idx:
        outs, _ = run_control([txns[i] for i in c_idx], PaymentGateway(seed=seed))
        for i, o in zip(c_idx, outs):
            y[i] = int(o.recovered)
    return y


def counterfactual(txns: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Both arms for every case. Evaluation only -- never available in production."""
    a_outs, _, _ = run_agent_batch(txns, PaymentGateway(seed=seed), scorer=get_scorer())
    c_outs, _ = run_control(txns, PaymentGateway(seed=seed))
    return (np.array([int(o.recovered) for o in a_outs]),
            np.array([int(o.recovered) for o in c_outs]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    train_df, test_df = load_split("train"), load_split("test")
    train_txns = to_transactions(train_df)
    test_txns = to_transactions(test_df)

    # ---- training: randomised assignment, one observed outcome per case -------
    treated = rng.random(len(train_txns)) < 0.5
    print(f"  training experiment: {len(train_txns):,} cases  "
          f"({int(treated.sum()):,} treated / {int((~treated).sum()):,} control)")
    y = observe(train_txns, treated, SEED)
    print(f"  observed recovery: treated {y[treated].mean():.1%}  "
          f"control {y[~treated].mean():.1%}  "
          f"(naive ATE {y[treated].mean() - y[~treated].mean():+.1%})")

    Xtr = build_features(train_df)
    model = UpliftModel().fit(Xtr, y, treated, seed=SEED)
    model.save(OUT)
    print(f"  saved -> {OUT}\n")

    # ---- evaluation: the counterfactual the model never saw -------------------
    print("  evaluating on the held-out split (both arms run, for ground truth)...")
    y_t, y_c = counterfactual(test_txns, SEED)
    Xte = build_features(test_df)
    amount = test_df["amount_usd"].to_numpy(dtype=float)

    u_pred = model.predict_uplift(Xte)
    scorer = get_scorer()
    p_out = scorer.predict_proba(test_df.drop(columns=["recovered", "recovery_days"]))
    ev = amount * p_out
    true_uplift = (y_t - y_c).astype(float)

    print(f"  true ATE on test: {true_uplift.mean():+.1%}   "
          f"predicted mean uplift: {u_pred.mean():+.1%}")
    corr = float(np.corrcoef(u_pred, true_uplift)[0, 1])
    print(f"  corr(predicted uplift, realised uplift): {corr:+.3f}\n")

    # ---- Qini, under a realistic single-observation regime --------------------
    assign = rng.random(len(test_txns)) < 0.5
    y_obs = np.where(assign, y_t, y_c)
    eiv = model.expected_incremental_value(Xte, amount)      # amount x uplift
    strategies = {
        "amount x uplift (new)": eiv,
        "expected value (current)": ev,
        "uplift only (pp)": u_pred,
        "amount only": amount,
        "random": rng.random(len(test_txns)),
    }
    print(f"  {'targeting strategy':<26}{'Qini':>8}{'uplift@250':>13}{'uplift@500':>13}")
    qini = {}
    for name, s in strategies.items():
        q = qini_coefficient(qini_curve(s, y_obs, assign, amount))
        u250 = uplift_at_k(s, y_obs, assign, amount, 250)
        u500 = uplift_at_k(s, y_obs, assign, amount, 500)
        qini[name] = {"qini": round(q, 4), "uplift_at_250": round(u250, 2),
                      "uplift_at_500": round(u500, 2)}
        print(f"  {name:<26}{q:>8.3f}{u250:>13,.0f}{u500:>13,.0f}")

    # ---- the decision that matters: a fixed contact budget --------------------
    # Exact, because the simulator gives both outcomes for every case.
    print(f"\n  MONEY CAUSED under a fixed contact budget "
          f"(exact, using the counterfactual)")
    print(f"  {'budget':>8}" + "".join(f"{n:>22}" for n in
                                       ("by amount x uplift", "by expected value", "by random")))
    budget_rows = []
    for k in (100, 250, 500, 1000, len(test_txns)):
        row = {"budget": k}
        for name, s in (("uplift", eiv), ("ev", ev), ("random", rng.random(len(test_txns)))):
            top = np.argsort(-s)[:k]
            mask = np.zeros(len(test_txns), dtype=bool)
            mask[top] = True
            caused = float((amount * (np.where(mask, y_t, y_c) - y_c)).sum())
            row[name] = round(caused, 2)
        budget_rows.append(row)
        print(f"  {k:>8}" + "".join(f"${row[n]:>21,.0f}" for n in ("uplift", "ev", "random")))

    best = budget_rows[1]
    gain = best["uplift"] - best["ev"]
    print(f"\n  at a 250-contact budget: amount x uplift causes ${gain:+,.0f} vs "
          f"expected-value targeting ({gain / max(best['ev'], 1):+.1%})")

    # ---- segments -------------------------------------------------------------
    seg = model.segment(Xte)
    print("\n  population segmented by predicted treatment effect:")
    for name in ("persuadable", "sure_thing", "lost_cause", "sleeping_dog"):
        m = seg == name
        if m.sum():
            print(f"    {name:<14} {int(m.sum()):>5} cases  "
                  f"realised uplift {true_uplift[m].mean():+.1%}  "
                  f"${amount[m].sum():>10,.0f} at risk")

    (RUN_DIR / "uplift_evaluation.json").write_text(json.dumps({
        "n_train": len(train_txns), "n_test": len(test_txns),
        "true_ate": round(float(true_uplift.mean()), 4),
        "predicted_mean_uplift": round(float(u_pred.mean()), 4),
        "correlation_with_realised": round(corr, 4),
        "qini": qini, "budget": budget_rows,
        "segments": {n: int((seg == n).sum()) for n in
                     ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")},
    }, indent=2))
    print(f"\n  saved -> {RUN_DIR}/uplift_evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
