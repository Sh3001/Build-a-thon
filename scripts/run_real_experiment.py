"""Three-arm experiment on REAL delinquency cases.

    STATUS QUO   the outcome that ACTUALLY happened to these customers. Not simulated
                 at all -- it is the observed label. Note this is *not* a no-touch arm:
                 the issuer certainly dunned these accounts, we simply do not observe
                 how. It is therefore the strongest possible baseline -- "beat what is
                 already being done" -- rather than "beat doing nothing".

    BASELINE     a fixed dunning schedule applied to the same real cases.
    RECOVERAI    the policy-gated agent applied to the same real cases.

Where simulation enters, and only here: the *effect of intervening*. No public dataset
records interventions, so an uplift must be assumed. Rather than pick one number and
present it as fact, the effect size is a parameter and the script sweeps it, reporting
the break-even point at which the agent stops beating the status quo.

The uplift is applied in log-odds space on top of each case's REAL calibrated probability,
so a case the data says is hopeless stays hopeless: an intervention scales the odds, it
does not manufacture them.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from backend.app.config import ACTION_COST_USD, MODEL_DIR, RUN_DIR, SEED
from backend.app.services.results import CaseOutcome, bootstrap_incremental, summarize
from ml.train_real import FEATURES, REAL_DIR

#: Log-odds lift of a well-targeted intervention at the reference effect size (1.0).
#: Anchored on published industry reporting that smart recovery roughly doubles the
#: capture of a fixed dunning schedule; treated as an assumption and swept below.
BASE_LIFT = {
    "send_reminder": 0.45,
    "send_payment_link": 0.70,
    "retry_payment": 0.30,
}

#: Each additional contact to the same customer is worth less than the one before it.
#: Omitting this is not neutral -- it hands an indiscriminate strategy unlimited free
#: upside and guarantees that contacting everyone beats contacting selectively.
CONTACT_FATIGUE = 0.55

#: Probability per contact that the customer disengages (opts out, ignores, complains).
#: An opted-out customer cannot be recovered by any later contact, so over-contacting
#: destroys future value rather than merely wasting a message.
OPTOUT_PER_CONTACT = 0.07

#: Interventions cannot rescue deeply delinquent accounts. Taken from the REAL decay
#: curve in the source data: cure falls 25.9% -> 15.9% -> 4.2% -> 0% as months late rise.
DEPTH_DAMPING = {1: 1.0, 2: 1.0, 3: 0.55, 4: 0.20, 5: 0.15, 6: 0.08, 7: 0.0, 8: 0.0}

#: Beyond this, no automated action -- the account goes to collections/human review.
ESCALATE_MONTHS_LATE = 5
#: No automated recovery attempt above this exposure.
MAX_AUTO_USD = 5000.0


def apply_plan(p_base: float, actions: list[str], damp: float, effect: float,
               rng) -> tuple[float, bool]:
    """Combined lift of a plan, with diminishing returns and opt-out risk.

    Contacts are applied in order. The nth contact contributes CONTACT_FATIGUE**(n-1) of
    its nominal lift, and each carries an opt-out hazard that ends the sequence.
    Retries are not customer contacts and neither fatigue nor opt-out applies to them.
    """
    total, n_contacts, opted_out = 0.0, 0, False
    for a in actions:
        if a == "retry_payment":
            total += BASE_LIFT[a] * (0.0 if opted_out else 1.0)
            continue
        if opted_out:
            continue
        total += BASE_LIFT[a] * (CONTACT_FATIGUE ** n_contacts)
        n_contacts += 1
        if rng.random() < OPTOUT_PER_CONTACT:
            opted_out = True
    return _sigmoid(_logit(p_base) + effect * damp * total), opted_out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_model():
    b = joblib.load(MODEL_DIR / "real" / "model.joblib")
    def predict(df: pd.DataFrame) -> np.ndarray:
        X = df[b["features"]].astype(float).fillna(0.0)
        raw = (b["xgb"].predict_proba(X)[:, 1] if b["kind"] == "xgboost"
               else b["lr"].predict_proba(b["scaler"].transform(X))[:, 1])
        return np.clip(b["calibrator"].predict(raw), 0.0005, 0.9995)
    return predict


def _outcome(row, strategy, recovered, actions, cost, status, reason, escalated=False):
    from backend.app.models.enums import FailureCode, category_of
    code = FailureCode(row["failure_code"])
    return CaseOutcome(
        transaction_id=row["transaction_id"], customer_id=row["customer_id"],
        amount_usd=float(row["amount_usd"]), failure_code=code,
        failure_category=category_of(code), strategy=strategy,
        recovered=bool(recovered), amount_recovered=float(row["amount_usd"]) if recovered else 0.0,
        recovery_hours=None, retries=actions.count("retry_payment"),
        contacts=sum(1 for a in actions if a != "retry_payment"),
        actions=actions, cost=round(cost, 4), status=status, stop_reason=reason,
        escalated=escalated, recovery_probability=float(row.get("p_base", 0.0)),
        expected_recovery=round(float(row["amount_usd"]) * float(row.get("p_base", 0.0)), 2),
    )


def run_status_quo(df: pd.DataFrame) -> tuple[list[CaseOutcome], dict]:
    """Zero simulation: the observed label is the outcome."""
    outs = [_outcome(r, "status_quo", bool(r["recovered"]), [], 0.0,
                     "recovered" if r["recovered"] else "stopped",
                     "observed outcome in the source data")
            for _, r in df.iterrows()]
    for o in outs:
        o.passive_recovery = o.recovered      # not caused by us
    return outs, summarize(outs, "status_quo")


def run_baseline(df: pd.DataFrame, rng, effect: float) -> tuple[list[CaseOutcome], dict]:
    """A fixed schedule: contact then retry, on every case, regardless of cause."""
    outs = []
    for _, r in df.iterrows():
        actions = ["send_reminder", "retry_payment", "retry_payment"]
        damp = DEPTH_DAMPING.get(int(r["months_late"]), 0.0)
        p, _ = apply_plan(float(r["p_base"]), actions, damp, effect, rng)
        cost = sum(ACTION_COST_USD[a] for a in actions)
        won = rng.random() < p
        outs.append(_outcome(r, "baseline", won, actions, cost,
                             "recovered" if won else "exhausted",
                             "fixed dunning schedule exhausted"))
    return outs, summarize(outs, "baseline")


def run_agent(df: pd.DataFrame, rng, effect: float) -> tuple[list[CaseOutcome], dict]:
    """Policy-gated, value-targeted. Spends effort where it can change the outcome."""
    outs = []
    for _, r in df.iterrows():
        months = int(r["months_late"])
        amount, p_base = float(r["amount_usd"]), float(r["p_base"])
        damp = DEPTH_DAMPING.get(months, 0.0)

        # --- guardrails, evaluated before anything is spent ---------------------
        if months >= ESCALATE_MONTHS_LATE or amount > MAX_AUTO_USD:
            outs.append(_outcome(r, "recoverai", bool(r["recovered"]), ["escalate_case"],
                                 ACTION_COST_USD["escalate_case"], "escalated",
                                 f"{months} months late / ${amount:,.0f} exposure: human review",
                                 escalated=True))
            outs[-1].passive_recovery = bool(r["recovered"])
            continue

        ev = amount * p_base
        if ev < 1.0 or damp == 0.0:
            outs.append(_outcome(r, "recoverai", bool(r["recovered"]), [], 0.0, "stopped",
                                 f"expected recovery ${ev:.2f} does not justify a contact"))
            outs[-1].passive_recovery = bool(r["recovered"])
            continue

        # --- targeted plan ------------------------------------------------------
        # Someone who paid *something* is engaged: a link converts. Someone who paid
        # nothing needs a nudge first. Retry alone is reserved for shallow delinquency.
        if r["payment_ratio"] > 0.05:
            actions = ["send_payment_link"]
        elif months <= 2:
            actions = ["send_reminder", "retry_payment"]
        else:
            actions = ["send_reminder", "send_payment_link"]

        p, _ = apply_plan(p_base, actions, damp, effect, rng)
        cost = sum(ACTION_COST_USD[a] for a in actions)
        won = rng.random() < p
        outs.append(_outcome(r, "recoverai", won, actions, cost,
                             "recovered" if won else "stopped",
                             "payment collected" if won else "plan exhausted"))
    return outs, summarize(outs, "recoverai")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the three-arm experiment on real data.")
    ap.add_argument("--effect", type=float, default=1.0,
                    help="intervention effect size (1.0 = reference assumption)")
    ap.add_argument("--sweep", action="store_true", help="sweep the effect size")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    test = pd.read_csv(REAL_DIR / "test.csv")
    test["p_base"] = load_model()(test)

    sq_outs, sq = run_status_quo(test)
    print(f"  real held-out cases   {len(test):,}   "
          f"${test['amount_usd'].sum():,.0f} at risk (real amounts)")
    print(f"  STATUS QUO (observed) {sq['cases_recovered']:,} recovered "
          f"({sq['recovery_rate']:.1%}), ${sq['revenue_recovered']:,.0f}  <- not simulated\n")

    effects = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] if a.sweep else [a.effect]
    rows = []
    for e in effects:
        rng = np.random.default_rng(a.seed)
        b_outs, b = run_baseline(test, rng, e)
        rng = np.random.default_rng(a.seed)
        g_outs, g = run_agent(test, rng, e)
        inc_sq = g["revenue_recovered"] - sq["revenue_recovered"]
        inc_b = g["revenue_recovered"] - b["revenue_recovered"]
        rows.append({
            "effect": e,
            "baseline_recovered": b["revenue_recovered"], "agent_recovered": g["revenue_recovered"],
            "incremental_vs_status_quo": round(inc_sq, 2),
            "incremental_vs_baseline": round(inc_b, 2),
            "agent_rate": g["recovery_rate"], "baseline_rate": b["recovery_rate"],
            "agent_cost": g["total_cost"], "baseline_cost": b["total_cost"],
            "agent_contacts": g["total_contacts"], "baseline_contacts": b["total_contacts"],
            "agent_escalated": g["cases_escalated"],
            "roi_vs_status_quo": round(inc_sq / g["total_cost"], 1) if g["total_cost"] else None,
            "bootstrap": bootstrap_incremental(sq_outs, g_outs, n_boot=1500, seed=a.seed),
        })

    print(f"  {'effect':>7} {'baseline $':>12} {'agent $':>12} {'vs status quo':>15} "
          f"{'vs baseline':>13} {'agent rate':>11} {'ROI':>8}")
    for r in rows:
        print(f"  {r['effect']:>7.2f} ${r['baseline_recovered']:>11,.0f} "
              f"${r['agent_recovered']:>11,.0f} ${r['incremental_vs_status_quo']:>14,.0f} "
              f"${r['incremental_vs_baseline']:>12,.0f} {r['agent_rate']:>10.1%} "
              f"{str(r['roi_vs_status_quo']) + 'x':>8}")

    ref = next((r for r in rows if abs(r["effect"] - 1.0) < 1e-9), rows[-1])
    ci = ref["bootstrap"]["incremental_revenue"]
    print(f"\n  at the reference effect size (1.0):")
    print(f"    incremental vs status quo  ${ref['incremental_vs_status_quo']:,.0f}   "
          f"90% CI ${ci['p05']:,.0f} to ${ci['p95']:,.0f}")
    print(f"    customer contacts          agent {ref['agent_contacts']:,} vs "
          f"baseline {ref['baseline_contacts']:,}")
    print(f"    escalated to human         {ref['agent_escalated']:,}")
    print(f"    action cost                ${ref['agent_cost']:,.2f}")

    out = {"source": "UCI default of credit card clients (real)",
           "n_cases": int(len(test)), "at_risk_usd": round(float(test["amount_usd"].sum()), 2),
           "status_quo": sq, "sweep": rows}
    (RUN_DIR / "real_experiment.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  saved -> {RUN_DIR}/real_experiment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
