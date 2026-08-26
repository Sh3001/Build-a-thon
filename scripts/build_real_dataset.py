"""Build the project's dataset from REAL data.

Source: UCI "Default of Credit Card Clients" (Yeh & Lien, 2009) -- 30,000 real Taiwanese
credit-card customers with six months of real repayment status, bill amounts and payments
(April-September 2005). Downloaded from the UCI ML Repository.

    https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

What is real here and what is not:

REAL  - the customers, their credit limits, ages, and six months of payment behaviour
REAL  - the delinquency events: a customer who is >=1 month late on a real balance
REAL  - the amount at risk: the actual bill statement amount at that moment
REAL  - the outcome label: whether they were current the following month (a cure)
REAL  - therefore the base cure rate, and how it decays with delinquency depth

NOT   - any intervention. This dataset records what happened; it does not record what the
        issuer did about it. There is no "we sent a reminder" column anywhere, and no
        public dataset has one, because it requires the issuer's own operational logs.

That boundary is the whole point of keeping this file separate from the synthetic
generator: everything below is measured, and the simulated part is confined to the
*effect of intervening*, which is stated explicitly wherever it is used.

Each row becomes one recovery case in the project's schema. A customer contributes up to
five events (one per month transition), so splits are taken on customer_id.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import DATA_PROCESSED, SEED

SOURCE_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
              "default%20of%20credit%20card%20clients.xls")

#: New Taiwan dollar -> USD. A fixed rate; FX drift is out of scope.
TWD_TO_USD = 0.031

# Chronological order. Note the source names the most recent month PAY_0, not PAY_1.
PAY = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
BILL = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
PAID = ["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]
MONTH_NAME = ["apr", "may", "jun", "jul", "aug", "sep"]

#: Delinquency depth -> the failure code it most nearly corresponds to.
#: This is an interpretation, not a measurement: the source records months-late, not a
#: processor decline reason. The mapping is monotone in severity and nothing downstream
#: depends on the specific labels beyond their category.
def _failure_code(months_late: int) -> str:
    if months_late <= 1:
        return "insufficient_funds"
    if months_late == 2:
        return "insufficient_funds"
    if months_late == 3:
        return "multiple_declines"
    if months_late <= 5:
        return "multiple_declines"
    return "closed_account"


def load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n  curl -L -o {path} '{SOURCE_URL}'")
    return pd.read_excel(path, header=1)


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per real delinquency event, with its real outcome."""
    out = []
    for t in range(len(PAY) - 1):
        cur, nxt = df[PAY[t]], df[PAY[t + 1]]
        m = cur >= 1                                   # real delinquency
        sub = df[m]
        if sub.empty:
            continue
        bill = sub[BILL[t]].clip(lower=0)
        prior = [PAY[i] for i in range(t)]
        out.append(pd.DataFrame({
            "customer_id": "uci_" + sub["ID"].astype(str),
            "transaction_id": "txn_uci_" + sub["ID"].astype(str) + "_" + MONTH_NAME[t],
            "month_index": t,
            # --- real, observed ---------------------------------------------
            "months_late": sub[PAY[t]].astype(int),
            "amount_twd": bill,
            "amount_usd": (bill * TWD_TO_USD).round(2),
            "credit_limit_twd": sub["LIMIT_BAL"],
            "paid_that_month_twd": sub[PAID[t]],
            "age": sub["AGE"],
            "education": sub["EDUCATION"],
            "marriage": sub["MARRIAGE"],
            # prior delinquency history available at decision time
            "prior_late_months": (sub[prior] >= 1).sum(axis=1) if prior else 0,
            "prior_observations": len(prior),
            "recovered": (nxt[m] <= 0).astype(int),     # REAL outcome
        }))
    ev = pd.concat(out, ignore_index=True)
    ev = ev[ev["amount_twd"] > 0].reset_index(drop=True)

    # --- derived, still from real quantities -----------------------------------
    ev["utilisation"] = (ev["amount_twd"] / ev["credit_limit_twd"].clip(lower=1)).clip(0, 3)
    ev["payment_ratio"] = (ev["paid_that_month_twd"] / ev["amount_twd"].clip(lower=1)).clip(0, 2)
    ev["paid_anything"] = (ev["paid_that_month_twd"] > 0).astype(int)
    ev["prior_late_rate"] = np.where(ev["prior_observations"] > 0,
                                     ev["prior_late_months"] / ev["prior_observations"].clip(lower=1),
                                     0.0)
    ev["failure_code"] = ev["months_late"].map(_failure_code)
    ev["currency"] = "TWD"
    ev["amount"] = ev["amount_twd"].round(2)
    return ev


def split(ev: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Split on customer, never on row: one customer contributes up to five events."""
    def bucket(cid: str) -> float:
        return int(hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    b = ev["customer_id"].map(bucket)
    return {"train": ev[b < 0.70].reset_index(drop=True),
            "val": ev[(b >= 0.70) & (b < 0.85)].reset_index(drop=True),
            "test": ev[b >= 0.85].reset_index(drop=True)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the real-data splits from UCI.")
    ap.add_argument("--source", type=Path, default=Path("data/external/uci_credit.xls"))
    ap.add_argument("--out", type=Path, default=DATA_PROCESSED / "real")
    a = ap.parse_args()

    df = load_source(a.source)
    ev = extract_events(df)
    a.out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(a.out / "events.csv", index=False)

    parts = split(ev)
    for name, part in parts.items():
        part.to_csv(a.out / f"{name}.csv", index=False)

    print(f"  source        {len(df):,} real customers (UCI, Taiwan, 2005)")
    print(f"  events        {len(ev):,} real delinquency events "
          f"from {ev['customer_id'].nunique():,} customers")
    print(f"  at risk       NT${ev['amount_twd'].sum():,.0f}  "
          f"(~${ev['amount_usd'].sum():,.0f})")
    print(f"  cure rate     {ev['recovered'].mean():.1%}  (REAL observed outcome)\n")
    for name, part in parts.items():
        print(f"  {name:<6} {len(part):>6,} events  {part['customer_id'].nunique():>5,} customers  "
              f"cure {part['recovered'].mean():.1%}")
    overlap = set(parts["train"]["customer_id"]) & set(parts["test"]["customer_id"])
    print(f"  train/test customer overlap: {len(overlap)}")

    print("\n  REAL cure rate by delinquency depth (the empirical decay curve)")
    g = ev.groupby("months_late").agg(n=("recovered", "size"), cure=("recovered", "mean"))
    for d, r in g.iterrows():
        if r["n"] >= 20:
            print(f"    {int(d)} months late   n={int(r['n']):>6,}   cure {r['cure']:>6.1%}")
    print(f"\n  saved -> {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
