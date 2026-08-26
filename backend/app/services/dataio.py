"""Shared loading of the processed splits into plain dicts the simulator understands."""
from __future__ import annotations

import pandas as pd

from backend.app.config import DATA_PROCESSED, to_usd


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"{name}.csv")
    if "amount_usd" not in df.columns:
        df["amount_usd"] = [to_usd(a, c) for a, c in zip(df["amount"], df["currency"])]
    return df


def to_transactions(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts. The `recovered`/`recovery_days` labels are dropped: a live run must
    not be able to see the historical outcome of the case it is working."""
    drop = [c for c in ("recovered", "recovery_days") if c in df.columns]
    return df.drop(columns=drop).to_dict("records")
