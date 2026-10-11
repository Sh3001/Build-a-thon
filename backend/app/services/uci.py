"""Historical UCI account snapshots, without a database or invented decline labels."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import ROOT
from backend.app.services.dataio import SqlSourceError

PAGE_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
TWD_TO_USD = 0.031  # Fixed illustrative rate, consistent with the offline UCI track.
COLUMNS = ["ID", "PAY_0", "BILL_AMT1"]


def source_path() -> Path:
    return Path(os.environ.get("RECOVERAI_UCI_PATH") or ROOT / "data/external/uci_credit.xls")


@lru_cache(maxsize=2)
def _snapshot(path: str, modified: int, size: int) -> tuple[list[dict], int]:
    if size > 20_000_000:
        raise ValueError("UCI file exceeds 20 MB")
    frame = pd.read_excel(path, header=1, usecols=COLUMNS)
    if frame.empty or len(frame) > 30000 or set(frame.columns) != set(COLUMNS):
        raise ValueError("Invalid UCI schema or row count")
    values = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Non-finite UCI values")
    if (values.ID <= 0).any() or (values.ID % 1 != 0).any() or values.ID.duplicated().any():
        raise ValueError("Invalid UCI account identifiers")
    if not values.PAY_0.between(-2, 9).all() or (values.PAY_0 % 1 != 0).any():
        raise ValueError("Invalid repayment status")
    eligible = values[(values.PAY_0 >= 1) & (values.BILL_AMT1 > 0)].sort_values("ID")
    rows = []
    for item in eligible.itertuples(index=False):
        identifier = str(int(item.ID))
        rows.append(dict(
            transaction_id=f"uci_{identifier}_2005_09", customer_id=f"uci_{identifier}",
            amount=float(item.BILL_AMT1), currency="TWD",
            amount_usd=round(float(item.BILL_AMT1) * TWD_TO_USD, 2),
            failure_code="historical_delinquency", failure_category="delinquency",
            root_cause=f"Recorded repayment delay: {int(item.PAY_0)} month(s). Gateway cause unknown.",
            status="historical", months_late=int(item.PAY_0), snapshot_date="2005-09",
            data_source="uci", recovery_probability=None, expected_recovery=None,
            risk_score=None, amount_recovered=0, actions=[], retries=0, contacts=0,
        ))
    return rows, len(frame)


def queue() -> tuple[list[dict], dict]:
    try:
        path = source_path().resolve()
        stat = path.stat()
        rows, customers = _snapshot(str(path), stat.st_mtime_ns, stat.st_size)
        # Copies protect the shared cached snapshot from filtering/sorting by callers.
        return [dict(row, actions=[]) for row in rows], dict(
            fetched=customers, accepted=len(rows), rejected=0,
            excluded=customers - len(rows), source="uci", snapshot_date="2005-09",
            source_url=PAGE_URL, license="CC BY 4.0",
            attribution="Yeh, I. (2009). Default of Credit Card Clients. UCI ML Repository.",
            total_balance_twd=round(sum(r["amount"] for r in rows), 2),
            usd_conversion_rate=TWD_TO_USD,
            note="Accounts at least one month late with a positive September bill. "
                 "Bill balances are not measured overdue amounts. USD values use a fixed illustrative rate. "
                 "No gateway causes, recovery predictions or intervention outcomes are available.",
        )
    except Exception as exc:
        raise SqlSourceError("UCI dataset unavailable or invalid; run python -m scripts.fetch_uci") from exc
