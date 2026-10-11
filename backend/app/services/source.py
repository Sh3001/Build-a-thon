"""Read-only external payments, independent of the recovery-state database."""
from __future__ import annotations

import os
from dataclasses import asdict

from backend.app.models.enums import category_of
from backend.app.services.dataio import SqlSource, SqlSourceError


def kind() -> str:
    return (os.environ.get("RECOVERAI_SOURCE") or "sql").strip().lower()


def configured() -> bool:
    return bool(os.environ.get("RECOVERAI_SOURCE") or os.environ.get("RECOVERAI_SOURCE_URL") or os.environ.get("RECOVERAI_SOURCE_QUERY"))


def fetch(tenant_id: str):
    # A configured view belongs to one merchant. Multi-tenant deployments must expose
    # separately scoped source credentials/views; a request cannot choose a source URL.
    owner = os.environ.get("RECOVERAI_SOURCE_TENANT", "default")
    if tenant_id != owner:
        raise SqlSourceError("No payment source is configured for this tenant")
    try:
        return SqlSource.from_env().fetch()
    except SqlSourceError:
        raise
    except Exception as exc:
        raise SqlSourceError("External source is unavailable") from exc


def queue(tenant_id: str) -> tuple[list[dict], dict]:
    if kind() == "uci":
        if tenant_id != os.environ.get("RECOVERAI_SOURCE_TENANT", "default"):
            raise SqlSourceError("No source is configured for this tenant")
        from backend.app.services.uci import queue as uci_queue
        return uci_queue()
    if kind() != "sql":
        raise SqlSourceError("Unknown data source")
    result = fetch(tenant_id)
    rows = []
    for txn in result.transactions:
        row = txn.model_dump(mode="json")
        row.update(failure_category=category_of(txn.failure_code).value, status="pending",
                   recovery_probability=None, expected_recovery=None, risk_score=None,
                   amount_recovered=0.0, actions=[], retries=0, contacts=0)
        rows.append(row)
    # No raw rejected records or exception strings cross the API boundary.
    report = asdict(result.report)
    report.pop("reject_reasons", None)
    return rows, report


def status() -> dict:
    from backend.app.database.db import state_configured
    return {"source_configured": configured(), "storage_configured": state_configured(),
            "mode": "external" if configured() else "state" if state_configured() else "setup",
            "source_read_only": True, "source_kind": kind() if configured() else None,
            "message": ("UCI historical repayment data · Taiwan, September 2005." if kind() == "uci" else
                        "External payment source configured." if configured() else
                        "Connect an existing PostgreSQL database to view failed payments. "
                        "No local database or demo data is created.")}
