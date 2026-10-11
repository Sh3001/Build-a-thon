"""UCI browsing must stay historical, isolated, and independent of writable storage."""
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.api import deps
from backend.app.api.jobs import submit
from backend.app.main import app
from backend.app.profiles import validate
from backend.app.security.auth import AuthPolicy
from backend.app.services import source, uci
from backend.app.services.dataio import SqlSourceError


@pytest.fixture
def uci_source(tmp_path, monkeypatch):
    path = tmp_path / "uci.xls"
    path.write_bytes(b"test")
    monkeypatch.setenv("RECOVERAI_SOURCE", "uci")
    monkeypatch.setenv("RECOVERAI_UCI_PATH", str(path))
    monkeypatch.setenv("RECOVERAI_SOURCE_TENANT", "default")
    for name in ("RECOVERAI_DB", "RECOVERAI_DB_URL", "RECOVERAI_ALLOW_SQLITE"):
        monkeypatch.delenv(name, raising=False)
    frame = pd.DataFrame({"ID": [1, 2, 3, 4], "PAY_0": [2, 0, 1, 3],
                          "BILL_AMT1": [1000, 500, -10, 2000]})
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: frame.copy())
    uci._snapshot.cache_clear()
    old = deps.get_auth_policy()
    deps.set_auth_policy(AuthPolicy.from_env({"RECOVERAI_PROFILE": "testing"}))
    yield frame
    deps.set_auth_policy(old)
    deps.get_rate_limiter().reset()
    uci._snapshot.cache_clear()


def test_api_without_database(uci_source):
    client = TestClient(app)
    assert client.get("/api/health/ready").status_code == 200
    status = client.get("/api/source/status").json()
    assert status["source_kind"] == "uci"
    assert not status["storage_configured"]
    response = client.get("/api/recovery-queue?sort=amount_usd&limit=1")
    assert response.status_code == 200
    row = response.json()[0]
    assert row["transaction_id"] == "uci_4_2005_09"
    assert row["amount_usd"] == 62
    assert row["months_late"] == 3
    assert row["recovery_probability"] is None
    assert row["failure_code"] == "historical_delinquency"
    assert row["status"] == "historical"
    assert client.get("/api/recovery-queue?sort=amount_usd&limit=1&offset=1").json()[0]["customer_id"] == "uci_1"
    detail = client.get(f"/api/cases/{row['transaction_id']}").json()
    assert detail["case"] == row
    assert detail["audit_events"] == []
    assert client.get("/api/cases/nonexistent").status_code == 404
    report = client.get("/api/source/report").json()
    assert (report["fetched"], report["accepted"], report["excluded"]) == (4, 2, 2)
    assert report["model_monitoring"]["status"] == "not_applicable"
    assert "recovered" not in row


def test_isolation_and_execution_guard(uci_source):
    with pytest.raises(SqlSourceError):
        source.queue("other-tenant")
    with pytest.raises(HTTPException) as exc:
        submit({}, None, None)
    assert exc.value.status_code == 409
    rows, _ = source.queue("default")
    rows[0]["amount"] = -1
    assert source.queue("default")[0][0]["amount"] == 1000


@pytest.mark.parametrize("column,value", [("ID", 1), ("PAY_0", 99), ("BILL_AMT1", float("nan"))])
def test_malformed_file_rejected(uci_source, column, value):
    uci_source.loc[3, column] = value
    assert TestClient(app).get("/api/recovery-queue").status_code == 503


def test_missing_file_is_unavailable(uci_source, monkeypatch):
    monkeypatch.setenv("RECOVERAI_UCI_PATH", "/nonexistent/uci.xls")
    client = TestClient(app)
    assert client.get("/api/recovery-queue").status_code == 503
    assert client.get("/api/health/ready").status_code == 503


def test_authenticated_readonly_profile_supports_uci():
    assert validate({"RECOVERAI_PROFILE": "readonly", "RECOVERAI_SOURCE": "uci",
                     "RECOVERAI_JWT_SECRET": "test-secret-value"}) == []


def test_failed_refresh_preserves_existing_file(tmp_path, monkeypatch):
    from scripts import fetch_uci
    path = tmp_path / "uci.xls"
    path.write_bytes(b"existing valid data")
    monkeypatch.setattr("sys.argv", ["fetch_uci", "--refresh", "--output", str(path)])

    class Download:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b"invalid replacement"

    monkeypatch.setattr(fetch_uci.httpx, "stream", lambda *a, **k: Download())

    def invalid(*args):
        raise ValueError("Invalid spreadsheet")

    monkeypatch.setattr(fetch_uci, "_snapshot", invalid)
    with pytest.raises(ValueError, match="Invalid spreadsheet"):
        fetch_uci.main()
    assert path.read_bytes() == b"existing valid data"
    assert list(tmp_path.iterdir()) == [path]
