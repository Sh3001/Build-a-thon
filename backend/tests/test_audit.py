"""Phase 7 tests -- the audit log's tamper-evidence and append-only guarantees."""
from __future__ import annotations

import pytest

from backend.app.audit.store import FIELDS, AuditStore
from backend.app.database.store import CaseStore
from backend.app.models.enums import FailureCategory, FailureCode
from backend.app.models.schemas import AuditEvent
from backend.app.services.results import CaseOutcome


@pytest.fixture
def store(tmp_path):
    s = AuditStore(tmp_path / "audit.db")
    yield s
    s.close()


def ev(txn="t1", n=0, **kw):
    base = dict(transaction_id=txn, customer_id="c1", agent_decision="select_intervention",
                reason=f"step {n}", action="retry_payment", policy_result="approve",
                rules_fired="R-COOLDOWN", recovery_probability=0.5, expected_recovery=50.0,
                amount_recovered=0.0, attempt_count=n)
    return AuditEvent(**{**base, **kw})


# ------------------------------------------------------------------ chaining
def test_a_clean_chain_verifies(store):
    store.append_many(ev(n=i) for i in range(20))
    store.commit()
    ok, bad = store.verify()
    assert ok and bad is None
    assert store.count() == 20


def test_editing_a_row_breaks_the_chain(store):
    store.append_many(ev(n=i) for i in range(10))
    store.commit()
    store.conn.execute("UPDATE audit_log SET reason = 'forged' WHERE seq = 4")
    store.commit()
    ok, bad = store.verify()
    assert not ok and bad == 4, "tampering was not detected at the right row"


def test_deleting_a_row_breaks_the_chain(store):
    store.append_many(ev(n=i) for i in range(10))
    store.commit()
    store.conn.execute("DELETE FROM audit_log WHERE seq = 5")
    store.commit()
    ok, bad = store.verify()
    assert not ok


def test_changing_an_amount_breaks_the_chain(store):
    """Financial fields are covered by the hash, not just free text."""
    store.append_many(ev(n=i) for i in range(6))
    store.commit()
    store.conn.execute("UPDATE audit_log SET amount_recovered = '999999' WHERE seq = 2")
    store.commit()
    assert store.verify() == (False, 2)


def test_restoring_the_original_value_repairs_verification(store):
    """Proves the check is a real hash re-derivation, not a stored 'tampered' flag."""
    store.append_many(ev(n=i) for i in range(5))
    store.commit()
    original = store.conn.execute("SELECT reason FROM audit_log WHERE seq = 3").fetchone()["reason"]
    store.conn.execute("UPDATE audit_log SET reason = 'x' WHERE seq = 3")
    store.commit()
    assert not store.verify()[0]
    store.conn.execute("UPDATE audit_log SET reason = ? WHERE seq = 3", (original,))
    store.commit()
    assert store.verify()[0]


def test_every_row_links_to_its_predecessor(store):
    store.append_many(ev(n=i) for i in range(8))
    store.commit()
    rows = [dict(r) for r in store.conn.execute("SELECT * FROM audit_log ORDER BY seq")]
    assert rows[0]["prev_hash"] == "0" * 64
    for prev, cur in zip(rows, rows[1:]):
        assert cur["prev_hash"] == prev["row_hash"]


# ------------------------------------------------------------------ append-only
def test_the_store_exposes_no_way_to_rewrite_history():
    """'Never overwrite previous audit events' is enforced by the API surface."""
    api = {m for m in dir(AuditStore) if not m.startswith("_")}
    for forbidden in ("update", "delete", "edit", "remove", "truncate", "clear"):
        assert not any(forbidden in m for m in api), f"AuditStore exposes {forbidden}"


def test_appending_never_mutates_earlier_rows(store):
    store.append_many(ev(n=i) for i in range(5))
    store.commit()
    before = [dict(r) for r in store.conn.execute("SELECT * FROM audit_log ORDER BY seq")]
    store.append_many(ev(n=i) for i in range(5, 10))
    store.commit()
    after = [dict(r) for r in store.conn.execute(
        "SELECT * FROM audit_log ORDER BY seq LIMIT 5")]
    assert before == after


# ------------------------------------------------------------------ content
def test_every_required_field_is_persisted(store):
    """The spec's audit schema, checked field by field."""
    required = {"timestamp", "transaction_id", "agent_decision", "reason", "risk_score",
                "recovery_probability", "expected_recovery", "policy_result", "action",
                "action_result", "amount_recovered", "next_step"}
    assert required <= set(FIELDS)
    store.append(ev(action_result="success", next_step="END", amount_recovered=100.0))
    store.commit()
    row = store.timeline("t1")[0]
    for f in required:
        assert f in row


def test_timeline_is_ordered_and_scoped(store):
    store.append_many(ev("t1", n=i) for i in range(4))
    store.append_many(ev("t2", n=i) for i in range(3))
    store.commit()
    t1 = store.timeline("t1")
    assert len(t1) == 4 and all(r["transaction_id"] == "t1" for r in t1)
    assert [r["seq"] for r in t1] == sorted(r["seq"] for r in t1)


def test_rule_and_decision_counts(store):
    store.append(ev(rules_fired="R-COOLDOWN,R-CHANNEL", policy_result="modify"))
    store.append(ev(rules_fired="R-COOLDOWN", policy_result="approve"))
    store.append(ev(rules_fired="", policy_result="reject"))
    store.commit()
    assert store.rule_counts()["R-COOLDOWN"] == 2
    assert store.rule_counts()["R-CHANNEL"] == 1
    assert store.decision_counts() == {"modify": 1, "approve": 1, "reject": 1}


# ------------------------------------------------------------------ case store
def test_case_store_round_trip(tmp_path):
    cs = CaseStore(tmp_path / "c.db")
    outs = [CaseOutcome(transaction_id=f"t{i}", failure_code=FailureCode.BANK_TIMEOUT,
                        failure_category=FailureCategory.TEMPORARY, amount_usd=100.0 * (i + 1),
                        expected_recovery=50.0 * (i + 1), recovered=i % 2 == 0,
                        amount_recovered=100.0 * (i + 1) if i % 2 == 0 else 0.0)
            for i in range(5)]
    assert cs.save_cases(outs, "recoverai") == 5
    assert cs.count("recoverai") == 5
    q = cs.queue("recoverai", limit=3)
    assert [r["transaction_id"] for r in q] == ["t4", "t3", "t2"], "queue not sorted by EV"
    assert cs.get_case("t0")["failure_code"] == "bank_timeout"
    cs.close()


def test_case_store_rejects_unknown_sort_column(tmp_path):
    """The order-by column is whitelisted -- no SQL injection through a query parameter."""
    cs = CaseStore(tmp_path / "c.db")
    cs.save_cases([CaseOutcome(transaction_id="t1", failure_code=FailureCode.BANK_TIMEOUT,
                               failure_category=FailureCategory.TEMPORARY, amount_usd=10.0)],
                  "recoverai")
    rows = cs.queue("recoverai", order_by="1; DROP TABLE cases--")
    assert len(rows) == 1
    assert cs.count("recoverai") == 1, "table survived"
    cs.close()


def test_run_payload_round_trip(tmp_path):
    cs = CaseStore(tmp_path / "c.db")
    cs.save_run("metrics", {"revenue_recovered": 123.45, "nested": {"a": [1, 2]}})
    assert cs.get_run("metrics")["revenue_recovered"] == 123.45
    assert "metrics" in cs.list_runs()
    cs.close()


def test_migrations_run_on_sqlite_and_are_idempotent(tmp_path):
    """The SQLite path goes through the same runner, so it gets the same guarantee."""
    from backend.app.database.db import connect
    from backend.app.database.migrations import migrate, pending

    c = connect(tmp_path / "m.db", url=None)
    assert pending(c)
    assert migrate(c)
    assert pending(c) == []
    assert migrate(c) == []
    tables = {r["name"] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"cases", "runs", "audit_log", "schema_migrations"} <= tables
    c.close()
