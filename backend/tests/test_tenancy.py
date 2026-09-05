"""Tenant isolation.

The requirement is negative -- "tenant A must never see tenant B's rows" -- so every test
here checks an absence. That is deliberately harder to satisfy by accident than a test
that checks a row comes back.
"""
from __future__ import annotations

import pytest

from backend.app.database.review import ConsentStore, ReviewStore
from backend.app.database.store import CaseStore
from backend.app.models.enums import (
    FailureCategory,
    FailureCode,
    InterventionType,
    PolicyDecision,
    Role,
)
from backend.app.models.schemas import PolicyResult, ProposedAction
from backend.app.services.results import CaseOutcome


@pytest.fixture
def db(tmp_path):
    return tmp_path / "tenancy.db"


def _outcome(tid: str, amount: float = 100.0) -> CaseOutcome:
    return CaseOutcome(
        transaction_id=tid, customer_id="c1", amount_usd=amount,
        failure_code=FailureCode.BANK_TIMEOUT,
        failure_category=FailureCategory.TEMPORARY, strategy="recoverai")


def test_cases_are_invisible_across_tenants(db):
    acme = CaseStore(db, tenant_id="acme")
    globex = CaseStore(db, tenant_id="globex")
    acme.save_cases([_outcome("txn_1")], "recoverai")

    assert acme.get_case("txn_1") is not None
    assert globex.get_case("txn_1") is None, "cross-tenant read"
    assert globex.count("recoverai") == 0
    assert not globex.queue("recoverai")


def test_the_same_transaction_id_can_exist_in_two_tenants_without_collision(db):
    """The reason tenant_id is in the primary key rather than a plain column: with the
    old key, the second write would overwrite the first."""
    acme = CaseStore(db, tenant_id="acme")
    globex = CaseStore(db, tenant_id="globex")
    acme.save_cases([_outcome("shared_id", amount=100.0)], "recoverai")
    globex.save_cases([_outcome("shared_id", amount=999.0)], "recoverai")

    assert acme.get_case("shared_id")["amount_usd"] == 100.0
    assert globex.get_case("shared_id")["amount_usd"] == 999.0


def test_run_reports_are_scoped(db):
    acme = CaseStore(db, tenant_id="acme")
    globex = CaseStore(db, tenant_id="globex")
    acme.save_run("recoverai", {"revenue_recovered": 1234.0})
    assert acme.get_run("recoverai")["revenue_recovered"] == 1234.0
    assert globex.get_run("recoverai") is None
    assert globex.list_runs() == []


def test_review_tasks_are_scoped(db):
    verdict = PolicyResult(
        decision=PolicyDecision.HUMAN_REVIEW,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT),
        rules_fired=["R-AMOUNT-CAP"], reason="over the ceiling")
    acme = ReviewStore(db, tenant_id="acme")
    globex = ReviewStore(db, tenant_id="globex")
    rid = acme.open_task("txn_1", verdict, amount_usd=9000.0)

    assert acme.get(rid) is not None
    assert globex.get(rid) is None, "cross-tenant review read"
    assert globex.pending() == []
    assert globex.stats()["pending"] == 0


def test_another_tenant_cannot_resolve_a_review_task(db):
    verdict = PolicyResult(
        decision=PolicyDecision.HUMAN_REVIEW,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT),
        rules_fired=["R-AMOUNT-CAP"], reason="over the ceiling")
    acme = ReviewStore(db, tenant_id="acme")
    globex = ReviewStore(db, tenant_id="globex")
    rid = acme.open_task("txn_1", verdict, amount_usd=9000.0)

    with pytest.raises(KeyError):
        globex.approve(rid, "mallory", Role.OPERATOR, "not mine to approve")
    assert acme.get(rid)["status"] == "pending", "the task must be untouched"


def test_overrides_are_scoped(db):
    verdict = PolicyResult(
        decision=PolicyDecision.HUMAN_REVIEW,
        effective_action=ProposedAction(action=InterventionType.RETRY_PAYMENT),
        rules_fired=["R-AMOUNT-CAP"], reason="over the ceiling")
    acme = ReviewStore(db, tenant_id="acme")
    rid = acme.open_task("txn_1", verdict, amount_usd=9000.0)
    acme.approve(rid, "alice", Role.OPERATOR, "confirmed with the merchant")

    assert len(acme.overrides()) == 1
    assert ReviewStore(db, tenant_id="globex").overrides() == []


def test_consent_is_scoped(db):
    """An opt-out with one merchant must not silence another merchant's messages, and a
    customer who opted out of one must not still be contactable by them."""
    acme = ConsentStore(db, tenant_id="acme")
    globex = ConsentStore(db, tenant_id="globex")
    acme.opt_out("cust_1")
    assert acme.is_opted_out("cust_1")
    assert not globex.is_opted_out("cust_1")


def test_a_blanket_opt_out_beats_a_per_channel_record(db):
    """Someone who said "stop contacting me" has not agreed to be reached on a channel
    they did not name."""
    store = ConsentStore(db, tenant_id="acme")
    store.opt_out("cust_1", channel="*")
    for channel in ("email", "sms", "whatsapp", "in_app"):
        assert store.is_opted_out("cust_1", channel)


def test_a_per_channel_opt_out_does_not_silence_every_channel(db):
    store = ConsentStore(db, tenant_id="acme")
    store.opt_out("cust_1", channel="sms")
    assert store.is_opted_out("cust_1", "sms")
    assert not store.is_opted_out("cust_1", "email")
