"""The model-facing view must never carry identifiers.

The allowlist in `_case_view` is the real defence; these tests are what make it a
guarantee rather than a comment. `test_case_view_carries_no_identifiers` is the one that
matters - it fails the build if someone widens the allowlist.
"""
from __future__ import annotations

import pytest

from backend.app.agents.llm import LLMPlanner
from backend.app.agents.redaction import REDACTED, is_forbidden_key, scrub
from backend.app.models.enums import Channel, FailureCode, PaymentMethod
from backend.app.models.schemas import Transaction


def _txn() -> Transaction:
    return Transaction(
        customer_id="cust_000123", transaction_id="txn_000999",
        invoice_id="inv_5", subscription_id="sub_9",
        amount=100.0, currency="USD", payment_method=PaymentMethod.CARD,
        failure_code=FailureCode.INSUFFICIENT_FUNDS, preferred_channel=Channel.EMAIL,
    )


def test_case_view_carries_no_identifiers():
    """If this fails, the allowlist has drifted and identifiers are about to leave the box."""
    view = LLMPlanner._case_view(_txn())
    assert scrub(view, strict=True) == view


def test_case_view_omits_the_specific_ids():
    view = LLMPlanner._case_view(_txn())
    blob = str(view)
    for leaked in ("cust_000123", "txn_000999", "inv_5", "sub_9"):
        assert leaked not in blob


@pytest.mark.parametrize("key", [
    "customer_id", "customerId", "billing_email", "phone_number",
    "card_number", "full_name", "postcode", "transaction_id",
])
def test_forbidden_keys_are_recognised(key):
    assert is_forbidden_key(key)


@pytest.mark.parametrize("key", [
    "amount_usd", "failure_code", "segment", "previous_success_rate", "overdue_days",
])
def test_behavioural_fields_are_allowed(key):
    assert not is_forbidden_key(key)


def test_scrub_drops_forbidden_keys():
    assert scrub({"customer_id": "c1", "amount_usd": 10}) == {"amount_usd": 10}


def test_scrub_masks_contact_shaped_values():
    out = scrub({"note": "reach them at a.b@example.com"})
    assert out["note"] == REDACTED


def test_scrub_masks_card_shaped_values():
    assert scrub({"note": "4111 1111 1111 1111"})["note"] == REDACTED


def test_scrub_leaves_ordinary_text_alone():
    text = "instrument is dead; replace before retrying"
    assert scrub({"reason": text})["reason"] == text


def test_strict_mode_raises_instead_of_cleaning():
    with pytest.raises(ValueError):
        scrub({"customer_id": "c1"}, strict=True)
    with pytest.raises(ValueError):
        scrub({"note": "a.b@example.com"}, strict=True)
