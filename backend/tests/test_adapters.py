"""Processor-code normalisation and the payment-rail abstraction."""
from __future__ import annotations

import pytest

from backend.app.adapters import mock_rail, sandbox_rails  # noqa: F401  -- registers rails
from backend.app.adapters.gateway import (
    GatewayRequest,
    PaymentRail,
    available,
    build_rail,
)
from backend.app.adapters.processor_codes import (
    CONFIDENCE_EXACT,
    CONFIDENCE_TEXT,
    PROVIDER_CODES,
    CanonicalCause,
    coverage,
    normalise,
)
from backend.app.adapters.sandbox_rails import LiveCredentialRefused
from backend.app.config import REVIEW_MIN_DIAGNOSIS_CONFIDENCE
from backend.app.models.enums import FailureCode


@pytest.mark.parametrize("provider,code,expected", [
    ("stripe", "insufficient_funds", CanonicalCause.INSUFFICIENT_FUNDS),
    ("stripe", "expired_card", CanonicalCause.EXPIRED_PAYMENT_METHOD),
    ("stripe", "fraudulent", CanonicalCause.FRAUD_SUSPECTED),
    ("razorpay", "GATEWAY_ERROR", CanonicalCause.NETWORK_FAILURE),
    ("razorpay", "card_expired", CanonicalCause.EXPIRED_PAYMENT_METHOD),
    ("adyen", "ExpiredCard", CanonicalCause.EXPIRED_PAYMENT_METHOD),
    ("adyen", "NotEnoughBalance", CanonicalCause.INSUFFICIENT_FUNDS),
    ("mock", "closed_account", CanonicalCause.ACCOUNT_CLOSED),
])
def test_three_vocabularies_collapse_to_one_taxonomy(provider, code, expected):
    n = normalise(provider, code)
    assert n.cause is expected
    assert n.confidence == CONFIDENCE_EXACT


def test_lookup_is_case_and_separator_insensitive():
    for spelling in ("expired_card", "EXPIRED_CARD", "Expired-Card", "expired card"):
        assert normalise("stripe", spelling).cause is CanonicalCause.EXPIRED_PAYMENT_METHOD


def test_an_unmapped_code_is_unknown_rather_than_guessed():
    """Mapping the unrecognised remainder to TEMPORARY_DECLINE because most declines are
    transient would turn a mapping gap into confident retries."""
    n = normalise("stripe", "some_code_invented_in_2027")
    assert n.is_unknown and n.failure_code is None
    assert "refusing to guess" in " ".join(n.evidence)


def test_an_unknown_provider_is_unknown():
    assert normalise("worldpay", "insufficient_funds").is_unknown


def test_an_unknown_cause_lands_below_the_automated_action_floor():
    """The link between normalisation and the policy engine: an unmappable code cannot
    clear the confidence bar for automated action."""
    assert normalise("stripe", "mystery").confidence < REVIEW_MIN_DIAGNOSIS_CONFIDENCE


def test_a_text_match_is_a_hint_not_a_determination():
    """Matching on prose is guessing with extra steps unless it is priced as a guess."""
    n = normalise("stripe", None, "Issuer timed out, please try again later")
    assert n.cause is CanonicalCause.NETWORK_FAILURE
    assert n.confidence == CONFIDENCE_TEXT
    assert n.confidence < REVIEW_MIN_DIAGNOSIS_CONFIDENCE


def test_a_structured_code_beats_a_free_text_message():
    """Reversing the order would let a decline string override an unambiguous code."""
    n = normalise("stripe", "expired_card", "insufficient funds in account")
    assert n.cause is CanonicalCause.EXPIRED_PAYMENT_METHOD


def test_every_canonical_cause_except_unknown_maps_to_an_internal_code():
    from backend.app.adapters.processor_codes import TO_FAILURE_CODE
    for cause, code in TO_FAILURE_CODE.items():
        if cause is CanonicalCause.UNKNOWN:
            assert code is None
        else:
            assert isinstance(code, FailureCode)


def test_provider_coverage_is_reported_rather_than_claimed():
    """"We support Stripe" is a claim; this is the number behind it."""
    c = coverage("stripe")
    assert c["codes_mapped"] == len(PROVIDER_CODES["stripe"])
    assert "unmapped_causes" in c


# ---------------------------------------------------------------- rails
def test_the_mock_rail_satisfies_the_protocol():
    assert isinstance(build_rail("mock"), PaymentRail)


def test_an_unknown_rail_refuses_rather_than_falling_back_to_the_mock():
    """A deployment that believes it is talking to Stripe and is talking to a simulator
    is the worst failure this abstraction can have."""
    with pytest.raises(KeyError, match="no fallback"):
        build_rail("definitely_not_registered")


def test_the_registry_lists_what_is_available():
    assert "mock" in available()
    assert "stripe_sandbox" in available()


def test_a_rail_request_cannot_be_built_without_an_idempotency_key():
    with pytest.raises(ValueError):
        GatewayRequest(idempotency_key="", transaction_id="t", customer_id="c",
                       amount_minor=1, currency="USD")


def test_a_sandbox_adapter_refuses_a_live_credential():
    """A live key reaching a system whose safety envelope has never been exercised
    against real money is the worst outcome available here, so it is refused at
    construction rather than trusted to configuration."""
    from backend.app.adapters.sandbox_rails import (
        RazorpaySandboxRail,
        StripeSandboxRail,
    )
    with pytest.raises(LiveCredentialRefused):
        StripeSandboxRail(api_key="sk_live_abc123")
    with pytest.raises(LiveCredentialRefused):
        RazorpaySandboxRail(key_id="rzp_live_abc", key_secret="x")
    # A test-mode key with no SDK installed constructs fine and simply has no client.
    assert StripeSandboxRail(api_key="").client is None
