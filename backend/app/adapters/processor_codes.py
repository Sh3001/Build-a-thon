"""Normalising processor vocabularies into one taxonomy.

Every provider names the same handful of realities differently, and none of them names
them the way this system does. Stripe says `insufficient_funds`; Razorpay says
`BAD_REQUEST_PAYMENT_FAILED` with a sub-reason; Adyen says `Not enough balance`. A
diagnosis engine written against one vocabulary is a diagnosis engine that silently
mis-handles every other provider.

Two design decisions carry the weight:

**Unknown is a first-class answer.** `CanonicalCause.UNKNOWN` is returned when a code
does not map, and it flows to `R-REVIEW-UNKNOWN-CAUSE`, which withholds automated action.
The tempting alternative -- mapping the unrecognised remainder to `TEMPORARY_DECLINE`
because most declines are transient -- would turn a mapping gap into confident retries
against instruments nobody has classified. Guessing is worse than admitting.

**The mapping is data.** A table, not a chain of `if`s, so adding a provider is a dict
entry and `coverage()` can report how much of a provider's published vocabulary is
actually handled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.app.models.enums import FailureCategory, FailureCode


class CanonicalCause(str, Enum):
    """The provider-independent taxonomy. Broader than `FailureCode`, which is this
    system's *internal* vocabulary -- the two are deliberately separate so a provider
    quirk never forces a change to the ML feature space."""
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    EXPIRED_PAYMENT_METHOD = "EXPIRED_PAYMENT_METHOD"
    TEMPORARY_DECLINE = "TEMPORARY_DECLINE"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    INVALID_PAYMENT_METHOD = "INVALID_PAYMENT_METHOD"
    ACCOUNT_CLOSED = "ACCOUNT_CLOSED"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"
    COMPLIANCE_HOLD = "COMPLIANCE_HOLD"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    DO_NOT_HONOR = "DO_NOT_HONOR"
    UNKNOWN = "UNKNOWN"


#: Canonical cause -> this system's internal failure code. `UNKNOWN` maps to nothing on
#: purpose: there is no internal code meaning "we do not know", and inventing one would
#: put an unclassifiable case into the model's feature vocabulary.
TO_FAILURE_CODE: dict[CanonicalCause, FailureCode | None] = {
    CanonicalCause.INSUFFICIENT_FUNDS: FailureCode.INSUFFICIENT_FUNDS,
    CanonicalCause.EXPIRED_PAYMENT_METHOD: FailureCode.EXPIRED_CARD,
    CanonicalCause.TEMPORARY_DECLINE: FailureCode.TEMPORARY_DECLINE,
    CanonicalCause.NETWORK_FAILURE: FailureCode.NETWORK_ERROR,
    CanonicalCause.INVALID_PAYMENT_METHOD: FailureCode.INVALID_PAYMENT_METHOD,
    CanonicalCause.ACCOUNT_CLOSED: FailureCode.CLOSED_ACCOUNT,
    CanonicalCause.FRAUD_SUSPECTED: FailureCode.SUSPECTED_FRAUD,
    CanonicalCause.COMPLIANCE_HOLD: FailureCode.COMPLIANCE_HOLD,
    CanonicalCause.LIMIT_EXCEEDED: FailureCode.PAYMENT_LIMIT_EXCEEDED,
    CanonicalCause.DO_NOT_HONOR: FailureCode.MULTIPLE_DECLINES,
    CanonicalCause.UNKNOWN: None,
}

C = CanonicalCause

#: Provider vocabularies, lowercased at lookup. Sourced from each provider's published
#: decline-code documentation; incomplete by construction, which is what `UNKNOWN` and
#: `coverage()` exist to make visible rather than to hide.
PROVIDER_CODES: dict[str, dict[str, CanonicalCause]] = {
    "stripe": {
        "insufficient_funds": C.INSUFFICIENT_FUNDS,
        "card_decline_rate_limit_exceeded": C.TEMPORARY_DECLINE,
        "expired_card": C.EXPIRED_PAYMENT_METHOD,
        "incorrect_number": C.INVALID_PAYMENT_METHOD,
        "invalid_account": C.ACCOUNT_CLOSED,
        "account_closed": C.ACCOUNT_CLOSED,
        "issuer_not_available": C.NETWORK_FAILURE,
        "processing_error": C.TEMPORARY_DECLINE,
        "try_again_later": C.TEMPORARY_DECLINE,
        "do_not_honor": C.DO_NOT_HONOR,
        "generic_decline": C.DO_NOT_HONOR,
        "fraudulent": C.FRAUD_SUSPECTED,
        "lost_card": C.FRAUD_SUSPECTED,
        "stolen_card": C.FRAUD_SUSPECTED,
        "merchant_blacklist": C.COMPLIANCE_HOLD,
        "card_velocity_exceeded": C.LIMIT_EXCEEDED,
        "withdrawal_count_limit_exceeded": C.LIMIT_EXCEEDED,
        "transaction_not_allowed": C.COMPLIANCE_HOLD,
    },
    "razorpay": {
        "bad_request_payment_failed": C.DO_NOT_HONOR,
        "payment_failed_insufficient_balance": C.INSUFFICIENT_FUNDS,
        "gateway_error": C.NETWORK_FAILURE,
        "server_error": C.NETWORK_FAILURE,
        "payment_timeout": C.NETWORK_FAILURE,
        "card_expired": C.EXPIRED_PAYMENT_METHOD,
        "invalid_card": C.INVALID_PAYMENT_METHOD,
        "payment_limit_exceeded": C.LIMIT_EXCEEDED,
        "upi_mandate_revoked": C.ACCOUNT_CLOSED,
        "payment_declined_by_bank": C.DO_NOT_HONOR,
        "suspected_fraud": C.FRAUD_SUSPECTED,
        "compliance_check_failed": C.COMPLIANCE_HOLD,
    },
    "adyen": {
        "notenoughbalance": C.INSUFFICIENT_FUNDS,
        "expiredcard": C.EXPIRED_PAYMENT_METHOD,
        "invalidcardnumber": C.INVALID_PAYMENT_METHOD,
        "closedaccount": C.ACCOUNT_CLOSED,
        "refused": C.DO_NOT_HONOR,
        "declined": C.DO_NOT_HONOR,
        "acquirererror": C.NETWORK_FAILURE,
        "issuerunavailable": C.NETWORK_FAILURE,
        "fraud": C.FRAUD_SUSPECTED,
        "fraud-cancelled-order": C.FRAUD_SUSPECTED,
        "blockedcard": C.COMPLIANCE_HOLD,
        "restrictedcard": C.COMPLIANCE_HOLD,
        "transactionnotpermitted": C.COMPLIANCE_HOLD,
        "withdrawalamountexceeded": C.LIMIT_EXCEEDED,
    },
    # The simulator speaks this system's own vocabulary, so its mapping is the identity
    # over `FailureCode`. Listed rather than special-cased so `normalise("mock", ...)`
    # goes down the same path as every other provider and the path stays tested.
    "mock": {
        "insufficient_funds": C.INSUFFICIENT_FUNDS,
        "expired_card": C.EXPIRED_PAYMENT_METHOD,
        "invalid_payment_method": C.INVALID_PAYMENT_METHOD,
        "payment_limit_exceeded": C.LIMIT_EXCEEDED,
        "temporary_decline": C.TEMPORARY_DECLINE,
        "bank_timeout": C.NETWORK_FAILURE,
        "network_error": C.NETWORK_FAILURE,
        "processor_unavailable": C.NETWORK_FAILURE,
        "multiple_declines": C.DO_NOT_HONOR,
        "invalid_account": C.ACCOUNT_CLOSED,
        "closed_account": C.ACCOUNT_CLOSED,
        "suspected_fraud": C.FRAUD_SUSPECTED,
        "high_risk_transaction": C.FRAUD_SUSPECTED,
        "compliance_hold": C.COMPLIANCE_HOLD,
    },
}

#: Last-resort patterns over free-text provider messages. Applied only after the code
#: table misses, and they lower the returned confidence -- matching on prose is a hint,
#: not a determination, and a system that acts on it as though it were one is guessing
#: with extra steps.
TEXT_PATTERNS: tuple[tuple[re.Pattern[str], CanonicalCause], ...] = (
    (re.compile(r"insufficient|not enough (balance|funds)|low balance", re.I),
     C.INSUFFICIENT_FUNDS),
    (re.compile(r"expired?\b.*(card|method)|card.*expired", re.I),
     C.EXPIRED_PAYMENT_METHOD),
    (re.compile(r"time(d)? ?out|unavailable|unreachable|network|connection reset", re.I),
     C.NETWORK_FAILURE),
    (re.compile(r"closed|terminated|revoked", re.I), C.ACCOUNT_CLOSED),
    (re.compile(r"fraud|stolen|lost card", re.I), C.FRAUD_SUSPECTED),
    (re.compile(r"sanction|aml|kyc|compliance|blocked", re.I), C.COMPLIANCE_HOLD),
    (re.compile(r"limit|velocity|exceed", re.I), C.LIMIT_EXCEEDED),
    (re.compile(r"do not honou?r|refused|declined", re.I), C.DO_NOT_HONOR),
)


@dataclass(frozen=True)
class Normalised:
    """The result of mapping one provider signal into the taxonomy."""
    cause: CanonicalCause
    confidence: float
    evidence: list[str]
    provider: str
    raw_code: str | None = None

    @property
    def is_unknown(self) -> bool:
        return self.cause is CanonicalCause.UNKNOWN

    @property
    def failure_code(self) -> FailureCode | None:
        return TO_FAILURE_CODE[self.cause]


#: Confidence by how the match was made. An exact code match is near-certain; a regex
#: over a human-readable message is a hint that lands below the review floor on its own.
CONFIDENCE_EXACT = 0.95
CONFIDENCE_TEXT = 0.45
CONFIDENCE_UNKNOWN = 0.0


def normalise(provider: str, raw_code: str | None = None,
              message: str | None = None) -> Normalised:
    """Map one provider signal into the taxonomy.

    Order matters: the structured code is tried first because it is what the provider
    committed to, and the free-text message only afterwards because it is what the
    provider happened to write. Reversing them would let a marketing-friendly decline
    string override an unambiguous code.
    """
    key = (provider or "").strip().lower()
    table = PROVIDER_CODES.get(key)
    evidence: list[str] = [f"provider={key or 'unspecified'}"]

    if table is None:
        return Normalised(CanonicalCause.UNKNOWN, CONFIDENCE_UNKNOWN,
                          [*evidence, f"no code table registered for {key!r}"],
                          key, raw_code)

    if raw_code:
        lookup = raw_code.strip().lower().replace("-", "_").replace(" ", "_")
        evidence.append(f"raw_code={raw_code}")
        if lookup in table:
            return Normalised(table[lookup], CONFIDENCE_EXACT,
                              [*evidence, "exact code match"], key, raw_code)

    if message:
        for pattern, cause in TEXT_PATTERNS:
            if pattern.search(message):
                return Normalised(
                    cause, CONFIDENCE_TEXT,
                    [*evidence, f"matched provider message on /{pattern.pattern}/",
                     "text match only -- below the automated-action confidence floor"],
                    key, raw_code)

    return Normalised(
        CanonicalCause.UNKNOWN, CONFIDENCE_UNKNOWN,
        [*evidence, "no mapping for this code or message; refusing to guess a cause"],
        key, raw_code)


def coverage(provider: str) -> dict:
    """How much of a provider's vocabulary is mapped. Reported rather than assumed --
    "we support Stripe" is a claim, and this is the number behind it."""
    table = PROVIDER_CODES.get(provider.lower(), {})
    by_cause: dict[str, int] = {}
    for cause in table.values():
        by_cause[cause.value] = by_cause.get(cause.value, 0) + 1
    return {
        "provider": provider.lower(),
        "codes_mapped": len(table),
        "causes_covered": sorted(by_cause),
        "unmapped_causes": sorted(
            c.value for c in CanonicalCause
            if c is not CanonicalCause.UNKNOWN and c.value not in by_cause),
    }


def category_of_cause(cause: CanonicalCause) -> FailureCategory | None:
    """Where a canonical cause lands in the internal category taxonomy, or None when the
    cause has no internal code (only `UNKNOWN`)."""
    from backend.app.models.enums import category_of
    code = TO_FAILURE_CODE[cause]
    return category_of(code) if code is not None else None
