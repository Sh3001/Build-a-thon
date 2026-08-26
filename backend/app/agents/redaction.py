"""Enforcement for the model-facing case view.

`LLMPlanner._case_view` is already an allowlist - it names the fields it sends rather
than removing the ones it must not, which is the right way round. What was missing is
anything that *fails* when the allowlist drifts: a docstring saying "deliberately
excludes customer identifiers" does not stop the next person adding `customer_id` to the
dict, and nothing downstream would notice.

`scrub()` is that backstop. It is not the primary defence and is not a substitute for the
allowlist; it exists so an accidental identifier is dropped and, under tests, loudly
fails instead of quietly reaching a third-party API.
"""
from __future__ import annotations

import re

#: Keys that must never be sent to a model, matched case-insensitively as substrings so
#: `billing_email` and `customerId` are caught alongside the exact names.
FORBIDDEN_KEY_PARTS: tuple[str, ...] = (
    "customer_id", "customerid", "transaction_id", "transactionid",
    "invoice_id", "invoiceid", "subscription_id", "subscriptionid",
    "email", "phone", "mobile", "address", "postcode", "zip",
    "name", "card", "pan", "iban", "account_number", "ssn", "tax_id",
)

#: Values that look like contact details or card numbers regardless of their key.
VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                 # email address
    re.compile(r"\+?\d[\d\s-]{8,}\d"),                      # phone-ish run of digits
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),                 # card-ish digit run
)

REDACTED = "[redacted]"


def is_forbidden_key(key: str) -> bool:
    k = key.lower()
    return any(part in k for part in FORBIDDEN_KEY_PARTS)


def scrub(view: dict, *, strict: bool = False) -> dict:
    """Drop forbidden keys and mask identifier-shaped values.

    `strict=True` raises instead of dropping - used in tests so a regression in the
    allowlist fails the build rather than being silently cleaned up in production.
    """
    out: dict = {}
    for key, value in view.items():
        if is_forbidden_key(key):
            if strict:
                raise ValueError(f"{key!r} must never be sent to a model")
            continue
        if isinstance(value, str) and any(p.search(value) for p in VALUE_PATTERNS):
            if strict:
                raise ValueError(f"value of {key!r} looks like contact or card data")
            out[key] = REDACTED
            continue
        out[key] = value
    return out
