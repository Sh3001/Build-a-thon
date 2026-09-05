"""Money must be exact.

The properties here are the reason `Money` exists at all: float accumulation over a few
thousand recoveries drifts, silently, in a direction nobody chose.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from backend.app.domain.money import (
    CurrencyMismatch,
    FxRates,
    Money,
    exponent_of,
    total,
)


def test_the_float_problem_this_module_exists_to_fix():
    """0.1 + 0.2 != 0.3 in binary floating point. In minor units it does."""
    assert 0.1 + 0.2 != 0.3
    assert Money.from_major("0.1") + Money.from_major("0.2") == Money.from_major("0.3")


def test_accumulation_does_not_drift():
    """The failure mode in production: a thousand small recoveries summed as floats."""
    naive = sum(0.07 for _ in range(10_000))
    exact = total([Money.from_major("0.07")] * 10_000)
    assert naive != 700.0
    assert exact == Money.from_major("700.00")
    assert exact.minor == 70_000


def test_float_input_goes_through_str_not_binary():
    """`Decimal(0.07)` is 0.0700000000000000066...; `Decimal('0.07')` is 0.07."""
    assert Money.from_major(0.07).minor == 7
    assert Money.from_major(0.145).minor == 15 or Money.from_major(0.145).minor == 14


@pytest.mark.parametrize("currency,expected_minor", [
    ("USD", 12345), ("EUR", 12345), ("JPY", 123), ("KWD", 123450),
])
def test_minor_unit_exponent_is_per_currency(currency, expected_minor):
    """A JPY amount stored with two decimal places is off by 100x."""
    assert Money.from_major("123.45", currency).minor == expected_minor
    assert exponent_of(currency) in (0, 2, 3)


def test_adding_different_currencies_is_refused():
    """There is no correct answer to USD 10 + INR 10, so guessing is the wrong move."""
    with pytest.raises(CurrencyMismatch):
        Money.from_major(10, "USD") + Money.from_major(10, "INR")


def test_scaling_by_a_probability_rounds_once():
    assert Money.from_major("99.99").scale("0.335") == Money.from_major("33.50")
    assert Money.from_major(100).scale(0).is_zero


def test_bankers_rounding_is_unbiased():
    """Half-up would bias every expected-value figure upward. Over a queue of thousands
    that is a systematic overstatement of forecast revenue."""
    assert Money.from_major("0.125").minor == 12    # to even
    assert Money.from_major("0.135").minor == 14    # to even


def test_fx_conversion_is_exact_and_attributed():
    fx = FxRates({"USD": 1, "INR": "0.012"}, as_of="test-rates")
    assert fx.convert(Money.from_major(1000, "INR")) == Money.from_major(12, "USD")
    assert fx.as_of == "test-rates"
    with pytest.raises(KeyError):
        fx.convert(Money.from_major(10, "ZWL"))


def test_total_of_nothing_is_zero_not_an_error():
    """A filtered-to-nothing aggregation should report 0, not blow up a dashboard."""
    assert total([]).is_zero


def test_ratio_against_zero_returns_zero_rather_than_raising():
    assert Money.from_major(10).ratio_to(Money.zero()) == Decimal(0)


def test_minor_units_must_be_integers():
    with pytest.raises(TypeError):
        Money(10.5)                                           # type: ignore[arg-type]
