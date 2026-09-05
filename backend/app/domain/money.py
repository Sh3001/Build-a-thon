"""Exact money.

Revenue was accumulated in `float` throughout, and costs were papered over with
`round(x, 4)`. That is survivable for a demo and wrong for a payment system: 0.1 + 0.2
is not 0.3 in binary floating point, and an experiment that sums 1,842 recoveries
accumulates a drift that no reviewer can account for. Worse, the drift is silent -- the
number is simply a little wrong, every time, in a direction nobody chose.

`Money` fixes the representation: an integer count of minor units (cents, paise) plus a
currency. Integers are exact under addition, which is the operation revenue aggregation
actually performs. Multiplication by a probability leaves the integers, so the rounding
mode is stated at exactly one place -- `Money.scale` -- instead of being wherever a
`round()` happened to land.

Deliberately NOT a wholesale rewrite of the wire schema. `Transaction.amount` stays a
float because JSON has no decimal type and every existing caller passes one; the value
is converted at the boundary. What changed is that every path where money is *summed*
or *converted* now goes through this module, which is where the drift was.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

#: Minor units per major unit. Currencies without a minor unit (JPY, KRW) are exponent 0;
#: getting this wrong inflates a JPY total by 100x, so it is data, not an assumption.
CURRENCY_EXPONENT: dict[str, int] = {
    "USD": 2, "EUR": 2, "GBP": 2, "INR": 2, "SGD": 2, "AUD": 2, "CAD": 2, "CHF": 2,
    "AED": 2, "BRL": 2, "MXN": 2, "ZAR": 2,
    "JPY": 0, "KRW": 0, "VND": 0, "CLP": 0, "ISK": 0,
    "BHD": 3, "KWD": 3, "OMR": 3, "TND": 3,
}
DEFAULT_EXPONENT = 2

#: Banker's rounding. Chosen over ROUND_HALF_UP because half-up is biased upward, and a
#: systematic upward bias on a recovered-revenue figure is precisely the kind of error a
#: reviewer would be right to distrust.
ROUNDING = ROUND_HALF_EVEN


def exponent_of(currency: str) -> int:
    return CURRENCY_EXPONENT.get(currency.upper(), DEFAULT_EXPONENT)


class CurrencyMismatch(ValueError):
    """Raised when two different currencies are added. There is no correct answer to
    `USD 10 + INR 10`, so the only safe behaviour is to refuse rather than to guess."""


@dataclass(frozen=True, order=False)
class Money:
    """An exact amount. `minor` is the authoritative value; everything else derives."""
    minor: int
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", self.currency.upper())
        if not isinstance(self.minor, int) or isinstance(self.minor, bool):
            raise TypeError(f"minor units must be int, got {type(self.minor).__name__}")

    # ------------------------------------------------------------------ builders
    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        return cls(0, currency)

    @classmethod
    def from_major(cls, amount: float | int | str | Decimal, currency: str = "USD") -> Money:
        """Parse a major-unit amount. Floats are routed through `str` deliberately:
        `Decimal(0.07)` is 0.070000000000000006661338147750939242541790008544921875,
        while `Decimal("0.07")` is 0.07, and the second is what the caller meant."""
        d = amount if isinstance(amount, Decimal) else Decimal(str(amount))
        scale = Decimal(10) ** exponent_of(currency)
        with localcontext() as ctx:
            ctx.prec = 34
            return cls(int((d * scale).quantize(Decimal(1), rounding=ROUNDING)), currency)

    # ------------------------------------------------------------------ views
    @property
    def decimal(self) -> Decimal:
        return Decimal(self.minor) / (Decimal(10) ** exponent_of(self.currency))

    def as_float(self) -> float:
        """For JSON and for the existing float-typed schemas. Lossy by definition, which
        is why it is a method and not the storage format."""
        return float(self.decimal)

    def __str__(self) -> str:
        return f"{self.currency} {self.decimal:,.{exponent_of(self.currency)}f}"

    def __repr__(self) -> str:
        return f"Money({self.minor}, {self.currency!r})"

    # ------------------------------------------------------------------ arithmetic
    def _check(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(
                f"cannot combine {self.currency} and {other.currency}; convert first")

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor - other.minor, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.minor, self.currency)

    def scale(self, factor: float | int | str | Decimal) -> Money:
        """Multiply by a dimensionless factor (a probability, an uplift, a fee rate).

        The single place a monetary rounding decision is taken. Everything downstream --
        expected value, expected profit, fee models -- funnels through here, so changing
        the rounding policy is a one-line change with one test to update.
        """
        f = factor if isinstance(factor, Decimal) else Decimal(str(factor))
        with localcontext() as ctx:
            ctx.prec = 34
            return Money(int((Decimal(self.minor) * f).quantize(Decimal(1), rounding=ROUNDING)),
                         self.currency)

    def ratio_to(self, other: Money) -> Decimal:
        """Dimensionless ratio, exact. Returns 0 against a zero denominator rather than
        raising: callers computing rates want a number, and a division guard at each of
        the twenty call sites is how a missing one gets shipped."""
        self._check(other)
        if other.minor == 0:
            return Decimal(0)
        return Decimal(self.minor) / Decimal(other.minor)

    # ------------------------------------------------------------------ comparison
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.minor == other.minor and self.currency == other.currency

    def __hash__(self) -> int:
        return hash((self.minor, self.currency))

    def __lt__(self, other: Money) -> bool:
        self._check(other)
        return self.minor < other.minor

    def __le__(self, other: Money) -> bool:
        self._check(other)
        return self.minor <= other.minor

    def __gt__(self, other: Money) -> bool:
        self._check(other)
        return self.minor > other.minor

    def __ge__(self, other: Money) -> bool:
        self._check(other)
        return self.minor >= other.minor

    @property
    def is_zero(self) -> bool:
        return self.minor == 0

    @property
    def is_positive(self) -> bool:
        return self.minor > 0


def total(amounts: Iterable[Money], currency: str = "USD") -> Money:
    """Exact sum. An empty iterable yields zero in `currency` rather than raising, so a
    filtered-to-nothing aggregation reports 0 instead of blowing up a dashboard."""
    acc = Money.zero(currency)
    for a in amounts:
        if acc.is_zero and acc.currency != a.currency:
            acc = Money.zero(a.currency)
        acc = acc + a
    return acc


# ---------------------------------------------------------------- FX
class FxRates:
    """Static rate table to one reporting currency.

    Real FX is out of scope, and pretending otherwise would be the more dishonest
    choice. What this fixes is that the rates were floats multiplied into float amounts:
    the conversion is now exact-in, exact-out, with the rounding stated once.

    `as_of` exists so a converted figure can be attributed to a rate set. A revenue
    total with no rate provenance cannot be reconciled by anyone.
    """

    def __init__(self, rates: dict[str, float | str | Decimal], base: str = "USD",
                 as_of: str = "static-2026-01-01"):
        self.base = base.upper()
        self.as_of = as_of
        self.rates: dict[str, Decimal] = {
            k.upper(): (v if isinstance(v, Decimal) else Decimal(str(v)))
            for k, v in rates.items()
        }
        self.rates.setdefault(self.base, Decimal(1))

    def rate(self, currency: str) -> Decimal:
        c = currency.upper()
        if c not in self.rates:
            raise KeyError(f"no FX rate for {c} -> {self.base} in rate set {self.as_of!r}")
        return self.rates[c]

    def convert(self, amount: Money, to: str | None = None) -> Money:
        """Convert exactly, then round once to the target currency's minor unit."""
        target = (to or self.base).upper()
        if amount.currency == target:
            return amount
        with localcontext() as ctx:
            ctx.prec = 34
            major = amount.decimal * self.rate(amount.currency) / self.rate(target)
            return Money.from_major(major, target)

    def has(self, currency: str) -> bool:
        return currency.upper() in self.rates

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.rates))
