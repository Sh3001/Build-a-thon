"""Simulator parameters as configuration.

Every number that governs the simulated world lived as a module-level constant in
`payment_gateway.py`. That made the world honest -- the constants are visible and
commented -- and it made it a single world. There was no way to ask "does the agent still
beat the baseline if self-cure is twice as common?", which is exactly the question a
reviewer should ask, because the answer is where a simulation-based result is most
fragile.

So the parameters move here, into a dataclass with a YAML loader and named scenarios.
Three properties are load-bearing:

* **The defaults reproduce the previous world exactly.** `SimulationConfig()` with no
  arguments gives the constants that were in the module, so every existing result and
  test is unaffected. A configuration system that changes the baseline while introducing
  itself is impossible to trust.
* **Scenarios are declared, not coded.** `config/simulation.yaml` names a handful of
  worlds -- pessimistic, high-fatigue, high-self-cure -- and the sensitivity harness
  sweeps them. A finding that survives all of them is worth more than one tuned to a
  single parameterisation.
* **The config is hashable.** `SimulationConfig.fingerprint()` goes into the run record,
  so a result can be tied to the world that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "simulation.yaml"


def _default_base_retry() -> dict[str, float]:
    return {
        "bank_timeout": 0.62, "network_error": 0.68, "processor_unavailable": 0.55,
        "temporary_decline": 0.38,
        "insufficient_funds": 0.20, "payment_limit_exceeded": 0.24,
        "expired_card": 0.0, "invalid_payment_method": 0.0,
        "multiple_declines": 0.02, "invalid_account": 0.0, "closed_account": 0.0,
        "suspected_fraud": 0.0, "high_risk_transaction": 0.0, "compliance_hold": 0.0,
    }


def _default_passive() -> dict[str, float]:
    return {
        "bank_timeout": 0.26, "network_error": 0.29, "processor_unavailable": 0.23,
        "temporary_decline": 0.17,
        "insufficient_funds": 0.15, "payment_limit_exceeded": 0.13,
        "expired_card": 0.07, "invalid_payment_method": 0.05,
        "multiple_declines": 0.04, "invalid_account": 0.01, "closed_account": 0.01,
        "suspected_fraud": 0.02, "high_risk_transaction": 0.03, "compliance_hold": 0.05,
    }


def _default_reach() -> dict[str, float]:
    return {"email": 0.42, "sms": 0.63, "whatsapp": 0.71, "in_app": 0.55}


#: Codes whose retry probability is a structural zero: the instrument cannot be
#: re-presented at all. A scenario may not raise these -- allowing it would let a sweep
#: produce a world in which brute force beats diagnosis, and every conclusion drawn from
#: this simulator rests on that being impossible.
STRUCTURAL_ZEROS = frozenset({
    "expired_card", "invalid_payment_method", "invalid_account", "closed_account",
    "suspected_fraud", "high_risk_transaction", "compliance_hold",
})


class InvalidScenario(ValueError):
    """A scenario that would break an invariant the evaluation depends on."""


@dataclass
class SimulationConfig:
    """The whole simulated world, in one object."""

    name: str = "default"
    description: str = "the shipped parameterisation"

    #: P(a bare retry of the unchanged instrument succeeds), before modifiers.
    base_retry: dict[str, float] = field(default_factory=_default_base_retry)
    #: P(the case resolves with no intervention at all within the horizon).
    passive_cure_rate: dict[str, float] = field(default_factory=_default_passive)
    #: Mean days until a self-cure lands, once one is going to happen.
    passive_cure_mean_days: float = 8.0
    #: P(a message on this channel is seen at all).
    channel_reach: dict[str, float] = field(default_factory=_default_reach)

    # ---- timing --------------------------------------------------------------
    #: Exponential decay per day for transient faults. Higher = the window closes faster.
    temporary_decay_per_day: float = 0.16
    #: Where the insufficient-funds recovery probability peaks, in days (payday).
    payday_peak_days: float = 6.0
    payday_falloff_per_day: float = 0.055
    payday_boost: float = 1.35
    customer_action_decay_per_day: float = 0.02
    persistent_decay_per_day: float = 0.05

    # ---- customer behaviour --------------------------------------------------
    #: Per-contact multiplier on response probability. Drawn per case in this range, so
    #: fatigue varies by customer rather than being a single global constant.
    fatigue_range: tuple[float, float] = (0.55, 0.82)
    #: Multiplier applied per additional retry attempt.
    retry_attempt_decay: float = 0.88
    #: Multiplier per prior failure on the case.
    failure_count_decay: float = 0.90
    #: A retry on a repaired instrument succeeds at this rate before modifiers.
    repaired_instrument_retry: float = 0.58
    reminder_effectiveness: float = 0.85
    reminder_ability_boost: float = 1.25
    method_update_boost: float = 1.45
    link_boost_dead_instrument: float = 1.25

    # ---- delivery ------------------------------------------------------------
    #: Share of (customer, channel) pairs that hard-bounce. Seeded per pair, so a bad
    #: address is consistently bad -- which is what makes a consecutive-failure counter
    #: meaningful rather than noise.
    delivery_failure_rate: float = 0.06

    #: Share of retry attempts on which the rail is unreachable. Distinct from a decline:
    #: the payment state is unknown, and the caller must re-present under the same
    #: idempotency key rather than treating it as a refusal.
    gateway_unavailable_rate: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        for code in STRUCTURAL_ZEROS:
            if self.base_retry.get(code, 0.0) > 0.0:
                raise InvalidScenario(
                    f"scenario {self.name!r} gives {code!r} a non-zero retry probability. "
                    f"That is a structural impossibility on real rails -- you cannot debit "
                    f"a closed account or re-present a dead card -- and allowing it would "
                    f"produce a world where brute force beats diagnosis, invalidating "
                    f"every comparison this simulator is used for.")
        for name, table in (("base_retry", self.base_retry),
                            ("passive_cure_rate", self.passive_cure_rate),
                            ("channel_reach", self.channel_reach)):
            for k, v in table.items():
                if not 0.0 <= float(v) <= 1.0:
                    raise InvalidScenario(f"{name}[{k}] = {v} is not a probability")
        if not 0.0 <= self.delivery_failure_rate <= 1.0:
            raise InvalidScenario("delivery_failure_rate is not a probability")
        lo, hi = self.fatigue_range
        if not 0.0 < lo <= hi <= 1.0:
            raise InvalidScenario(f"fatigue_range {self.fatigue_range} must satisfy 0 < lo <= hi <= 1")

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, scenario: str = "default",
             path: Path | str | None = None) -> SimulationConfig:
        """Load a named scenario, falling back to the coded defaults.

        An unknown scenario name raises rather than silently returning the default: a
        sweep that thinks it ran `pessimistic` and actually ran `default` would report a
        robustness result it never tested.
        """
        p = Path(path or DEFAULT_PATH)
        if not p.exists():
            if scenario != "default":
                raise InvalidScenario(
                    f"no scenario file at {p}; only the built-in 'default' is available")
            return cls()
        import yaml
        raw = yaml.safe_load(p.read_text()) or {}
        scenarios = raw.get("scenarios") or {}
        if scenario not in scenarios:
            raise InvalidScenario(
                f"unknown scenario {scenario!r}; {p} defines {sorted(scenarios)}")
        return cls.from_dict({"name": scenario, **(scenarios[scenario] or {})})

    @classmethod
    def scenario_names(cls, path: Path | str | None = None) -> list[str]:
        p = Path(path or DEFAULT_PATH)
        if not p.exists():
            return ["default"]
        import yaml
        return sorted((yaml.safe_load(p.read_text()) or {}).get("scenarios") or {})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimulationConfig:
        """Overlay `raw` on the defaults. Dict-valued fields merge key by key, so a
        scenario can change one failure code without restating the whole table."""
        base = cls()
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise InvalidScenario(
                f"scenario {raw.get('name', '?')!r} sets unknown parameters "
                f"{sorted(unknown)}; a typo here would silently do nothing")
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            current = getattr(base, f.name)
            value = raw[f.name]
            if isinstance(current, dict) and isinstance(value, dict):
                kwargs[f.name] = {**current, **{k: float(v) for k, v in value.items()}}
            elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
                kwargs[f.name] = tuple(float(v) for v in value)
            else:
                kwargs[f.name] = value
        return cls(**{**{f.name: getattr(base, f.name) for f in fields(cls)}, **kwargs})

    # ------------------------------------------------------------------ identity
    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Content hash of the whole world. Goes into the run record so a result can be
        tied to the parameterisation that produced it."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_all(path: Path | str | None = None) -> dict[str, SimulationConfig]:
    return {n: SimulationConfig.load(n, path) for n in SimulationConfig.scenario_names(path)}
