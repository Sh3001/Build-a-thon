"""What an action costs, and what a recovery is worth.

The original objective was `amount x P(recovery)`. Three things are wrong with it as a
decision rule, and each has a business consequence:

1. **It ignores the counterfactual.** A customer who was going to pay anyway scores at
   the top. Contacting them is pure cost, and on real delinquency data 44.8% of the
   top-250 by expected recovery paid on their own.
2. **It ignores cost.** Every candidate looks free, so "send everything to everyone"
   maximises it. The contact budget is then spent by whoever asks first.
3. **It ignores what a recovery nets.** A $100 card recovery is not $100: the processor
   takes a fee, the contact cost money, and a fraction of recovered payments are later
   disputed.

So the objective becomes expected *incremental* profit:

    E[profit | action]
      = amount x (P(recover | action) - P(recover | no action))    incremental revenue
      - processing_fee(amount) x P(recover | action)               only paid if it lands
      - action_cost                                                the send/retry itself
      - contact_cost                                               goodwill, per touch
      - risk_cost                                                  disputes, opt-outs
      - operational_cost                                           human minutes
      - inference_cost                                             tokens, if any

Every coefficient is configuration, loaded from `config/economics.yaml`. They are
*assumptions*, and naming them in a file a reviewer can open is the difference between an
assumption and a hidden constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from backend.app.config import ROOT
from backend.app.domain.money import Money
from backend.app.models.enums import CONTACT_ACTIONS, InterventionType

DEFAULT_ECONOMICS_PATH = ROOT / "config" / "economics.yaml"


@dataclass(frozen=True)
class ActionCost:
    """The direct cost of running one action once."""
    direct_usd: Decimal = Decimal("0")
    #: Goodwill/annoyance priced per customer touch. Not a real invoice, which is exactly
    #: why it must be explicit: an un-priced externality is one the optimiser spends
    #: freely.
    contact_usd: Decimal = Decimal("0")
    #: Human minutes, priced. Escalation is the expensive action and should look it.
    operational_usd: Decimal = Decimal("0")
    #: Expected cost of the downside this action can cause -- a dispute, an opt-out.
    risk_usd: Decimal = Decimal("0")

    @property
    def total(self) -> Decimal:
        return self.direct_usd + self.contact_usd + self.operational_usd + self.risk_usd


@dataclass
class CostModel:
    """Prices for the profit equation. Loaded from YAML; defaults match the shipped file
    so the system runs with no configuration present."""

    #: Proportional processor fee on a successful recovery.
    processing_fee_rate: Decimal = Decimal("0.029")
    #: Fixed per-successful-transaction fee.
    processing_fee_fixed_usd: Decimal = Decimal("0.30")
    #: Share of recovered payments later disputed or refunded, times what that costs.
    #: Applied to recovered revenue, not to the attempt: an attempt that fails cannot be
    #: charged back.
    chargeback_rate: Decimal = Decimal("0.004")
    chargeback_cost_usd: Decimal = Decimal("15.00")
    #: LLM spend attributable to one case, when a model was consulted.
    inference_usd: Decimal = Decimal("0")

    actions: dict[InterventionType, ActionCost] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def default(cls) -> CostModel:
        return cls(actions={
            InterventionType.RETRY_PAYMENT: ActionCost(
                direct_usd=Decimal("0.02"), risk_usd=Decimal("0.01")),
            InterventionType.SEND_REMINDER: ActionCost(
                direct_usd=Decimal("0.01"), contact_usd=Decimal("0.05"),
                risk_usd=Decimal("0.02")),
            InterventionType.SEND_PAYMENT_LINK: ActionCost(
                direct_usd=Decimal("0.04"), contact_usd=Decimal("0.05"),
                risk_usd=Decimal("0.02")),
            InterventionType.REQUEST_PAYMENT_METHOD_UPDATE: ActionCost(
                direct_usd=Decimal("0.04"), contact_usd=Decimal("0.06"),
                risk_usd=Decimal("0.03")),
            InterventionType.ESCALATE_CASE: ActionCost(
                direct_usd=Decimal("0"), operational_usd=Decimal("2.50")),
            InterventionType.WAIT: ActionCost(),
            InterventionType.STOP: ActionCost(),
        })

    @classmethod
    def load(cls, path: Path | str | None = None) -> CostModel:
        """Read `config/economics.yaml`, falling back to `default()` when absent.

        A missing file is not an error: the defaults are the shipped values and the
        project must run with no configuration. A *malformed* file is an error -- silently
        ignoring a typo in a cost coefficient would change every decision the optimiser
        makes with no signal at all.
        """
        p = Path(path or DEFAULT_ECONOMICS_PATH)
        if not p.exists():
            return cls.default()
        import yaml
        raw = yaml.safe_load(p.read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CostModel:
        base = cls.default()
        fees = raw.get("fees", {})
        model = cls(
            processing_fee_rate=Decimal(str(fees.get("processing_rate",
                                                     base.processing_fee_rate))),
            processing_fee_fixed_usd=Decimal(str(fees.get("processing_fixed_usd",
                                                          base.processing_fee_fixed_usd))),
            chargeback_rate=Decimal(str(fees.get("chargeback_rate", base.chargeback_rate))),
            chargeback_cost_usd=Decimal(str(fees.get("chargeback_cost_usd",
                                                     base.chargeback_cost_usd))),
            inference_usd=Decimal(str(raw.get("inference", {}).get("usd_per_case", 0))),
            actions=dict(base.actions),
        )
        for name, spec in (raw.get("actions") or {}).items():
            try:
                kind = InterventionType(name)
            except ValueError as exc:
                raise ValueError(
                    f"economics.yaml prices an unknown action {name!r}; the action space "
                    f"is a closed enum, so this is a typo rather than an extension "
                    f"({[a.value for a in InterventionType]})") from exc
            spec = spec or {}
            model.actions[kind] = ActionCost(
                direct_usd=Decimal(str(spec.get("direct_usd", 0))),
                contact_usd=Decimal(str(spec.get("contact_usd", 0))),
                operational_usd=Decimal(str(spec.get("operational_usd", 0))),
                risk_usd=Decimal(str(spec.get("risk_usd", 0))),
            )
        return model

    # ------------------------------------------------------------------ pricing
    def action_cost(self, action: InterventionType, *, fatigue_multiplier: float = 1.0,
                    with_inference: bool = False) -> Money:
        """Total cost of taking `action` once.

        `fatigue_multiplier` raises the *goodwill* component for a repeat contact only.
        The direct cost of an SMS does not rise with the fourth send; the damage does,
        and pricing that is what stops the optimiser treating contact as free.
        """
        c = self.actions.get(action, ActionCost())
        contact = c.contact_usd * Decimal(str(fatigue_multiplier)) \
            if action in CONTACT_ACTIONS else c.contact_usd
        total = c.direct_usd + contact + c.operational_usd + c.risk_usd
        if with_inference:
            total += self.inference_usd
        return Money.from_major(total)

    def processing_cost(self, recovered: Money) -> Money:
        """Fee on a successful recovery. Zero on a failed attempt -- a decline costs the
        action, not the fee."""
        if not recovered.is_positive:
            return Money.zero(recovered.currency)
        return recovered.scale(self.processing_fee_rate) + Money.from_major(
            self.processing_fee_fixed_usd, recovered.currency)

    def chargeback_cost(self, recovered: Money) -> Money:
        """Expected cost of the share of recoveries that are later disputed."""
        if not recovered.is_positive:
            return Money.zero(recovered.currency)
        return recovered.scale(self.chargeback_rate) + Money.from_major(
            self.chargeback_cost_usd * self.chargeback_rate, recovered.currency)

    def net_of_fees(self, recovered: Money) -> Money:
        """What a recovery is actually worth once the rails have taken their share."""
        return recovered - self.processing_cost(recovered) - self.chargeback_cost(recovered)

    def describe(self) -> dict:
        return {
            "processing_fee_rate": float(self.processing_fee_rate),
            "processing_fee_fixed_usd": float(self.processing_fee_fixed_usd),
            "chargeback_rate": float(self.chargeback_rate),
            "chargeback_cost_usd": float(self.chargeback_cost_usd),
            "inference_usd_per_case": float(self.inference_usd),
            "actions": {k.value: {
                "direct_usd": float(v.direct_usd), "contact_usd": float(v.contact_usd),
                "operational_usd": float(v.operational_usd), "risk_usd": float(v.risk_usd),
                "total_usd": float(v.total),
            } for k, v in sorted(self.actions.items(), key=lambda kv: kv[0].value)},
        }


_default: CostModel | None = None


def get_cost_model() -> CostModel:
    global _default
    if _default is None:
        _default = CostModel.load()
    return _default
