"""Phase 2 -- the mock payment rail. This is ground truth for every live run.

Both the baseline and RecoverAI act against this one simulator on the same cases, which
is the only reason the comparison between them means anything.

Design rules, chosen so the evaluation cannot flatter the agent:

1. **Independent of the ML model.** Outcomes are sampled from latents derived from the
   transaction, never from the model's predicted probability. The agent cannot be right
   merely because it is confident.
2. **Hard failures have zero retry success.** `invalid_account` and `closed_account`
   return 0.0 forever. Brute force cannot beat diagnosis; only changing the instrument can.
3. **Timing matters, and differently per cause.** Temporary faults decay; insufficient
   funds improves for about a week. Same structure as the training data, independently
   parameterised.
4. **Contact fatigue is real.** Each customer touch lowers the response rate of the next,
   so over-contacting destroys value instead of adding it.
5. **Deterministic.** Every draw is seeded by (seed, transaction_id, tag, attempt), so a
   run replays exactly and an idempotent retry returns the same answer.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from backend.app.models.enums import FailureCategory, FailureCode, category_of

#: Probability a *bare retry of the unchanged instrument* succeeds, before modifiers.
BASE_RETRY: dict[str, float] = {
    "bank_timeout": 0.62, "network_error": 0.68, "processor_unavailable": 0.55,
    "temporary_decline": 0.38,
    "insufficient_funds": 0.20, "payment_limit_exceeded": 0.24,
    "expired_card": 0.0,             # the instrument is dead; retrying it is theatre
    "invalid_payment_method": 0.0,
    "multiple_declines": 0.02, "invalid_account": 0.0, "closed_account": 0.0,
    "suspected_fraud": 0.0, "high_risk_transaction": 0.0, "compliance_hold": 0.0,
}

#: Probability a case self-cures within the horizon with ZERO intervention -- the
#: customer notices and pays, or the bank re-presents on its own. Without this the
#: no-touch control arm recovers nothing by construction, and every dollar the agent
#: collects looks causal when some of it would have arrived regardless.
PASSIVE_CURE_RATE: dict[str, float] = {
    "bank_timeout": 0.26, "network_error": 0.29, "processor_unavailable": 0.23,
    "temporary_decline": 0.17,
    "insufficient_funds": 0.15, "payment_limit_exceeded": 0.13,
    "expired_card": 0.07, "invalid_payment_method": 0.05,
    "multiple_declines": 0.04, "invalid_account": 0.01, "closed_account": 0.01,
    "suspected_fraud": 0.02, "high_risk_transaction": 0.03, "compliance_hold": 0.05,
}

#: Mean days until a self-cure lands, once one is going to happen.
PASSIVE_CURE_MEAN_DAYS = 8.0

#: Reach of each channel -- probability the message is seen at all.
CHANNEL_REACH: dict[str, float] = {
    "email": 0.42, "sms": 0.63, "whatsapp": 0.71, "in_app": 0.55,
}


@dataclass
class CaseLatents:
    """Hidden ground truth for one transaction. Never exposed to any agent component."""
    ability_to_pay: float
    responsiveness: float
    fatigue: float
    method_update_willingness: float
    payday_offset: float


@dataclass
class GatewayResult:
    success: bool
    detail: str
    probability: float
    amount: float = 0.0


@dataclass
class PaymentGateway:
    """Deterministic mock rail. `seed` fixes the whole world."""
    seed: int = 20260822
    _latents: dict[str, CaseLatents] = field(default_factory=dict, repr=False)
    #: transaction_id -> whether the customer has fixed their payment instrument
    _instrument_fixed: dict[str, bool] = field(default_factory=dict, repr=False)
    _contacts: dict[str, int] = field(default_factory=dict, repr=False)
    #: transaction_id -> hour of spontaneous resolution, or None
    _self_cure: dict[str, float | None] = field(default_factory=dict, repr=False)
    calls: int = 0

    # ------------------------------------------------------------------ internals
    def _rng(self, txn_id: str, tag: str, nonce: int = 0) -> np.random.Generator:
        h = hashlib.sha256(f"{self.seed}:{txn_id}:{tag}:{nonce}".encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "big"))

    def latents(self, txn: dict) -> CaseLatents:
        tid = txn["transaction_id"]
        if tid not in self._latents:
            r = self._rng(tid, "latents")
            psr = float(txn.get("previous_success_rate", 0.5))
            self._latents[tid] = CaseLatents(
                # Anchored on payment history, but not equal to it.
                ability_to_pay=float(np.clip(r.beta(2.0 + 4.0 * psr, 2.2), 0.03, 0.98)),
                responsiveness=float(np.clip(r.beta(2.2, 3.0), 0.05, 0.92)),
                fatigue=float(r.uniform(0.55, 0.82)),
                method_update_willingness=float(np.clip(r.beta(2.4, 2.4), 0.05, 0.95)),
                payday_offset=float(r.uniform(-2.0, 2.0)),
            )
        return self._latents[tid]

    def _fatigue_factor(self, txn_id: str) -> float:
        lat = self._latents.get(txn_id)
        n = self._contacts.get(txn_id, 0)
        return (lat.fatigue if lat else 0.7) ** n

    def _timing_factor(self, code: str, hours: float, lat: CaseLatents) -> float:
        """Cause-specific decay. Mirrors the structure of the training data without
        sharing its coefficients."""
        days = hours / 24.0
        cat = category_of(code)
        if cat is FailureCategory.TEMPORARY:
            return float(np.clip(np.exp(-0.16 * days), 0.05, 1.0))
        if code == "insufficient_funds":
            peak = 6.0 + lat.payday_offset          # payday lands in the first week
            return float(np.clip(1.0 - 0.055 * abs(days - peak), 0.25, 1.0) * 1.35)
        if cat is FailureCategory.CUSTOMER_ACTION:
            return float(np.clip(np.exp(-0.02 * days), 0.4, 1.0))
        return float(np.clip(np.exp(-0.05 * days), 0.2, 1.0))

    # ------------------------------------------------------------------ tools
    def retry_payment(self, txn: dict, hours_since_failure: float, attempt: int) -> GatewayResult:
        """Re-present the *existing* instrument. Cannot fix a dead one."""
        self.calls += 1
        tid, code = txn["transaction_id"], txn["failure_code"]
        lat = self.latents(txn)
        fixed = self._instrument_fixed.get(tid, False)

        if code in ("expired_card", "invalid_payment_method") and not fixed:
            return GatewayResult(False, f"{code}: instrument unchanged, retry cannot succeed", 0.0)
        if BASE_RETRY[code] <= 0.0 and not fixed:
            return GatewayResult(False, f"{code}: not recoverable by retry", 0.0)
        base = 0.58 if fixed else BASE_RETRY[code]

        p = base * self._timing_factor(code, hours_since_failure, lat)
        p *= 0.45 + 0.85 * lat.ability_to_pay
        p *= 0.88 ** max(0, attempt - 1)            # each re-presentment is worth less
        p *= 0.90 ** max(0, int(txn.get("failure_count", 1)) - 1)
        p = float(np.clip(p, 0.0, 0.95))

        ok = bool(self._rng(tid, "retry", attempt).random() < p)
        return GatewayResult(ok, f"retry {'succeeded' if ok else 'declined'} (p={p:.3f})", p,
                             float(txn.get("amount_usd", 0.0)) if ok else 0.0)

    def customer_pays_via_link(self, txn: dict, hours: float, channel: str, nonce: int) -> GatewayResult:
        """A payment link routes around the failed instrument entirely -- but needs the
        customer to act, so reach, responsiveness and fatigue all apply."""
        self.calls += 1
        tid, code = txn["transaction_id"], txn["failure_code"]
        lat = self.latents(txn)
        if category_of(code) is FailureCategory.RISK_COMPLIANCE:
            return GatewayResult(False, "risk hold: no customer-initiated path offered", 0.0)

        p = CHANNEL_REACH.get(channel, 0.45) * lat.responsiveness * self._fatigue_factor(tid)
        p *= 0.40 + 0.95 * lat.ability_to_pay
        p *= self._timing_factor(code, hours, lat)
        p *= 1.25 if code in ("expired_card", "invalid_payment_method") else 1.0
        p = float(np.clip(p, 0.0, 0.90))

        self._contacts[tid] = self._contacts.get(tid, 0) + 1
        ok = bool(self._rng(tid, "link", nonce).random() < p)
        return GatewayResult(ok, f"payment link {'paid' if ok else 'not paid'} (p={p:.3f})", p,
                             float(txn.get("amount_usd", 0.0)) if ok else 0.0)

    def customer_responds_to_reminder(self, txn: dict, hours: float, channel: str, nonce: int) -> GatewayResult:
        """A reminder does not itself collect. It raises the odds the *next* retry lands,
        which is why the agent must pair it with one."""
        self.calls += 1
        tid = txn["transaction_id"]
        lat = self.latents(txn)
        p = CHANNEL_REACH.get(channel, 0.45) * lat.responsiveness * self._fatigue_factor(tid)
        p = float(np.clip(p * 0.85, 0.0, 0.85))
        self._contacts[tid] = self._contacts.get(tid, 0) + 1
        ok = bool(self._rng(tid, "reminder", nonce).random() < p)
        if ok:
            # Acknowledged: the customer tops up, improving the next re-presentment.
            lat.ability_to_pay = float(np.clip(lat.ability_to_pay * 1.25, 0.0, 0.98))
        return GatewayResult(ok, f"reminder {'acknowledged' if ok else 'ignored'} (p={p:.3f})", p)

    def request_method_update(self, txn: dict, channel: str, nonce: int) -> GatewayResult:
        """The only action that can revive a dead instrument."""
        self.calls += 1
        tid = txn["transaction_id"]
        lat = self.latents(txn)
        p = CHANNEL_REACH.get(channel, 0.45) * lat.method_update_willingness * self._fatigue_factor(tid)
        p = float(np.clip(p * 1.45, 0.0, 0.88))
        self._contacts[tid] = self._contacts.get(tid, 0) + 1
        ok = bool(self._rng(tid, "update", nonce).random() < p)
        if ok:
            self._instrument_fixed[tid] = True
        return GatewayResult(ok, f"payment method {'updated' if ok else 'not updated'} (p={p:.3f})", p)

    # ------------------------------------------------------------------ counterfactual
    def self_cure_hour(self, txn: dict, horizon_hours: float = 720.0) -> float | None:
        """Hour (measured from the ORIGINAL failure) at which this case would resolve on
        its own with no intervention at all, or None if it never would.

        Drawn once per transaction from a dedicated RNG stream, so it is identical in
        every arm. That is what makes the control arm a true counterfactual rather than
        a separate population: the same case either self-cures or does not, whatever
        strategy is working it.
        """
        tid = txn["transaction_id"]
        if tid in self._self_cure:
            return self._self_cure[tid]

        r = self._rng(tid, "passive")
        code = txn["failure_code"]
        lat = self.latents(txn)
        rate = PASSIVE_CURE_RATE.get(code, 0.05) * (0.55 + 0.9 * lat.ability_to_pay)

        cure: float | None = None
        if r.random() < min(rate, 0.95):
            start = float(txn.get("days_since_failure", 0.0)) * 24.0
            when = start + float(r.exponential(PASSIVE_CURE_MEAN_DAYS * 24.0))
            cure = when if when <= horizon_hours else None
        self._self_cure[tid] = cure
        return cure

    def check_self_cure(self, txn: dict, hours: float) -> GatewayResult | None:
        """Has this case already resolved on its own by `hours`? Returns a result if so.

        Every arm must call this, or an arm that acts slowly would be credited with
        recoveries that simply happened while it waited.
        """
        cure = self.self_cure_hour(txn)
        if cure is not None and hours >= cure:
            return GatewayResult(True, f"self-cured with no intervention at hour {cure:.0f}",
                                 1.0, float(txn.get("amount_usd", 0.0)))
        return None

    def instrument_fixed(self, txn_id: str) -> bool:
        return self._instrument_fixed.get(txn_id, False)

    def contacts(self, txn_id: str) -> int:
        return self._contacts.get(txn_id, 0)

    def reset_case(self, txn_id: str) -> None:
        self._instrument_fixed.pop(txn_id, None)
        self._contacts.pop(txn_id, None)
        self._self_cure.pop(txn_id, None)
