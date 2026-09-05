"""Phase 6 -- the recovery strategy agent: which intervention, and *when*.

This proposes only. Everything it returns goes through the policy engine before it can
run, so a bad proposal is a wasted cycle, never an unsafe action.

Three ideas carry most of the value over a fixed retry schedule:

* **Repair before retry.** A dead instrument is fixed by asking the customer to replace
  it. Retrying it first is guaranteed waste.
* **Route around.** When the instrument cannot be repaired -- a closed account -- a payment
  link lets the customer pay by other means. Retrying cannot ever work; a link can.
* **Time the retry to the cause.** Transient faults decay, so retry immediately. Insufficient
  funds does the opposite: the account refills around payday, so the correct move is to
  *wait* into that window rather than burn attempts in the first 72 hours.
"""
from __future__ import annotations

from backend.app.config import (
    MAX_AUTO_RECOVERY_AMOUNT_USD,
    MAX_CONTACTS_PER_CASE,
    MIN_EXPECTED_RECOVERY_USD,
    MIN_RETRY_INTERVAL_HOURS,
)
from backend.app.models.enums import (
    Channel,
    FailureCategory,
    FailureCode,
    InterventionType,
)
from backend.app.models.schemas import AgentState, Diagnosis, ProposedAction

#: Hours after the original failure at which an insufficient-funds retry is most likely to
#: land. The account refills around payday; retrying at hour 24 mostly wastes an attempt.
PAYDAY_TARGET_HOURS = 132.0        # ~5.5 days

#: Retry caps mirrored from policy so the agent proposes within its envelope rather than
#: being corrected. Policy remains the enforcer.
SOFT_RETRY_CAP = {
    FailureCode.INSUFFICIENT_FUNDS: 2,
    FailureCode.PAYMENT_LIMIT_EXCEEDED: 2,
    FailureCode.MULTIPLE_DECLINES: 1,
}

DEAD_INSTRUMENT = {FailureCode.EXPIRED_CARD, FailureCode.INVALID_PAYMENT_METHOD}
UNREACHABLE = {FailureCode.CLOSED_ACCOUNT, FailureCode.INVALID_ACCOUNT}


def _done(state: AgentState, action: InterventionType) -> bool:
    """An action already taken *or* already refused by policy is off the table. Without
    the second half, a deterministic planner would re-propose a rejected action forever."""
    return action.value in state.actions_taken or action.value in state.blocked_actions


def _may_retry(state: AgentState) -> bool:
    return InterventionType.RETRY_PAYMENT.value not in state.blocked_actions


def _channel(state: AgentState) -> Channel:
    return state.transaction.preferred_channel if state.transaction else Channel.EMAIL


def _stop(reason: str) -> ProposedAction:
    return ProposedAction(action=InterventionType.STOP, reason=reason)


def _retry(reason: str, delay: float = 0.0) -> ProposedAction:
    return ProposedAction(action=InterventionType.RETRY_PAYMENT,
                          delay_hours=max(0.0, delay), reason=reason)


def select_intervention(state: AgentState, dx: Diagnosis) -> ProposedAction:
    """Choose the next action for this case. Pure, deterministic, always available."""
    root = dx.root_cause
    cat = dx.category
    ch = _channel(state)
    contacts_left = state.contact_count < MAX_CONTACTS_PER_CASE

    # --- never automate risk ---------------------------------------------------
    if cat is FailureCategory.RISK_COMPLIANCE:
        return ProposedAction(action=InterventionType.ESCALATE_CASE, reason=(
            f"{root.value} requires human review; no automated recovery is permitted"))

    # --- above the ceiling, a human decides ------------------------------------
    if state.amount_usd > MAX_AUTO_RECOVERY_AMOUNT_USD:
        return ProposedAction(action=InterventionType.ESCALATE_CASE, reason=(
            f"${state.amount_usd:,.2f} exceeds the automatic recovery ceiling"))

    # --- not worth the contact cost --------------------------------------------
    # Free actions (a retry costs no customer goodwill) are still worth taking, so this
    # floor only rules out cases where the ONLY remaining move is a paid contact.
    if state.expected_recovery < MIN_EXPECTED_RECOVERY_USD and state.attempt_count == 0 \
            and not dx.retry_viable:
        return _stop(f"expected recovery ${state.expected_recovery:.2f} does not justify "
                     f"a paid intervention")

    # --- dead instrument: repair, then retry; if unrepaired, route around -------
    if root in DEAD_INSTRUMENT:
        if state.instrument_fixed and _may_retry(state):
            return _retry("payment method was replaced; re-presenting the invoice")
        if not _done(state, InterventionType.REQUEST_PAYMENT_METHOD_UPDATE) and contacts_left:
            return ProposedAction(
                action=InterventionType.REQUEST_PAYMENT_METHOD_UPDATE, channel=ch,
                reason=f"{root.value}: the instrument is dead, so it must be replaced "
                       f"before any retry can succeed")
        if not _done(state, InterventionType.SEND_PAYMENT_LINK) and contacts_left:
            return ProposedAction(
                action=InterventionType.SEND_PAYMENT_LINK, channel=ch,
                reason="method not updated; a payment link lets the customer pay by "
                       "another means without touching the dead instrument")
        return _stop("instrument unrepaired and contact budget spent")

    # --- unreachable account: retry is impossible, only a link can work --------
    if root in UNREACHABLE:
        if not _done(state, InterventionType.SEND_PAYMENT_LINK) and contacts_left:
            return ProposedAction(
                action=InterventionType.SEND_PAYMENT_LINK, channel=ch,
                reason=f"{root.value}: the account cannot be debited, so recovery depends "
                       f"entirely on the customer paying by another method")
        return _stop(f"{root.value}: no remaining path to collection")

    # --- chronic decliner: one attempt, then hand the choice to the customer ----
    if root is FailureCode.MULTIPLE_DECLINES:
        if _may_retry(state) and state.attempt_count < SOFT_RETRY_CAP[root]:
            return _retry("single re-presentment before switching approach",
                          delay=MIN_RETRY_INTERVAL_HOURS)
        if not _done(state, InterventionType.SEND_PAYMENT_LINK) and contacts_left:
            return ProposedAction(action=InterventionType.SEND_PAYMENT_LINK, channel=ch,
                                  reason="issuer keeps declining; offer an alternative rail")
        return _stop("repeated declines with no alternative taken up")

    # --- insufficient funds: nudge, then retry into the payday window ----------
    if root is FailureCode.INSUFFICIENT_FUNDS:
        if state.attempt_count == 0 and not _done(state, InterventionType.SEND_REMINDER) \
                and contacts_left:
            return ProposedAction(
                action=InterventionType.SEND_REMINDER, channel=ch,
                reason="warn the customer before re-presenting, so the balance is there "
                       "when the debit lands")
        if _may_retry(state) and state.attempt_count < SOFT_RETRY_CAP[root]:
            # Wait into the payday window rather than spending the attempt early.
            wait = PAYDAY_TARGET_HOURS - state.elapsed_hours
            if state.attempt_count == 0 and wait > MIN_RETRY_INTERVAL_HOURS:
                return _retry(
                    f"defer the re-presentment to ~{PAYDAY_TARGET_HOURS/24:.1f} days after "
                    f"failure, when the account is most likely to be funded", delay=wait)
            return _retry("re-present now that the account has had time to refill",
                          delay=MIN_RETRY_INTERVAL_HOURS)
        if not _done(state, InterventionType.SEND_PAYMENT_LINK) and contacts_left:
            return ProposedAction(
                action=InterventionType.SEND_PAYMENT_LINK, channel=ch,
                reason="retries exhausted; let the customer choose the timing and method")
        return _stop("funding never materialised within the horizon")

    # --- limit exceeded: a different rail beats the same one -------------------
    if root is FailureCode.PAYMENT_LIMIT_EXCEEDED:
        if not _done(state, InterventionType.SEND_PAYMENT_LINK) and contacts_left:
            return ProposedAction(
                action=InterventionType.SEND_PAYMENT_LINK, channel=ch,
                reason="the transaction breaches a per-payment limit, so the same rail "
                       "will keep failing; offer an alternative")
        if _may_retry(state) and state.attempt_count < SOFT_RETRY_CAP[root]:
            return _retry("re-present after the limit window resets",
                          delay=MIN_RETRY_INTERVAL_HOURS)
        return _stop("payment limit not resolved")

    # --- transient faults: retry, and retry early ------------------------------
    if cat is FailureCategory.TEMPORARY:
        if not dx.retry_viable:
            return _stop("transient window closed")
        if _may_retry(state) and state.attempt_count == 0:
            # No prior attempt means no cooldown applies: go immediately, because the
            # success probability of a transient fault decays from the moment it happens.
            return _retry("transient fault: re-present immediately, before the recovery "
                          "window decays")
        if _may_retry(state) and state.attempt_count < 3:
            return _retry("re-present after the mandatory cooldown",
                          delay=MIN_RETRY_INTERVAL_HOURS)
        if not _done(state, InterventionType.SEND_PAYMENT_LINK) and contacts_left \
                and state.expected_recovery > MIN_EXPECTED_RECOVERY_USD * 5:
            return ProposedAction(action=InterventionType.SEND_PAYMENT_LINK, channel=ch,
                                  reason="automated retries exhausted on a valuable case")
        return _stop("retries exhausted on a transient fault")

    # --- anything else -------------------------------------------------------
    if _may_retry(state) and state.attempt_count < 2:
        return _retry("default re-presentment", delay=MIN_RETRY_INTERVAL_HOURS)
    return _stop("no remaining strategy for this cause")
