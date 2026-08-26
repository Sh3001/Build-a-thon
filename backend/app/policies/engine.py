"""Phase 4 -- the deterministic policy validator.

**The model proposes; this engine disposes.** No component may execute an action that has
not been returned as `effective_action` by `validate()`. The LLM never holds a tool.

Every rule is one function with a stable ID, registered here. Rules are pure: given the
same state and proposed action they always return the same verdict, so a decision can be
re-derived from an audit row months later.

Evaluation order is deliberate:
  1. REJECT rules run first and short-circuit -- a blocked action is never modified into
     an allowed one.
  2. MODIFY rules then run cumulatively, so a deferral and a channel change compose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from backend.app.config import (
    MAX_AGENT_STEPS, MAX_AUTO_RECOVERY_AMOUNT_USD, MAX_CONTACTS_PER_CASE, MAX_RETRIES,
    MIN_EXPECTED_RECOVERY_USD, MIN_RETRY_INTERVAL_HOURS, RECOVERY_HORIZON_DAYS,
)
from backend.app.models.enums import (
    CONTACT_ACTIONS, EXECUTABLE_ACTIONS, Channel, FailureCategory, FailureCode, InterventionType,
    PolicyDecision, category_of,
)
from backend.app.models.schemas import AgentState, PolicyResult, ProposedAction

#: Per-cause retry ceilings, tighter than the global cap where the cause warrants it.
RETRY_CAP_BY_CODE: dict[FailureCode, int] = {
    FailureCode.INSUFFICIENT_FUNDS: 2,      # spec: reminder, retry, then a link
    FailureCode.PAYMENT_LIMIT_EXCEEDED: 2,
    FailureCode.TEMPORARY_DECLINE: 3,
    FailureCode.BANK_TIMEOUT: 3,
    FailureCode.NETWORK_ERROR: 3,
    FailureCode.PROCESSOR_UNAVAILABLE: 3,
    FailureCode.MULTIPLE_DECLINES: 1,
}

#: Instruments that cannot be re-presented unchanged.
DEAD_INSTRUMENT_CODES = frozenset({FailureCode.EXPIRED_CARD, FailureCode.INVALID_PAYMENT_METHOD})


@dataclass
class PolicyContext:
    """Everything a rule may look at beyond the agent state itself."""
    hours_since_last_attempt: float = 1e9
    instrument_fixed: bool = False
    elapsed_hours: float = 0.0
    already_executed: frozenset[str] = field(default_factory=frozenset)
    #: Channels quarantined for this customer after repeated hard delivery failures.
    #: Empty when no DLQ is attached, which leaves R-DLQ inert.
    quarantined_channels: frozenset[str] = field(default_factory=frozenset)
    #: The channel a contact would actually go out on, resolved by the caller.
    contact_channel: str = ""


@dataclass
class Verdict:
    decision: PolicyDecision
    reason: str
    action: ProposedAction | None = None


Rule = Callable[[AgentState, ProposedAction, PolicyContext], Verdict | None]

REJECT_RULES: dict[str, Rule] = {}
MODIFY_RULES: dict[str, Rule] = {}


def reject_rule(rule_id: str):
    def deco(fn: Rule) -> Rule:
        REJECT_RULES[rule_id] = fn
        fn.rule_id = rule_id  # type: ignore[attr-defined]
        return fn
    return deco


def modify_rule(rule_id: str):
    def deco(fn: Rule) -> Rule:
        MODIFY_RULES[rule_id] = fn
        fn.rule_id = rule_id  # type: ignore[attr-defined]
        return fn
    return deco


def all_rule_ids() -> list[str]:
    return sorted(REJECT_RULES) + sorted(MODIFY_RULES)


# ===================================================================== REJECT rules
@reject_rule("R-ALLOWLIST")
def _allowlist(state, action, ctx):
    """Only actions in the enum allowlist may ever run. This is the hard boundary the
    LLM cannot cross -- an invented tool name dies here."""
    if not isinstance(action.action, InterventionType):
        # Defence in depth: schema validation should have caught this, but the gate does
        # not rely on that. An unrecognised action name is refused outright.
        return Verdict(PolicyDecision.REJECT,
                       f"action {action.action!r} is not on the allowlist")
    if action.action in (InterventionType.WAIT, InterventionType.STOP):
        return None
    if action.action not in EXECUTABLE_ACTIONS:
        return Verdict(PolicyDecision.REJECT, f"action {action.action.value} is not executable")
    return None


@reject_rule("R-TERMINAL")
def _terminal(state, action, ctx):
    """A closed case takes no further action. Guarantees the automatic stop after success."""
    if state.is_terminal:
        return Verdict(PolicyDecision.REJECT, f"case is terminal ({state.status.value})")
    return None


@reject_rule("R-RISK-BLOCK")
def _risk_block(state, action, ctx):
    """Fraud, high-risk and compliance holds are never automated. Escalation is the only
    permitted action, no matter what the model proposed."""
    if category_of(state.failure_code) is not FailureCategory.RISK_COMPLIANCE:
        return None
    if action.action in (InterventionType.ESCALATE_CASE, InterventionType.STOP):
        return None
    return Verdict(PolicyDecision.REJECT,
                   f"{state.failure_code.value} is risk/compliance: only human escalation is permitted")


@reject_rule("R-DEAD-INSTRUMENT")
def _dead_instrument(state, action, ctx):
    """Never re-present an instrument that is known dead. Retrying an expired card is
    pure waste and annoys the issuer."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.failure_code in DEAD_INSTRUMENT_CODES and not ctx.instrument_fixed:
        return Verdict(PolicyDecision.REJECT,
                       f"{state.failure_code.value}: payment method unchanged, retry cannot succeed")
    return None


@reject_rule("R-PERSISTENT-NORETRY")
def _persistent_noretry(state, action, ctx):
    """Closed and invalid accounts cannot be retried into success."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.failure_code in (FailureCode.INVALID_ACCOUNT, FailureCode.CLOSED_ACCOUNT):
        return Verdict(PolicyDecision.REJECT,
                       f"{state.failure_code.value}: account is not reachable by retry")
    return None


@reject_rule("R-DLQ")
def _dlq(state, action, ctx):
    """A channel that has hard-bounced repeatedly for this customer is quarantined, and
    no further contact goes out on it. Retrying a dead address costs money per attempt,
    delivers nothing, and on a real ESP damages sender reputation for every other
    customer -- so the pair is parked for human review instead of retried."""
    if action.action not in CONTACT_ACTIONS:
        return None
    channel = (action.channel.value if action.channel else ctx.contact_channel)
    if channel and channel in ctx.quarantined_channels:
        return Verdict(PolicyDecision.REJECT,
                       f"{channel} is quarantined for this customer after repeated "
                       f"delivery failures; queued for review")
    return None


@reject_rule("R-MAX-RETRIES")
def _max_retries(state, action, ctx):
    """Global ceiling. Nothing may raise it."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.attempt_count >= MAX_RETRIES:
        return Verdict(PolicyDecision.REJECT,
                       f"global retry limit reached ({state.attempt_count}/{MAX_RETRIES})")
    return None


@reject_rule("R-CAUSE-RETRY-CAP")
def _cause_retry_cap(state, action, ctx):
    """Per-cause ceiling, tighter than the global one where the cause warrants it."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    cap = RETRY_CAP_BY_CODE.get(state.failure_code)
    if cap is not None and state.attempt_count >= cap:
        return Verdict(PolicyDecision.REJECT,
                       f"{state.failure_code.value} retry cap reached ({state.attempt_count}/{cap})")
    return None


@reject_rule("R-AMOUNT-CAP")
def _amount_cap(state, action, ctx):
    """Above the configured ceiling, no money moves without a human."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.amount_usd > MAX_AUTO_RECOVERY_AMOUNT_USD:
        return Verdict(PolicyDecision.REJECT,
                       f"${state.amount_usd:,.2f} exceeds the automatic recovery ceiling "
                       f"of ${MAX_AUTO_RECOVERY_AMOUNT_USD:,.2f}: human approval required")
    return None


@reject_rule("R-CONTACT-CAP")
def _contact_cap(state, action, ctx):
    """Bounded customer contact. Over-contacting destroys more value than it recovers."""
    if action.action in CONTACT_ACTIONS and state.contact_count >= MAX_CONTACTS_PER_CASE:
        return Verdict(PolicyDecision.REJECT,
                       f"contact limit reached ({state.contact_count}/{MAX_CONTACTS_PER_CASE})")
    return None


@reject_rule("R-HORIZON")
def _horizon(state, action, ctx):
    if action.action in (InterventionType.STOP, InterventionType.ESCALATE_CASE):
        return None
    if ctx.elapsed_hours > RECOVERY_HORIZON_DAYS * 24:
        return Verdict(PolicyDecision.REJECT,
                       f"past the {RECOVERY_HORIZON_DAYS}-day recovery horizon")
    return None


@reject_rule("R-STEP-CAP")
def _step_cap(state, action, ctx):
    """Defence against a planner that will not settle. Bounds the agent loop absolutely."""
    if action.action is InterventionType.STOP:
        return None
    if state.step_count >= MAX_AGENT_STEPS:
        return Verdict(PolicyDecision.REJECT,
                       f"agent step ceiling reached ({state.step_count}/{MAX_AGENT_STEPS})")
    return None


@reject_rule("R-MIN-VALUE")
def _min_value(state, action, ctx):
    """Do not spend a paid contact chasing less than it costs to chase."""
    if action.action in CONTACT_ACTIONS and state.expected_recovery < MIN_EXPECTED_RECOVERY_USD:
        return Verdict(PolicyDecision.REJECT,
                       f"expected recovery ${state.expected_recovery:.2f} is below the "
                       f"${MIN_EXPECTED_RECOVERY_USD:.2f} floor for a paid contact")
    return None


@reject_rule("R-IDEMPOTENT")
def _idempotent(state, action, ctx):
    """The same contact must not be sent twice for one case. Retries are exempt: a second
    re-presentment is a distinct, intended event."""
    once_only = {InterventionType.REQUEST_PAYMENT_METHOD_UPDATE, InterventionType.ESCALATE_CASE}
    if action.action in once_only and action.action.value in ctx.already_executed:
        return Verdict(PolicyDecision.REJECT,
                       f"{action.action.value} already performed for this case")
    return None


# ===================================================================== MODIFY rules
@modify_rule("R-COOLDOWN")
def _cooldown(state, action, ctx):
    """Enforce the minimum retry interval by deferring, not by rejecting -- the action is
    legitimate, just early."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    wait = MIN_RETRY_INTERVAL_HOURS - ctx.hours_since_last_attempt
    if wait > 0 and action.delay_hours < wait:
        return Verdict(PolicyDecision.MODIFY,
                       f"deferred {wait:.1f}h to respect the {MIN_RETRY_INTERVAL_HOURS}h "
                       f"minimum retry interval",
                       action.model_copy(update={"delay_hours": round(wait, 2)}))
    return None


@modify_rule("R-CHANNEL")
def _channel(state, action, ctx):
    """Contact on the customer's stated preferred channel."""
    if action.channel is None or state.transaction is None:
        return None
    pref = state.transaction.preferred_channel
    if action.channel != pref:
        return Verdict(PolicyDecision.MODIFY,
                       f"channel switched to the customer's preference ({pref.value})",
                       action.model_copy(update={"channel": pref}))
    return None


@modify_rule("R-ESCALATE-HIGH-VALUE")
def _escalate_high_value(state, action, ctx):
    """A high-value case that has exhausted its automated options goes to a human rather
    than being silently dropped."""
    if action.action is not InterventionType.STOP:
        return None
    if state.amount_usd > MAX_AUTO_RECOVERY_AMOUNT_USD and state.amount_recovered <= 0:
        return Verdict(PolicyDecision.MODIFY,
                       f"${state.amount_usd:,.2f} case routed to human review instead of being dropped",
                       action.model_copy(update={"action": InterventionType.ESCALATE_CASE}))
    return None


# ===================================================================== entry point
def validate(state: AgentState, action: ProposedAction,
             ctx: PolicyContext | None = None) -> PolicyResult:
    """The single gate. Returns the only action that may be executed."""
    ctx = ctx or PolicyContext()
    fired: list[str] = []

    for rid in sorted(REJECT_RULES):
        v = REJECT_RULES[rid](state, action, ctx)
        if v is not None:
            fired.append(rid)
            return PolicyResult(decision=PolicyDecision.REJECT, effective_action=None,
                                rules_fired=fired, reason=f"{rid}: {v.reason}")

    effective, reasons = action, []
    for rid in sorted(MODIFY_RULES):
        v = MODIFY_RULES[rid](state, effective, ctx)
        if v is not None and v.action is not None:
            fired.append(rid)
            effective = v.action
            reasons.append(f"{rid}: {v.reason}")

    if reasons:
        # A rewritten action must clear the same bar as an original one. Without this
        # pass a MODIFY rule can synthesise an action the REJECT rules never saw --
        # R-ESCALATE-HIGH-VALUE turning STOP into ESCALATE_CASE, for instance, slipped
        # past R-IDEMPOTENT and could escalate the same case twice.
        for rid in sorted(REJECT_RULES):
            v = REJECT_RULES[rid](state, effective, ctx)
            if v is not None:
                fired.append(rid)
                return PolicyResult(
                    decision=PolicyDecision.REJECT, effective_action=None,
                    rules_fired=fired,
                    reason=f"{rid}: {v.reason} (the rewritten action was refused)")
        return PolicyResult(decision=PolicyDecision.MODIFY, effective_action=effective,
                            rules_fired=fired, reason="; ".join(reasons))
    return PolicyResult(decision=PolicyDecision.APPROVE, effective_action=effective,
                        rules_fired=fired, reason="approved: no rule objected")
