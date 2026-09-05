"""Phase 4 -- the deterministic policy validator.

**The model proposes; this engine disposes.** No component may execute an action that has
not been returned as `effective_action` by `validate()`. The LLM never holds a tool.

Every rule is one function with a stable ID, registered here. Rules are pure: given the
same state and proposed action they always return the same verdict, so a decision can be
re-derived from an audit row months later.

Evaluation order is deliberate:
  1. REJECT rules run first and short-circuit -- a blocked action is never modified into
     an allowed one.
  2. REVIEW rules run next and also short-circuit. An action a human must approve is not
     worth rewriting first, and rewriting it would change what the human is approving.
  3. MODIFY rules then run cumulatively, so a deferral and a channel change compose.

Three verdicts stop execution -- REJECT, HUMAN_REVIEW, and any error inside the gate --
and only two let it proceed. That asymmetry is the point: uncertainty resolves to *not
acting*, never to acting anyway.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from backend.app.config import (
    MAX_AGENT_STEPS,
    MAX_AUTO_RECOVERY_AMOUNT_USD,
    MAX_CONTACTS_PER_CASE,
    MAX_RETRIES,
    MIN_EXPECTED_RECOVERY_USD,
    MIN_RETRY_INTERVAL_HOURS,
    RECOVERY_HORIZON_DAYS,
    REVIEW_MAX_EXECUTION_FAILURES,
    REVIEW_MIN_DIAGNOSIS_CONFIDENCE,
)
from backend.app.models.enums import (
    CONTACT_ACTIONS,
    EXECUTABLE_ACTIONS,
    FailureCategory,
    FailureCode,
    InterventionType,
    PolicyDecision,
    category_of,
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
    #: Confidence of the diagnosis the proposal was built on. Below the configured floor
    #: the case goes to a human: acting decisively on a guess is how an automated system
    #: does confident damage. Defaults to 1.0 so a caller that does not supply a
    #: diagnosis is not silently routed to review.
    diagnosis_confidence: float = 1.0
    #: True when the diagnosis could not be mapped to a known cause at all.
    diagnosis_unknown: bool = False
    #: The customer has withdrawn consent to be contacted. Hard stop, not a preference.
    customer_opted_out: bool = False
    #: Consecutive executor failures on this case. Repeated failure is a signal that the
    #: world disagrees with the plan, and retrying harder is the wrong response.
    consecutive_execution_failures: int = 0
    #: The customer explicitly asked for a human.
    support_requested: bool = False
    #: Tenant the case belongs to. Carried so a rule can be tenant-specific without the
    #: engine reaching into a database.
    tenant_id: str = "default"


@dataclass
class Verdict:
    decision: PolicyDecision
    reason: str
    action: ProposedAction | None = None


Rule = Callable[[AgentState, ProposedAction, PolicyContext], Verdict | None]

REJECT_RULES: dict[str, Rule] = {}
REVIEW_RULES: dict[str, Rule] = {}
MODIFY_RULES: dict[str, Rule] = {}


def reject_rule(rule_id: str) -> Callable[[Rule], Rule]:
    def deco(fn: Rule) -> Rule:
        REJECT_RULES[rule_id] = fn
        fn.rule_id = rule_id  # type: ignore[attr-defined]
        return fn
    return deco


def review_rule(rule_id: str) -> Callable[[Rule], Rule]:
    """Registers a rule that routes to a human instead of refusing outright."""
    def deco(fn: Rule) -> Rule:
        REVIEW_RULES[rule_id] = fn
        fn.rule_id = rule_id  # type: ignore[attr-defined]
        return fn
    return deco


def modify_rule(rule_id: str) -> Callable[[Rule], Rule]:
    def deco(fn: Rule) -> Rule:
        MODIFY_RULES[rule_id] = fn
        fn.rule_id = rule_id  # type: ignore[attr-defined]
        return fn
    return deco


def all_rule_ids() -> list[str]:
    return sorted(REJECT_RULES) + sorted(REVIEW_RULES) + sorted(MODIFY_RULES)


# ===================================================================== REJECT rules
@reject_rule("R-ALLOWLIST")
def _allowlist(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
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
def _terminal(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """A closed case takes no further action. Guarantees the automatic stop after success."""
    if state.is_terminal:
        return Verdict(PolicyDecision.REJECT, f"case is terminal ({state.status.value})")
    return None


@reject_rule("R-RISK-BLOCK")
def _risk_block(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Fraud, high-risk and compliance holds are never automated. Escalation is the only
    permitted action, no matter what the model proposed."""
    if category_of(state.failure_code) is not FailureCategory.RISK_COMPLIANCE:
        return None
    if action.action in (InterventionType.ESCALATE_CASE, InterventionType.STOP):
        return None
    return Verdict(PolicyDecision.REJECT,
                   f"{state.failure_code.value} is risk/compliance: only human escalation is permitted")


@reject_rule("R-DEAD-INSTRUMENT")
def _dead_instrument(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Never re-present an instrument that is known dead. Retrying an expired card is
    pure waste and annoys the issuer."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.failure_code in DEAD_INSTRUMENT_CODES and not ctx.instrument_fixed:
        return Verdict(PolicyDecision.REJECT,
                       f"{state.failure_code.value}: payment method unchanged, retry cannot succeed")
    return None


@reject_rule("R-PERSISTENT-NORETRY")
def _persistent_noretry(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Closed and invalid accounts cannot be retried into success."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.failure_code in (FailureCode.INVALID_ACCOUNT, FailureCode.CLOSED_ACCOUNT):
        return Verdict(PolicyDecision.REJECT,
                       f"{state.failure_code.value}: account is not reachable by retry")
    return None


@reject_rule("R-DLQ")
def _dlq(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
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
def _max_retries(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Global ceiling. Nothing may raise it."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.attempt_count >= MAX_RETRIES:
        return Verdict(PolicyDecision.REJECT,
                       f"global retry limit reached ({state.attempt_count}/{MAX_RETRIES})")
    return None


@reject_rule("R-CAUSE-RETRY-CAP")
def _cause_retry_cap(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Per-cause ceiling, tighter than the global one where the cause warrants it."""
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    cap = RETRY_CAP_BY_CODE.get(state.failure_code)
    if cap is not None and state.attempt_count >= cap:
        return Verdict(PolicyDecision.REJECT,
                       f"{state.failure_code.value} retry cap reached ({state.attempt_count}/{cap})")
    return None


@reject_rule("R-OPT-OUT")
def _opt_out(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """A customer who has withdrawn consent is not contacted again, by any channel, for
    any reason, at any value. This is the one limit that no expected-profit calculation
    may outbid -- which is why it is a rule and not a term in the objective. A retry of
    an existing mandate is not a contact and is not blocked here."""
    if action.action in CONTACT_ACTIONS and ctx.customer_opted_out:
        return Verdict(PolicyDecision.REJECT,
                       "customer has opted out of contact; no automated message may be sent")
    return None


@reject_rule("R-CONTACT-CAP")
def _contact_cap(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Bounded customer contact. Over-contacting destroys more value than it recovers."""
    if action.action in CONTACT_ACTIONS and state.contact_count >= MAX_CONTACTS_PER_CASE:
        return Verdict(PolicyDecision.REJECT,
                       f"contact limit reached ({state.contact_count}/{MAX_CONTACTS_PER_CASE})")
    return None


@reject_rule("R-HORIZON")
def _horizon(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    if action.action in (InterventionType.STOP, InterventionType.ESCALATE_CASE):
        return None
    if ctx.elapsed_hours > RECOVERY_HORIZON_DAYS * 24:
        return Verdict(PolicyDecision.REJECT,
                       f"past the {RECOVERY_HORIZON_DAYS}-day recovery horizon")
    return None


@reject_rule("R-STEP-CAP")
def _step_cap(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Defence against a planner that will not settle. Bounds the agent loop absolutely."""
    if action.action is InterventionType.STOP:
        return None
    if state.step_count >= MAX_AGENT_STEPS:
        return Verdict(PolicyDecision.REJECT,
                       f"agent step ceiling reached ({state.step_count}/{MAX_AGENT_STEPS})")
    return None


@reject_rule("R-MIN-VALUE")
def _min_value(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Do not spend a paid contact chasing less than it costs to chase."""
    if action.action in CONTACT_ACTIONS and state.expected_recovery < MIN_EXPECTED_RECOVERY_USD:
        return Verdict(PolicyDecision.REJECT,
                       f"expected recovery ${state.expected_recovery:.2f} is below the "
                       f"${MIN_EXPECTED_RECOVERY_USD:.2f} floor for a paid contact")
    return None


@reject_rule("R-IDEMPOTENT")
def _idempotent(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """The same contact must not be sent twice for one case. Retries are exempt: a second
    re-presentment is a distinct, intended event."""
    once_only = {InterventionType.REQUEST_PAYMENT_METHOD_UPDATE, InterventionType.ESCALATE_CASE}
    if action.action in once_only and action.action.value in ctx.already_executed:
        return Verdict(PolicyDecision.REJECT,
                       f"{action.action.value} already performed for this case")
    return None


# ===================================================================== REVIEW rules
# These do not refuse an action -- they withhold it pending an authorised human. Nothing
# executes on a HUMAN_REVIEW verdict, so the system still fails closed; the difference is
# that the case stays alive in an operator's queue instead of being dropped.
@review_rule("R-AMOUNT-CAP")
def _amount_cap(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Above the configured ceiling, no money moves without a human.

    Previously a flat refusal. That is the wrong verdict: the largest recoveries in the
    queue are exactly the ones worth a human minute, and refusing them meant the system's
    safety limit was also its biggest revenue leak. The message always said "human
    approval required"; now the decision says it too.
    """
    if action.action is not InterventionType.RETRY_PAYMENT:
        return None
    if state.amount_usd > MAX_AUTO_RECOVERY_AMOUNT_USD:
        return Verdict(PolicyDecision.HUMAN_REVIEW,
                       f"${state.amount_usd:,.2f} exceeds the automatic recovery ceiling "
                       f"of ${MAX_AUTO_RECOVERY_AMOUNT_USD:,.2f}: human approval required")
    return None


@review_rule("R-REVIEW-LOW-CONFIDENCE")
def _low_confidence(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """An uncertain diagnosis must not drive a confident action.

    Control flow is exempt: WAIT and STOP are what an uncertain system *should* do, and
    routing "do nothing" to a human would flood the queue with cases needing no decision.
    """
    if action.action in (InterventionType.WAIT, InterventionType.STOP,
                         InterventionType.ESCALATE_CASE):
        return None
    if ctx.diagnosis_confidence < REVIEW_MIN_DIAGNOSIS_CONFIDENCE:
        return Verdict(PolicyDecision.HUMAN_REVIEW,
                       f"diagnosis confidence {ctx.diagnosis_confidence:.2f} is below the "
                       f"{REVIEW_MIN_DIAGNOSIS_CONFIDENCE:.2f} floor for automated action")
    return None


@review_rule("R-REVIEW-UNKNOWN-CAUSE")
def _unknown_cause(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """A processor code we cannot map is not evidence of anything. Guessing a cause and
    acting on it is how a normalisation gap becomes a customer-visible mistake."""
    if action.action in (InterventionType.WAIT, InterventionType.STOP,
                         InterventionType.ESCALATE_CASE):
        return None
    if ctx.diagnosis_unknown:
        return Verdict(PolicyDecision.HUMAN_REVIEW,
                       "root cause could not be determined from the available signals")
    return None


@review_rule("R-REVIEW-REPEATED-FAILURE")
def _repeated_failure(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """Repeated execution failure on one case means the world disagrees with the plan.
    The correct response is to ask someone, not to try the same thing more forcefully."""
    if action.action in (InterventionType.WAIT, InterventionType.STOP,
                         InterventionType.ESCALATE_CASE):
        return None
    if ctx.consecutive_execution_failures >= REVIEW_MAX_EXECUTION_FAILURES:
        return Verdict(PolicyDecision.HUMAN_REVIEW,
                       f"{ctx.consecutive_execution_failures} consecutive execution "
                       f"failures on this case; automated handling suspended")
    return None


@review_rule("R-REVIEW-SUPPORT-REQUESTED")
def _support_requested(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
    """The customer asked for a person. Continuing to automate at them after that is the
    behaviour that makes people hate dunning systems."""
    if action.action in (InterventionType.WAIT, InterventionType.STOP,
                         InterventionType.ESCALATE_CASE):
        return None
    if ctx.support_requested:
        return Verdict(PolicyDecision.HUMAN_REVIEW,
                       "customer has requested human support on this case")
    return None


# ===================================================================== MODIFY rules
@modify_rule("R-COOLDOWN")
def _cooldown(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
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
def _channel(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
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
def _escalate_high_value(state: AgentState, action: ProposedAction,
        ctx: PolicyContext) -> Verdict | None:
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
def _first_verdict(rules: dict[str, Rule], state: AgentState, action: ProposedAction,
                   ctx: PolicyContext) -> tuple[str, Verdict] | None:
    """First rule (in stable ID order) that objects, or None. Sorted rather than
    insertion-ordered so the rule that fires is a function of the inputs alone -- import
    order must never be able to change a recorded verdict."""
    for rid in sorted(rules):
        v = rules[rid](state, action, ctx)
        if v is not None:
            return rid, v
    return None


def validate(state: AgentState, action: ProposedAction,
             ctx: PolicyContext | None = None) -> PolicyResult:
    """The single gate. Returns the only action that may be executed.

    Four verdicts are possible and exactly two of them permit execution:

        REJECT        refused outright; the case may try a different route
        HUMAN_REVIEW  withheld pending an authorised human; nothing runs now
        MODIFY        rewritten, then re-validated from scratch
        APPROVE       no rule objected

    Any exception raised inside a rule is caught and converted to HUMAN_REVIEW. A gate
    that crashes must not be a gate that is bypassed, and in a financial system the safe
    reading of "the validator broke" is "do not act", not "act unvalidated".
    """
    ctx = ctx or PolicyContext()
    fired: list[str] = []

    try:
        hit = _first_verdict(REJECT_RULES, state, action, ctx)
        if hit is not None:
            rid, v = hit
            fired.append(rid)
            return PolicyResult(decision=PolicyDecision.REJECT, effective_action=None,
                                rules_fired=fired, reason=f"{rid}: {v.reason}")

        hit = _first_verdict(REVIEW_RULES, state, action, ctx)
        if hit is not None:
            rid, v = hit
            fired.append(rid)
            return PolicyResult(
                decision=PolicyDecision.HUMAN_REVIEW,
                # The action is carried so an operator can see and approve exactly what
                # was proposed. It is NOT executable: `allowed` is False on this verdict,
                # and the executor refuses anything whose decision is not APPROVE/MODIFY.
                effective_action=action, rules_fired=fired,
                reason=f"{rid}: {v.reason}")

        effective, reasons = action, []
        for rid in sorted(MODIFY_RULES):
            rewrite = MODIFY_RULES[rid](state, effective, ctx)
            if rewrite is not None and rewrite.action is not None:
                fired.append(rid)
                effective = rewrite.action
                reasons.append(f"{rid}: {rewrite.reason}")

        if reasons:
            # A rewritten action must clear the same bar as an original one. Without this
            # pass a MODIFY rule can synthesise an action the REJECT rules never saw --
            # R-ESCALATE-HIGH-VALUE turning STOP into ESCALATE_CASE, for instance, slipped
            # past R-IDEMPOTENT and could escalate the same case twice. The review tier is
            # re-run for the same reason: a rewrite must not launder an action past it.
            hit = _first_verdict(REJECT_RULES, state, effective, ctx)
            if hit is not None:
                rid, v = hit
                fired.append(rid)
                return PolicyResult(
                    decision=PolicyDecision.REJECT, effective_action=None,
                    rules_fired=fired,
                    reason=f"{rid}: {v.reason} (the rewritten action was refused)")
            hit = _first_verdict(REVIEW_RULES, state, effective, ctx)
            if hit is not None:
                rid, v = hit
                fired.append(rid)
                return PolicyResult(
                    decision=PolicyDecision.HUMAN_REVIEW, effective_action=effective,
                    rules_fired=fired,
                    reason=f"{rid}: {v.reason} (the rewritten action needs approval)")
            return PolicyResult(decision=PolicyDecision.MODIFY, effective_action=effective,
                                rules_fired=fired, reason="; ".join(reasons))
        return PolicyResult(decision=PolicyDecision.APPROVE, effective_action=effective,
                            rules_fired=fired, reason="approved: no rule objected")

    except Exception as exc:
        return PolicyResult(
            decision=PolicyDecision.HUMAN_REVIEW, effective_action=None,
            rules_fired=[*fired, "R-ENGINE-ERROR"],
            reason=f"R-ENGINE-ERROR: policy evaluation raised "
                   f"{type(exc).__name__}: {exc}; failing closed to human review")
