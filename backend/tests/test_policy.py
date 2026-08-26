"""Phase 4 tests -- one per policy rule, plus a meta-test that keeps it that way.

If you add a rule to the engine and no test names its ID, `test_every_rule_has_a_test`
fails. The safety envelope cannot grow untested.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import (
    MAX_AGENT_STEPS, MAX_AUTO_RECOVERY_AMOUNT_USD, MAX_CONTACTS_PER_CASE, MAX_RETRIES,
    MIN_RETRY_INTERVAL_HOURS, RECOVERY_HORIZON_DAYS,
)
from backend.app.models.enums import (
    CaseStatus, Channel, FailureCode, InterventionType, PolicyDecision,
)
from backend.app.models.schemas import AgentState, ProposedAction, Transaction
from backend.app.policies.engine import (
    MODIFY_RULES, REJECT_RULES, PolicyContext, all_rule_ids, validate,
)


def state(code="bank_timeout", amount=100.0, ev=50.0, **kw) -> AgentState:
    return AgentState(transaction_id="t1", customer_id="c1", amount=amount, currency="USD",
                      failure_code=code, expected_recovery=ev, **kw)


def act(a=InterventionType.RETRY_PAYMENT, **kw) -> ProposedAction:
    return ProposedAction(action=a, **kw)


PASSING = PolicyContext(hours_since_last_attempt=999.0)


# ------------------------------------------------------------------ meta
def test_every_rule_has_a_test():
    src = Path(__file__).read_text()
    missing = [r for r in all_rule_ids() if r not in src]
    assert not missing, f"policy rules with no test: {missing}"


def test_rule_ids_are_unique():
    assert not (set(REJECT_RULES) & set(MODIFY_RULES))


def test_validate_never_returns_an_action_when_rejecting():
    """A rejected proposal must yield nothing executable. This is the whole guarantee."""
    r = validate(state("suspected_fraud"), act(), PASSING)
    assert r.decision is PolicyDecision.REJECT
    assert r.effective_action is None
    assert not r.allowed


# ------------------------------------------------------------------ REJECT rules
def test_R_ALLOWLIST_blocks_non_executable_actions():
    r = validate(state(), act(InterventionType.WAIT), PASSING)
    assert r.decision is not PolicyDecision.REJECT      # WAIT is control flow, allowed
    r2 = validate(state(), act(InterventionType.RETRY_PAYMENT), PASSING)
    assert r2.allowed


def test_R_ALLOWLIST_rejects_an_invented_tool():
    """The LLM cannot conjure a tool. Schema validation refuses an unknown action name,
    and if one were ever smuggled past it, "R-ALLOWLIST" refuses it at the gate."""
    with pytest.raises(Exception):
        ProposedAction(action="wire_transfer_to_attacker")

    # Bypass validation to prove the gate does not depend on the schema catching it.
    smuggled = ProposedAction.model_construct(action="wire_transfer_to_attacker",
                                              channel=None, delay_hours=0.0,
                                              message=None, reason="", source="llm")
    r = validate(state(), smuggled, PASSING)
    assert r.decision is PolicyDecision.REJECT
    assert "R-ALLOWLIST" in r.rules_fired
    assert r.effective_action is None


def test_R_TERMINAL_blocks_action_on_a_closed_case():
    r = validate(state(status=CaseStatus.RECOVERED), act(), PASSING)
    assert r.decision is PolicyDecision.REJECT and "R-TERMINAL" in r.rules_fired


def test_R_RISK_BLOCK_forbids_everything_but_escalation():
    for code in ("suspected_fraud", "high_risk_transaction", "compliance_hold"):
        for a in (InterventionType.RETRY_PAYMENT, InterventionType.SEND_PAYMENT_LINK,
                  InterventionType.SEND_REMINDER):
            r = validate(state(code), act(a), PASSING)
            assert r.decision is PolicyDecision.REJECT, f"{code}/{a} was not blocked"
            assert "R-RISK-BLOCK" in r.rules_fired
        assert validate(state(code), act(InterventionType.ESCALATE_CASE), PASSING).allowed


def test_R_DEAD_INSTRUMENT_blocks_retry_until_the_method_is_updated():
    for code in ("expired_card", "invalid_payment_method"):
        r = validate(state(code), act(), PASSING)
        assert "R-DEAD-INSTRUMENT" in r.rules_fired
        ok = validate(state(code), act(), PolicyContext(hours_since_last_attempt=999,
                                                        instrument_fixed=True))
        assert ok.allowed, "retry must be permitted once the instrument is replaced"


def test_R_PERSISTENT_NORETRY_blocks_closed_and_invalid_accounts():
    for code in ("invalid_account", "closed_account"):
        r = validate(state(code), act(), PASSING)
        assert r.decision is PolicyDecision.REJECT and "R-PERSISTENT-NORETRY" in r.rules_fired
    # a link routes around the instrument, so it stays available
    assert validate(state("closed_account"), act(InterventionType.SEND_PAYMENT_LINK), PASSING).allowed


def test_R_MAX_RETRIES_enforces_the_global_ceiling():
    r = validate(state("temporary_decline", attempt_count=MAX_RETRIES), act(), PASSING)
    assert r.decision is PolicyDecision.REJECT
    assert {"R-MAX-RETRIES", "R-CAUSE-RETRY-CAP"} & set(r.rules_fired)


def test_R_CAUSE_RETRY_CAP_is_tighter_for_insufficient_funds():
    """Spec: insufficient funds gets at most two retries, not the global three."""
    assert validate(state("insufficient_funds", attempt_count=1), act(), PASSING).allowed
    r = validate(state("insufficient_funds", attempt_count=2), act(), PASSING)
    assert r.decision is PolicyDecision.REJECT and "R-CAUSE-RETRY-CAP" in r.rules_fired
    # ...while a bank timeout still gets three
    assert validate(state("bank_timeout", attempt_count=2), act(), PASSING).allowed


def test_R_AMOUNT_CAP_blocks_automated_money_movement_above_the_ceiling():
    over = MAX_AUTO_RECOVERY_AMOUNT_USD + 1
    r = validate(state("bank_timeout", amount=over), act(), PASSING)
    assert r.decision is PolicyDecision.REJECT and "R-AMOUNT-CAP" in r.rules_fired
    assert validate(state("bank_timeout", amount=MAX_AUTO_RECOVERY_AMOUNT_USD - 1), act(), PASSING).allowed


def test_R_CONTACT_CAP_bounds_customer_contact():
    s = state("insufficient_funds", contact_count=MAX_CONTACTS_PER_CASE)
    r = validate(s, act(InterventionType.SEND_REMINDER), PASSING)
    assert r.decision is PolicyDecision.REJECT and "R-CONTACT-CAP" in r.rules_fired
    # retries are not customer contact, so they remain available
    assert validate(s, act(InterventionType.RETRY_PAYMENT), PASSING).allowed


def test_R_HORIZON_stops_work_after_the_recovery_window():
    ctx = PolicyContext(hours_since_last_attempt=999, elapsed_hours=RECOVERY_HORIZON_DAYS * 24 + 1)
    r = validate(state(), act(), ctx)
    assert r.decision is PolicyDecision.REJECT and "R-HORIZON" in r.rules_fired
    assert validate(state(), act(InterventionType.ESCALATE_CASE), ctx).allowed


def test_R_STEP_CAP_bounds_the_agent_loop():
    r = validate(state(step_count=MAX_AGENT_STEPS), act(), PASSING)
    assert r.decision is PolicyDecision.REJECT and "R-STEP-CAP" in r.rules_fired
    assert validate(state(step_count=MAX_AGENT_STEPS), act(InterventionType.STOP), PASSING).allowed


def test_R_MIN_VALUE_refuses_paid_contact_below_the_floor():
    r = validate(state("insufficient_funds", ev=0.10), act(InterventionType.SEND_REMINDER), PASSING)
    assert r.decision is PolicyDecision.REJECT and "R-MIN-VALUE" in r.rules_fired


def test_R_IDEMPOTENT_blocks_a_repeated_one_shot_action():
    ctx = PolicyContext(hours_since_last_attempt=999,
                        already_executed=frozenset({"request_payment_method_update"}))
    r = validate(state("expired_card", ev=50), act(InterventionType.REQUEST_PAYMENT_METHOD_UPDATE), ctx)
    assert r.decision is PolicyDecision.REJECT and "R-IDEMPOTENT" in r.rules_fired


# ------------------------------------------------------------------ MODIFY rules
def test_R_COOLDOWN_defers_rather_than_rejects():
    r = validate(state(), act(), PolicyContext(hours_since_last_attempt=2.0))
    assert r.decision is PolicyDecision.MODIFY and "R-COOLDOWN" in r.rules_fired
    assert r.effective_action.delay_hours == pytest.approx(MIN_RETRY_INTERVAL_HOURS - 2.0)
    assert r.allowed, "a deferred action is still permitted, just later"


def test_R_COOLDOWN_is_satisfied_once_the_interval_has_passed():
    r = validate(state(), act(), PolicyContext(hours_since_last_attempt=MIN_RETRY_INTERVAL_HOURS + 1))
    assert "R-COOLDOWN" not in r.rules_fired


def test_R_CHANNEL_switches_to_the_customer_preference():
    txn = Transaction(customer_id="c1", transaction_id="t1", amount=100, currency="USD",
                      payment_method="card", failure_code="insufficient_funds",
                      preferred_channel=Channel.WHATSAPP)
    s = state("insufficient_funds", ev=50.0)
    s.transaction = txn
    r = validate(s, act(InterventionType.SEND_REMINDER, channel=Channel.EMAIL), PASSING)
    assert r.decision is PolicyDecision.MODIFY and "R-CHANNEL" in r.rules_fired
    assert r.effective_action.channel is Channel.WHATSAPP


def test_R_ESCALATE_HIGH_VALUE_reroutes_a_dropped_big_case():
    s = state("bank_timeout", amount=MAX_AUTO_RECOVERY_AMOUNT_USD + 500)
    r = validate(s, act(InterventionType.STOP), PASSING)
    assert "R-ESCALATE-HIGH-VALUE" in r.rules_fired
    assert r.effective_action.action is InterventionType.ESCALATE_CASE


# ------------------------------------------------------------------ properties
def test_rejection_short_circuits_before_modification():
    """A blocked action must never be modified into an allowed one."""
    s = state("suspected_fraud", contact_count=0)
    r = validate(s, act(InterventionType.SEND_REMINDER, channel=Channel.EMAIL),
                 PolicyContext(hours_since_last_attempt=0.0))
    assert r.decision is PolicyDecision.REJECT
    assert r.effective_action is None
    assert "R-CHANNEL" not in r.rules_fired


def test_validation_is_pure_and_repeatable():
    s, a = state("insufficient_funds"), act()
    ctx = PolicyContext(hours_since_last_attempt=3.0)
    first = validate(s, a, ctx)
    for _ in range(5):
        again = validate(s, a, ctx)
        assert again.decision == first.decision
        assert again.rules_fired == first.rules_fired
        assert again.reason == first.reason


@pytest.mark.parametrize("code", [c.value for c in FailureCode])
def test_no_risk_case_can_ever_be_retried(code):
    """Swept across the entire taxonomy: fraud/compliance never gets an executable retry."""
    from backend.app.models.enums import FailureCategory, category_of
    r = validate(state(code), act(InterventionType.RETRY_PAYMENT), PASSING)
    if category_of(code) is FailureCategory.RISK_COMPLIANCE:
        assert r.decision is PolicyDecision.REJECT and r.effective_action is None


# ---------------------------------------------------------------- R-DLQ
def test_dlq_rule_blocks_contact_on_a_quarantined_channel():
    """R-DLQ: a channel that has hard-bounced repeatedly takes no further contact."""
    st = state(code="insufficient_funds")
    action = act(InterventionType.SEND_REMINDER)
    ctx = PolicyContext(quarantined_channels=frozenset({"email"}), contact_channel="email")
    res = validate(st, action, ctx)
    assert not res.allowed
    assert "R-DLQ" in res.rules_fired


def test_dlq_rule_allows_a_healthy_channel():
    st = state(code="insufficient_funds")
    action = act(InterventionType.SEND_REMINDER)
    ctx = PolicyContext(quarantined_channels=frozenset({"sms"}), contact_channel="email")
    res = validate(st, action, ctx)
    assert "R-DLQ" not in res.rules_fired


def test_dlq_rule_does_not_block_a_retry():
    """Quarantine is about messaging a dead address, not about charging a card."""
    st = state(code="insufficient_funds")
    action = act(InterventionType.RETRY_PAYMENT)
    ctx = PolicyContext(quarantined_channels=frozenset({"email"}), contact_channel="email")
    res = validate(st, action, ctx)
    assert "R-DLQ" not in res.rules_fired
