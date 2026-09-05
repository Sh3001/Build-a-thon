"""The expected-incremental-profit objective, and the honesty of its estimates."""
from __future__ import annotations

import pytest

from backend.app.adapters.processor_codes import CanonicalCause
from backend.app.agents.diagnose import diagnose
from backend.app.decision.economics import CostModel
from backend.app.decision.optimizer import (
    MEASURED,
    PRIOR,
    EffectModel,
    ProfitOptimizer,
)
from backend.app.domain.money import Money
from backend.app.models.enums import InterventionType
from backend.app.models.schemas import AgentState, Transaction


def _case(code: str, amount: float = 400.0, **kw) -> tuple[AgentState, object]:
    t = Transaction(customer_id="c", transaction_id="t", amount=amount,
                    payment_method="card", failure_code=code,
                    avg_transaction_value=200.0)
    s = AgentState(transaction_id="t", customer_id="c", amount=amount,
                   failure_code=code, transaction=t, recovery_probability=0.45,
                   expected_recovery=amount * 0.45, **kw)
    return s, diagnose(t)


@pytest.fixture
def opt():
    return ProfitOptimizer()


def test_a_dead_instrument_is_never_retried(opt):
    """A structural claim about the rails, not a statistical one: no coefficient can
    make re-presenting an expired card work."""
    for code in ("expired_card", "invalid_payment_method", "closed_account"):
        state, dx = _case(code)
        retry = next(c for c in opt.candidates(state, dx)
                     if c.action is InterventionType.RETRY_PAYMENT)
        assert not retry.feasible
        assert "structural" in retry.infeasible_reason
        assert opt.best(state, dx).action is not InterventionType.RETRY_PAYMENT


def test_a_dead_card_is_repaired_before_anything_else(opt):
    state, dx = _case("expired_card")
    assert opt.best(state, dx).action is InterventionType.REQUEST_PAYMENT_METHOD_UPDATE


def test_a_transient_fault_is_retried(opt):
    state, dx = _case("bank_timeout")
    assert opt.best(state, dx).action is InterventionType.RETRY_PAYMENT


def test_risk_and_compliance_score_no_automated_action(opt):
    """Two independent mechanisms refuse this -- the empty effect table here and
    `R-RISK-BLOCK` at the gate -- because one of them being wrong should not be enough."""
    for code in ("suspected_fraud", "high_risk_transaction", "compliance_hold"):
        state, dx = _case(code, amount=50_000.0)
        assert opt.best(state, dx) is None


def test_nothing_worth_doing_returns_none(opt):
    """Most of the value in dunning is in the cases you leave alone, so the optimiser
    must be able to name none."""
    state, dx = _case("closed_account", amount=0.50)
    assert opt.best(state, dx) is None


def test_a_tiny_case_does_not_justify_a_paid_contact(opt):
    state, dx = _case("insufficient_funds", amount=0.20)
    best = opt.best(state, dx)
    assert best is None or best.expected_profit.is_positive


def test_profit_subtracts_fees_and_costs(opt):
    state, dx = _case("bank_timeout", amount=1000.0)
    c = next(x for x in opt.candidates(state, dx)
             if x.action is InterventionType.RETRY_PAYMENT)
    assert c.expected_profit < c.expected_incremental_revenue
    assert (c.expected_incremental_revenue - c.processing_cost - c.action_cost
            == c.expected_profit)


def test_fees_are_charged_against_incremental_not_gross_revenue(opt):
    """Charging fees on money that would have arrived anyway makes every action look
    unprofitable on a case with a high passive-recovery rate."""
    state, dx = _case("bank_timeout", amount=1000.0)
    c = next(x for x in opt.candidates(state, dx)
             if x.action is InterventionType.RETRY_PAYMENT)
    assert c.processing_cost < opt.costs.processing_cost(c.expected_gross_revenue) \
        + opt.costs.chargeback_cost(c.expected_gross_revenue)


def test_contact_cost_grows_with_fatigue(opt):
    """An un-priced externality is one the optimiser spends freely."""
    fresh, dx = _case("insufficient_funds", contact_count=0)
    tired, _ = _case("insufficient_funds", contact_count=3)
    a = next(c for c in opt.candidates(fresh, dx)
             if c.action is InterventionType.SEND_REMINDER)
    b = next(c for c in opt.candidates(tired, dx)
             if c.action is InterventionType.SEND_REMINDER)
    assert b.action_cost > a.action_cost


def test_a_retry_does_not_pay_the_fatigue_penalty(opt):
    """A re-presentment is not a customer touch."""
    fresh, dx = _case("bank_timeout", contact_count=0)
    tired, _ = _case("bank_timeout", contact_count=3)
    a = next(c for c in opt.candidates(fresh, dx)
             if c.action is InterventionType.RETRY_PAYMENT)
    b = next(c for c in opt.candidates(tired, dx)
             if c.action is InterventionType.RETRY_PAYMENT)
    assert a.action_cost == b.action_cost


def test_escalation_is_not_priced_against_the_profit_objective(opt):
    """Whether a human looks at a case is not a trade the optimiser may make."""
    state, dx = _case("bank_timeout")
    esc = next(c for c in opt.candidates(state, dx)
               if c.action is InterventionType.ESCALATE_CASE)
    assert "not scored on expected profit" in esc.rationale


# ---------------------------------------------------------------- provenance
def test_per_action_estimates_are_labelled_as_priors_not_measurements(opt):
    """No randomised multi-armed data exists in this project. Calling the per-action
    figure a measurement would be the single most misleading thing this module could do."""
    state, dx = _case("insufficient_funds")
    for c in opt.candidates(state, dx):
        assert c.p_with_action.provenance == PRIOR
        assert not c.p_with_action.is_measured


def test_a_non_randomised_uplift_model_is_not_labelled_measured():
    class FakeUplift:
        def predict_arms(self, X):
            return [0.6], [0.3]

        def predict_uplift(self, X):
            return [0.3]

    import pandas as pd
    model = EffectModel(uplift=FakeUplift(), uplift_is_randomised=False)
    state, _ = _case("bank_timeout")
    est = model.p_no_action(state, CanonicalCause.NETWORK_FAILURE, pd.DataFrame([{}]))
    assert est.provenance == PRIOR
    assert "not a verified counterfactual" in est.detail


def test_a_randomised_uplift_model_is_labelled_measured():
    class FakeUplift:
        def predict_arms(self, X):
            return [0.6], [0.3]

        def predict_uplift(self, X):
            return [0.3]

    import pandas as pd
    model = EffectModel(uplift=FakeUplift(), uplift_is_randomised=True)
    state, _ = _case("bank_timeout")
    est = model.p_no_action(state, CanonicalCause.NETWORK_FAILURE, pd.DataFrame([{}]))
    assert est.provenance == MEASURED


def test_the_explanation_is_generated_from_stored_numbers(opt):
    """An explanation that can invent a figure is worse than none, because it is trusted."""
    state, dx = _case("expired_card")
    best = opt.best(state, dx)
    text = best.explain()
    assert f"{best.incremental_probability:+.1%}" in text
    assert str(best.expected_profit) in text
    assert best.p_with_action.provenance in text


# ---------------------------------------------------------------- cost model
def test_an_unknown_action_in_the_cost_config_is_refused():
    """The action space is a closed enum, so this is a typo, not an extension."""
    with pytest.raises(ValueError, match="unknown action"):
        CostModel.from_dict({"actions": {"send_carrier_pigeon": {"direct_usd": 1}}})


def test_a_missing_economics_file_falls_back_to_the_shipped_defaults(tmp_path):
    model = CostModel.load(tmp_path / "absent.yaml")
    assert model.actions[InterventionType.ESCALATE_CASE].operational_usd > 0


def test_a_recovery_is_worth_less_than_its_face_value():
    model = CostModel.default()
    assert model.net_of_fees(Money.from_major(100)) < Money.from_major(100)


def test_a_failed_attempt_pays_no_processing_fee():
    model = CostModel.default()
    assert model.processing_cost(Money.zero()).is_zero
