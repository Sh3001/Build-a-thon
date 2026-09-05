"""Phase 6 tests -- the bounded LLM planner, against a stub client. Nothing calls out."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.agents.llm import LLMAction, LLMDiagnosis, LLMPlanner, build_planner
from backend.app.config import LLM_ESCALATE_EV_USD
from backend.app.models.enums import Channel, InterventionType
from backend.app.models.schemas import AgentState, Diagnosis, Transaction


def txn(code="insufficient_funds", amount=100.0, **kw):
    base = dict(customer_id="cust_secret_123", transaction_id="t1", amount=amount,
                currency="USD", payment_method="card", failure_code=code,
                avg_transaction_value=100.0, previous_success_rate=0.7, failure_count=1,
                days_since_failure=1.0, preferred_channel="sms")
    return Transaction(**{**base, **kw})


class StubClient:
    """Records what it was asked and returns whatever the test wants."""
    def __init__(self, parsed=None, raise_exc=None, text=None, usage=(100, 50)):
        self.parsed, self.raise_exc, self.text = parsed, raise_exc, text
        self.usage = usage
        self.prompts: list[str] = []
        self.models: list[str] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, *, model, max_tokens, system, messages, response_format):
        self.models.append(model)
        self.prompts.append(messages[0]["content"])
        if self.raise_exc:
            raise self.raise_exc
        content = [SimpleNamespace(text=self.text)] if self.text else []
        return SimpleNamespace(
            parsed=self.parsed, content=content,
            usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]),
        )


def planner(client):
    return LLMPlanner(api_key="test-key", client=client)


def state(ev=50.0, **kw):
    t = kw.pop("transaction", txn())
    return AgentState(transaction_id="t1", customer_id="c1", amount=100.0,
                      failure_code=t.failure_code, transaction=t, expected_recovery=ev, **kw)


# ------------------------------------------------------------------ availability
def test_disabled_without_a_key():
    p = LLMPlanner(api_key=None)
    assert not p.enabled
    assert p.diagnose(txn()) is None
    assert p.select(state(), Diagnosis(root_cause="insufficient_funds",
                                       category="CUSTOMER_ACTION", confidence=0.9)) is None


def test_build_planner_modes(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert build_planner("rules") is None
    assert build_planner("auto") is None
    with pytest.raises(RuntimeError):
        build_planner("llm")


# ------------------------------------------------------------------ happy path
def test_diagnosis_is_parsed_into_the_domain_schema():
    c = StubClient(parsed=LLMDiagnosis(root_cause="multiple_declines", confidence=0.71,
                                       recoverable=True, rationale="chronic failures",
                                       evidence=["5 failures"]))
    d = planner(c).diagnose(txn())
    assert d is not None
    assert d.root_cause.value == "multiple_declines"
    assert d.category.value == "PERSISTENT"      # derived, not taken from the model
    assert d.source == "llm"


def test_action_is_parsed_into_a_proposal():
    c = StubClient(parsed=LLMAction(action=InterventionType.SEND_PAYMENT_LINK,
                                    channel=Channel.WHATSAPP, delay_hours=12.0,
                                    reason="route around the dead instrument"))
    a = planner(c).select(state(), Diagnosis(root_cause="expired_card",
                                             category="CUSTOMER_ACTION", confidence=0.9))
    assert a is not None
    assert a.action is InterventionType.SEND_PAYMENT_LINK
    assert a.channel is Channel.WHATSAPP and a.delay_hours == 12.0
    assert a.source == "llm"


# ------------------------------------------------------------------ boundedness
def test_an_invented_action_cannot_be_constructed():
    """The action space is a closed enum: the model cannot name a tool that does not exist."""
    with pytest.raises(Exception):
        LLMAction(action="wire_funds_offshore", reason="nope")


def test_planner_never_receives_a_tool():
    """Boundedness by construction -- the call carries no tools parameter at all."""
    import inspect
    src = inspect.getsource(LLMPlanner._parse)
    assert "tools" not in src
    assert "tool_choice" not in src


def test_prompt_excludes_customer_identifiers():
    c = StubClient(parsed=LLMDiagnosis(root_cause="insufficient_funds", confidence=0.8,
                                       recoverable=True, rationale="ok"))
    planner(c).diagnose(txn())
    assert "cust_secret_123" not in c.prompts[0]
    assert "customer_id" not in c.prompts[0]


# ------------------------------------------------------------------ fallbacks
@pytest.mark.parametrize("exc", [
    RuntimeError("api down"), TimeoutError("timeout"), ValueError("bad"),
])
def test_api_errors_fall_back_to_rules(exc):
    p = planner(StubClient(raise_exc=exc))
    assert p.diagnose(txn()) is None
    assert p.failures == 1


def test_a_refusal_falls_back():
    """A text refusal instead of structured output must not crash or be misread."""
    p = planner(StubClient(parsed=None, text="I can't help with that."))
    assert p.diagnose(txn()) is None


def test_malformed_json_falls_back():
    p = planner(StubClient(parsed=None, text='{"root_cause": "not_a_real_code"}'))
    assert p.diagnose(txn()) is None
    assert p.failures == 1


def test_empty_response_falls_back():
    assert planner(StubClient(parsed=None)).diagnose(txn()) is None


def test_wrong_schema_type_falls_back():
    """A diagnosis object returned where an action was asked for is rejected."""
    c = StubClient(parsed=LLMDiagnosis(root_cause="expired_card", confidence=0.9,
                                       recoverable=True, rationale="x"))
    assert planner(c).select(state(), Diagnosis(root_cause="expired_card",
                                                category="CUSTOMER_ACTION", confidence=0.9)) is None


# ------------------------------------------------------------------ routing & cost
def test_high_value_cases_route_to_the_stronger_model():
    c = StubClient(parsed=LLMAction(action=InterventionType.RETRY_PAYMENT, reason="x"))
    p = planner(c)
    p.select(state(ev=LLM_ESCALATE_EV_USD + 1), Diagnosis(root_cause="bank_timeout",
                                                          category="TEMPORARY", confidence=0.9))
    p.select(state(ev=1.0), Diagnosis(root_cause="bank_timeout",
                                      category="TEMPORARY", confidence=0.9))
    assert c.models[0] == p.model
    assert c.models[1] == p.fast_model


def test_cost_is_metered_from_real_usage():
    c = StubClient(parsed=LLMDiagnosis(root_cause="bank_timeout", confidence=0.9,
                                       recoverable=True, rationale="x"),
                   usage=(1_000_000, 1_000_000))
    p = planner(c)
    p.model = "claude-opus-5"
    p.diagnose(txn(amount=100_000))            # routes to opus
    assert p.cost_usd == pytest.approx(5.0 + 25.0, rel=1e-6)


def test_pop_cost_drains_the_pending_charge():
    c = StubClient(parsed=LLMDiagnosis(root_cause="bank_timeout", confidence=0.9,
                                       recoverable=True, rationale="x"))
    p = planner(c)
    p.diagnose(txn())
    first = p.pop_cost()
    assert first > 0
    assert p.pop_cost() == 0.0, "cost was charged twice"


def test_call_counter_tracks_every_attempt():
    c = StubClient(parsed=None)
    p = planner(c)
    for _ in range(3):
        p.diagnose(txn())
    assert p.calls == 3
