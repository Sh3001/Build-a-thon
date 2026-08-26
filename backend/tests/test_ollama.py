"""The local-LLM planner.

Most of these run offline against a stub transport - the point is the contract (schema
enforcement, fall-back on failure, no identifiers in the payload), not whether a
particular model is any good. The tests that need a live daemon skip without one, so the
default suite stays offline.
"""
from __future__ import annotations

import json
import os

import pytest

from backend.app.agents.llm import build_planner
from backend.app.agents.ollama import OllamaPlanner
from backend.app.models.enums import Channel, FailureCode, PaymentMethod
from backend.app.models.schemas import Transaction

LIVE = os.environ.get("RECOVERAI_TEST_OLLAMA")


def _txn(code=FailureCode.INSUFFICIENT_FUNDS) -> Transaction:
    return Transaction(
        customer_id="cust_000123", transaction_id="txn_000999", invoice_id="inv_5",
        amount=500.0, currency="USD", payment_method=PaymentMethod.CARD,
        failure_code=code, preferred_channel=Channel.EMAIL,
        previous_success_rate=0.8, failure_count=1, days_since_failure=1.0,
    )


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Transport:
    """Records what was posted and returns a canned body."""

    def __init__(self, content):
        self.content, self.posted = content, []

    def post(self, url, **kw):
        self.posted.append(kw.get("json", {}))
        return _Response({"message": {"content": self.content}})


def _planner_with(content, monkeypatch) -> OllamaPlanner:
    p = OllamaPlanner.__new__(OllamaPlanner)          # skip the probe
    OllamaPlanner.__init__(p, model="m", fast_model="m")
    p._available = True
    p.calls = p.failures = 0
    p.seconds = 0.0
    t = _Transport(content)
    monkeypatch.setattr("httpx.post", t.post)
    p._transport = t
    return p


# ---------------------------------------------------------------- offline contract
def test_disabled_planner_returns_none():
    p = OllamaPlanner.__new__(OllamaPlanner)
    OllamaPlanner.__init__(p, model="m", fast_model="m")
    p._available = False
    assert p.diagnose(_txn()) is None


def test_valid_response_parses_into_a_diagnosis(monkeypatch):
    body = json.dumps({"root_cause": "insufficient_funds", "confidence": 0.9,
                       "recoverable": True, "rationale": "balance", "evidence": []})
    p = _planner_with(body, monkeypatch)
    dx = p.diagnose(_txn())
    assert dx is not None
    assert dx.root_cause is FailureCode.INSUFFICIENT_FUNDS
    assert dx.source == "llm"
    assert p.calls == 1 and p.failures == 0


def test_malformed_json_falls_back_rather_than_raising(monkeypatch):
    p = _planner_with("not json at all", monkeypatch)
    assert p.diagnose(_txn()) is None
    assert p.failures == 1


def test_action_outside_the_enum_is_refused(monkeypatch):
    """A model inventing a tool name must not reach the executor."""
    body = json.dumps({"root_cause": "wire_transfer_to_me", "confidence": 1.0,
                       "recoverable": True, "rationale": "x", "evidence": []})
    p = _planner_with(body, monkeypatch)
    assert p.diagnose(_txn()) is None
    assert p.failures == 1


def test_risk_cause_cannot_be_marked_recoverable(monkeypatch):
    """Even if the model insists, a fraud hold is not recoverable."""
    body = json.dumps({"root_cause": "suspected_fraud", "confidence": 1.0,
                       "recoverable": True, "rationale": "x", "evidence": []})
    p = _planner_with(body, monkeypatch)
    dx = p.diagnose(_txn(FailureCode.SUSPECTED_FRAUD))
    assert dx is not None and dx.recoverable is False


def test_dead_instrument_is_never_retry_viable(monkeypatch):
    body = json.dumps({"root_cause": "expired_card", "confidence": 1.0,
                       "recoverable": True, "rationale": "x", "evidence": []})
    p = _planner_with(body, monkeypatch)
    dx = p.diagnose(_txn(FailureCode.EXPIRED_CARD))
    assert dx is not None and dx.retry_viable is False


def test_request_carries_a_json_schema(monkeypatch):
    """Structured output is the contract; without a schema the response is a guess."""
    body = json.dumps({"root_cause": "bank_timeout", "confidence": 0.8,
                       "recoverable": True, "rationale": "x", "evidence": []})
    p = _planner_with(body, monkeypatch)
    p.diagnose(_txn())
    sent = p._transport.posted[0]
    assert isinstance(sent["format"], dict) and "properties" in sent["format"]
    assert sent["options"]["temperature"] == 0     # determinism for reproducible runs


def test_no_identifiers_are_sent_to_the_local_model(monkeypatch):
    """The redaction guarantee must hold on this transport too, not just the hosted one."""
    body = json.dumps({"root_cause": "bank_timeout", "confidence": 0.8,
                       "recoverable": True, "rationale": "x", "evidence": []})
    p = _planner_with(body, monkeypatch)
    p.diagnose(_txn())
    blob = json.dumps(p._transport.posted[0])
    for leaked in ("cust_000123", "txn_000999", "inv_5"):
        assert leaked not in blob


def test_local_inference_is_not_charged_a_token_price(monkeypatch):
    body = json.dumps({"root_cause": "bank_timeout", "confidence": 0.8,
                       "recoverable": True, "rationale": "x", "evidence": []})
    p = _planner_with(body, monkeypatch)
    p.diagnose(_txn())
    assert p.cost_usd == 0.0
    assert p.seconds >= 0.0


def test_build_planner_rejects_ollama_when_unavailable(monkeypatch):
    monkeypatch.setattr(OllamaPlanner, "_probe", lambda self: False)
    with pytest.raises(RuntimeError, match="ollama"):
        build_planner("ollama")


# ---------------------------------------------------------------- live daemon
@pytest.mark.skipif(not LIVE, reason="RECOVERAI_TEST_OLLAMA not set")
def test_live_daemon_answers_with_a_valid_enum_member():
    p = OllamaPlanner(timeout=90)
    if not p.enabled:
        pytest.skip("no Ollama daemon")
    dx = p.diagnose(_txn(FailureCode.EXPIRED_CARD))
    # It may fall back; what it must never do is return something off-enum.
    assert dx is None or isinstance(dx.root_cause, FailureCode)


def test_auto_never_silently_selects_a_local_model(monkeypatch):
    """Regression: `auto` once fell back to Ollama whenever a daemon was reachable.

    Benchmarked diagnosis accuracy is 100% for the deterministic planner and 25-65% for
    the local models, so that fallback silently degraded every run on any machine with
    Ollama installed, with nothing in the output to say so. Local inference is opt-in.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(OllamaPlanner, "_probe", lambda self: True)   # daemon IS up
    assert build_planner("auto") is None
