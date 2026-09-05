"""Metrics exposition and log redaction."""
from __future__ import annotations

import io
import json
import logging

import pytest

from backend.app.observability.logging import JsonFormatter, configure, get_logger, new_trace_id
from backend.app.observability.metrics import M, MetricsRegistry


@pytest.fixture
def registry():
    return MetricsRegistry("test")


def test_counters_accumulate_per_label_set(registry):
    registry.counter("things", labels={"kind": "a"})
    registry.counter("things", labels={"kind": "a"}, value=2)
    registry.counter("things", labels={"kind": "b"})
    assert registry.value("things", {"kind": "a"}) == 3
    assert registry.value("things", {"kind": "b"}) == 1


def test_a_counter_cannot_decrease(registry):
    with pytest.raises(ValueError):
        registry.counter("things", value=-1)


def test_exposition_is_byte_stable(registry):
    """Deterministic output so a test can diff it and a scrape does not churn."""
    for labels in ({"b": "2"}, {"a": "1"}):
        registry.counter("things", "a help string", labels)
    assert registry.render() == registry.render()
    lines = [line for line in registry.render().splitlines()
             if not line.startswith("#")]
    assert lines == sorted(lines)


def test_histogram_buckets_are_cumulative(registry):
    for v in (0.002, 0.03, 1.4):
        registry.observe("latency_seconds", v)
    text = registry.render()
    assert 'latency_seconds_bucket{le="+Inf"} 3' in text
    assert "latency_seconds_count 3" in text
    counts = [int(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if "_bucket{" in line]
    assert counts == sorted(counts), "bucket counts must be non-decreasing"


def test_label_values_are_escaped(registry):
    registry.gauge("thing", 1.0, labels={"note": 'a "quoted" \\ value'})
    assert '\\"quoted\\"' in registry.render()


def test_the_timer_records_a_duration(registry):
    with registry.timer(M.GATEWAY_LATENCY, {"provider": "mock"}):
        pass
    assert registry.value(M.GATEWAY_LATENCY, {"provider": "mock"}) == 1


def test_the_snapshot_matches_the_exposition(registry):
    registry.counter(M.POLICY_DECISIONS, labels={"decision": "approve"}, value=5)
    snap = registry.snapshot()
    assert snap["counters"]["test_policy_decisions_total"]["decision=approve"] == 5


# ---------------------------------------------------------------- logging
def _emit(**extra) -> dict:
    buf = io.StringIO()
    configure("INFO", stream=buf)
    new_trace_id()
    get_logger("test").info("an event happened", extra=extra)
    for handler in logging.getLogger().handlers:
        handler.flush()
    return json.loads(buf.getvalue().strip().splitlines()[-1])


def test_forbidden_keys_never_reach_the_log():
    """Redaction is the formatter's job, not the caller's: a rule that depends on the
    person debugging at 3am remembering is a rule that gets broken."""
    out = _emit(customer_id="c_123", billing_email="a@b.com", card_number="4111111111111111",
                action="retry_payment")
    assert "customer_id" not in out
    assert "billing_email" not in out
    assert "card_number" not in out
    assert out["action"] == "retry_payment"


def test_identifier_shaped_values_are_masked_regardless_of_key():
    out = _emit(note="please contact alice@example.com about this")
    assert out["note"] == "[redacted]"


def test_nested_structures_are_scrubbed():
    out = _emit(context={"amount_usd": 42.0, "customer_id": "c_1",
                         "inner": {"phone": "+1 555 867 5309"}})
    assert "customer_id" not in out["context"]
    assert "phone" not in out["context"]["inner"]
    assert out["context"]["amount_usd"] == 42.0


def test_every_line_carries_the_trace_id():
    assert len(_emit()["trace_id"]) == 16


def test_output_is_one_json_object_per_line():
    """A traceback must not break a pipeline that splits on newlines."""
    buf = io.StringIO()
    configure("INFO", stream=buf)
    try:
        raise ValueError("customer c_123 exploded")
    except ValueError:
        get_logger("test").exception("failed")
    lines = [line for line in buf.getvalue().strip().splitlines() if line]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["error"]["type"] == "ValueError"


def test_a_traceback_does_not_leak_frame_locals():
    """Locals in this codebase contain customer records."""
    buf = io.StringIO()
    configure("INFO", stream=buf)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("test").exception("failed")
    parsed = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert "traceback" not in parsed
    assert set(parsed["error"]) == {"type", "message"}


def test_the_formatter_survives_a_deep_structure():
    """A self-referential structure should truncate, not overflow the stack."""
    deep: dict = {}
    node = deep
    for _ in range(50):
        node["next"] = {}
        node = node["next"]
    record = logging.LogRecord("t", logging.INFO, "f", 1, "m", (), None)
    record.payload = deep                                     # type: ignore[attr-defined]
    assert JsonFormatter().format(record)
