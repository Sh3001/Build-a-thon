"""A Prometheus-compatible metrics registry.

Written rather than imported for two reasons that are about testability, not
minimalism: `prometheus_client` keeps a process-global default registry, which makes
"assert this counter incremented" tests order-dependent; and its multiprocess mode needs
a shared directory that Docker-composed local runs do not have. A registry that can be
constructed per test, and per app, avoids both.

The exposition format is the text format Prometheus actually scrapes -- `# HELP`, `# TYPE`,
one sample per line, labels sorted so output is byte-stable and diffable in a test.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

Labels = tuple[tuple[str, str], ...]


def _labels(pairs: dict[str, str] | None) -> Labels:
    return tuple(sorted((str(k), str(v)) for k, v in (pairs or {}).items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: Labels, extra: tuple[tuple[str, str], ...] = ()) -> str:
    items = [*labels, *extra]
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in items) + "}"


@dataclass
class _Metric:
    name: str
    help: str
    kind: str
    values: dict[Labels, float] = field(default_factory=dict)


@dataclass
class _Histogram:
    name: str
    help: str
    buckets: tuple[float, ...]
    counts: dict[Labels, list[int]] = field(default_factory=dict)
    sums: dict[Labels, float] = field(default_factory=dict)
    totals: dict[Labels, int] = field(default_factory=dict)


#: Latency buckets in seconds. Chosen around the decisions this system actually makes:
#: a policy evaluation is microseconds, a gateway call is tens of milliseconds, an LLM
#: call is seconds. Default Prometheus buckets bunch everything interesting into one bin.
LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class MetricsRegistry:
    """Thread-safe. Counters, gauges and histograms; nothing else is needed here."""

    def __init__(self, namespace: str = "recoverai"):
        self.namespace = namespace
        self._lock = threading.Lock()
        self._counters: dict[str, _Metric] = {}
        self._gauges: dict[str, _Metric] = {}
        self._histograms: dict[str, _Histogram] = {}

    def _full(self, name: str) -> str:
        return name if name.startswith(f"{self.namespace}_") else f"{self.namespace}_{name}"

    # ------------------------------------------------------------------ counters
    def counter(self, name: str, help: str = "", labels: dict[str, str] | None = None,
                value: float = 1.0) -> None:
        if value < 0:
            raise ValueError("a counter may not decrease; use a gauge")
        key = self._full(name)
        with self._lock:
            m = self._counters.setdefault(key, _Metric(key, help, "counter"))
            if help and not m.help:
                m.help = help
            lk = _labels(labels)
            m.values[lk] = m.values.get(lk, 0.0) + value

    # ------------------------------------------------------------------ gauges
    def gauge(self, name: str, value: float, help: str = "",
              labels: dict[str, str] | None = None) -> None:
        key = self._full(name)
        with self._lock:
            m = self._gauges.setdefault(key, _Metric(key, help, "gauge"))
            if help and not m.help:
                m.help = help
            m.values[_labels(labels)] = float(value)

    # ------------------------------------------------------------------ histograms
    def observe(self, name: str, value: float, help: str = "",
                labels: dict[str, str] | None = None,
                buckets: Iterable[float] = LATENCY_BUCKETS) -> None:
        key = self._full(name)
        with self._lock:
            h = self._histograms.get(key)
            if h is None:
                h = _Histogram(key, help, tuple(sorted(buckets)))
                self._histograms[key] = h
            lk = _labels(labels)
            counts = h.counts.setdefault(lk, [0] * len(h.buckets))
            for i, edge in enumerate(h.buckets):
                if value <= edge:
                    counts[i] += 1
            h.sums[lk] = h.sums.get(lk, 0.0) + float(value)
            h.totals[lk] = h.totals.get(lk, 0) + 1

    def timer(self, name: str, labels: dict[str, str] | None = None) -> _Timer:
        return _Timer(self, name, labels)

    # ------------------------------------------------------------------ read
    def value(self, name: str, labels: dict[str, str] | None = None) -> float:
        key, lk = self._full(name), _labels(labels)
        with self._lock:
            for store in (self._counters, self._gauges):
                if key in store:
                    return store[key].values.get(lk, 0.0)
            if key in self._histograms:
                return float(self._histograms[key].totals.get(lk, 0))
        return 0.0

    def render(self) -> str:
        """Prometheus text exposition. Deterministic ordering so it can be diffed."""
        out: list[str] = []
        with self._lock:
            for store in (self._counters, self._gauges):
                for key in sorted(store):
                    m = store[key]
                    if m.help:
                        out.append(f"# HELP {m.name} {m.help}")
                    out.append(f"# TYPE {m.name} {m.kind}")
                    for lk in sorted(m.values):
                        out.append(f"{m.name}{_render_labels(lk)} {_num(m.values[lk])}")
            for key in sorted(self._histograms):
                h = self._histograms[key]
                if h.help:
                    out.append(f"# HELP {h.name} {h.help}")
                out.append(f"# TYPE {h.name} histogram")
                for lk in sorted(h.counts):
                    counts = h.counts[lk]
                    for edge, c in zip(h.buckets, counts):
                        out.append(f"{h.name}_bucket"
                                   f"{_render_labels(lk, (('le', _num(edge)),))} {c}")
                    total = h.totals.get(lk, 0)
                    out.append(f"{h.name}_bucket{_render_labels(lk, (('le', '+Inf'),))} {total}")
                    out.append(f"{h.name}_sum{_render_labels(lk)} {_num(h.sums.get(lk, 0.0))}")
                    out.append(f"{h.name}_count{_render_labels(lk)} {total}")
        return "\n".join(out) + "\n"

    def snapshot(self) -> dict:
        """The same data as JSON, for the dashboard, which does not speak Prometheus."""
        with self._lock:
            return {
                "counters": {m.name: {"|".join(f"{k}={v}" for k, v in lk): val
                                      for lk, val in m.values.items()}
                             for m in self._counters.values()},
                "gauges": {m.name: {"|".join(f"{k}={v}" for k, v in lk): val
                                    for lk, val in m.values.items()}
                           for m in self._gauges.values()},
                "histograms": {h.name: {"|".join(f"{k}={v}" for k, v in lk): {
                    "count": h.totals[lk], "sum": round(h.sums[lk], 6),
                    "mean": round(h.sums[lk] / h.totals[lk], 6) if h.totals[lk] else 0.0,
                } for lk in h.totals} for h in self._histograms.values()},
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


def _num(x: float) -> str:
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    if isinstance(x, float) and math.isinf(x):
        return "+Inf" if x > 0 else "-Inf"
    return repr(round(x, 9))


class _Timer:
    """`with registry.timer("gateway_latency_seconds"):`"""

    def __init__(self, registry: MetricsRegistry, name: str,
                 labels: dict[str, str] | None):
        self.registry, self.name, self.labels = registry, name, labels
        self.started = 0.0

    def __enter__(self) -> _Timer:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.registry.observe(self.name, time.perf_counter() - self.started,
                              labels=self.labels)


#: Names used across the codebase, in one place so a metric cannot be spelled two ways
#: in two files and silently become two series.
class M:
    CASES_PROCESSED = "cases_processed_total"
    RECOVERIES = "recoveries_total"
    RECOVERED_USD = "recovered_usd_total"
    INCREMENTAL_USD = "incremental_usd_total"
    ACTION_COST_USD = "action_cost_usd_total"
    POLICY_DECISIONS = "policy_decisions_total"
    POLICY_RULE_FIRED = "policy_rule_fired_total"
    HUMAN_REVIEWS = "human_reviews_total"
    HUMAN_REVIEW_PENDING = "human_review_pending"
    LLM_CALLS = "llm_calls_total"
    LLM_FAILURES = "llm_failures_total"
    LLM_LATENCY = "llm_latency_seconds"
    LLM_COST_USD = "llm_cost_usd_total"
    GATEWAY_CALLS = "gateway_calls_total"
    GATEWAY_LATENCY = "gateway_latency_seconds"
    EXECUTOR_FAILURES = "executor_failures_total"
    NOTIFICATION_SENDS = "notification_sends_total"
    DEAD_LETTERS = "dead_letters_total"
    DLQ_QUARANTINED = "dlq_quarantined"
    API_REQUESTS = "api_requests_total"
    API_LATENCY = "api_request_seconds"
    AUTH_FAILURES = "auth_failures_total"
    RATE_LIMITED = "rate_limited_total"
    WEBHOOK_REJECTED = "webhook_rejected_total"
    MODEL_PREDICTION = "model_prediction"
    MODEL_DRIFT_PSI = "model_drift_psi"


_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _registry
