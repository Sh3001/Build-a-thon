"""Metrics and structured logging.

Both are dependency-free. Prometheus scrapes a text format that is forty lines to emit,
and `prometheus_client` brings a global registry that fights with the test suite's need
to construct isolated ones. Structured logging is a `json.dumps` and a redaction pass.
"""
