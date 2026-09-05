"""Structured logging with redaction that is on by default.

Two things a payment system's logs must not be: unparseable, and full of card numbers.
The first is solved by emitting JSON. The second is solved by making redaction the
*formatter's* job rather than the caller's -- a log line is written by whoever is
debugging at the time, and a rule that depends on them remembering is a rule that will be
broken at 3am.

So `scrub()` runs over every structured field on the way out. It is the same function the
LLM boundary uses, which means one list of forbidden keys governs both places customer
data can escape.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from backend.app.agents.redaction import REDACTED, VALUE_PATTERNS, is_forbidden_key

#: Correlation id for the current request or agent run. A ContextVar rather than a
#: parameter so every line emitted while handling one request carries it without each
#: call site threading it through.
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

RESERVED = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "message",
    "taskName",
})


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def set_trace_id(tid: str) -> None:
    _trace_id.set(tid)


def current_trace_id() -> str:
    return _trace_id.get()


def _redact(value: Any, depth: int = 0) -> Any:
    """Recursively drop forbidden keys and mask identifier-shaped values.

    Depth-limited: a self-referential structure in a log call should produce a truncated
    line, not a stack overflow inside the logger.
    """
    if depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        return {k: _redact(v, depth + 1) for k, v in value.items()
                if not is_forbidden_key(str(k))}
    if isinstance(value, (list, tuple)):
        return [_redact(v, depth + 1) for v in value][:100]
    if isinstance(value, str):
        if any(p.search(value) for p in VALUE_PATTERNS):
            return REDACTED
        return value if len(value) <= 2000 else value[:2000] + "...[truncated]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:500]


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Never multi-line, so a traceback cannot break a log
    pipeline that splits on newlines."""

    def __init__(self, service: str = "recoverai", **kw):
        super().__init__(**kw)
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": _redact(record.getMessage()),
        }
        tid = current_trace_id()
        if tid:
            payload["trace_id"] = tid
        for key, value in record.__dict__.items():
            if key in RESERVED or key.startswith("_") or key in payload:
                continue
            if is_forbidden_key(key):
                continue
            payload[key] = _redact(value)
        if record.exc_info:
            # The type and message, not the frames: a traceback can contain local
            # variables, and locals in this codebase contain customer records.
            exc_type, exc_value, _ = record.exc_info
            payload["error"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": _redact(str(exc_value)),
            }
        return json.dumps(payload, separators=(",", ":"), default=str)


def configure(level: str | None = None, service: str = "recoverai",
              stream=None) -> logging.Logger:
    """Install the JSON formatter on the root logger. Idempotent."""
    lvl = (level or os.environ.get("RECOVERAI_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(lvl)
    for h in list(root.handlers):
        if getattr(h, "_recoverai", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter(service=service))
    handler._recoverai = True                                 # type: ignore[attr-defined]
    root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
