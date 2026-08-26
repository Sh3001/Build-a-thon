"""Local-LLM planner backed by Ollama.

Subclasses `LLMPlanner` rather than reimplementing it, so the parts that matter for
safety are inherited unchanged: the allowlisted `_case_view` (no identifiers reach the
model), the closed `InterventionType` enum the response must parse into, the expected-value
model routing, and the silent fall-back to the deterministic planner on any failure. Only
the transport differs.

Why this is worth having beyond cost: it makes the LLM path **demonstrable offline**. The
architecture claims a reasoning layer; without a key that claim ships unexercised, and
`llm_calls: 0` is not evidence of anything.

Ollama's `format` parameter takes a JSON Schema, which Pydantic emits directly, so the
structured-output contract is the same one the Anthropic path uses -- a response that does
not fit the schema is a parse failure, not a mystery string.

    ollama serve
    ollama pull llama3.1:8b
    RECOVERAI_OLLAMA_MODEL=llama3.1:8b .venv/bin/python scripts/run_experiment.py \
        --planner ollama --fresh
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.app.agents.llm import SYSTEM as _SYSTEM
from backend.app.agents.llm import LLMPlanner

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
#: Small model for bulk triage, larger one where the decision is worth more.
OLLAMA_MODEL = os.environ.get("RECOVERAI_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_MODEL_FAST = os.environ.get("RECOVERAI_OLLAMA_MODEL_FAST", "llama3.2:3b")
#: Local inference is slow relative to a hosted API; a stuck request must not wedge a run.
OLLAMA_TIMEOUT = float(os.environ.get("RECOVERAI_OLLAMA_TIMEOUT", 90))


@dataclass
class OllamaPlanner(LLMPlanner):
    """Same contract as LLMPlanner, served by a local Ollama daemon.

    `cost_usd` stays 0.0 by design -- local inference has no per-token price, and
    inventing one would corrupt the ROI figure the dashboard reports.
    """

    model: str = OLLAMA_MODEL
    fast_model: str = OLLAMA_MODEL_FAST
    host: str = OLLAMA_HOST
    timeout: float = OLLAMA_TIMEOUT
    #: Wall-clock seconds spent in inference. The local analogue of cost.
    seconds: float = 0.0
    _available: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # Deliberately does NOT call super().__post_init__(): that looks for an Anthropic
        # key and would build a hosted client we are not using.
        self._available = self._probe()

    def _probe(self) -> bool:
        """One cheap call at construction. If the daemon is down we want the deterministic
        planner immediately, not a timeout per case."""
        try:
            import httpx
            r = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            if r.status_code != 200:
                return False
            names = {m.get("name", "") for m in r.json().get("models", [])}
            wanted = {self.model, self.fast_model}
            missing = {w for w in wanted if w not in names}
            if missing:
                # Fall back to whatever is actually pulled rather than failing outright.
                if not names:
                    return False
                pick = sorted(names)[0]
                for attr in ("model", "fast_model"):
                    if getattr(self, attr) in missing:
                        setattr(self, attr, pick)
            return True
        except Exception:
            return False

    @property
    def enabled(self) -> bool:
        return self._available

    def _meter(self, model: str, usage: Any) -> None:
        """No token price for local inference. Time is the resource that is actually
        consumed, so that is what gets recorded."""
        return None

    def _parse(self, model: str, schema: type[BaseModel], prompt: str) -> BaseModel | None:
        if not self.enabled:
            return None
        import time

        import httpx
        started = time.time()
        try:
            self.calls += 1
            r = httpx.post(
                f"{self.host}/api/chat",
                timeout=self.timeout,
                json={
                    "model": model,
                    "stream": False,
                    # A JSON Schema here is what makes the response parseable rather than
                    # hopeful: the same structured-output contract as the hosted path.
                    "format": schema.model_json_schema(),
                    "options": {"temperature": 0, "num_predict": 700},
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            return schema.model_validate(json.loads(content))
        except (ValidationError, json.JSONDecodeError, KeyError):
            # A small model that returns malformed or out-of-enum JSON is a normal event,
            # not an exception: the deterministic planner takes the case.
            self.failures += 1
            return None
        except Exception:
            self.failures += 1
            return None
        finally:
            self.seconds += time.time() - started

    def stats(self) -> dict:
        return {
            "backend": "ollama", "host": self.host,
            "model": self.model, "fast_model": self.fast_model,
            "calls": self.calls, "failures": self.failures,
            "seconds": round(self.seconds, 1),
            "avg_seconds_per_call": round(self.seconds / self.calls, 2) if self.calls else 0.0,
        }
