#!/usr/bin/env python3
"""Benchmark diagnosis quality: deterministic rules vs one or more local LLMs.

The claim under test is the project's central one - *diagnosis is what pays* - so the
thing to measure is whether a model diagnoses better than the error-code rules do, not
whether it produces fluent text.

Two metrics, and the second is the one that matters:

  accuracy        share of cases whose root cause matches ground truth
  unsafe errors   cases where the model's diagnosis would UNBLOCK a dangerous action or
                  BLOCK a recoverable one -- e.g. calling insufficient_funds a
                  closed_account (kills a recoverable case) or calling suspected_fraud a
                  bank_timeout (would automate a fraud hold if the gate let it through)

An accuracy number alone hides the asymmetry: not all mistakes cost the same.

    .venv/bin/python scripts/bench_planner.py --n 40
    .venv/bin/python scripts/bench_planner.py --n 40 --models llama3.2:3b,qwen2.5:3b-instruct
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agents.diagnose import diagnose as rules_diagnose
from backend.app.agents.ollama import OllamaPlanner
from backend.app.models.enums import FailureCategory, FailureCode, category_of
from backend.app.services.dataio import load_split, to_transactions

#: Causes on which no automated money movement or messaging may happen.
RISK = FailureCategory.RISK_COMPLIANCE
#: Causes where a retry can never work, so calling something else one kills the case.
NO_RETRY = {FailureCode.CLOSED_ACCOUNT, FailureCode.INVALID_ACCOUNT,
            FailureCode.EXPIRED_CARD, FailureCode.INVALID_PAYMENT_METHOD}


def unsafe(truth: FailureCode, pred: FailureCode) -> str | None:
    """Classify a mistake by what it would cost, not by whether it is wrong."""
    if truth == pred:
        return None
    t_cat, p_cat = category_of(truth), category_of(pred)
    if t_cat is RISK and p_cat is not RISK:
        return "risk-downgraded"        # would have automated a fraud/compliance hold
    if truth not in NO_RETRY and pred in NO_RETRY:
        return "recoverable-killed"     # abandons a case that could have been recovered
    if truth in NO_RETRY and pred not in NO_RETRY:
        return "dead-instrument-retried"
    return None                         # wrong, but not dangerous


def evaluate(name: str, predict, txns) -> dict:
    correct = 0
    harms: dict[str, int] = {}
    fell_back = 0
    started = time.time()
    for t in txns:
        pred = predict(t)
        if pred is None:
            fell_back += 1
            continue
        if pred == t.failure_code:
            correct += 1
        else:
            kind = unsafe(t.failure_code, pred)
            if kind:
                harms[kind] = harms.get(kind, 0) + 1
    scored = len(txns) - fell_back
    return {
        "planner": name,
        "n": len(txns), "scored": scored, "fell_back": fell_back,
        "accuracy": correct / scored if scored else 0.0,
        "unsafe_total": sum(harms.values()),
        "unsafe": harms,
        "seconds": round(time.time() - started, 1),
        "per_case": round((time.time() - started) / max(len(txns), 1), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="cases to score")
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", default="", help="comma-separated Ollama tags")
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()

    # to_transactions yields dicts; the planners take a typed Transaction.
    from backend.app.models.schemas import Transaction
    txns = [Transaction(**d) for d in to_transactions(load_split(a.split))[: a.n]]
    print(f"  {len(txns)} cases from the {a.split} split\n")

    rows = [evaluate("rules (deterministic)",
                     lambda t: rules_diagnose(t).root_cause, txns)]

    for tag in [m.strip() for m in a.models.split(",") if m.strip()]:
        p = OllamaPlanner(model=tag, fast_model=tag, timeout=a.timeout)
        if not p.enabled:
            print(f"  skipping {tag}: Ollama unavailable")
            continue
        print(f"  running {tag} ...", flush=True)
        rows.append(evaluate(f"ollama {tag}",
                             lambda t, pl=p: (d.root_cause if (d := pl.diagnose(t)) else None),
                             txns))

    print(f"\n  {'planner':<30}{'acc':>7}{'unsafe':>8}{'fallback':>10}{'s/case':>9}")
    print("  " + "-" * 64)
    for r in rows:
        print(f"  {r['planner']:<30}{r['accuracy']:>6.1%}{r['unsafe_total']:>8}"
              f"{r['fell_back']:>10}{r['per_case']:>9.2f}")
    for r in rows:
        if r["unsafe"]:
            print(f"\n  {r['planner']} unsafe breakdown: {r['unsafe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
