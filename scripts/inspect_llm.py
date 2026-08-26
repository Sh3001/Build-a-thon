#!/usr/bin/env python3
"""Inspect one LLM response end to end: payload out, raw answer back, what the gate did.

The automated tests answer "does the contract hold?". This answers "what did the model
actually say, and did it matter?" - which is what you need when a model is behaving oddly,
and what you want on screen when someone asks whether the LLM is really wired in.

Five things are printed, in the order they happen:

  1. the exact JSON sent to the model      - proof that no identifier leaves the box
  2. the raw response text                 - before any parsing
  3. the parsed diagnosis                  - or the reason it was rejected
  4. what the deterministic planner said   - the comparison that matters
  5. the policy verdict on the LLM's own proposal, with the rule IDs that fired

    .venv/bin/python scripts/inspect_llm.py --case txn_0001586
    .venv/bin/python scripts/inspect_llm.py --code expired_card --model qwen2.5:3b-instruct
    .venv/bin/python scripts/inspect_llm.py --code suspected_fraud --raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agents.diagnose import diagnose as rules_diagnose  # noqa: E402
from backend.app.agents.llm import LLMDiagnosis  # noqa: E402
from backend.app.agents.ollama import OllamaPlanner  # noqa: E402
from backend.app.models.enums import FailureCode  # noqa: E402
from backend.app.models.schemas import AgentState, Transaction  # noqa: E402
from backend.app.policies.engine import PolicyContext, validate  # noqa: E402
from backend.app.services.dataio import load_split, to_transactions  # noqa: E402

RULE = "-" * 78


def head(title: str) -> None:
    print(f"\n{title}\n{RULE}")


def pick(case_id: str | None, code: str | None, split: str) -> Transaction:
    rows = to_transactions(load_split(split))
    if case_id:
        row = next((r for r in rows if r["transaction_id"] == case_id), None)
        if row is None:
            raise SystemExit(f"no case {case_id} in the {split} split")
        return Transaction(**row)
    if code:
        row = next((r for r in rows if str(r["failure_code"]) == code), None)
        if row is None:
            raise SystemExit(f"no case with failure_code={code} in the {split} split")
        return Transaction(**row)
    return Transaction(**rows[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="transaction id")
    ap.add_argument("--code", help="pick the first case with this failure_code")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", help="Ollama tag (default: whatever the planner picks)")
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--raw", action="store_true", help="also dump the full HTTP body")
    a = ap.parse_args()

    txn = pick(a.case, a.code, a.split)
    kw = {"model": a.model, "fast_model": a.model} if a.model else {}
    planner = OllamaPlanner(timeout=a.timeout, **kw)
    if not planner.enabled:
        raise SystemExit("no Ollama daemon reachable - `ollama serve`")

    print(f"  case {txn.transaction_id}   reported code: {txn.failure_code.value}   "
          f"${txn.amount_usd:,.2f}")
    print(f"  model {planner.model}")

    # ---------------------------------------------------------------- 1. payload
    view = planner._case_view(txn)
    head("1. SENT TO THE MODEL  (allowlisted view - no identifiers)")
    print(json.dumps(view, indent=2))
    leaked = [v for v in (txn.customer_id, txn.transaction_id, txn.invoice_id)
              if v and v in json.dumps(view)]
    print(f"\n  identifiers present: {leaked or 'none'}")

    # ---------------------------------------------------------------- 2. raw answer
    import time

    import httpx
    prompt = (
        "Diagnose the true root cause of this failed payment. The processor code is "
        "evidence, not truth: repeated failures on a customer who never pays are a "
        "dead relationship, not a temporary fault.\n\n"
        f"{json.dumps(view, indent=2)}"
    )
    from backend.app.agents.llm import SYSTEM
    started = time.time()
    r = httpx.post(f"{planner.host}/api/chat", timeout=a.timeout, json={
        "model": planner.model, "stream": False,
        "format": LLMDiagnosis.model_json_schema(),
        "options": {"temperature": 0, "num_predict": 700},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
    })
    elapsed = time.time() - started
    body = r.json()
    head(f"2. RAW RESPONSE  ({elapsed:.1f}s, HTTP {r.status_code})")
    print(body.get("message", {}).get("content", "<empty>"))
    if a.raw:
        print("\n  full body:")
        print(json.dumps({k: v for k, v in body.items() if k != "message"}, indent=2))

    # ---------------------------------------------------------------- 3. parsed
    head("3. PARSED")
    dx = planner.diagnose(txn)
    if dx is None:
        print("  REJECTED - did not fit the schema, or the action was off-enum.")
        print("  The deterministic planner takes this case. This is a normal outcome,")
        print("  not an error: a model that cannot answer in the contract does not get")
        print("  to answer at all.")
    else:
        print(f"  root_cause    {dx.root_cause.value}")
        print(f"  category      {dx.category.value}")
        print(f"  confidence    {dx.confidence:.2f}")
        print(f"  recoverable   {dx.recoverable}   (derived, not taken from the model)")
        print(f"  retry_viable  {dx.retry_viable}   (derived - a fact about the rails)")
        print(f"  rationale     {dx.rationale[:160]}")

    # ---------------------------------------------------------------- 4. comparison
    rules_dx = rules_diagnose(txn)
    head("4. WHAT THE DETERMINISTIC PLANNER SAID")
    print(f"  root_cause    {rules_dx.root_cause.value}   confidence {rules_dx.confidence:.2f}")
    if dx is not None:
        same = dx.root_cause is rules_dx.root_cause
        print(f"\n  agreement: {'MATCH' if same else 'DISAGREE'}"
              + ("" if same else f"  ({dx.root_cause.value} vs {rules_dx.root_cause.value})"))
        if not same:
            truth_risk = rules_dx.category.value == "RISK_COMPLIANCE"
            pred_risk = dx.category.value == "RISK_COMPLIANCE"
            if truth_risk and not pred_risk:
                print("  ** the model downgraded a fraud/compliance hold **")

    # ---------------------------------------------------------------- 5. the gate
    head("5. POLICY VERDICT ON THE MODEL'S OWN PROPOSAL")
    state = AgentState(transaction_id=txn.transaction_id, customer_id=txn.customer_id,
                       amount=txn.amount, currency=txn.currency,
                       failure_code=txn.failure_code, transaction=txn,
                       diagnosis=dx or rules_dx, recovery_probability=0.5,
                       expected_recovery=txn.amount_usd * 0.5)
    proposal = planner.select(state, dx or rules_dx)
    if proposal is None:
        print("  model proposed nothing parseable; deterministic planner would take over.")
        return 0
    verdict = validate(state, proposal, PolicyContext(hours_since_last_attempt=999.0))
    print(f"  proposed   {proposal.action.value}")
    print(f"  verdict    {verdict.decision.value.upper()}")
    print(f"  rules      {', '.join(verdict.rules_fired) or 'none fired'}")
    print(f"  reason     {verdict.reason}")
    if not verdict.allowed:
        print("\n  The proposal never reaches the executor. This is the architecture:")
        print("  the gate's output is the executor's input, so a wrong model cannot act.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
