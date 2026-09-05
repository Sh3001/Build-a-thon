"""Throughput and latency at increasing case counts.

    python scripts/bench_throughput.py
    python scripts/bench_throughput.py --sizes 100 1000 10000

Measures the two things that decide whether this design survives a real queue: how long
one case takes end to end, and whether that number is flat as the population grows. A
per-case cost that rises with N means something is quadratic and will fall over at scale;
a flat one means the batch runner is the only thing that needs to change.

Deliberately measured **without** the LLM. Inference latency dominates everything else and
would hide the cost of the parts this project actually owns. `scripts/bench_planner.py`
measures the planner separately.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from backend.app.agents.runner import run_agent_batch
from backend.app.config import RUN_DIR, SEED
from backend.app.decision.optimizer import ProfitOptimizer
from backend.app.ml.scorer import get_scorer
from backend.app.policies.engine import PolicyContext, validate
from backend.app.services.dataio import load_split, to_transactions
from simulation.payment_gateway import PaymentGateway


def peak_rss_mb() -> float:
    """Process high-water-mark RSS in MB.

    `ru_maxrss` is bytes on macOS and kilobytes on Linux, so the unit is decided by the
    platform and never by the magnitude. The first version of this function guessed from
    the magnitude -- anything over 1GB must be bytes -- which silently reported 200MB as
    "201040M" and only looked right once the process happened to cross a gigabyte. A
    memory figure wrong by 1000x is easy to miss in a table, which is exactly why it must
    not be inferred.

    Note this is a *high-water mark for the whole process*, so in a run that measures
    several batch sizes the later figures include the earlier batches' peak. It is
    reported per row anyway because the useful signal is whether it keeps climbing.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def grow(txns: list[dict], n: int) -> list[dict]:
    """Repeat the population to reach `n`, giving each copy a distinct transaction id.

    The ids must differ or the simulator's per-case latents collide and the copies stop
    being independent cases -- which would make the benchmark measure a much easier
    workload than the real one.
    """
    out: list[dict] = []
    i = 0
    while len(out) < n:
        for t in txns:
            if len(out) >= n:
                break
            copy = dict(t)
            copy["transaction_id"] = f"{t['transaction_id']}_r{i}"
            out.append(copy)
        i += 1
    return out


def bench_policy_engine(iterations: int = 50_000) -> dict:
    """The gate on its own. Every case passes through it several times, so its per-call
    cost multiplies through the whole system."""
    from backend.app.models.enums import InterventionType
    from backend.app.models.schemas import AgentState, ProposedAction

    state = AgentState(transaction_id="t", customer_id="c", amount=250.0,
                       failure_code="insufficient_funds", expected_recovery=100.0)
    action = ProposedAction(action=InterventionType.RETRY_PAYMENT)
    ctx = PolicyContext(hours_since_last_attempt=999.0)

    t0 = time.perf_counter()
    for _ in range(iterations):
        validate(state, action, ctx)
    elapsed = time.perf_counter() - t0
    return {"iterations": iterations,
            "microseconds_per_call": round(elapsed / iterations * 1e6, 2),
            "calls_per_second": round(iterations / elapsed)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 1_000, 10_000, 100_000])
    ap.add_argument("--split", default="test")
    ap.add_argument("--optimizer", action="store_true")
    ap.add_argument("--out", type=Path, default=RUN_DIR / "throughput.json")
    a = ap.parse_args()

    scorer = get_scorer()
    if not scorer.is_trained:
        print("  !! no trained model -- run `python ml/train.py` first")
        return 1

    base = to_transactions(load_split(a.split))
    print(f"  base population {len(base):,} cases from the {a.split} split")
    print(f"  model           {scorer.model_version}")
    print(f"  optimizer       {'on' if a.optimizer else 'off'}")
    print("  planner         deterministic (LLM latency would hide everything else)\n")

    gate = bench_policy_engine()
    print(f"  policy engine   {gate['microseconds_per_call']} us/call "
          f"({gate['calls_per_second']:,}/s)\n")

    rows = []
    print(f"  {'cases':>9}{'seconds':>10}{'cases/s':>11}{'ms/case':>10}"
          f"{'peak RSS':>11}{'recovered':>13}")
    print("  " + "-" * 66)
    for n in a.sizes:
        txns = grow(base, n)
        t0 = time.perf_counter()
        _, report, _ = run_agent_batch(
            txns, PaymentGateway(seed=SEED), scorer=scorer, planner=None,
            optimizer=ProfitOptimizer() if a.optimizer else None)
        elapsed = time.perf_counter() - t0
        row = {
            "cases": n,
            "seconds": round(elapsed, 3),
            "cases_per_second": round(n / elapsed, 1),
            "ms_per_case": round(elapsed / n * 1000, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "revenue_recovered": report["revenue_recovered"],
        }
        rows.append(row)
        print(f"  {n:>9,}{row['seconds']:>10.2f}{row['cases_per_second']:>11,.0f}"
              f"{row['ms_per_case']:>10.3f}{row['peak_rss_mb']:>10.1f}M"
              f"{row['revenue_recovered']:>13,.0f}")

    per_case = [r["ms_per_case"] for r in rows]
    verdict = "flat" if len(per_case) < 2 or max(per_case) / min(per_case) < 1.5 \
        else "RISING -- something scales worse than linearly"
    print("\n  " + "-" * 66)
    print(f"  per-case cost across sizes: {verdict}")
    if len(per_case) > 1:
        print(f"  spread {min(per_case):.3f} to {max(per_case):.3f} ms/case "
              f"(ratio {max(per_case) / min(per_case):.2f}x)")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": scorer.model_version,
        "optimizer_enabled": a.optimizer,
        "planner": "deterministic",
        "policy_engine": gate,
        "batches": rows,
        "per_case_ms_spread": [min(per_case), max(per_case)],
        "verdict": verdict,
        "note": ("single process, one machine, mock rails. Not a distributed throughput "
                 "figure and not a claim about production latency, where a real gateway "
                 "call would dominate. peak_rss_mb is a process high-water mark, so later "
                 "rows include earlier batches."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"\n  saved -> {a.out}\n")
    print("  " + payload["note"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
