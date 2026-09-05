"""Multi-seed, multi-scenario evaluation.

`scripts/run_experiment.py` runs one seed and reports a number. That number is a draw
from a distribution, and quoting it as *the* result is how a simulated finding gets
overstated. This harness runs the same comparison across many independent seeds and
reports the distribution: mean, median, spread, a confidence interval, and -- the two
figures that matter most for an honest read -- the worst and best seed.

    python scripts/run_multiseed.py --seeds 20
    python scripts/run_multiseed.py --seeds 10 --scenario pessimistic
    python scripts/run_multiseed.py --seeds 10 --sweep          # every scenario

**Everything this produces is a simulation result.** The output is stamped
`SIMULATED_INTERVENTION_DATA` and every artefact carries the claim string that goes with
it. It says what happens inside this simulator under these parameters. It is not evidence
about real customers, and the report says so rather than leaving it to be inferred.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from backend.app.agents.runner import run_agent_batch
from backend.app.config import RUN_DIR, SEED
from backend.app.decision.optimizer import ProfitOptimizer
from backend.app.domain.provenance import Provenance, label
from backend.app.experiments.stats import MultiSeedReport, compare_arms_across_seeds
from backend.app.ml.scorer import get_scorer
from backend.app.ml.targeting import get_targeter
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.results import compare_to_control
from backend.app.services.strategies import BaselineSuite
from simulation.config import SimulationConfig
from simulation.payment_gateway import PaymentGateway

#: Metrics tracked per seed. Anything not listed here is not reported with an interval,
#: and a metric reported without one should be treated as decorative.
TRACKED = [
    "revenue_recovered", "recovery_rate", "cases_recovered", "total_cost",
    "total_retries", "total_contacts", "net_recovered", "risk_actions_taken",
    "value_recovery_rate",
]


def git_commit() -> str:
    """The code that produced the result. Part of reproducibility, not decoration."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip() or "unknown"
        return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except Exception:
        return "unknown"


def run_one_seed(txns, seed: int, config: SimulationConfig, scorer, targeter,
                 use_optimizer: bool) -> dict[str, dict]:
    """Every arm, one seed, one population. Returns arm name -> summary report."""
    suite = BaselineSuite(seed=seed, scorer=scorer, targeter=targeter, config=config)
    arms = {name: res.report for name, res in suite.run(txns).items()}

    # The agent arm, on a gateway seeded identically to every other arm's.
    _, agent_report, _ = run_agent_batch(
        txns, PaymentGateway(seed=seed, config=config), scorer=scorer, planner=None,
        optimizer=ProfitOptimizer() if use_optimizer else None)
    arms["recoverai"] = agent_report
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=20, help="number of independent seeds")
    ap.add_argument("--seed0", type=int, default=SEED, help="first seed; the rest follow")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="cap cases per seed (0 = all)")
    ap.add_argument("--scenario", default="default",
                    help=f"simulation scenario; one of {SimulationConfig.scenario_names()}")
    ap.add_argument("--sweep", action="store_true",
                    help="run every scenario instead of one")
    ap.add_argument("--optimizer", action="store_true",
                    help="give the agent the expected-profit arbiter")
    ap.add_argument("--out", type=Path, default=RUN_DIR / "multiseed.json")
    a = ap.parse_args()

    scorer = get_scorer()
    if not scorer.is_trained:
        print("  !! no trained model -- run `python ml/train.py` first")
        return 1
    targeter = get_targeter()

    txns = to_transactions(load_split(a.split))
    if a.limit:
        txns = txns[: a.limit]

    scenarios = SimulationConfig.scenario_names() if a.sweep else [a.scenario]
    seeds = [a.seed0 + i for i in range(a.seeds)]

    print(f"  cases      {len(txns):,} from the held-out {a.split} split")
    print(f"  seeds      {a.seeds}  ({seeds[0]} .. {seeds[-1]})")
    print(f"  scenarios  {scenarios}")
    print(f"  model      {scorer.model_version}")
    print(f"  uplift     {'loaded' if targeter.has_uplift else 'absent (arm 6 skipped)'}")
    print(f"  optimizer  {'on' if a.optimizer else 'off'}\n")

    all_results: dict[str, dict] = {}
    t0 = time.time()

    for scenario in scenarios:
        config = SimulationConfig.load(scenario)
        reports: dict[str, MultiSeedReport] = {}
        for i, seed in enumerate(seeds, 1):
            print(f"\r  {scenario:<16} seed {i}/{len(seeds)}", end="", flush=True)
            arms = run_one_seed(txns, seed, config, scorer, targeter, a.optimizer)
            for name, report in arms.items():
                reports.setdefault(name, MultiSeedReport(name)).record(seed, report, TRACKED)
                # Incremental revenue against the untouched control, per seed. This is
                # the causal quantity within the simulator, and the only one worth an
                # interval -- gross recovered includes money that arrived anyway.
                if name != "control" and "control" in arms:
                    vs = compare_to_control(arms["control"], report)
                    reports[name].metrics.setdefault(
                        "incremental_vs_control",
                        __import__("backend.app.experiments.stats", fromlist=["SeedResults"])
                        .SeedResults("incremental_vs_control")
                    ).add(seed, vs["incremental_recovered_revenue"])
        print()

        summaries = {n: r.summary() for n, r in reports.items()}
        comparisons = {}
        for arm in reports:
            if arm == "recoverai":
                continue
            comparisons[f"recoverai_vs_{arm}"] = compare_arms_across_seeds(
                reports["recoverai"], reports[arm], "revenue_recovered")
        all_results[scenario] = {
            "config_fingerprint": config.fingerprint(),
            "config_description": config.description,
            "arms": summaries,
            "paired_comparisons": comparisons,
        }
        _print_scenario(scenario, summaries, comparisons)

    payload = label({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "split": a.split, "cases_per_seed": len(txns),
        "seeds": seeds, "n_seeds": len(seeds),
        "model_version": scorer.model_version,
        "uplift_model_loaded": targeter.has_uplift,
        "optimizer_enabled": a.optimizer,
        "elapsed_seconds": round(time.time() - t0, 1),
        "scenarios": all_results,
    }, Provenance.SIMULATED_INTERVENTION)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  {payload['claim']}")
    print(f"  saved -> {a.out}\n")
    return 0


def _print_scenario(scenario: str, summaries: dict, comparisons: dict) -> None:
    print(f"\n  == {scenario} " + "=" * (66 - len(scenario)))
    print(f"  {'arm':<16}{'mean recovered':>16}{'95% CI':>26}{'worst seed':>14}")
    print("  " + "-" * 72)
    for name in ("control", "naive_retry", "smart_retry", "ml_probability",
                 "expected_value", "uplift", "recoverai"):
        m = summaries.get(name, {}).get("metrics", {}).get("revenue_recovered")
        if not m:
            continue
        ci = f"[{m['ci_low']:,.0f} .. {m['ci_high']:,.0f}]"
        print(f"  {name:<16}{m['mean']:>16,.0f}{ci:>26}{m['worst']:>14,.0f}")

    print(f"\n  {'incremental vs no-touch control':<40}{'mean':>12}{'95% CI':>26}")
    print("  " + "-" * 78)
    for name in ("naive_retry", "smart_retry", "ml_probability", "expected_value",
                 "uplift", "recoverai"):
        m = summaries.get(name, {}).get("metrics", {}).get("incremental_vs_control")
        if not m:
            continue
        ci = f"[{m['ci_low']:,.0f} .. {m['ci_high']:,.0f}]"
        flag = "" if m["excludes_zero"] else "  (interval includes zero)"
        print(f"  {name:<40}{m['mean']:>12,.0f}{ci:>26}{flag}")

    print(f"\n  {'RecoverAI vs arm, paired across seeds':<40}{'mean diff':>12}{'95% CI':>26}")
    print("  " + "-" * 78)
    for key, c in comparisons.items():
        if not c.get("seeds"):
            continue
        ci = f"[{c['ci_low']:,.0f} .. {c['ci_high']:,.0f}]"
        note = "" if c["excludes_zero"] else "  (not distinguishable from zero)"
        print(f"  {key.replace('recoverai_vs_', 'vs '):<40}"
              f"{c['mean_difference']:>12,.0f}{ci:>26}{note}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
