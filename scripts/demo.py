"""Phase 10 -- one command that produces a populated demo environment.

    python scripts/demo.py            # full pipeline, then serve the dashboard
    python scripts/demo.py --no-serve # pipeline only
    python scripts/demo.py --quick    # smaller dataset, for a fast smoke run

Steps: dataset -> model -> held-out evaluation -> baseline -> RecoverAI -> API.
Nothing is cached and nothing is hard-coded: every number in the final summary is
computed by this run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def step(n: int, total: int, title: str) -> None:
    print(f"\n\033[1m[{n}/{total}] {title}\033[0m")
    print("-" * 70)


def run(cmd: list[str], title: str) -> None:
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"\n  !! {title} failed (exit {r.returncode})")
        raise SystemExit(r.returncode)
    print(f"  ({time.time() - t0:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the whole RecoverAI demo.")
    ap.add_argument("--no-serve", action="store_true", help="skip starting the API")
    ap.add_argument("--quick", action="store_true", help="smaller dataset and case count")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--planner", choices=["rules", "auto", "llm", "ollama"], default="auto")
    a = ap.parse_args()

    n_records = 4000 if a.quick else 12000
    limit = ["--limit", "400"] if a.quick else []
    total = 5

    print("\n\033[1mRecoverAI -- autonomous revenue recovery\033[0m")
    print(f"  python  {sys.version.split()[0]}")
    print(f"  mode    {'quick' if a.quick else 'full'}   planner: {a.planner}")

    step(1, total, "Generate the dataset")
    run([PY, "scripts/generate_dataset.py", "--n", str(n_records)], "dataset generation")

    step(2, total, "Train the recovery-probability model")
    run([PY, "ml/train.py"], "training")

    step(3, total, "Evaluate on the held-out test split")
    run([PY, "ml/evaluate.py"], "evaluation")

    step(4, total, "Run the baseline and RecoverAI on the same cases")
    run([PY, "scripts/run_experiment.py", "--fresh", "--planner", a.planner, *limit],
        "experiment")

    step(5, total, "Summary")
    exp = json.loads((ROOT / "data" / "runs" / "experiment.json").read_text())
    ev = json.loads((ROOT / "data" / "runs" / "model_evaluation.json").read_text())
    c, ag, base = exp["comparison"], exp["recoverai"], exp["baseline"]

    print(f"  model      ROC-AUC {ev['roc_auc']}  PR-AUC {ev['pr_auc']}  "
          f"F1 {ev['f1']}  (held-out, n={ev['n']:,})")
    print(f"  at risk    ${ag['revenue_at_risk']:,.2f} across {ag['cases']:,} cases")
    print(f"  baseline   ${base['revenue_recovered']:,.2f} recovered "
          f"({base['recovery_rate']:.1%}), {base['total_retries']:,} retries, "
          f"{base['risk_actions_taken']} unsafe actions")
    print(f"  RecoverAI  ${ag['revenue_recovered']:,.2f} recovered "
          f"({ag['recovery_rate']:.1%}), {ag['total_retries']:,} retries, "
          f"{ag['risk_actions_taken']} unsafe actions")
    print(f"\n  \033[1mINCREMENTAL RECOVERED REVENUE  "
          f"${c['incremental_recovered_revenue']:,.2f}  "
          f"({c['recovery_uplift_pct']}% uplift)\033[0m")
    bs = c.get("bootstrap") or {}
    if bs:
        ci, rc = bs["incremental_revenue"], bs["incremental_recovery_rate"]
        print(f"  90% interval  ${ci['p05']:,.0f} to ${ci['p95']:,.0f}"
              f"{'  (excludes zero)' if bs['excludes_zero'] else '  (includes zero)'}")
        print(f"  rate lift     {rc['p05']*100:+.1f} to {rc['p95']*100:+.1f} pp"
              f"   - the stable figure; dollar totals swing with a few large invoices")
    print(f"  audit      {exp['meta']['audit_rows']:,} rows, "
          f"chain valid: {exp['meta']['audit_chain_valid']}")

    if a.no_serve:
        print(f"\n  Start the dashboard with:\n"
              f"    .venv/bin/python -m uvicorn backend.app.main:app --port {a.port}\n")
        return 0

    dist = ROOT / "frontend" / "dist"
    if not dist.exists():
        print("\n  note: frontend/dist not built -- the API will serve JSON only.")
        print("        build it with:  cd frontend && npm install && npm run build")
    print(f"\n  Dashboard  ->  http://127.0.0.1:{a.port}/")
    print(f"  API docs   ->  http://127.0.0.1:{a.port}/docs")
    print("  Ctrl-C to stop.\n")
    subprocess.run([PY, "-m", "uvicorn", "backend.app.main:app",
                    "--port", str(a.port), "--host", "127.0.0.1"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
