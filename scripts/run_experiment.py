"""Phase 7/10 -- the end-to-end experiment.

Runs the baseline and RecoverAI over the *same* held-out cases against the *same* seeded
simulator, writes every decision to the hash-chained audit log, and persists the results
the API and dashboard read. Nothing here is hard-coded: every number in the final report
comes out of this run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.agents.llm import build_planner
from backend.app.agents.runner import run_agent_batch
from backend.app.audit.store import AuditStore
from backend.app.config import DB_PATH, DB_URL, RUN_DIR, SEED
from backend.app.database.operational import ActionLedger, DLQStore
from backend.app.database.store import CaseStore
from backend.app.ml.scorer import get_scorer
from backend.app.services.baseline import run_baseline
from backend.app.services.control import run_control
from backend.app.services.dataio import load_split, to_transactions
from backend.app.services.results import (
    bootstrap_incremental,
    compare,
    compare_to_control,
)
from simulation.payment_gateway import PaymentGateway


def money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full RecoverAI experiment.")
    ap.add_argument("--split", default="test", help="held-out split to evaluate on")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of cases (0 = all)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--planner", choices=["rules", "auto", "llm", "ollama"], default="auto")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--fresh", action="store_true", help="drop the existing database first")
    a = ap.parse_args()

    if a.fresh:
        # Deleting the file only clears SQLite. Under Postgres the database is not a file
        # we own, so "fresh" has to mean truncating the tables -- without this, --fresh
        # silently appended to the previous run and every metric was computed over two
        # runs' worth of rows.
        if DB_URL:
            from backend.app.database.db import connect
            from backend.app.database.migrations import migrate
            conn = connect(url=DB_URL)
            migrate(conn)
            for table in ("audit_log", "cases", "runs", "action_ledger", "dlq"):
                conn.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY")
            conn.commit()
            conn.close()
            print(f"  fresh: truncated {DB_URL}")
        elif a.db.exists():
            a.db.unlink()
            for suffix in ("-wal", "-shm"):
                p = Path(str(a.db) + suffix)
                if p.exists():
                    p.unlink()

    df = load_split(a.split)
    txns = to_transactions(df)
    if a.limit:
        txns = txns[: a.limit]

    scorer = get_scorer()
    if not scorer.is_trained:
        print("  !! no trained model -- run `python ml/train.py` first")
        return 1

    planner = build_planner(a.planner)
    print(f"  cases      {len(txns):,} from the held-out {a.split} split")
    print(f"  model      {scorer.model_version}")
    print(f"  planner    {'claude (' + planner.model + ')' if planner else 'deterministic rules'}")
    print(f"  seed       {a.seed}\n")

    # --- control: the counterfactual ----------------------------------------
    # Runs first because every later number is quoted relative to it.
    control_outcomes, control_report = run_control(txns, PaymentGateway(seed=a.seed))

    # --- baseline -----------------------------------------------------------
    t0 = time.time()
    base_outcomes, base_report = run_baseline(txns, PaymentGateway(seed=a.seed))
    t_base = time.time() - t0

    # --- RecoverAI ----------------------------------------------------------
    audit = AuditStore(a.db)
    # Operational state shares the audit connection so everything a run produces lands in
    # one transaction-consistent place. `run_id` scopes it, so re-running never replays
    # the previous run's cached action results.
    run_id = f"{a.split}-{a.seed}-{int(time.time())}"
    ledger = ActionLedger(run_id, conn=audit.conn)
    dlq = DLQStore(run_id, conn=audit.conn)
    events: list = []
    t0 = time.time()
    agent_outcomes, agent_report, states = run_agent_batch(
        txns, PaymentGateway(seed=a.seed), scorer=scorer, planner=planner,
        on_audit=events.append, ledger=ledger, dlq=dlq,
        progress=lambda i, n: print(f"\r  running agent  {i}/{n}", end="", flush=True),
    )
    t_agent = time.time() - t0
    print()

    audit.append_many(events)
    audit.commit()
    chain_ok, bad_seq = audit.verify()

    comparison = compare(base_report, agent_report)
    comparison["bootstrap"] = bootstrap_incremental(base_outcomes, agent_outcomes)

    vs_control = {
        "recoverai": compare_to_control(control_report, agent_report),
        "baseline": compare_to_control(control_report, base_report),
    }
    vs_control["recoverai"]["bootstrap"] = bootstrap_incremental(control_outcomes, agent_outcomes)
    vs_control["baseline"]["bootstrap"] = bootstrap_incremental(control_outcomes, base_outcomes)

    # --- persist ------------------------------------------------------------
    cases = CaseStore(a.db, conn=audit.conn)
    extra = {
        s.transaction_id: {
            "currency": s.currency, "amount": s.amount,
            "root_cause": s.root_cause.value if s.root_cause else None,
            "recommended_action": s.actions_taken[0] if s.actions_taken else None,
        } for s in states
    }
    cases.save_cases(agent_outcomes, "recoverai", extra)
    cases.save_cases(base_outcomes, "baseline")
    cases.save_cases(control_outcomes, "control")

    meta = {
        "split": a.split, "cases": len(txns), "seed": a.seed,
        "model_version": scorer.model_version,
        "planner": "llm" if planner else "rules",
        "llm_enabled": bool(planner),
        "llm_calls": getattr(planner, "calls", 0) if planner else 0,
        "llm_cost_usd": round(getattr(planner, "cost_usd", 0.0), 4) if planner else 0.0,
        "llm_failures": getattr(planner, "failures", 0) if planner else 0,
        "baseline_seconds": round(t_base, 2), "agent_seconds": round(t_agent, 2),
        "audit_rows": audit.count(), "audit_chain_valid": chain_ok,
        "audit_first_bad_seq": bad_seq,
        "policy_decisions": audit.decision_counts(),
        "rules_fired": audit.rule_counts(),
        "run_id": run_id,
        "dlq": dlq.stats(),
        "idempotency_ledger_rows": ledger.count(),
    }
    cases.save_run("control", control_report)
    cases.save_run("vs_control", vs_control)
    cases.save_run("baseline", base_report)
    cases.save_run("recoverai", agent_report)
    cases.save_run("comparison", comparison)
    cases.save_run("meta", meta)
    # The JSON goes next to the database it describes. Writing it unconditionally to
    # RUN_DIR meant that a scratch run with `--db /tmp/scratch.db` still clobbered the
    # canonical experiment.json -- so a 40-case probe silently replaced the 1,842-case
    # result every dashboard number is quoted from.
    out_json = (a.db.parent / "experiment.json") if a.db != DB_PATH \
        else (RUN_DIR / "experiment.json")
    out_json.write_text(json.dumps(
        {"meta": meta, "control": control_report, "baseline": base_report,
         "recoverai": agent_report, "comparison": comparison,
         "vs_control": vs_control}, indent=2, default=str))
    audit.close()

    # --- report -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  {'metric':<28}{'control':>12}{'baseline':>14}{'RecoverAI':>14}")
    print("-" * 78)
    rows = [
        ("revenue at risk", "revenue_at_risk", money), ("revenue recovered", "revenue_recovered", money),
        ("cases recovered", "cases_recovered", str), ("recovery rate", "recovery_rate", lambda v: f"{v:.1%}"),
        ("value recovery rate", "value_recovery_rate", lambda v: f"{v:.1%}"),
        ("avg recovery hours", "avg_recovery_hours", lambda v: f"{v:.1f}" if v else "n/a"),
        ("retries", "total_retries", str), ("customer contacts", "total_contacts", str),
        ("cases escalated", "cases_escalated", str), ("policy-blocked actions", "policy_blocked_actions", str),
        ("unsafe risk actions", "risk_actions_taken", str), ("action cost", "total_cost", money),
    ]
    for label, key, fmt in rows:
        print(f"  {label:<28}{fmt(control_report[key]):>12}"
              f"{fmt(base_report[key]):>14}{fmt(agent_report[key]):>14}")
    print("-" * 78)
    vc = vs_control["recoverai"]
    vcb = vs_control["baseline"]
    print("  IMPACT vs NO-TOUCH CONTROL (the causal number)")
    print(f"  {'incremental revenue':<28}{money(vcb['incremental_recovered_revenue']):>26}"
          f"{money(vc['incremental_recovered_revenue']):>14}")
    print(f"  {'incremental cases':<28}{vcb['incremental_cases_recovered']!s:>26}"
          f"{vc['incremental_cases_recovered']!s:>14}")
    causal_b = f"{vcb['share_of_revenue_that_is_causal']:.1%}"
    causal_a = f"{vc['share_of_revenue_that_is_causal']:.1%}"
    print(f"  {'share of revenue that is causal':<28}{causal_b:>26}{causal_a:>14}")
    print(f"  {'ROI vs control':<28}{str(vcb['roi_vs_control']) + 'x':>26}"
          f"{str(vc['roi_vs_control']) + 'x':>14}")
    cb = vc["bootstrap"]["incremental_revenue"]
    print(f"  {'90% interval (vs control)':<28}"
          f"{money(cb['p05']) + ' to ' + money(cb['p95']):>40}")
    print("-" * 78)
    print(f"  INCREMENTAL RECOVERED REVENUE   {money(comparison['incremental_recovered_revenue']):>32}")
    print(f"  recovery uplift                 {str(comparison['recovery_uplift_pct']) + '%':>32}")
    print(f"  incremental cases               {comparison['incremental_cases_recovered']!s:>32}")
    print(f"  incremental ROI                 {str(comparison['incremental_roi']) + 'x':>32}")
    bs = comparison.get("bootstrap") or {}
    if bs:
        ci = bs["incremental_revenue"]
        rc = bs["incremental_recovery_rate"]
        print(f"  90% interval (paired bootstrap) {money(ci['p05']) + ' to ' + money(ci['p95']):>32}")
        rate_txt = f"{rc['p05'] * 100:+.1f} to {rc['p95'] * 100:+.1f} pp"
        print(f"  recovery-rate lift, 90%         {rate_txt:>32}")
        print(f"  resamples where the agent won   {str(round(bs['share_positive']*100, 1)) + '%':>32}")
    print("-" * 78)
    print(f"  audit rows {meta['audit_rows']:,}   chain valid: {chain_ok}   "
          f"policy: {meta['policy_decisions']}")
    print(f"  top rules fired: {dict(list(meta['rules_fired'].items())[:6])}")
    print("=" * 78)
    print(f"\n  saved -> {a.db}")
    print(f"  saved -> {out_json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
