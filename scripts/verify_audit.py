"""Verify the audit chain, from the command line.

    python scripts/verify_audit.py                    # verify and summarise
    python scripts/verify_audit.py --case txn_0001    # one case's decision timeline
    python scripts/verify_audit.py --json             # machine-readable

Exit code is 0 only when the chain verifies end to end. That makes it usable as a CI gate
and as a cron check -- a verification whose result you have to read is one nobody reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.audit.store import CURRENT_HASH_VERSION, AuditStore
from backend.app.config import DB_PATH
from backend.app.database.review import ReviewStore
from backend.app.policies.version import policy_version


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--case", help="print the decision timeline for one transaction")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--tenant", default="default")
    a = ap.parse_args()

    if not a.db.exists():
        print(f"  no database at {a.db}; run scripts/run_experiment.py first")
        return 2

    audit = AuditStore(a.db)
    ok, bad_seq = audit.verify()
    rows = audit.count()

    if a.case:
        timeline = audit.timeline(a.case)
        if a.as_json:
            print(json.dumps({"transaction_id": a.case, "chain_valid": ok,
                              "events": timeline}, indent=2, default=str))
        else:
            _print_timeline(a.case, timeline, ok)
        audit.close()
        return 0 if ok else 1

    reviews = ReviewStore(a.db, conn=audit.conn, tenant_id=a.tenant)
    payload = {
        "database": str(a.db),
        "rows": rows,
        "chain_valid": ok,
        "first_bad_seq": bad_seq,
        "hash_version": CURRENT_HASH_VERSION,
        "policy_version": policy_version(),
        "policy_decisions": audit.decision_counts(),
        "rules_fired": audit.rule_counts(),
        "human_review": reviews.stats(),
        "overrides": len(reviews.overrides(limit=10_000)),
    }
    audit.close()

    if a.as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _print_summary(payload)
    return 0 if ok else 1


def _print_summary(p: dict) -> None:
    mark = "VERIFIED" if p["chain_valid"] else "BROKEN"
    print(f"\n  audit chain      {mark}")
    print(f"  rows             {p['rows']:,}")
    if not p["chain_valid"]:
        print(f"  first bad seq    {p['first_bad_seq']}")
        print("\n  Every row from that sequence number on is unverifiable. The chain "
              "covers\n  each row's own field set, so this is tampering or corruption, "
              "not a schema\n  change.")
    print(f"  policy version   {p['policy_version']}")
    print(f"  decisions        {p['policy_decisions']}")
    top = dict(list(p["rules_fired"].items())[:8])
    print(f"  top rules        {top}")
    hr = p["human_review"]
    print(f"  human review     {hr['pending']} pending "
          f"(${hr['pending_value_usd']:,.2f}), {hr['total']} total")
    print(f"  human overrides  {p['overrides']}\n")


def _print_timeline(case: str, events: list[dict], ok: bool) -> None:
    if not events:
        print(f"  no audit trail for {case}")
        return
    print(f"\n  decision timeline for {case}   (chain {'VERIFIED' if ok else 'BROKEN'})")
    print("  " + "-" * 96)
    for e in events:
        step = str(e.get("agent_decision") or "")
        verdict = e.get("policy_result") or ""
        rules = e.get("rules_fired") or ""
        print(f"  {e['seq']!s:>5}  {step:<26}{verdict:<14}{rules:<28}"
              f"{str(e.get('reason') or '')[:60]}")
    print("  " + "-" * 96)
    last = events[-1]
    print(f"  model {last.get('model_version') or 'n/a'}   "
          f"policy {last.get('policy_version') or 'n/a'}   "
          f"run {last.get('agent_run_id') or 'n/a'}\n")


if __name__ == "__main__":
    sys.exit(main())
