"""A content hash of the safety envelope.

A decision recorded six months ago is only reproducible if you can tell which rules were
in force when it was taken. A hand-maintained version string cannot do that -- it says
whatever the last person remembered to type -- so the version is *derived* from the rules
themselves: their IDs plus the source of every rule function.

Two consequences worth stating:

* editing a rule's logic changes `POLICY_VERSION`, even if nobody bumps anything;
* every audit row stamped with a given version was evaluated by exactly that rule set.

The threshold *values* are configuration, not code, so they are hashed separately and
reported alongside. A run that only changed `MAX_RETRIES` gets a new limits hash and the
same rules hash, which is the distinction an auditor actually cares about.
"""
from __future__ import annotations

import hashlib
import inspect


def _rules_hash() -> str:
    from backend.app.policies.engine import MODIFY_RULES, REJECT_RULES, REVIEW_RULES

    h = hashlib.sha256()
    for tier, rules in (("reject", REJECT_RULES), ("review", REVIEW_RULES),
                        ("modify", MODIFY_RULES)):
        for rid in sorted(rules):
            h.update(f"{tier}:{rid}\n".encode())
            try:
                h.update(inspect.getsource(rules[rid]).encode())
            except (OSError, TypeError):
                # A dynamically defined rule (a test monkeypatch) has no source. Hash its
                # identity instead of silently ignoring it -- an unhashable rule must not
                # be an invisible one.
                h.update(repr(rules[rid]).encode())
    return h.hexdigest()[:12]


def _limits_hash() -> str:
    from backend.app import config as cfg

    keys = [
        "MAX_RETRIES", "MIN_RETRY_INTERVAL_HOURS", "MAX_AUTO_RECOVERY_AMOUNT_USD",
        "MAX_CONTACTS_PER_CASE", "MAX_AGENT_STEPS", "RECOVERY_HORIZON_DAYS",
        "MIN_EXPECTED_RECOVERY_USD", "REVIEW_MIN_DIAGNOSIS_CONFIDENCE",
        "REVIEW_MAX_EXECUTION_FAILURES",
    ]
    blob = ";".join(f"{k}={getattr(cfg, k)}" for k in keys)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def policy_version() -> str:
    """`rules@<hash>+limits@<hash>`. Computed fresh; cheap enough not to cache, and
    caching it would hide a monkeypatched rule from the very log meant to catch it."""
    return f"rules@{_rules_hash()}+limits@{_limits_hash()}"


def describe() -> dict:
    from backend.app import config as cfg
    from backend.app.policies.engine import MODIFY_RULES, REJECT_RULES, REVIEW_RULES

    return {
        "policy_version": policy_version(),
        "rules": {
            "reject": sorted(REJECT_RULES),
            "review": sorted(REVIEW_RULES),
            "modify": sorted(MODIFY_RULES),
        },
        "limits": {
            "max_retries": cfg.MAX_RETRIES,
            "min_retry_interval_hours": cfg.MIN_RETRY_INTERVAL_HOURS,
            "max_auto_recovery_amount_usd": cfg.MAX_AUTO_RECOVERY_AMOUNT_USD,
            "max_contacts_per_case": cfg.MAX_CONTACTS_PER_CASE,
            "max_agent_steps": cfg.MAX_AGENT_STEPS,
            "recovery_horizon_days": cfg.RECOVERY_HORIZON_DAYS,
            "min_expected_recovery_usd": cfg.MIN_EXPECTED_RECOVERY_USD,
            "review_min_diagnosis_confidence": cfg.REVIEW_MIN_DIAGNOSIS_CONFIDENCE,
            "review_max_execution_failures": cfg.REVIEW_MAX_EXECUTION_FAILURES,
        },
    }


class _Lazy(str):
    """`POLICY_VERSION` reads as a string but is computed on use, so importing this
    module does not freeze a value before the rules have finished registering."""

    def __new__(cls) -> _Lazy:
        return super().__new__(cls, "")

    def __str__(self) -> str:
        return policy_version()

    def __repr__(self) -> str:
        return policy_version()


POLICY_VERSION = _Lazy()
