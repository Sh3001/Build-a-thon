"""Phase 1 -- configuration. Every guardrail threshold lives here, not inline in logic,
so a reviewer can audit the safety envelope by reading one file."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "model"
RUN_DIR = ROOT / "data" / "runs"

for _d in (DATA_RAW, DATA_PROCESSED, MODEL_DIR, RUN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.environ.get("RECOVERAI_DB", RUN_DIR / "recoverai.db"))
#: Set to a postgresql:// URL to run the stores on Postgres instead of SQLite. Unset
#: (the default) keeps the zero-setup SQLite path. See backend/app/database/db.py.
DB_URL = os.environ.get("RECOVERAI_DB_URL") or None

# ---------------------------------------------------------------- dataset
SEED = int(os.environ.get("RECOVERAI_SEED", 20260822))
N_RECORDS = int(os.environ.get("RECOVERAI_N", 12000))
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# ---------------------------------------------------------------- guardrails
MAX_RETRIES = int(os.environ.get("RECOVERAI_MAX_RETRIES", 3))
MIN_RETRY_INTERVAL_HOURS = int(os.environ.get("RECOVERAI_MIN_RETRY_HOURS", 24))
#: Above this, no automatic money movement -- the case goes to a human instead.
MAX_AUTO_RECOVERY_AMOUNT_USD = float(os.environ.get("RECOVERAI_MAX_AUTO_AMOUNT", 5000.0))
#: Total customer contacts (reminders + links + update requests) per case.
MAX_CONTACTS_PER_CASE = int(os.environ.get("RECOVERAI_MAX_CONTACTS", 4))
#: Hard ceiling on agent loop iterations. Defence against a planner that will not settle.
MAX_AGENT_STEPS = int(os.environ.get("RECOVERAI_MAX_STEPS", 12))
#: Horizon after the original failure past which we stop trying.
RECOVERY_HORIZON_DAYS = int(os.environ.get("RECOVERAI_HORIZON_DAYS", 30))
#: Cases below this expected value are not worth a paid contact.
MIN_EXPECTED_RECOVERY_USD = float(os.environ.get("RECOVERAI_MIN_EV", 1.0))
#: Consecutive hard delivery failures on one (customer, channel) before that channel is
#: quarantined. Retrying a dead address forever burns budget and, on a real ESP, sender
#: reputation -- so the pair is parked for human review instead.
DLQ_FAILURE_THRESHOLD = int(os.environ.get("RECOVERAI_DLQ_THRESHOLD", 3))
#: Below this diagnosis confidence, no automated action runs -- the case goes to a human.
#: The floor is a business decision, so it is configuration, not a constant in a rule.
REVIEW_MIN_DIAGNOSIS_CONFIDENCE = float(os.environ.get("RECOVERAI_REVIEW_MIN_CONFIDENCE", 0.55))
#: Consecutive executor failures on one case before automated handling is suspended.
REVIEW_MAX_EXECUTION_FAILURES = int(os.environ.get("RECOVERAI_REVIEW_MAX_FAILURES", 3))
#: Hours a pending human-review task stays actionable before it expires. An unbounded
#: queue is not a safety mechanism; it is a place decisions go to be forgotten.
REVIEW_SLA_HOURS = int(os.environ.get("RECOVERAI_REVIEW_SLA_HOURS", 72))

#: Share of sends that hard-bounce, seeded per (customer, channel) so a given address is
#: consistently bad rather than randomly bad. 0 disables the failure mode entirely.
DELIVERY_FAILURE_RATE = float(os.environ.get("RECOVERAI_DELIVERY_FAILURE_RATE", 0.06))

# ---------------------------------------------------------------- economics
#: Static FX to a single reporting currency (USD). Real FX is out of scope; this keeps
#: cross-currency revenue aggregation well defined and auditable.
FX_TO_USD: dict[str, float] = {
    "USD": 1.0, "EUR": 1.09, "GBP": 1.27, "INR": 0.012, "SGD": 0.74, "AUD": 0.66,
}

#: Unit cost of each action in USD. Rolled into ROI so contact spend is never free.
ACTION_COST_USD: dict[str, float] = {
    "retry_payment": 0.02,
    "send_payment_link": 0.04,
    "send_reminder": 0.01,
    "request_payment_method_update": 0.04,
    "escalate_case": 2.50,          # a human minute is the expensive resource
    "wait": 0.0,
    "stop": 0.0,
}

# ---------------------------------------------------------------- llm
ANTHROPIC_MODEL = os.environ.get("RECOVERAI_MODEL", "claude-opus-5")
ANTHROPIC_MODEL_FAST = os.environ.get("RECOVERAI_MODEL_FAST", "claude-haiku-4-5-20251001")
LLM_ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY"))
#: Only cases above this expected value get the expensive model.
LLM_ESCALATE_EV_USD = float(os.environ.get("RECOVERAI_LLM_EV", 200.0))


def to_usd(amount: float, currency: str) -> float:
    return round(amount * FX_TO_USD.get(currency.upper(), 1.0), 2)
