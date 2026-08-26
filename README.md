# RecoverAI - Autonomous AI Revenue Recovery

An agent that works a queue of failed payments end to end: **score, diagnose, choose an
intervention, execute it under hard limits, monitor, and stop when it should** - measured
as dollars recovered against a baseline dunning strategy on a held-out test set, not cases
flagged.

```
1,842 held-out cases · $346,631 at risk · 30-day horizon · three arms, one population

                        control      baseline     RecoverAI
money recovered        $45,955      $107,344      $144,167
recovery rate            16.2%         35.9%         44.3%
retries used                 0         4,563         2,610
unsafe risk actions          0           392             0
cost                     $0.00        $91.26       $442.47

MONEY WE CAUSED (vs no-touch control)
  baseline              $61,389   - 57.2% of its gross was causal
  RecoverAI             $98,213   - 68.1% of its gross was causal
                        90% CI $74,145 to $123,000 · ROI 222x
```

**The control arm is the point.** 16.2% of these cases self-cure with no intervention at
all - $45,955 arrives whether or not anyone lifts a finger. Quoting the $144,167 gross
figure as "recovered by the agent" would overstate impact by **32%**. The number that
survives scrutiny is **$98,213 caused**, and it is the one the dashboard leads with.

More money, on **43% fewer retries** than the baseline, with **zero** automated actions on
fraud or compliance holds. Every figure here is produced by `scripts/run_experiment.py`;
nothing is hard-coded.

> **Read the case-rate lift, not the dollar total.** Recovery amounts are lognormal, so
> dollar totals swing on whether a few large invoices land. The case-rate figures are
> stable.

---

## Contents

| | |
|---|---|
| **Start here** | [Quickstart](#quickstart) · [Setup](#setup) · [Commands](#commands) |
| **Understand it** | [Architecture](#architecture) · [The one design decision](#the-one-design-decision-that-matters) · [Workflow](#the-langgraph-workflow) |
| **Use it** | [Dashboard](#the-dashboard) · [API](#api-reference) · [Configuration](#configuration-reference) |
| **Operate it** | [Storage](#storage-sqlite-or-postgres) · [Migrations](#schema-migrations) · [Testing & CI](#testing-ci) |
| **Trust it** | [Safety](#safety-and-guardrails) · [Audit log](#audit-log) · [Dead letter queue](#dead-letter-queue) |
| **Check the claims** | [Model evaluation](#model-evaluation) · [Real data](#results-on-real-data) · [Uplift](#uplift-targeting) · [Results](#baseline-vs-recoverai) |
| **The caveats** | [Known limitations](#known-limitations) |

---

## Quickstart

The repository ships with a populated database and a built dashboard, so nothing needs to
be generated first:

```bash
cd recoverai
./run.sh serve
```

Open **<http://127.0.0.1:8000/>**. That is the whole thing - no database server, no API
key, no network.

---

## Setup

**Requirements**: Python 3.11+. Node 18+ only if you intend to modify the dashboard - the
repo ships a built `frontend/dist`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

cd frontend && npm install && npm run build && cd ..   # optional
```

Two optional integrations, both off by default and both reported by `/api/health` so a
fallback can never be mistaken for the real thing:

| Variable | Effect when unset |
|---|---|
| `ANTHROPIC_API_KEY` | Agent uses the deterministic planner (`llm_enabled: false`) |
| `RECOVERAI_DB_URL` | Storage is SQLite at `data/runs/recoverai.db` (`db_engine: "sqlite"`) |

`psycopg` is in `requirements.txt` but is imported only when a Postgres URL is set.

---

## Commands

`run.sh` resolves its own interpreter, so it works from any directory and cannot pick up a
sibling project's virtualenv.

| Command | What it does |
|---|---|
| `./run.sh serve [PORT]` | Serve dashboard + API (default 8000) |
| `./run.sh demo [ARGS]` | Full pipeline, then serve - `--quick`, `--no-serve` |
| `./run.sh test [ARGS]` | The test suite |
| `./run.sh pipeline` | dataset → train → evaluate → experiment, no server |
| `./run.sh shell` | Python REPL with the project importable |

Step by step, if you want to watch each stage:

```bash
.venv/bin/python scripts/generate_dataset.py       # 12,000 records + customer-disjoint splits
.venv/bin/python ml/train.py                       # XGBoost vs logistic, selected on validation
.venv/bin/python ml/evaluate.py                    # held-out metrics
.venv/bin/python scripts/run_experiment.py --fresh # control vs baseline vs RecoverAI
.venv/bin/python -m uvicorn backend.app.main:app --port 8000
```

With Docker:

```bash
docker compose up --build                      # SQLite, one command
docker compose --profile postgres up --build   # brings up Postgres alongside
```

Frontend dev server (proxies `/api` to :8000): `cd frontend && npm run dev`

---

## Architecture

```
Transaction events            data/processed/*.csv (customer-disjoint splits)
        │
        ▼
Feature engineering           backend/app/ml/features.py    55 features, enum-driven vocab
        │
        ▼
Recovery probability model    ml/train.py → XGBoost vs logistic, selected on validation
        │
        ▼
Root cause engine             backend/app/agents/diagnose.py   may OVERRULE the error code
        │
        ▼
Recovery strategy agent       backend/app/agents/strategy.py   proposes only
        │                     backend/app/agents/llm.py        optional Claude planner
        ▼
╔═══════════════════════╗
║   POLICY VALIDATOR    ║     backend/app/policies/engine.py   17 deterministic rules
╚═══════════════════════╝
        │  approve / modify / reject  ← the ONLY path to a side effect
        ▼
Action executor               backend/app/tools/executor.py    idempotent, allowlisted
        │
        ▼
Outcome monitor               backend/app/agents/graph.py      the only place a case closes
        │
        ▼
Audit store                   backend/app/audit/store.py       append-only, hash-chained
        │
        ▼
FastAPI  →  React dashboard
```

### The one design decision that matters

**The model proposes; the policy engine disposes.**

The LLM is handed **no tools**. It returns a parsed Pydantic object - a diagnosis or a
proposed action - and every proposal is evaluated by a deterministic rule engine that
returns approve / modify / reject plus the rule IDs that fired. The executor's *input* is
the policy engine's *output*:

```python
def execute(self, state, policy: PolicyResult, txn) -> ActionResult:
    if not policy.allowed or policy.effective_action is None:
        raise PolicyViolation(...)
    action = policy.effective_action     # never the action that was proposed
```

There is no code path from a proposal to a side effect that skips the gate. That is
asserted directly in `backend/tests/test_safety.py`, which drives the agent with a
deliberately hostile planner - see [Safety](#proving-the-gate-actually-works).

### The LangGraph workflow

```
START → load_transaction → score_recovery → diagnose_root_cause
      → calculate_expected_recovery → select_intervention → validate_policy
      → [approved?] → execute_action → monitor_outcome
      → {success | retry | stop | escalate} → END
```

The loop is bounded three independent ways - per-cause retry caps, `MAX_AGENT_STEPS`, and
the 30-day horizon - so it terminates even if the planner never proposes `STOP`.

---

## The dashboard

Six pages, all reachable with number keys **1**–**6**. Rows in the Recovery Queue and
Audit Log open the case behind them, so you can get from a headline number to the decision
chain that produced it in two clicks.

**Overview** - the scoreboard. Tiles showing **View cases →** on hover are drill-downs:
clicking *Cases Escalated* opens the queue filtered to exactly those cases. *Money We
Caused* and *Would Have Arrived Anyway* carry tooltips explaining the control arm, because
they are the two figures most easily misread. Intervention bars are clickable too.

**Recovery Queue** - the work list, ranked by expected recovery.

| Control | Behaviour |
|---|---|
| Status / Failure cause | Filter, both defaulting to **All**. Server-side. |
| Search | Transaction, customer, cause, action, status. Underscore-insensitive, so `retry payment` and `retry_payment` both match. |
| Column headers | Click to sort, click again to reverse. Server-side, so ascending really is the global minimum, not the bottom of the fetched page. |
| Cause / status cells | Click a value in any row to filter by it. |
| Reset *n* filters | Appears only when something is filtered, and says how many. |

**Agent Trace** - one case end to end, alongside **the same transaction under the baseline
strategy**, which is what lets a single case argue for itself: `txn_0001586` - baseline
burned three retries and collected $0; RecoverAI replaced the dead card first and collected
$4,580.25. Where the baseline did better, the page says so rather than staying quiet.
*Actions & policy only* collapses a 20-step trace to the steps where something happened.
`j` / `k` step through the queue's current order.

**Revenue Analytics** - cumulative recovery over time with a crosshair, recovery rate by
category, and the baseline-vs-RecoverAI delta table. The delta column knows which direction
is an improvement, so fewer retries reads as a win and extra cost does not.

**Audit Log** - every decision and policy verdict, newest first. Filter by stage or policy
verdict (both server-side, so **Reject** searches all rows rather than the visible page)
and search the reason text.

**Dead Letter Queue** - quarantined (customer, channel) pairs with a Release button. See
[Dead letter queue](#dead-letter-queue).

**Keyboard**

| Key | Action |
|---|---|
| `1`–`6` | Switch page |
| `↑` `↓` | Move between table rows |
| `Enter` | Open the focused row |
| `/` | Jump to the search box |
| `Esc` | Clear the search, or deselect |
| `j` / `k` | Next / previous case (Agent Trace) |

Shortcuts are suppressed while you are typing in a filter. **Theme** (Auto / Light / Dark)
sits in the header and is remembered per browser; filter and sort choices persist in
`localStorage`.

> **If the dashboard looks unchanged after a rebuild**, load it once with a query string
> (`http://127.0.0.1:8000/?fresh=1`). `index.html` is served `no-cache`, but a copy cached
> before that fix can still be held by the browser. Assets are content-hashed and cache
> normally.

---

## API reference

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/api/health` | model/dataset/experiment readiness, audit rows, **chain validity**, live DB engine |
| GET | `/api/overview` | the dashboard cards |
| GET | `/api/metrics` | full per-arm metrics, model card, comparison, bootstrap intervals |
| GET | `/api/baseline` | baseline-arm metrics alone |
| GET | `/api/revenue-at-risk` | at-risk totals by category and failure code, plus top cases |
| GET | `/api/recovery-queue` | ranked queue; `limit`, `offset`, `status`, `failure_code`, `sort`, `direction` |
| GET | `/api/cases/{transaction_id}` | case detail + full audit trail + baseline counterpart |
| GET | `/api/cases/{transaction_id}/audit` | just the hash-chained trace |
| GET | `/api/audit` | audit rows; `limit`, `decision`, `policy_result` |
| GET | `/api/dlq` | quarantined pairs + stats; `quarantined_only`, `limit` |
| POST | `/api/dlq/release` | put a quarantined pair back in service |
| POST | `/api/recovery/run` | run a batch through the agent |
| POST | `/api/recovery/run/{transaction_id}` | re-run one case live |

```json
GET /api/health
{"status":"ok","model_trained":true,"model_version":"logistic-recovery-v1",
 "dataset_ready":true,"experiment_ready":true,"audit_rows":28432,
 "audit_chain_valid":true,"llm_enabled":false,"db_engine":"postgres"}
```

---

## Storage: SQLite or Postgres

SQLite is the default and needs no setup - a single file at `data/runs/recoverai.db`. The
same stores run on Postgres when given a URL:

```bash
createdb recoverai
.venv/bin/python scripts/migrate_to_postgres.py --url postgresql://localhost/recoverai

export RECOVERAI_DB_URL=postgresql://localhost/recoverai
./run.sh serve
```

The migration copies `cases`, `audit_log` and `runs`, preserves each audit row's `seq`, and
**re-verifies the hash chain on the destination** before reporting success - a migration
that reordered history fails there rather than on the dashboard. `--replace` makes it
re-runnable.

**There is no silent fallback.** With the URL set and the server unreachable, the app
refuses to start rather than quietly writing to a local file:

```
RuntimeError: RECOVERAI_DB_URL is set but the database is unreachable: connection refused
Start Postgres, or unset RECOVERAI_DB_URL to fall back to SQLite.
```

The dashboard header and `/api/health` both name the live engine.

### Schema migrations

`CREATE TABLE IF NOT EXISTS` is not a migration strategy - on a database that already has
the table it silently does nothing, so a changed column type or a new column never lands
and the app fails later with a missing-column error. Tolerable when the database was a
disposable local file; not once it started outliving the code.

Schema changes are a numbered list in `backend/app/database/migrations.py` with a
`schema_migrations` ledger. `migrate()` runs on every store construction, is idempotent,
and an existing database adopts the ledger without being rebuilt. Deliberately not Alembic:
three tables and no ORM do not justify the dependency.

> Dropping a table by hand no longer recreates it - the ledger correctly believes it is
> already applied. Drop `schema_migrations` too if you want a genuine rebuild.

---

## Safety and guardrails

All 17 policy rules live in one file and each has a dedicated test. A meta-test
(`test_every_rule_has_a_test`) **fails the build if a rule is added without one**.

| Control | Rule ID | Behaviour |
|---|---|---|
| Action allowlist | `R-ALLOWLIST` | closed enum; an invented tool name is refused even if it bypasses schema validation |
| Compliance/risk block | `R-RISK-BLOCK` | fraud / high-risk / compliance → escalation is the *only* permitted action |
| Dead instrument | `R-DEAD-INSTRUMENT` | never re-present an expired/invalid method until it is replaced |
| Unreachable account | `R-PERSISTENT-NORETRY` | closed/invalid accounts are never retried |
| Max retry count | `R-MAX-RETRIES` | global ceiling of 3 |
| Per-cause retry cap | `R-CAUSE-RETRY-CAP` | insufficient funds → 2; multiple declines → 1 |
| Amount threshold | `R-AMOUNT-CAP` | no automated money movement above $5,000 |
| Contact limit | `R-CONTACT-CAP` | ≤4 customer touches per case |
| Minimum retry interval | `R-COOLDOWN` | 24h, enforced by **deferring** rather than rejecting |
| Recovery horizon | `R-HORIZON` | stop after 30 days |
| Agent step ceiling | `R-STEP-CAP` | bounds the loop absolutely |
| Value floor | `R-MIN-VALUE` | no paid contact below the expected-value floor |
| Idempotency | `R-IDEMPOTENT` | one-shot actions cannot be re-sent |
| Human escalation | `R-ESCALATE-HIGH-VALUE` | a high-value case is escalated, never silently dropped |
| Channel consent | `R-CHANNEL` | contact on the customer's stated preferred channel |
| Terminal case | `R-TERMINAL` | automatic stop after success - a closed case takes no action |
| Dead address | `R-DLQ` | a channel that hard-bounces repeatedly is quarantined; no further contact goes out on it |

### Proving the gate actually works

In a normal run the policy engine rejects only 80 of 9,552 policy verdicts (and modifies
another 130) - because the deterministic planner is compliant by construction. That proves
the *planner* is well behaved; it does not prove the *gate* is. So `test_safety.py` drives
the whole agent with a `RoguePlanner` that proposes the most damaging available action at
every step:

- cannot retry a fraud case (`attempt_count == 0` on every risk case)
- cannot message a compliance hold (`contact_count == 0`)
- cannot exceed the retry ceiling, the contact cap, or the amount ceiling
- cannot loop forever
- **the payment rail receives zero calls** across a hostile run on risk cases

### Audit log

Every decision is one row whose hash covers the row before it. Editing, deleting, or
altering an amount in any row breaks the chain, and `verify()` reports the first bad
sequence number - each asserted by its own test. The store exposes **no update and no
delete method**, so "never overwrite history" is enforced by the type, not by convention.
The dashboard header and `/api/health` surface live chain validity.

### Dead letter queue

Delivery can fail. A hard bounce is modelled as a property of the **(customer, channel)
pair**, seeded by hash, so a bad address bounces every time rather than intermittently -
which is what makes a consecutive-failure counter meaningful at all. Three consecutive
bounces quarantine the pair, `R-DLQ` refuses further contact on it, and a *successful*
delivery resets the counter so a transient outage never quarantines anyone.

Retrying a dead address costs money per attempt, delivers nothing, and on a real provider
damages sender reputation for every other customer. The **Dead Letter Queue** page lists
quarantined pairs and releases them after review.

Current run: 97 hard bounces across 53 pairs, 13 quarantined, `R-DLQ` blocking 7 contact
attempts.

### Idempotency ledger

Keys are derived from (transaction, action, attempt) and mirrored to an `action_ledger`
table, so "a replay never re-charges" survives a process restart rather than living only in
one object's memory.

The ledger is scoped by `run_id`, and that scoping is load-bearing: without it the second
experiment run would replay the first run's cached results and recover nothing, turning a
re-run into a silent no-op.

### The LLM path, exercised locally

The architecture claims a reasoning layer. Without an API key that claim ships
unexercised, and `llm_calls: 0` is not evidence of anything. `backend/app/agents/ollama.py`
runs the same planner against a local Ollama daemon, so the LLM path can be demonstrated
offline and for free:

```bash
ollama serve && ollama pull llama3.2:3b
.venv/bin/python scripts/run_experiment.py --planner ollama --limit 40 --fresh
.venv/bin/python scripts/bench_planner.py --n 20 --models llama3.2:3b,qwen2.5:3b-instruct
```

It subclasses `LLMPlanner` rather than reimplementing it, so the parts that matter are
inherited unchanged: the allowlisted case view, the closed action enum, the EV-based model
routing, and the silent fall-back. Only the transport differs. Ollama's `format` parameter
takes a JSON Schema, which Pydantic emits directly, so the structured-output contract is
the same one the hosted path uses.

**Measured diagnosis quality** (`scripts/bench_planner.py`, held-out split):

| planner | accuracy | unsafe errors | s/case |
|---|---|---|---|
| rules (deterministic) | **100.0%** | 0 | 0.00 |
| ollama `llama3.2:3b` | 65.0% | 0 | 8.69 |
| ollama `llama3.1:8b` | 30.0% | 3 | 23.27 |
| ollama `qwen2.5:3b-instruct` | 25.0% | 8 | 7.78 |

Two things to read carefully. First, **the rules planner scores 100% by construction** -
on synthetic data `failure_code` *is* ground truth, so reading it is not a skill. This
benchmark cannot show an LLM winning; it can only show how much one loses. Second,
**accuracy hides the asymmetry**: `qwen2.5:3b` produced 7 *recoverable-killed* errors
(diagnosing a live case as a closed account, abandoning recoverable money) and 1
*risk-downgraded* error - diagnosing a fraud/compliance hold as something automatable.

**And that is the point.** Driving the full agent over fraud and compliance cases with the
worst model available:

```
6 fraud/compliance cases, planner = qwen2.5:3b-instruct (25% accurate)

  retry attempts on those cases : 0
  customer contacts on those cases : 0
  actions the gate refused : 36
     28x retry_payment    R-RISK-BLOCK
      7x wait             R-RISK-BLOCK
      1x send_reminder    R-MIN-VALUE
```

A model that is wrong three times out of four proposed retrying a fraud hold **28 times**,
and not one reached the payment rail. This is the architecture's central claim tested
against a genuinely bad reasoner rather than a compliant one - the adversarial
`RoguePlanner` in `test_safety.py` proves the gate holds against a *deliberately* hostile
planner; this proves it holds against an *accidentally* hostile one.

**Local models are not currently good enough to run this agent's diagnosis**, and the
project does not pretend otherwise. The deterministic planner remains the default, and
`--planner auto` will **never** select a local model on its own - with accuracy at 100%
for rules and 25-65% for these models, silently preferring Ollama merely because a daemon
happened to be running would degrade every run on that machine with no signal in the
output. Local inference is opt-in via `--planner ollama`
(`test_auto_never_silently_selects_a_local_model`). What
the local path buys is a demonstrable, zero-cost LLM integration and a live proof that the
gate contains a bad model.

### PII on the LLM path

`LLMPlanner._case_view` is an **allowlist** - it names the 13 behavioural fields that go
out, so adding a field to `Transaction` never silently widens what reaches a third-party
API. No customer, transaction, invoice or subscription identifier is ever sent.
`backend/app/agents/redaction.py` is a backstop against that list drifting, and
`test_case_view_carries_no_identifiers` fails the build if it does.

---

## Testing & CI

```bash
./run.sh test                                    # 282 offline
RECOVERAI_TEST_DB_URL=postgresql://localhost/recoverai_test \
  .venv/bin/python -m pytest backend/tests/ -q   # 291 including the Postgres round-trip
```

The offline suite is the gate: no database server, no API key, no network. Postgres
round-trip tests skip unless `RECOVERAI_TEST_DB_URL` points at a **scratch** database - its
tables are dropped at the start of every test.

`.github/workflows/ci.yml` runs four jobs: the offline suite, the Postgres suite against a
service container, a pipeline smoke test that asserts the audit chain verifies after a real
run, and the frontend build.

---

## Configuration reference

Every guardrail threshold lives in `backend/app/config.py`, not inline in logic, so the
safety envelope can be audited by reading one file. All are environment-overridable.

| Variable | Default | Meaning |
|---|---|---|
| `RECOVERAI_MAX_RETRIES` | 3 | Global retry ceiling |
| `RECOVERAI_MIN_RETRY_HOURS` | 24 | Minimum interval between attempts |
| `RECOVERAI_MAX_AUTO_AMOUNT` | 5000 | Above this, no automated money movement |
| `RECOVERAI_MAX_CONTACTS` | 4 | Customer touches per case |
| `RECOVERAI_MAX_STEPS` | 12 | Hard ceiling on agent loop iterations |
| `RECOVERAI_HORIZON_DAYS` | 30 | Stop trying after this |
| `RECOVERAI_MIN_EV` | 1.0 | Expected-value floor for a paid contact |
| `RECOVERAI_DLQ_THRESHOLD` | 3 | Consecutive bounces before quarantine |
| `RECOVERAI_DELIVERY_FAILURE_RATE` | 0.06 | Simulated hard-bounce rate; `0` disables |
| `RECOVERAI_SEED` | 20260822 | Simulator seed |
| `RECOVERAI_N` | 12000 | Generated dataset size |
| `RECOVERAI_DB` / `RECOVERAI_DB_URL` | - | Storage location / Postgres URL |
| `RECOVERAI_MODEL` / `_MODEL_FAST` / `_LLM_EV` | - | Claude model routing |

---

## Model evaluation

Held-out test split, n=1,842.

| | value |
|---|---|
| ROC-AUC | **0.7764** |
| PR-AUC | **0.6625** |
| Precision / Recall / F1 | 0.5730 / 0.8806 / **0.6943** |
| Brier (calibrated) | 0.1907 |

Threshold (0.330) is chosen on **validation**, never on test. Splits are
**customer-disjoint** - the same customer never appears in both train and test
(`test_split_is_customer_disjoint`).

### Recovery@K - ranking by expected recovery

| K | share of queue | precision@K | revenue captured | of all recoverable | lift |
|---|---|---|---|---|---|
| 50 | 2.7% | 0.620 | $78,582 | **50.4%** | 18.6× |
| 100 | 5.4% | 0.630 | $100,682 | **64.5%** | 11.9× |
| 250 | 13.6% | 0.588 | $124,370 | 79.7% | 5.9× |
| 500 | 27.1% | 0.566 | $137,965 | 88.4% | 3.3× |

**Working the top 5% of the queue captures 64% of all recoverable revenue.**

### The model is selected, not asserted

`ml/train.py` trains **both** a gradient-boosted and a regularised logistic model and ships
whichever wins on validation ROC-AUC:

| candidate | validation ROC-AUC |
|---|---|
| XGBoost | 0.8056 |
| logistic regression | **0.8106** ← shipped |

XGBoost does *not* win here, and this README does not pretend otherwise. The label comes
from a logistic model plus a piecewise timing term, and `features.py` hands both candidates
that interaction explicitly. Remove those two columns and XGBoost wins by +0.007 - but
absolute AUC is lower, so the features stay and the linear model ships.
`test_the_shipped_model_is_the_one_that_won_validation` enforces this.

The **Bayes ceiling** - scoring the generator's own noise-free probability against the
sampled labels - is **0.8139**. The shipped model reaches 0.8106, ~99.6% of the signal that
exists. The ceiling is low because the labels are deliberately stochastic, not because the
model is weak.

---

## Results on real data

Everything above comes from a simulator. This section does not.

**Source:** [UCI "Default of Credit Card Clients"](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(Yeh & Lien, 2009) - 30,000 real Taiwanese credit-card customers with six months of real
repayment status. `scripts/build_real_dataset.py` extracts **17,829 real delinquency
events** across 7,622 customers carrying **NT$890.7M (~$27.6M) of real money at risk**.

```bash
curl -L -o data/external/uci_credit.xls \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
.venv/bin/python scripts/build_real_dataset.py     # 17,829 real events
.venv/bin/python ml/train_real.py                  # real model, real metrics
.venv/bin/python scripts/real_prioritization.py    # assumption-free result
.venv/bin/python scripts/run_real_experiment.py --sweep
```

### The real decay curve

The "recovery probability decays with time" mechanic the synthetic simulator *assumes* is
**measured** here:

| months late | events | real cure rate |
|---|---|---|
| 2 | 15,939 | 25.9% |
| 3 | 1,108 | 15.9% |
| 4 | 377 | 4.2% |
| 5 | 111 | 5.4% |
| 7 | 209 | 0.0% |

### Real held-out metrics (n=2,739, customer-disjoint)

| | value |
|---|---|
| ROC-AUC | **0.8394** |
| PR-AUC | 0.6215 |
| Precision / Recall / F1 | 0.525 / 0.709 / 0.603 |
| Brier | 0.1321 |

**On real data XGBoost wins: 0.8365 vs logistic 0.7831 (+0.053).** That reverses the
synthetic finding - the tree model was losing there because the synthetic label *was* a
logistic model with the interactions handed over as columns. Given real feature
interactions, the gradient-boosted model earns its place decisively.

### The assumption-free result

If a collections team can only work K cases, ranking by expected recovery vs the
alternatives - **real amounts, real observed outcomes, no intervention model at all**:

| ranking | top 100 | top 250 | top 500 | top 1000 |
|---|---|---|---|---|
| **expected recovery (model)** | **26.7%** | **53.6%** | **70.0%** | **87.5%** |
| biggest amount first | 20.8% | 37.6% | 52.1% | 74.5% |
| freshest delinquency first | 2.7% | 9.7% | 21.9% | 27.8% |
| random | 3.4% | 9.3% | 18.3% | 36.5% |

**Working 9% of the queue captures 53.6% of all recoverable revenue - 1.43× better than
"chase the biggest bills" and 5.75× better than random.** This is the strongest claim in
the project, because it depends on no assumption whatsoever.

### Where real data runs out - and a negative result

No public dataset records **interventions**. The source tells us what happened, never what
the issuer did about it. So `run_real_experiment.py` layers a *modelled* uplift on real base
probabilities and sweeps the effect size rather than asserting one:

| effect | agent vs status quo | agent vs blanket dunning |
|---|---|---|
| 0.25 | +$46,020 | **+$73,289** |
| 0.50 | +$120,667 | +$12,238 |
| 0.75 | +$203,057 | **−$91,656** |
| 1.00 | +$255,350 | **−$179,317** |

The agent beats the observed status quo at every effect size. **But it loses to
indiscriminate dunning once interventions are strong (effect > ~0.5)** - reported here
rather than tuned away. Two reasons, both honest:

1. **This dataset cannot test the agent's main mechanism.** Its biggest synthetic win is
   refusing to retry a dead instrument (`expired_card`: 0% by construction). Revolving
   credit delinquency has no dead-instrument analogue.
2. **Selectivity only pays when contact is costly.** Contact fatigue and opt-out hazard are
   modelled, and adding them moved break-even from 0.25 to ~0.5 - but if contacts are cheap
   and effective, contacting everyone wins. That is a real finding about when this class of
   agent is worth building.

**What survives contact with real data: the prioritisation model. What does not: the
intervention policy** - not because it is wrong, but because no public data can validate
it. That needs a pilot with a processor's own operational logs.

---

## Uplift targeting

The recovery model answers **"who will pay?"**. For spending a contact budget that is the
wrong question, and measurably so: on real delinquency data, **44.3% of the top 250 cases
ranked by expected recovery paid on their own.** Roughly half the budget goes to people who
needed nothing.

`backend/app/ml/uplift.py` models the treatment effect instead:

```
uplift(x) = P(recover | contacted, x) − P(recover | not contacted, x)
```

A **T-learner**: two outcome models, one per arm, differenced. Each sees only its own arm,
so neither can learn the treatment indicator as a shortcut. It sorts the population into
**persuadables**, **sure things**, **lost causes** and **sleeping dogs** (pay *unless*
contacted, where a contact actively loses money). An outcome model cannot separate any of
these; it ranks sure things at the very top.

Trained inside the simulator, which is a genuine randomised experiment: **each training case
is assigned to exactly one arm and only that arm's outcome is observed**, as in a live
holdout. The counterfactual is used only at evaluation.

- true ATE on the held-out split **+28.7%**, predicted mean uplift **+27.5%**
- correlation between predicted and realised per-case uplift **+0.22** (binary outcomes)

### The ranking bug this caught

The first implementation ranked by **percentage-point uplift** and lost badly - $17,149
against expected value's $74,265 at a 250 budget. It optimises the wrong unit and quietly
selects the smallest balances (mean **$206** vs **$917**). The objective is revenue, so the
score has to carry the money:

```
incremental_value = amount × uplift(x)
```

### Result - money caused, under a fixed contact budget

| contact budget | amount × uplift | expected value | random |
|---|---|---|---|
| 100 | **$60,915** | $58,639 | $5,985 |
| 250 | **$76,760** | $74,265 | $7,776 |
| 500 | **$87,251** | $83,484 | $20,505 |

Qini coefficient **0.404** vs **0.387** for expected value, **−0.152** for random.

**The gain is real but modest here (+3.4% at a 250 budget), and the reason matters:** this
simulator's control arm recovers only 16%, so it contains almost no sure things - the test
split has just **3**. On real data, where 44.3% of the top-EV queue self-recovers, the same
mechanism has roughly a hundred times more to work with. **The synthetic environment
systematically understates the value of uplift modelling.**

`backend/app/ml/targeting.py` ranks by `incremental_value` when an uplift model exists and
falls back to `expected_recovery` when it does not, reporting which mode is active.
`Targeter.select(df, budget)` drops non-positive-uplift cases rather than padding the budget
- contacting someone the model expects to be unmoved, or harmed, is worse than leaving the
budget unspent.

---

## Baseline vs RecoverAI

Three arms over one population:

- **control** - never touched. Its recoveries are pure self-cure. **This is the only reason
  the other two numbers mean anything.**
- **baseline** - retry every eligible failed payment after 24h, stop after 3. No diagnosis,
  so it re-presents dead cards, hammers closed accounts, and retries fraud holds.
- **RecoverAI** - the full loop.

All three run on the **same cases** against the **same seeded simulator**. `self_cure_hour`
is drawn once per transaction from a dedicated RNG stream, so *the same case either
self-cures or does not, whatever arm is working it* - a true counterfactual over one
population (`test_self_cure_is_identical_across_arms`). Every active arm checks for
self-cure **before** each action; without that, an arm that waits would be credited with
money that arrived while it waited.

| | control | baseline | RecoverAI |
|---|---|---|---|
| Money recovered (gross) | $45,955 | $107,344 | **$144,167** |
| **Money caused** (vs control) | - | $61,389 | **$98,213** |
| Share of gross that is causal | - | 57.2% | **68.1%** |
| Cases recovered | 298 | 661 | **816** |
| Recovery rate | 16.2% | 35.9% | **44.3%** |
| Retries | 0 | 4,563 | **2,610** |
| Customer contacts | 0 | 0 | 1,519 |
| Cases escalated | 0 | 0 | 137 |
| Policy-blocked actions | 0 | 0 | 80 |
| **Unsafe risk actions** | 0 | **392** | **0** |
| Cost | $0.00 | $91.26 | $442.47 |

**Primary business metric - money caused vs the no-touch control: $98,213, ROI 222×, 90%
paired-bootstrap interval $74,145 to $123,000.**

Against the baseline specifically, the agent adds **$36,824** of gross recovery (+34.3%,
90% CI $10,393–$62,276) on 43% fewer retries.

### Where the gain comes from

| failure category | baseline | RecoverAI | why |
|---|---|---|---|
| TEMPORARY | 49.4% | **58.4%** | retries immediately; transient faults decay from the moment they happen, and the baseline always waits 24h |
| CUSTOMER_ACTION | 32.5% | **47.9%** | repairs dead instruments before retrying, and times insufficient-funds retries to the payday window (~day 5.5) instead of burning attempts at 24/48/72h |
| PERSISTENT | 2.9% | **5.9%** | never retries a closed account - sends a payment link that routes around the instrument entirely |
| RISK_COMPLIANCE | 0.0% | 0.0% | escalates immediately; **takes no automated action at all** |

The mechanism is visible in the simulator: for `expired_card`, bare retry succeeds **0.0%**
of the time by construction, while request-update → retry recovers **22.2%** end to end.
Diagnosis is what pays, not effort.

---

## Demo script

**~4 minutes.**

1. **Health** - open `http://127.0.0.1:8000/`. Header reads `model logistic-recovery-v1 ·
   planner deterministic rules · db postgres · audit 28,432 rows · chain verified`. The
   `db` field names the live engine and `chain verified` is recomputed on every call.
2. **Overview** - $346,631 at risk → $144,167 recovered gross → but the control arm shows
   $45,955 would have arrived anyway, so **$98,213 is what we caused**. Point at *Unsafe
   Actions Prevented: 392*: the baseline retried fraud and compliance holds 392 times;
   RecoverAI did it zero times. Click *Cases Escalated* to land in the queue filtered to
   those 137.
3. **The chart** - baseline and RecoverAI are close on TEMPORARY (blind retry works there)
   and nearly 2× apart on CUSTOMER_ACTION (diagnosis works there). Nothing on
   RISK_COMPLIANCE for either.
4. **Recovery Queue** - sorted by expected recovery. Top rows recommend **escalate case**,
   because they exceed the $5,000 automatic ceiling.
5. **The money shot** - open a dead-card case. This is the whole thesis in one case:
   ```
   invalid payment method  ·  $4,035.51 at risk  ·  P(recovery) 68.4%
     → root cause: the instrument is dead, so it must be replaced before any retry
     → request_payment_method_update   [approve]   → method updated
     → retry_payment                   [approve]   → succeeded   +$4,035.51
   ```
   A blind retry strategy scores **0%** on this case by construction.
6. **Audit Log** - every decision, its policy verdict, and the rule IDs that fired. Set
   verdict to *Reject* to see what the gate stopped.
7. **Dead Letter Queue** - 13 quarantined pairs the agent refuses to keep messaging.
8. **Live** - hit *Re-run this case* to watch the agent work it again in real time.

---

## Project layout

```
backend/app/models/      enums (the failure taxonomy), Pydantic schemas, API schemas
backend/app/ml/          feature engineering · serving scorer · uplift · targeting
backend/app/policies/    17 deterministic rules, one function each, stable IDs
backend/app/agents/      diagnosis · strategy · bounded LLM planner · redaction · LangGraph
backend/app/tools/       the action executor - the only source of side effects
backend/app/audit/       append-only hash-chained log
backend/app/database/    persistence - SQLite or Postgres · migrations · operational state
backend/app/services/    control arm, baseline strategy, shared metrics, data loading
backend/tests/           291 tests (unit, adversarial safety, regression, API, integration,
                         Postgres round-trip)
ml/                      train.py · train_real.py · train_uplift.py · evaluate.py · model/
simulation/              payment_gateway.py · notification_service.py
scripts/                 generate_dataset.py · run_experiment.py · demo.py ·
                         migrate_to_postgres.py · build_real_dataset.py
frontend/                React + Vite dashboard (6 pages, hand-rolled charts, no chart lib)
.github/workflows/       CI: offline suite · Postgres suite · pipeline smoke · frontend build
```

---
