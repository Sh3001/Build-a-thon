# RecoverAI - Autonomous AI Revenue Recovery

An agent that works a queue of failed payments end to end: **score, diagnose, choose an
intervention, execute it under hard limits, monitor, and stop when it should** - measured
as dollars recovered against a baseline dunning strategy on a held-out test set, not cases
flagged.

```
1,842 held-out cases · $346,631 at risk · 30-day horizon · one seed, three arms
(the seven-arm, 20-seed table is below and is the one to read)

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

## What is demonstrated, what is simulated, what is not validated

The distinction is load-bearing, so it is stated before anything else. It is also
enforced in code: every experiment artefact carries a `provenance` field and a `claim`
string, and `Provenance.combine()` refuses to pool kinds
(`backend/app/domain/provenance.py`).

| | Status | What it means |
|---|---|---|
| Policy engine cannot be bypassed | **demonstrated** | The executor's input is the gate's output; passing an unapproved verdict raises. One test per rule; a meta-test fails if a rule has no test. |
| Idempotent execution | **demonstrated** | Three layers: in-process cache, durable ledger, provider idempotency header. Replay returns the original result. |
| Audit chain verifies across a schema change | **demonstrated** | Per-row field-list versioning; an unknown version refuses rather than passing. |
| Tenant isolation | **demonstrated** | Tenant comes from the credential, never the request; `tenant_id` is in the primary key. Nine isolation tests. |
| Bounded termination | **demonstrated** | Three independent bounds; tested against a planner that will not settle. |
| Recovery lift over control and baselines | **simulated** | Holds inside this simulator, across seeds and nine scenarios. Not evidence about real customers. |
| Prediction quality on real delinquency data | **observational** | Real data, but nothing was assigned. Supports predictive claims only. |
| Per-action effect sizes | **configured prior** | No randomised multi-armed data exists here. Every such estimate is labelled `CONFIGURED_PRIOR` and `is_measured` is False. |
| Real-world causal effect | **not validated** | No live holdout, no real rail, no randomised production experiment. |
| Security | **not audited** | No external review, no penetration test. See `docs/threat_model.md`. |
| Throughput | **measured, single-process** | Flat at ~70 cases/s to 50,000 cases; 10.6 us per policy validation. Mock rails, one machine, not a distributed figure. |
| Memory at scale | **measured, and a known problem** | The batch runner holds every case's state: ~700 MB at 10,000 cases. Fine for evaluation, wrong for a production worker. |

---

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
| **Check the claims** | [Multi-seed results](#multi-seed-results) · [The seven arms](#the-seven-arm-ladder) · [Performance](#performance) · [Model evaluation](#model-evaluation) · [Real data](#results-on-real-data) · [Uplift](#uplift-targeting) |
| **The caveats** | [Known limitations](#known-limitations) · [Evidence status](#what-is-demonstrated-what-is-simulated-what-is-not-validated) |
| **Go deeper** | [`docs/`](docs/): [architecture](docs/architecture.md) · [policy engine](docs/policy_engine.md) · [ML](docs/ml.md) · [causal inference](docs/causal_inference.md) · [experiments](docs/experiments.md) · [security](docs/security.md) · [threat model](docs/threat_model.md) · [production](docs/production.md) |

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
| `./run.sh multiseed [N]` | Seven arms across N seeds with confidence intervals (default 20) |
| `./run.sh sweep [N]` | The same, across all nine simulator scenarios (default 8) |
| `./run.sh verify [CASE]` | Verify the audit chain; with a transaction id, print its timeline |
| `./run.sh bench` | Throughput and per-case latency at increasing batch sizes |
| `./run.sh check` | ruff + mypy on the safety-critical modules + the test suite |
| `./run.sh shell` | Python REPL with the project importable |

Step by step, if you want to watch each stage:

```bash
.venv/bin/python scripts/generate_dataset.py       # 12,000 records + customer-disjoint splits
.venv/bin/python ml/train.py                       # XGBoost vs logistic, selected on validation
.venv/bin/python ml/evaluate.py                    # held-out metrics
.venv/bin/python scripts/run_experiment.py --fresh # control vs baseline vs RecoverAI
.venv/bin/python -m uvicorn backend.app.main:app --port 8000
```

Evaluation beyond a single seed - this is where the honest numbers come from:

```bash
# seven arms, many seeds, intervals instead of a point estimate
.venv/bin/python scripts/run_multiseed.py --seeds 20

# does the finding survive a less flattering world?
.venv/bin/python scripts/run_multiseed.py --seeds 10 --scenario pessimistic
.venv/bin/python scripts/run_multiseed.py --seeds 10 --sweep      # all nine scenarios

# with expected-incremental-profit arbitration
.venv/bin/python scripts/run_multiseed.py --seeds 20 --optimizer

# verify the decision record; exit code 0 only if the chain verifies end to end
.venv/bin/python scripts/verify_audit.py
.venv/bin/python scripts/verify_audit.py --case txn_0001234
```

Quality gates:

```bash
.venv/bin/ruff check backend simulation scripts ml
.venv/bin/mypy backend/app/policies backend/app/domain backend/app/security backend/app/decision
.venv/bin/python -m pytest backend/tests -q
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
                              or an external SQL source (RECOVERAI_SOURCE_URL)
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
║   POLICY VALIDATOR    ║     backend/app/policies/engine.py   22 deterministic rules
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

Eight pages, all reachable with number keys **1** to **8**. Rows in the Recovery Queue and
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

**Human Review** - the queue of cases the policy engine *withheld* rather than dropped,
ordered by value, because an operator's minute is the scarce resource. Each row shows the
rule that fired and what was proposed; approving requires a written reason and records an
override. Approving does not execute: it authorises an action the executor will still run
through the same gate. Below the queue, the append-only override log shows who decided
what, when, and why.

**Audit Log** - every decision and policy verdict, newest first. Filter by stage or policy
verdict (both server-side, so **Reject** searches all rows rather than the visible page)
and search the reason text.

**Dead Letter Queue** - quarantined (customer, channel) pairs with a Release button. See
[Dead letter queue](#dead-letter-queue).

**Ask** - a natural-language query over the run. The router only ever picks an intent;
every figure in the reply is computed from the database, so a model cannot state a number
here and therefore cannot state a wrong one.

**Keyboard**

| Key | Action |
|---|---|
| `1` to `8` | Switch page |
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

### v1 surface added in this pass

Every route below takes its tenant from the caller's credential. None accepts a tenant
parameter, which is the whole of tenant isolation.

| Route | Capability | Notes |
|---|---|---|
| `GET /api/v1/reviews` | `read:review` | The human-review queue, ordered by value |
| `GET /api/v1/reviews/{id}` | `read:review` | 404 for another tenant's task, not 403 |
| `POST /api/v1/reviews/{id}/approve` | `write:review` | Reason mandatory; records an override; **does not execute** |
| `POST /api/v1/reviews/{id}/reject` | `write:review` | Reason mandatory |
| `GET /api/v1/reviews/audit/overrides` | `read:overrides` | Append-only; readable by AUDITOR, who can approve nothing |
| `POST /api/v1/consent/opt-out` | `write:review` | Enforced by `R-OPT-OUT` on the next evaluation |
| `GET /api/v1/policy` | `read:metrics` | Every rule by tier, every limit, the derived version hash |
| `GET /api/v1/models` | `read:models` | Registry contents and lifecycle status |
| `POST /api/v1/models/{v}/promote` | `write:models` | ADMIN only; gated on a quality floor |
| `GET /api/v1/metrics/prometheus` | none | Text exposition; no customer data, so no auth |
| `GET /api/v1/metrics/snapshot` | `read:metrics` | The same series as JSON |
| `POST /api/v1/webhooks/{provider}` | signature | Signature → freshness → replay → parse, in that order |
| `GET /api/v1/dlq/events` | `read:metrics` | Failed event handlers (distinct from the delivery DLQ) |
| `POST /api/v1/dlq/events/retry` | `write:dlq` | Quarantines after `max_attempts` |

Authentication is by `X-API-Key: <key_id>.<secret>` or `Authorization: Bearer <jwt>`.
Presenting both is refused. In the `development` profile an anonymous principal holds
`ANALYST` and `AUDITOR`, so the dashboard works with no setup, and cannot
execute anything.

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

## Reading cases from your own database

`RECOVERAI_DB_URL` is where *this system's* data lives. `RECOVERAI_SOURCE_URL` is different:
it is a read-only connection to **your** payments database, so a deployment works real
failed payments instead of the generated dataset. Both live in `SqlSource`
(`backend/app/services/dataio.py`), alongside the CSV loader every experiment uses.

```bash
export RECOVERAI_SOURCE_URL="postgresql://readonly@db.internal:5432/payments"
export RECOVERAI_SOURCE_QUERY="SELECT * FROM failed_payments WHERE failed_at > now() - interval '30 days'"
export RECOVERAI_SOURCE_PROVIDER=stripe        # stripe | razorpay | adyen
export RECOVERAI_SOURCE_COLUMNS='{"cust_ref":"customer_id","txn_ref":"transaction_id",
                                  "gross_amount":"amount","decline_reason":"failure_code"}'

python -m backend.app.services.dataio --limit 20   # dry run: fetch a sample, report on it
```

`RECOVERAI_SOURCE_COLUMNS` maps *your* column names to `Transaction` fields; anything you
do not name is assumed to match already. The preview command prints counts only, never row
content - the point is to check the mapping before pointing a run at the source, not to
dump customer data into a terminal.

### Four properties, none of them optional

**The connection is read-only.** The session is set read-only *and* the statement is
rejected unless it is a single SELECT. Two checks rather than one, because
`SELECT 1; DROP TABLE payments` passes the first and fails the second. A connection whose
driver cannot be made read-only is refused outright rather than used.

**Outcome columns never survive.** `recovered` and `recovery_days` are labels. A live run
that could see them would be scoring the answer it was handed. They are dropped on the way
in and the drop is reported, because a source carrying them is one somebody should look at.

**Only schema fields survive.** A payments table holds emails, names and card fingerprints.
The mapping is an allowlist of `Transaction` fields, so a column nobody mapped cannot ride
along into a prompt, a log, or a model feature. `Transaction` has no PII fields at all, so
this falls out of the schema rather than depending on a redaction pass.

**An unmappable failure code is quarantined, not guessed.** `processor_codes.normalise`
answers UNKNOWN when it does not recognise a code, and UNKNOWN has no `FailureCode`. Such a
row comes back as a reject carrying its reason. Defaulting the unrecognised remainder to
`temporary_decline` would turn a mapping gap into confident retries against instruments
nobody has classified.

### What a real fetch looks like

Against a Postgres table with a deliberately awkward schema - renamed columns, PII, an
outcome label, one unmappable code and one invalid payment method:

```
fetched 5, accepted 3, rejected 2
  dropped outcome columns: was_recovered
  ignored unmapped columns: cardholder_name, customer_email, decline_message
  failure_count: 5 rows fell back to the schema default
  ... 8 more features defaulted ...
  days_since_failure: 1 rows fell back to the schema default
  rejected 1: failure code 'wibble_unknown' is not mapped for provider 'stripe'
  rejected 1: payment_method: Input should be 'card', 'upi', 'upi_autopay', ...
```

Two things in that output matter more than the accept count.

**The defaulted-feature counts.** A real source rarely has `previous_success_rate` or
`customer_tenure`. Falling back to the schema default is reasonable; doing it invisibly
degrades every score with no trace, so the count is part of the report rather than a
footnote. Nine of ten features defaulting means the model is running on `amount` and
`failure_code` alone, and you should know that before reading its output.

**The rejects are the useful half.** A row that does not map is a row a person needs to
classify, not a row to drop quietly.

### Known limits

`fetchmany(limit)` caps a fetch; there is no cursor-based paging, so this reads a working
set rather than streaming a large table. Postgres is the tested engine - the adapter takes
a connection factory, so another driver is an injection rather than a rewrite, but nothing
else has been exercised. And the SQL guard is a keyword check over a statement you control;
it is a guard against mistakes in a mapping file, not a sandbox for hostile input.

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

### HUMAN_REVIEW: the fourth verdict

The gate used to return three verdicts. It now returns four, and only two of them permit
execution:

```
APPROVE · MODIFY   →  executes
HUMAN_REVIEW       →  withheld; a task appears in an operator's queue
REJECT             →  refused; the case may try a different route
```

`R-AMOUNT-CAP` moved from REJECT to HUMAN_REVIEW, and the reason is worth stating: a flat
refusal *dropped the most valuable cases in the queue*, so the system's safety limit was
also its biggest revenue leak. The rule's message always said "human approval required";
now the decision does too. Nothing executes on the verdict either way (`allowed` is
False), so the safety property is unchanged and the business outcome is not.

Four other rules route to a person: a diagnosis below the confidence floor, a processor
code that could not be mapped at all, repeated execution failures on one case, and a
customer who has asked for support.

Approving does **not** execute. It records an override (actor, role, timestamp, and a
mandatory reason) and returns the `PolicyResult` that permits the action. The executor
still requires a permitting verdict, so an operator's approval reaches the rails through
the same single gate as everything else rather than through a side door.

A resolved task cannot be re-decided; otherwise two operators can disagree and the last
writer silently wins. Tasks expire after an SLA, because an unbounded queue is not a
safety mechanism; it is where decisions go to be forgotten.

### Consent is a rule, not a term in the objective

`R-OPT-OUT` refuses every customer-facing action for a customer who has withdrawn consent,
at any value, regardless of expected profit. It is a hard rule precisely so that no
optimisation can outbid it: a *term* in an objective function can be outweighed by a large
enough number, and this one must not be. A blanket opt-out beats any per-channel record.
Someone who said "stop contacting me" has not agreed to be reached on a channel they did
not name. A retry of an existing mandate is not a customer contact and is unaffected.

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

## Multi-seed results

**Simulation result.** 20 independent seeds, all 1,842 held-out cases, `default` scenario.
Produced by `scripts/run_multiseed.py`; the artefact is `data/runs/multiseed.json` and
carries its provenance, git commit and config fingerprint.

```
arm               mean recovered            95% CI    worst seed  retries  contacts  risk    ROI
control                   49,356   [45,508 .. 53,204]     37,999        0         0     0      -
naive_retry              106,006  [101,156 ..110,855]     85,498    4,616         0   396    614x
smart_retry              104,477  [100,300 ..108,654]     90,009    2,506         0     0  1,100x
ml_probability            76,143   [70,956 .. 81,331]     53,466    1,323       446     0    605x
expected_value            95,979   [90,411 ..101,548]     72,191    1,323       479     0  1,022x
uplift                    98,297   [92,840 ..103,754]     74,238    1,323       496     0  1,057x
RecoverAI                128,717  [124,888 ..132,547]    115,720    2,649     1,536     0    179x

INCREMENTAL vs no-touch control          mean            95% CI      seeds won
  naive_retry                          56,650  [51,554 .. 61,745]        20/20
  smart_retry                          55,121  [50,650 .. 59,591]        20/20
  ml_probability                       26,787  [21,500 .. 32,074]        20/20
  expected_value                       46,623  [40,831 .. 52,415]        20/20
  uplift                               48,941  [43,112 .. 54,770]        20/20
  RecoverAI                            79,361  [74,841 .. 83,881]        20/20

RecoverAI paired against each arm, across the same 20 seeds
  vs control              +79,361  [74,841 .. 83,881]
  vs naive_retry          +22,712  [18,969 .. 26,454]
  vs smart_retry          +24,240  [20,991 .. 27,490]
  vs ml_probability       +52,574  [48,592 .. 56,557]
  vs expected_value       +32,738  [28,538 .. 36,939]
  vs uplift               +30,420  [26,295 .. 34,545]
```

### Three things in this table that are not flattering, and are the point

**Diagnosis does most of the work.** `smart_retry` is a cause-aware retry schedule with
**no model at all**, a lookup table a competent engineer writes in an afternoon. It
captures **$55,121** of incremental recovery against RecoverAI's $79,361, or **69% of the
lift**. Everything else (the calibrated model, the uplift ranking, the agent loop, the
optional LLM) accounts for the remaining 31% ($24,240, CI $20,991 to $27,490). That gap is
real and statistically distinguishable from zero, but "the ML is worth 31%" is a far
smaller claim than the gross figure implies, and it is the one the numbers support.

**RecoverAI has the worst ROI of every arm that acts.** 179× against `smart_retry`'s
1,100×. It recovers more and spends 8.9× as much doing it, because it contacts customers
and the retry-only arms do not. Whether that trade is worth taking depends on a merchant's
real contact costs, which is exactly why they live in `config/economics.yaml` rather than
in code, and why `--optimizer` exists to refuse contacts that do not pay for themselves.

**Ranking by `P(recover)` is actively bad.** `ml_probability` is the intuitive strategy,
contact whoever is likeliest to pay, and it is the *worst* acting arm at $26,787
incremental, well under `expected_value`'s $46,623 on an identical contact budget. It
spends the budget on people who were going to pay anyway. This is the sure-thing problem
the uplift work exists to address, reproduced as a measurement rather than asserted as a
motivation.

Two things RecoverAI wins outright: **zero** automated actions on fraud and compliance
holds where `naive_retry` takes 396, and a worst seed ($115,720) that still beats every
other arm's *mean*.

### Does it survive a less flattering world?

8 seeds × 9 scenarios (`scripts/run_multiseed.py --sweep`, `data/runs/multiseed_sweep.json`):

```
scenario          control    smart  RecovAI   RAI incremental vs control    RAI vs smart_retry
default            46,790  104,902  128,023   81,233  [72,836 .. 89,630]   +23,121  [15,698..30,544]
fast_decay         46,790   80,703  100,796   54,006  [48,005 .. 60,007]   +20,093  [12,483..27,704]
high_fatigue       46,790  104,902  123,449   76,659  [69,307 .. 84,010]   +18,547  [12,138..24,955]
high_self_cure    101,418  127,721  134,915   33,496  [25,087 .. 41,905]    +7,194   [-458..14,845] n.s.
low_fatigue        46,790  104,902  132,460   85,670  [77,541 .. 93,799]   +27,558  [19,972..35,144]
low_self_cure      26,955   96,589  124,739   97,785  [89,239 ..106,330]   +28,151  [19,265..37,036]
pessimistic        88,208   98,846  100,643   12,435   [6,304 .. 18,567]    +1,798  [-3,426.. 7,021] n.s.
slow_decay         46,790  130,621  152,388  105,598  [98,590 ..112,606]   +21,767  [15,446..28,088]
unreliable_rails   46,790  100,792  120,136   73,346  [64,972 .. 81,721]   +19,344  [12,561..26,127]
```

**RecoverAI beats the untouched control in all nine worlds**, every interval excluding
zero. But against `smart_retry`, the no-ML arm, the advantage is **not distinguishable
from zero in two of them**: `high_self_cure` and `pessimistic`. Both are worlds where most
of the money arrives on its own, and the honest reading is that when the counterfactual is
high, the marginal contribution of the model over a good lookup table is not something 8
seeds can resolve. That is reported here rather than left out.

---

## Performance

`scripts/bench_throughput.py`. Single process, one machine, mock rails, deterministic
planner (LLM latency would hide everything else).

```
policy engine       10.6 us/call   (94,000 validations/s)

      cases   seconds   cases/s   ms/case   peak RSS
        100      1.50        67    14.979     196 MB
      1,000     14.20        70    14.202     247 MB
     10,000    142.43        70    14.243     717 MB
     50,000    751.05        67    15.021    1,386 MB

per-case cost across sizes: FLAT (1.05x spread)
```

**Throughput is flat**, which is the property that matters: nothing here is quadratic, so
the batch runner is the only thing that would need to change to go faster. At ~70 cases/s
a 100,000-case queue is about 24 minutes on one core, and the work is embarrassingly
parallel: `RecoveryAgent.run` holds no cross-case state.

**Memory is not flat, and that is a real limitation.** `run_agent_batch` accumulates every
`AgentState`, each carrying its full audit-event list, so a 10,000-case batch peaks around
700 MB. That is fine for the evaluation harness, which wants the states for analysis, and
it is wrong for a production worker, which should stream outcomes and discard state. The
fix is a streaming variant of the runner; it is not written, and the figure above is why
it is on the list rather than assumed away.

Two caveats on the whole table: these are **mock rails**, so a real gateway call (tens to
hundreds of milliseconds) would dominate the 14 ms per case entirely; and this is one
process on one machine, not a distributed throughput figure.

---

## The seven-arm ladder

Two arms, a no-touch control and a fixed retry baseline, show that the agent beats
*doing nothing* and beats *the dumbest possible thing*. Neither shows that its
intelligence is what does the work, because almost anything beats a 24-hour retry loop.

Five more arms were added so each ingredient can be priced separately:

| # | Arm | What it adds | Prices |
|---|---|---|---|
| 1 | `control` | nothing | what arrives on its own |
| 2 | `naive_retry` | fixed 24h × 3, everyone | the untuned dunning system |
| 3 | `smart_retry` | cause-aware timing, **no model** | **diagnosis** |
| 4 | `ml_probability` | contact the likeliest to pay | prediction |
| 5 | `expected_value` | rank by `amount × P(recover)` | ranking by money |
| 6 | `uplift` | rank by `amount × uplift(x)` | causal targeting |
| 7 | `recoverai` | the full loop | the whole thing |

**Arm 3 is the one to read.** It has all the domain knowledge and none of the machine
learning, so whatever it recovers is the part of the lift a competent engineer could have
written as a lookup table. Only the gap above *that* line is attributable to the model.
Arms 4 to 6 share a contact budget, so "targeted better" cannot secretly mean "contacted
more". Each arm gets a freshly constructed, identically seeded gateway. Sharing one would
let an earlier arm's contact fatigue leak into a later one and make arm order part of the
result.

Run it: `python scripts/run_multiseed.py --seeds 20`.

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
90% CI $10,393 to $62,276) on 43% fewer retries.

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
backend/app/domain/      money (exact, minor units) · case state machine · events · provenance
backend/app/models/      enums (the failure taxonomy), Pydantic schemas, API schemas
backend/app/adapters/    PaymentRail protocol · mock rail · sandbox rails · processor codes
backend/app/decision/    cost model (config/economics.yaml) · expected-incremental-profit
backend/app/ml/          features · scorer · uplift · targeting · registry · drift
backend/app/policies/    22 deterministic rules (14 reject · 5 review · 3 modify) + a
                         version hash derived from the rules' own source
backend/app/agents/      diagnosis · strategy · bounded LLM planner · redaction · LangGraph
backend/app/tools/       the action executor - the only source of side effects
backend/app/security/    auth · roles · HS256 tokens · webhook verification · rate limiting
backend/app/observability/  Prometheus registry · structured logging with redaction
backend/app/experiments/ deterministic randomisation · intervals · multi-seed aggregation
backend/app/audit/       append-only hash-chained log, versioned payload
backend/app/database/    SQLite or Postgres · migrations · operational state · review queue
backend/app/services/    control · baselines · the seven-arm suite · shared metrics
backend/app/api/         auth dependencies · review router · ops router
backend/app/profiles.py  five deployment profiles; production refuses to boot misconfigured
backend/tests/           582 tests; 572 run fully offline, 10 skip without Postgres or
                         Ollama. Unit · policy (one per rule) · security · tenancy ·
                         chaos/failure-injection · statistical · API · Postgres round-trip
ml/                      train.py · train_real.py · train_uplift.py · evaluate.py · model/
simulation/              config.py (nine scenarios) · payment_gateway.py · notification_service.py
scripts/                 generate_dataset · run_experiment · run_multiseed · verify_audit ·
                         bench_throughput · bench_planner · demo · migrate_to_postgres ·
                         build_real_dataset
config/                  simulation.yaml · economics.yaml · effects.yaml
docs/                    architecture · policy_engine · ml · causal_inference ·
                         experiments · security · threat_model · production
frontend/                React + Vite dashboard (7 pages incl. human review, hand-rolled
                         charts, no chart lib)
.github/workflows/       CI: lint+types · offline suite · Postgres · security · pipeline ·
                         multi-seed smoke · audit verification · frontend · docker
```

---

## Known limitations

**Simulated, and honest about it.** The payment rails and messaging channels are mocks.
Outcomes are sampled from a model with hidden latents, seeded per case - reproducible and
genuinely comparable across strategies, but no real money moved. The channel adapters sit
behind an interface; wiring a sandbox is an adapter swap.

**The dataset is synthetic.** It is drawn from an explicit generative model, so the
*structure* the agent exploits (cause × timing interaction, dead instruments, contact
fatigue) is real structure - but it is structure I specified. On production data the
coefficients would differ and the strategy would need re-tuning. The Bayes-ceiling
comparison is only meaningful *within* this generator.

**Modelled delivery failure penalises the agent arm only.** Hard bounces are simulated at a
6% address-level rate, and the baseline arm sends *zero* customer contacts - it only
retries. So the failure mode costs RecoverAI ~$1,078 of gross recovery and costs the
baseline nothing. That is realistic but **not neutral**: it narrows the measured gap by
making the agent's world harder while leaving the baseline's untouched. Set
`RECOVERAI_DELIVERY_FAILURE_RATE=0` to reproduce the pre-DLQ numbers.

**The comparison is against one baseline.** "Retry every 24h, max 3" is real and common,
but a well-tuned commercial dunning tool would be a harder opponent.

**Escalation forfeits passive recovery, which understates the agent.** Escalating a case
closes it in our accounting, so it can no longer be credited with a later self-cure. On
risk/compliance cases the control arm collects $87 that the agent records as $0. The
artifact is small and runs *against* the agent, so it is left in rather than corrected into
a flattering direction.

**The self-cure rates are assumed, not measured.** `PASSIVE_CURE_RATE` encodes plausible
spontaneous-recovery rates per failure cause (26% for a bank timeout, 1% for a closed
account). The *shape* is defensible; the levels are estimates. Since the causal headline is
`agent − control`, a systematically wrong control rate moves that number directly. On real
data this is the first thing to calibrate.

**Confidence intervals now cover seeds as well as resampling.** The single-run figures
above carry a paired bootstrap interval over cases, which captures sampling variation
within one world. `scripts/run_multiseed.py` adds the other half: the same comparison
across independent seeds and across nine simulator scenarios, reported with a
t-distribution interval and, deliberately, the **worst seed** next to the mean. At low
seed counts several arm comparisons are *not distinguishable from zero*, and the harness
says so in the row rather than quoting the point estimate.

**Currency handling is simplified.** A static FX table converts to USD for aggregation; real
FX, rounding, and settlement timing are out of scope.

**The winning model is linear.** Training selects on validation, so the pipeline is honest
about this, but the "hybrid ML + rules + LLM" architecture is currently carrying a logistic
regression on the synthetic track. On real data the tree model wins decisively.

**The value floor is conservative.** `MIN_EXPECTED_RECOVERY_USD` is $1.00 against a
payment-link cost of $0.04 - a 25× margin. It correctly declines tiny cases, but on
unreachable accounts it leaves roughly two thirds of them unworked. Tuning the floor to the
*marginal* value of the specific action is the obvious next win.

**Authentication and multi-tenancy now exist, and have not been audited.** API keys and
HS256 JWTs, five roles enforced by capability, tenant taken from the credential and never
from the request, `tenant_id` in the primary key, nine isolation tests. The anonymous
principal granted in the development profile holds read roles only and can never execute
an action. What is still missing: encryption at rest, key rotation and expiry, dual
control on high-value approvals, and any review by someone other than the authors. See
`docs/threat_model.md` for the ten threats and which eight remain open.

**Temporal validation is not possible as built.** The synthetic dataset has no event
timestamp, only `days_since_failure`, which is a duration, not a point in time. Splits
are grouped by customer, which prevents customer leakage but not temporal leakage, and
cannot detect a model that has learned a regime that no longer holds. Closing this needs
an `occurred_at` column and a time-ordered split. Stated rather than glossed over
(`docs/ml.md`).

**Per-action effect sizes are priors, not measurements.** `config/effects.yaml` says how
much of an intervention's effect each action captures for each cause. No randomised
multi-armed experiment exists in this project, so these are informed guesses in a file
where they can be argued with. Every estimate derived from them is labelled
`CONFIGURED_PRIOR`, and `EffectEstimate.is_measured` is False for all of them. The zeros
are the exception: those are structural claims about the payment rails (you cannot debit
a closed account), not statistical ones.

**The event bus, gateway and rate limiter are in-process.** The bus is synchronous and
ordered; the rate limiter is per-process, so *N* replicas allow *N* times the configured
rate. Both are protocols with one implementation, and `bus_from_env` raises on an unknown
URL rather than silently downgrading, but neither is a distributed guarantee, and this
document does not claim one.

**Next**, in order of what would change a conclusion:

1. **A sandbox integration against one real rail.** The `PaymentRail` protocol is the seam
   and `StripeSandboxRail` shows the shape; what is missing is a full cycle against a real
   provider's test mode, including an ambiguous call whose response is lost, to exercise
   the idempotency path for real.
2. **A live holdout.** Without a randomly assigned fraction that receives nothing, no
   causal claim about real customers is available at any confidence.
3. **Multi-armed randomisation** across retry / email / SMS / payment link / support. That
   is the only thing that would replace `config/effects.yaml`'s priors with measurements.
4. **Event timestamps in the schema**, so validation can be temporal rather than grouped.
5. **Encryption at rest and key lifecycle**: the two largest open security gaps.
6. **A live Claude A/B** (`--planner llm`) to test whether the LLM beats the rules planner
   on the cases where the error code lies. The benchmark harness exists
   (`scripts/bench_planner.py`); the comparison has not been run at scale.
