# Production deployment

**Read this first: RecoverAI has never moved real money.** It has not been tested against
a real payment provider, it has not had a security audit, it has not been load-tested, and
no regulator has looked at it. This document describes what a production deployment would
require, not a deployment that has happened.

---

## Profiles

`backend/app/profiles.py`. Five environments, one place that says what each demands.
`validate_or_raise()` runs at startup and **raises** rather than warning, because a process that
boots misconfigured looks healthy while being unsafe, which is strictly worse than a crash
loop somebody has to look at.

| Profile | Auth | Rails | Database | Webhooks | LLM |
|---|---|---|---|---|---|
| `development` | optional | mock | SQLite | not required | allowed |
| `simulation` | optional | mock only | SQLite | not required | allowed |
| `testing` | optional | mock | SQLite | not required | **forbidden** |
| `staging` | **required** | sandbox | **Postgres** | **required** | allowed |
| `production` | **required** | **mock refused** | **Postgres** | **required** | allowed |

A typo'd profile name raises. Guessing `development` for `prodcution` is exactly the
failure this check exists to prevent.

---

## Configuration

```bash
# --- profile and security -------------------------------------------------
RECOVERAI_PROFILE=production
RECOVERAI_JWT_SECRET=<32+ chars, from a secret manager>
RECOVERAI_KEY_SALT=<distinct from the JWT secret, so rotating one does not break the other>
RECOVERAI_WEBHOOK_SECRET=<per provider>
RECOVERAI_API_KEYS=keyid:secret:tenant:role1|role2,...   # small deployments only

# --- persistence ----------------------------------------------------------
RECOVERAI_DB_URL=postgresql://user:pass@host:5432/recoverai

# --- rails ----------------------------------------------------------------
RECOVERAI_PAYMENT_RAIL=stripe_sandbox      # 'mock' is refused in production

# --- the safety envelope --------------------------------------------------
RECOVERAI_MAX_RETRIES=3
RECOVERAI_MIN_RETRY_HOURS=24
RECOVERAI_MAX_AUTO_AMOUNT=5000
RECOVERAI_MAX_CONTACTS=4
RECOVERAI_MAX_STEPS=12
RECOVERAI_HORIZON_DAYS=30
RECOVERAI_MIN_EV=1.0
RECOVERAI_REVIEW_MIN_CONFIDENCE=0.55
RECOVERAI_REVIEW_MAX_FAILURES=3
RECOVERAI_REVIEW_SLA_HOURS=72

# --- optional -------------------------------------------------------------
ANTHROPIC_API_KEY=<unset to run entirely on the deterministic planner>
RECOVERAI_LOG_LEVEL=INFO
```

No secret has a default. Nothing falls back silently.

---

## Local development

The whole system runs offline with no account anywhere:

```bash
docker compose up --build          # dashboard at http://localhost:8000
```

Mock rails, SQLite, no API key, no network. If a change makes the offline path need a
service, the project has lost a property worth keeping, and the CI `tests-sqlite` job is
the gate on that.

Postgres is opt-in:

```bash
docker compose --profile postgres up --build
```

---

## Integrating a real provider

The seam is `backend/app/adapters/gateway.PaymentRail`, which is three methods. `MockRail` is the
reference implementation and `StripeSandboxRail` / `RazorpaySandboxRail` show the shape
against a real SDK.

Two contracts an adapter must honour:

1. **Every call carries an idempotency key**, and it is passed to the provider's own
   idempotency header. `GatewayRequest` has no default for it. The executor's ledger
   protects against a repeated *call*; the provider's header protects against a repeated
   *charge* when the first call's response was lost in transit.
2. **A transport failure raises `GatewayUnavailable`, never a false decline.** The state of
   the payment is *unknown*. Collapsing outage and decline is how a five-minute processor
   blip burns a customer's entire retry budget, or worse, moves on while a charge landed.

The sandbox adapters **refuse a live credential at construction**. That is not
configuration; it is a `raise`.

```python
from backend.app.adapters.gateway import build_rail
rail = build_rail("stripe_sandbox", api_key=os.environ["STRIPE_TEST_KEY"])
```

`build_rail` raises on an unknown name rather than falling back to the mock. A deployment
that believes it is talking to Stripe and is running a simulator would report simulated
recoveries as real ones, the worst failure this abstraction can have.

---

## Observability

- `GET /api/v1/metrics/prometheus`: text exposition. Unauthenticated by design, and safe:
  counters, latencies and rates, no customer identifiers and no per-case amounts. A scrape
  endpoint behind auth is a scrape endpoint nobody configures.
- `GET /api/v1/metrics/snapshot`: the same series as JSON, for the dashboard.
- Structured JSON logs, one object per line, with redaction in the formatter and a
  `trace_id` on every line.

Series worth alerting on:

| Metric | Alert when |
|---|---|
| `recoverai_human_review_pending` | rising and not falling, so the queue is not being worked |
| `recoverai_policy_decisions_total{decision="reject"}` | a step change, so a rule is firing far more than usual |
| `recoverai_executor_failures_total` | any sustained rate |
| `recoverai_webhook_rejected_total` | a spike, so a forgery attempt or a rotated secret nobody updated |
| `recoverai_auth_failures_total` | a spike |
| `recoverai_dead_letters_total` | non-zero |
| `recoverai_model_drift_psi` | above 0.25 |
| `recoverai_llm_failures_total` | rising, so the fallback is working and something is wrong |

---

## Operational runbook

**Verify the decision record.** `python scripts/verify_audit.py`. Exit code 0 only when
the chain verifies end to end, so it works as a cron check and a CI gate.

**Work the review queue.** `GET /api/v1/reviews` (ordered by value), then approve or reject
with a reason. Tasks expire after the SLA; an unbounded queue is not a safety mechanism.

**Release a quarantined channel.** `POST /api/dlq/release` after confirming the address.

**Replay failed events.** `POST /api/v1/dlq/events/retry`. Letters that fail
`max_attempts` times are quarantined rather than retried forever, because a deterministic failure
will fail deterministically again, and the loop only burns budget while hiding the problem.

**Roll a model back.** Promote the previous version; the registry retires the incumbent
automatically and records who did it.

**Rotate a secret.** `RECOVERAI_KEY_SALT` and `RECOVERAI_JWT_SECRET` are independent, so
rotating one does not invalidate the other. There is **no key expiry or rotation workflow**
See T3 in the threat model.

---

## What must be true before real money moves

Not a wish list. Each of these is a gap that currently exists.

1. **Sandbox validation.** A full recovery cycle against a real provider's test mode,
   including an ambiguous call whose response is lost, to prove the idempotency path.
2. **Encryption at rest.** There is none. This is the largest open gap (T8).
3. **Key lifecycle.** Expiry, rotation, and anomaly detection on key use (T3).
4. **Reconciliation.** A pass that resolves payments left in an unknown state after a
   gateway outage (T7).
5. **External anchoring of the audit chain.** Tampering is currently detectable, not
   preventable; anyone with write access can truncate the table (T9).
6. **A live holdout.** Without it, no causal claim about real customers is available at
   any confidence (see `docs/causal_inference.md`).
7. **A streaming batch runner.** `scripts/bench_throughput.py` shows throughput is flat
   (~70 cases/s to 50,000 cases, 10.6 us per policy validation) but memory is not: the
   runner accumulates every `AgentState` with its full audit-event list, peaking around
   700 MB at 10,000 cases. Correct for the evaluation harness, wrong for a worker. Also,
   these are mock rails, so a real gateway call would dominate the 14 ms per case, and no
   distributed load test has been run.
8. **A security review** by someone other than the authors.
9. **Regulatory review.** Automated customer contact and automated re-presentment are
   regulated differently in every market this could run in. Consent, quiet hours, retry
   limits and dispute handling are jurisdictional, and the policy engine is where those
   rules would live, but nobody qualified has reviewed which rules belong there.
10. **A shared rate limiter.** The current one is per-process, so *N* replicas allow *N*
    times the configured rate (T13).

---

## Scaling

The system is a modular monolith and should stay one until there is a reason it cannot be.

Present shape: a single FastAPI process, SQLite or Postgres, an in-process event bus, and
a batch runner that works cases sequentially.

The seams that exist for when that stops being enough:

- `EventBus` is a protocol. A Redis or Kafka adapter is a class implementing
  `subscribe`/`publish`, not a rewrite. `bus_from_env` raises on an unknown URL rather
  than silently downgrading.
- `PaymentRail` is a protocol; adapters are independently deployable in principle.
- Stores take a connection, so a read replica is a constructor argument.
- The agent holds no cross-case state: `RecoveryAgent.run` is pure with respect to the
  case, so horizontal scaling is a queue away.

What would **not** help: introducing Kubernetes, Kafka or microservices before a measured
bottleneck. That is complexity bought for appearance.
