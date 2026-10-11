# Existing database setup and durable recovery

The normal server and Docker launch do not generate datasets, create a local SQLite file,
start a PostgreSQL container, or migrate an external database. With no configuration the
UI displays a connection setup screen. Existing repository data is left untouched.

## Browse existing failed payments

PostgreSQL is currently the supported external source. Once you have an existing database,
configure the server using `.env.example`. Docker Compose reads `.env`; a direct Python
launch needs the variables exported in its environment.

- `RECOVERAI_SOURCE_URL`: PostgreSQL URL for a SELECT-only role.
- `RECOVERAI_SOURCE_QUERY`: one SELECT over an existing failed-payment table or view.
- `RECOVERAI_SOURCE_COLUMNS`: JSON mapping from source column names to transaction fields.
- `RECOVERAI_SOURCE_PROVIDER`: `razorpay`, `stripe`, or another supported processor mapping.
- `RECOVERAI_SOURCE_TENANT`: tenant owning this source view, default `default`.
- `RECOVERAI_SOURCE_LIMIT`: bounded snapshot size, default 5000, maximum 10000.

For example, a query may select `payment_id, customer_id, amount, currency, payment_method,
failure_code, failed_at` from your existing failed-payment view. Set
`RECOVERAI_SOURCE_COLUMNS={"payment_id":"transaction_id"}` if those are its actual names.
The minimum transaction fields are `transaction_id`, `customer_id`, `amount`,
`payment_method`, and `failure_code`. Amounts must be in **major currency units**, not
provider minor units. Convert them explicitly in your view/query when necessary.

Connections enforce read-only sessions, a five-second connection timeout, a ten-second
statement timeout, and a SQL-level row limit. Unmapped fields are discarded. Unknown
failure codes and invalid records are rejected, with counts reported through
`/api/source/report`. The dashboard snapshot is not an aggregate of every row in the
source database. It does not invent recovery outcomes or causal comparisons.

A source belongs to one configured tenant; a credential for another tenant cannot query it.
For deployment, use `RECOVERAI_PROFILE=readonly`, a strong `RECOVERAI_JWT_SECRET` (at least
32 characters), and tenant-bound API keys. This profile requires authentication without
requiring recovery storage. Development permits anonymous reads and is for local use.

## Optional writable recovery state

Reading source payments needs no local or recovery database. Jobs, approvals, action
idempotency, checkpoints, audit history, and consent require durable writable storage.
Configure `RECOVERAI_DB_URL` for an existing, dedicated PostgreSQL database or schema/role.
Keep the source account read-only. A source table does not need to match RecoverAI's
internal schema.

An administrator provisions the **application tables**, explicitly:

```sh
python -m scripts.migrate_database
```

This command never issues `CREATE DATABASE`. It applies versioned migrations to the
configured existing recovery database. Ordinary requests validate schema versions and
refuse outdated/missing schemas; they do not apply PostgreSQL migrations.

For an explicitly requested local simulation/test only, set `RECOVERAI_ALLOW_SQLITE=1`
or `RECOVERAI_DB=/your/explicit/path.sqlite`. Simulation generation remains available via
the existing scripts, but never runs during ordinary startup.

## Job execution

`POST /api/recovery/run` and `/api/recovery/run/{id}` now return **202** with a job ID.
Send `Idempotency-Key` to deduplicate a submission; reusing it with different parameters
returns 409. Poll `GET /api/jobs/{id}`. Cancellation stops remaining work between stages;
an already dispatched action may still finish. The legacy `persist` request field is
accepted for compatibility, but durable execution always persists operational evidence.

Start the worker separately:

```sh
python -m scripts.worker
```

One worker owns the state database using a PostgreSQL session advisory lock (a file lock
for explicitly enabled SQLite). Restart releases/reacquires ownership and resumes saved
workflow stages. Completed cases are not executed again. A durable action claim is
committed before dispatch; uncertain outcomes remain blocked for reconciliation, rather
than being blindly retried. There is no automatic expiry/reclaim of uncertain actions.

**Execution still uses the repository's simulator.** Both job submission and worker
startup refuse execution when an external payment source is configured or the profile is
production. Integrating a real provider, its idempotency keys, and reconciliation is a
separate requirement before enabling real payment execution. No real payments were
performed while implementing these changes.

Approval of a review with saved state atomically creates a job and an outbox event. The
worker rechecks approval expiry, policy version, consent, case status, and policy rules.
Only the reviewed condition is waived; other review conditions and hard rejections remain
binding. Legacy reviews without a saved snapshot remain recorded approvals for manual
handling. A follow-up action is audited separately without reopening the terminal parent
case. Its financial outcome is available in the job result and audit, not automatically
merged into historical experiment aggregates.

## Operations and recovery

- `/api/health/live`: process liveness without scanning audit history.
- `/api/health/ready`: source connectivity and recovery-schema readiness; 503 in setup mode.
- `/api/v1/operations`: tenant-specific job counts, queue-delay alerts (five minutes),
  overdue review alerts, and uncertain/stale action alerts (ten minutes).
- `/api/v1/events`: recent tenant-scoped durable outbox events for inspection. This is
  not a broker delivery/acknowledgement protocol; external delivery adapters remain separate.
- Requests carry `X-Request-ID`; jobs preserve the request ID and actor for correlation.

Database snapshots/backups should include checkpoints, claims, jobs, reviews, audit and
outbox together. Test restores in an isolated existing database. After a crash around a
provider call, reconcile the provider's result before changing any uncertain action state.
The current API does not expose a button that clears such claims.

Rate limits remain per API process and are now synchronized across threads. Configure
shared ingress limits before deploying multiple API replicas. Actual request bytes are
bounded even when Content-Length is absent or incorrect.

CI includes isolated backend regression tests and frontend checks. Tests never require
access to a production payments database. PostgreSQL integration tests require an
explicit scratch `RECOVERAI_TEST_DB_URL`; their fixture provisions/drops test tables.

## Model monitoring

Optionally set `RECOVERAI_DRIFT_BASELINE_FILE` to a server-side JSON snapshot produced
by `DriftMonitor.to_dict()` from your training population. The source report evaluates
feature drift and prediction drift (only for a matching trained model version). Missing
labels are explicitly reported as performance not checked. Without a baseline, monitoring
is reported as not configured. Offline evaluation now also reports Brier score and
calibration error by failure category. These diagnostics do not establish causal uplift.
