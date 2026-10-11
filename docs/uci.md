# UCI data source

From the repository root, run `./run.sh uci` (optional port: `./run.sh uci 8001`).
The launcher validates the cached spreadsheet, downloads it if absent, and starts the
API/dashboard with `RECOVERAI_SOURCE=uci`. It does not create a database or train a model.
Build the dashboard with `cd frontend && npm ci && npm run build` when needed.

For separate fetch and server steps:

```sh
.venv/bin/python -m scripts.fetch_uci
RECOVERAI_SOURCE=uci .venv/bin/python -m uvicorn backend.app.main:app --port 8000
```

`RECOVERAI_UCI_PATH` selects an existing spreadsheet. `scripts.fetch_uci --refresh`
explicitly downloads a replacement, validates it, and replaces the file atomically.
The API never downloads on requests. Missing or malformed data returns 503; it does not
fall back to generated records. Restart after changing source environment settings.

For Docker, mount the spreadsheet read-only at `/app/data/external/uci_credit.xls`
and set `RECOVERAI_SOURCE=uci` on the service. No database service is needed.
For authenticated deployments use the `readonly` profile and the authentication settings
in [external database setup](external_database.md); UCI does not require SQL URL/query settings.
`RECOVERAI_SOURCE_TENANT` scopes access and defaults to `default`.

## What the dashboard shows

The source is [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
Yeh, I. (2009), UCI Machine Learning Repository, DOI 10.24432/C55S3H,
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It covers Taiwanese accounts in April–September 2005.

The dashboard selects September accounts with `PAY_0 >= 1` and `BILL_AMT1 > 0`:
5,129 of 30,000 accounts in the published file. Other accounts are excluded, not rejected.
Each account appears once, with a stable ID, recorded months late, and original TWD bill.
USD equivalents use an illustrative fixed 0.031 rate, matching the offline UCI workflow.
Bill balances are not measured overdue amounts; totals are historical, not current exposure.

No gateway decline reason is inferred. Default-next-month labels and demographic fields
are not exposed. Recovery scores, interventions, and audit events are unavailable;
execution is blocked. This is a historical browsing source, not evidence that a recovery
strategy works. The existing multi-month offline training scripts remain separate.
