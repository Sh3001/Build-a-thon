import { test, expect } from "@playwright/test";

const transaction = {
  transaction_id: "txn_1", customer_id: "customer_1", amount: 100, currency: "USD",
  amount_usd: 100, failure_code: "bank_timeout", status: "pending", actions: [],
  amount_recovered: 0, retries: 0, contacts: 0, stop_reason: "",
};
const health = { db_engine: "unconfigured", model_version: "fallback", audit_chain_valid: null };

async function mock(page, responder) {
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname;
    const data = responder(path, route.request());
    await route.fulfill({ status: data?.statusCode || 200, contentType: "application/json",
      body: JSON.stringify(data?.body ?? data ?? {}) });
  });
}

test("unconfigured app shows setup without querying or generating data", async ({ page }) => {
  const requests = [];
  await mock(page, path => {
    requests.push(path);
    if (path === "/api/health") return health;
    if (path === "/api/source/status") return {
      source_configured: false, storage_configured: false, message: "No database configured.",
    };
    throw new Error(`Unexpected data request: ${path}`);
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Connect your payment data" })).toBeVisible();
  await page.getByRole("button", { name: "Recovery Queue", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Connect your payment data" })).toBeVisible();
  expect(requests.every(p => ["/api/health", "/api/source/status"].includes(p))).toBeTruthy();
});

test("external source browses payments with execution disabled", async ({ page }) => {
  await mock(page, path => {
    if (path === "/api/health") return health;
    if (path === "/api/source/status") return { source_configured: true, storage_configured: false };
    if (path === "/api/source/report") return { fetched: 1, accepted: 1, rejected: 0 };
    if (path === "/api/recovery-queue") return [transaction];
    if (path === "/api/cases/txn_1") return { case: transaction, audit_events: [], chain_valid: true };
    throw new Error(`Unexpected request: ${path}`);
  });
  await page.goto("/");
  await page.getByRole("button", { name: "View failed payments" }).click();
  await expect(page.getByText("customer_1", { exact: true })).toBeVisible();
  await page.getByText("txn_1", { exact: true }).click();
  await expect(page.getByRole("button", { name: "Read-only source" })).toBeDisabled();
});

test("queued recovery exposes status and refreshes the completed case", async ({ page }) => {
  let completed = false;
  await mock(page, (path, request) => {
    if (path === "/api/health") return health;
    if (path === "/api/source/status") return { source_configured: false, storage_configured: true };
    if (path === "/api/overview" || path === "/api/metrics") return {};
    if (path === "/api/cases/txn_1") return {
      case: { ...transaction, status: completed ? "recovered" : "pending" }, audit_events: [], chain_valid: true,
    };
    if (path === "/api/recovery/run/txn_1") {
      expect(request.headers()["idempotency-key"]).toBeTruthy();
      return { statusCode: 202, body: { job_id: "job_1", status: "queued" } };
    }
    if (path === "/api/jobs/job_1") {
      completed = true;
      return { job_id: "job_1", status: "succeeded" };
    }
    throw new Error(`Unexpected request: ${path}`);
  });
  await page.goto("/#trace/txn_1");
  await page.getByRole("button", { name: "Start recovery" }).click();
  await expect(page.getByRole("status")).toContainText("queued");
  await expect(page.getByRole("status")).toContainText("succeeded");
  await expect(page.getByRole("button", { name: "Start recovery" })).toBeDisabled();
});

test("UCI accounts are labelled historical and cannot execute recovery", async ({ page }) => {
  const account = { ...transaction, data_source: "uci", status: "historical",
    snapshot_date: "2005-09", months_late: 2, failure_code: "historical_delinquency",
    currency: "TWD", root_cause: "Recorded repayment delay: 2 month(s). Gateway cause unknown." };
  await mock(page, path => {
    if (path === "/api/health") return health;
    if (path === "/api/source/status") return { source_configured: true, source_kind: "uci", storage_configured: false };
    if (path === "/api/source/report") return { fetched: 30000, accepted: 5129, excluded: 24871, rejected: 0 };
    if (path === "/api/recovery-queue") return [account];
    if (path === "/api/cases/txn_1") return { case: account, audit_events: [], chain_valid: true };
    throw new Error(`Unexpected request: ${path}`);
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "UCI credit-card repayment dataset" })).toBeVisible();
  await page.getByRole("button", { name: "View delinquent accounts" }).click();
  await expect(page.getByRole("heading", { name: "Historical delinquent accounts" })).toBeVisible();
  await expect(page.getByText("2 month(s)", { exact: true })).toBeVisible();
  await page.getByText("txn_1", { exact: true }).click();
  await expect(page.getByText(/Historical snapshot: 2005-09/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Read-only source" })).toBeDisabled();
});
