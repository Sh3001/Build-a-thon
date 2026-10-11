const BASE = "";

/**
 * The API key, when one is configured. Read from localStorage so a local run needs no
 * setup: in the development profile the backend grants an anonymous read-only principal,
 * so every page here works with no key at all.
 *
 * Note what this cannot do. Sending a key from the browser grants exactly the
 * capabilities the backend has attached to it; hiding a button in this file grants
 * nothing and prevents nothing. Every authorisation decision is made server-side, and a
 * 403 rendered below is the backend refusing, not the frontend deciding.
 */
function authHeaders() {
  try {
    const key = window.localStorage.getItem("recoverai_api_key");
    return key ? { "X-API-Key": key } : {};
  } catch {
    // Private browsing, or storage disabled. Anonymous is a valid state.
    return {};
  }
}

export async function request(path, { method = "GET", body, signal, headers = {}, timeout = 15000 } = {}) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal?.aborted) controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
  try {
    const response = await fetch(BASE + path, {
      method, signal: controller.signal,
      headers: { ...authHeaders(), ...(body ? { "Content-Type": "application/json" } : {}), ...headers },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail = typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status})`;
      const error = new Error(detail);
      error.status = response.status;
      error.requestId = response.headers.get("X-Request-ID");
      throw error;
    }
    return await response.json();
  } catch (error) {
    if (timedOut) throw new Error("The request timed out. Refresh to check its status.");
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}
const get = (path, options) => request(path, options);
const post = (path, body, options) => request(path, { ...options, method: "POST", body });

export const api = {
  health: () => get("/api/health"),
  sourceStatus: () => get("/api/source/status"),
  sourceReport: (options) => get("/api/source/report", options),
  job: (id) => get(`/api/jobs/${encodeURIComponent(id)}`),
  cancelJob: (id) => post(`/api/jobs/${encodeURIComponent(id)}/cancel`),
  overview: () => get("/api/overview"),
  metrics: () => get("/api/metrics"),
  baseline: () => get("/api/baseline"),
  revenueAtRisk: (top = 10) => get(`/api/revenue-at-risk?top=${top}`),
  queue: (p = {}, options) => {
    const q = new URLSearchParams({ limit: 100, sort: "expected_recovery", ...p });
    return get(`/api/recovery-queue?${q}`, options);
  },
  caseDetail: (id, options) => get(`/api/cases/${encodeURIComponent(id)}`, options),
  caseAudit: (id) => get(`/api/cases/${id}/audit`),
  auditLog: (limit = 200, decision, policyResult) => {
    const q = new URLSearchParams({ limit });
    if (decision) q.set("decision", decision);
    if (policyResult) q.set("policy_result", policyResult);
    return get(`/api/audit?${q}`);
  },
  dlq: (quarantinedOnly = true) =>
    get(`/api/dlq?quarantined_only=${quarantinedOnly}`),
  dlqRelease: (customerId, channel) =>
    post(`/api/dlq/release?customer_id=${encodeURIComponent(customerId)}`
         + `&channel=${encodeURIComponent(channel)}`),
  reviews: (limit = 100) => get(`/api/v1/reviews?limit=${limit}`),
  reviewStats: () => get("/api/v1/reviews/stats"),
  reviewApprove: (id, reason) => post(`/api/v1/reviews/${id}/approve`, { reason }),
  reviewReject: (id, reason) => post(`/api/v1/reviews/${id}/reject`, { reason }),
  overrides: (limit = 100) => get(`/api/v1/reviews/audit/overrides?limit=${limit}`),
  policy: () => get("/api/v1/policy"),
  models: () => get("/api/v1/models"),
  chat: (question, router = "keywords") => post("/api/chat", { question, router }),
  runOne: (id, key = crypto.randomUUID()) => post(`/api/recovery/run/${encodeURIComponent(id)}`, null, { headers: { "Idempotency-Key": key } }),
  runBatch: (limit) => post("/api/recovery/run", { limit, persist: true }),
};

export const fmtMoney = (v, dp = 0) =>
  v == null ? "-" : `$${Number(v).toLocaleString("en-US", {
    minimumFractionDigits: dp, maximumFractionDigits: dp })}`;
export const fmtPct = (v, dp = 1) => (v == null ? "-" : `${(v * 100).toFixed(dp)}%`);
export const fmtNum = (v) => (v == null ? "-" : Number(v).toLocaleString("en-US"));
export const titleize = (s) => (s || "").replace(/_/g, " ");
