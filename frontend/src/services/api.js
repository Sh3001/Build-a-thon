const BASE = "";

async function get(path) {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(`${path} -> ${r.status} ${await r.text()}`);
  return r.json();
}
async function post(path, body) {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  health: () => get("/api/health"),
  overview: () => get("/api/overview"),
  metrics: () => get("/api/metrics"),
  baseline: () => get("/api/baseline"),
  revenueAtRisk: (top = 10) => get(`/api/revenue-at-risk?top=${top}`),
  queue: (p = {}) => {
    const q = new URLSearchParams({ limit: 100, sort: "expected_recovery", ...p });
    return get(`/api/recovery-queue?${q}`);
  },
  caseDetail: (id) => get(`/api/cases/${id}`),
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
  chat: (question, router = "keywords") => post("/api/chat", { question, router }),
  runOne: (id) => post(`/api/recovery/run/${id}`),
  runBatch: (limit) => post("/api/recovery/run", { limit, persist: true }),
};

export const fmtMoney = (v, dp = 0) =>
  v == null ? "-" : `$${Number(v).toLocaleString("en-US", {
    minimumFractionDigits: dp, maximumFractionDigits: dp })}`;
export const fmtPct = (v, dp = 1) => (v == null ? "-" : `${(v * 100).toFixed(dp)}%`);
export const fmtNum = (v) => (v == null ? "-" : Number(v).toLocaleString("en-US"));
export const titleize = (s) => (s || "").replace(/_/g, " ");
