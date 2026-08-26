import React, { useEffect, useState } from "react";
import { api, fmtNum } from "./services/api";
import { Overview } from "./pages/Overview";
import { Queue } from "./pages/Queue";
import { Trace } from "./pages/Trace";
import { Analytics } from "./pages/Analytics";
import { AuditLog } from "./pages/AuditLog";
import { DLQ } from "./pages/DLQ";
import { Assistant } from "./pages/Assistant";

const TABS = [
  ["overview", "Overview"],
  ["queue", "Recovery Queue"],
  ["trace", "Agent Trace"],
  ["analytics", "Revenue Analytics"],
  ["audit", "Audit Log"],
  ["dlq", "Dead Letter Queue"],
  ["ask", "Ask"],
];

const VALID = new Set(TABS.map(([k]) => k));

const THEMES = [["system", "Auto"], ["light", "Light"], ["dark", "Dark"]];

/** The stylesheet defines all three states, so the toggle works in both directions. */
function applyTheme(t) {
  const el = document.documentElement;
  if (t === "system") el.removeAttribute("data-theme");
  else el.setAttribute("data-theme", t);
  try { localStorage.setItem("recoverai.theme", t); } catch { /* not persisted */ }
}

/** `#trace/txn_0006884` -> {tab: "trace", id: "txn_0006884"} */
function parseHash() {
  const [tab, id] = window.location.hash.slice(1).split("/");
  return { tab: VALID.has(tab) ? tab : "overview", id: id || null };
}

export default function App() {
  // Hash routing: tabs and individual cases are deep-linkable and survive a reload.
  const [tab, setTabState] = useState(() => parseHash().tab);
  const [health, setHealth] = useState(null);
  const [overview, setOverview] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [selected, setSelectedState] = useState(() => parseHash().id);
  const [err, setErr] = useState(null);
  // Set when a tile or bar on another page drills into the queue; `at` makes an identical
  // preset a new object, so clicking the same tile twice still re-applies it.
  const [queuePreset, setQueuePreset] = useState(null);
  // The queue's current ordering, so Agent Trace can step to the next case.
  const [caseIds, setCaseIds] = useState([]);
  const [theme, setThemeState] = useState(() => {
    try { return localStorage.getItem("recoverai.theme") || "system"; } catch { return "system"; }
  });

  const setTheme = (t) => { applyTheme(t); setThemeState(t); };
  useEffect(() => { applyTheme(theme); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const setTab = (t) => {
    window.location.hash = t === "trace" && selected ? `trace/${selected}` : t;
    setTabState(t);
  };
  const setSelected = (id) => {
    window.location.hash = `trace/${id}`;
    setSelectedState(id);
  };

  useEffect(() => {
    const onHash = () => {
      const { tab: t, id } = parseHash();
      setTabState(t);
      if (id) setSelectedState(id);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // 1-5 jump between tabs, but never while you are typing in a filter.
  useEffect(() => {
    const onKey = (e) => {
      const t = document.activeElement;
      if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const i = Number(e.key) - 1;
      if (Number.isInteger(i) && TABS[i]) { e.preventDefault(); setTab(TABS[i][0]); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setErr(String(e)));
    api.overview().then(setOverview).catch((e) => setErr(String(e)));
    api.metrics().then(setMetrics).catch((e) => setErr(String(e)));
  }, []);

  function openQueue(preset) {
    setQueuePreset({ ...preset, at: Date.now() });
    window.location.hash = "queue";
    setTabState("queue");
  }

  function openCase(id) {
    window.location.hash = `trace/${id}`;
    setSelectedState(id);
    setTabState("trace");
  }

  return (
    <div className="shell">
      <header className="top">
        <h1>RecoverAI</h1>
        <span className="sub">Autonomous revenue recovery</span>
        <span className="spacer" />
        {health && (
          <span className="sub">
            model <b>{health.model_version}</b> ·{" "}
            planner <b>{health.llm_enabled ? "claude + rules" : "deterministic rules"}</b> ·{" "}
            db <b>{health.db_engine}</b> ·{" "}
            audit <b>{fmtNum(health.audit_rows)}</b> rows ·{" "}
            chain{" "}
            <b style={{ color: health.audit_chain_valid ? "var(--good)" : "var(--critical)" }}>
              {health.audit_chain_valid ? "verified" : "BROKEN"}
            </b>
          </span>
        )}
        <div className="themes" role="group" aria-label="Colour theme">
          {THEMES.map(([k, label]) => (
            <button key={k} onClick={() => setTheme(k)} aria-pressed={theme === k}
                    title={`${label} theme`}>{label}</button>
          ))}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map(([k, label], i) => (
          <button key={k} onClick={() => setTab(k)} aria-current={tab === k}
                  title={`${label} - press ${i + 1}`}>
            {label}
          </button>
        ))}
      </nav>

      {err && <div className="err">{err} - is the backend running on :8000?</div>}

      {tab === "overview" && <Overview overview={overview} metrics={metrics} onDrill={openQueue} />}
      {tab === "queue" && <Queue onSelect={openCase} selected={selected} metrics={metrics}
                                 preset={queuePreset} onRows={setCaseIds} />}
      {tab === "trace" && <Trace transactionId={selected} onSelect={setSelected}
                                 caseIds={caseIds} onBack={() => setTab("queue")} />}
      {tab === "analytics" && <Analytics metrics={metrics} />}
      {tab === "audit" && <AuditLog onSelect={openCase} />}
      {tab === "dlq" && <DLQ />}
      {tab === "ask" && <Assistant onOpenCase={openCase} />}
    </div>
  );
}
