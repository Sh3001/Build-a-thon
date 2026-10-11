import React, { useEffect, useState } from "react";
import { api, fmtMoney, fmtPct, titleize } from "../services/api";
import { Chip } from "../components/Chip";

const STAGE_LABEL = {
  load_transaction: "Payment failed",
  score_recovery: "Recovery probability",
  diagnose_root_cause: "Root cause identified",
  calculate_expected_recovery: "Expected recovery",
  select_intervention: "Intervention selected",
  validate_policy: "Policy validation",
  execute_action: "Action executed",
  monitor_outcome: "Outcome monitored",
};

export function Trace({ transactionId, onSelect, caseIds = [], onBack, readOnly = false, onUpdated }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [onlyDecisions, setOnlyDecisions] = useState(false);

  useEffect(() => {
    if (!transactionId) return;
    const controller = new AbortController();
    setData(null); setBusy(true); setErr(null); setJob(null);
    api.caseDetail(transactionId, { signal: controller.signal }).then(setData)
      .catch(e => { if (e.name !== "AbortError") setErr(e.message); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [transactionId]);

  // Step through the queue in its current order without going back to the table.
  const idx = caseIds.indexOf(transactionId);
  const prev = idx > 0 ? caseIds[idx - 1] : null;
  const next = idx >= 0 && idx < caseIds.length - 1 ? caseIds[idx + 1] : null;

  useEffect(() => {
    const onKey = (e) => {
      const t = document.activeElement;
      if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (e.key === "j" && next) onSelect(next);
      if (e.key === "k" && prev) onSelect(prev);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [prev, next, onSelect]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    let active = true;
    const timer = setTimeout(async () => {
      try {
        const next = await api.job(job.job_id);
        if (!active) return;
        setJob(next);
        if (next.status === "succeeded") {
          setData(await api.caseDetail(transactionId));
          onUpdated?.();
        }
      } catch (e) { if (active) setErr(e.message); }
    }, 1500);
    return () => { active = false; clearTimeout(timer); };
  }, [job, transactionId]);

  async function replay() {
    setBusy(true); setErr(null);
    try { setJob(await api.runOne(transactionId)); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  async function copyId() {
    try {
      await navigator.clipboard.writeText(transactionId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch { /* clipboard blocked -- the id is selectable on the page anyway */ }
  }

  if (!transactionId) return (
    <div className="card"><div className="empty">
      Pick a case in the Recovery Queue to read its full decision chain.
    </div></div>
  );
  if (err) return <div className="err">{err}</div>;
  if (!data) return <div className="card"><div className="empty">Loading…</div></div>;

  const c = data.case;
  const b = data.baseline;
  const events = onlyDecisions
    ? data.audit_events.filter((e) => e.policy_result || e.action)
    : data.audit_events;

  return (
    <div className="card">
      {job && <p role="status">Recovery job: {job.status.replaceAll("_", " ")}. {job.error}
        {["queued", "running"].includes(job.status) && <button onClick={async () => {
          try { setJob(await api.cancelJob(job.job_id)); } catch (e) { setErr(e.message); }
        }}>Cancel remaining work</button>}
      </p>}
      <div className="trace-head">
        <button className="linky mono h2ish" onClick={copyId}
                title="Copy transaction id">
          {c.transaction_id}
        </button>
        {copied && <span className="copied">copied</span>}
        <Chip kind={c.status} />
        <span className="sub">
          {titleize(c.failure_code)}
          {c.root_cause && c.root_cause !== c.failure_code &&
            <> → diagnosed as <b>{titleize(c.root_cause)}</b></>}
        </span>
        <span className="spacer" style={{ flex: 1 }} />
        {onBack && <button className="act" onClick={onBack}>&larr; Queue</button>}
        <button className="act" onClick={() => prev && onSelect(prev)} disabled={!prev}
                title="Previous case in the queue (k)">&uarr; Prev</button>
        <button className="act" onClick={() => next && onSelect(next)} disabled={!next}
                title="Next case in the queue (j)">&darr; Next</button>
        <button className="act primary" onClick={replay} disabled={busy || readOnly || !["pending", "in_progress"].includes(c.status) || ["queued", "running"].includes(job?.status)}>
          {busy ? "Submitting…" : readOnly ? "Read-only source" : "Start recovery"}
        </button>
      </div>
      {idx >= 0 && (
        <div className="sub" style={{ marginTop: 6 }}>
          Case {idx + 1} of {caseIds.length} in the current queue view.
        </div>
      )}

      {c.data_source === "uci" && <p>Historical snapshot: {c.snapshot_date}. {c.root_cause}
        {" "}The bill balance is not a measured overdue amount. USD uses a fixed illustrative rate of 0.031.
        No recovery actions or outcomes were recorded by this application.</p>}
      <div className="tiles" style={{ marginTop: 14 }}>
        <div className="tile"><div className="k">{c.data_source === "uci" ? "Historical bill balance" : "Amount at risk"}</div>
          <div className="v">{fmtMoney(c.amount_usd, 2)}</div>
          <div className="s">{c.currency} {c.amount?.toLocaleString()}</div></div>
        <div className="tile"><div className="k">Recovery probability</div>
          <div className="v">{fmtPct(c.recovery_probability)}</div>
          <div className="s">expected {fmtMoney(c.expected_recovery, 2)}</div></div>
        <div className="tile"><div className="k">Recovered</div>
          <div className={`v${c.amount_recovered > 0 ? " good" : ""}`}>
            {readOnly ? "Not available" : fmtMoney(c.amount_recovered, 2)}</div>
          <div className="s">{c.retries} retries · {c.contacts} contacts</div></div>
        <div className="tile"><div className="k">Audit chain</div>
          <div className={`v${data.chain_valid ? " good" : " critical"}`}
               style={{ fontSize: 18 }}>{readOnly ? "Not available" : data.chain_valid ? "verified" : "BROKEN"}</div>
          <div className="s">{data.audit_events.length} events</div></div>
      </div>

      {/* The same transaction under the retry-every-24h baseline. Returned by the API all
          along; showing it is what makes a single case argue for itself. */}
      {b && (
        <>
          <h2 style={{ marginTop: 20, marginBottom: 3 }}>Same case, baseline strategy</h2>
          <div className="sub">What retry-every-24h did with this exact transaction.</div>
          <table className="vs">
            <thead><tr>
              <th></th><th className="num">Baseline</th><th className="num">RecoverAI</th>
            </tr></thead>
            <tbody>
              <tr><td>Outcome</td>
                <td className="num"><Chip kind={b.status} /></td>
                <td className="num"><Chip kind={c.status} /></td></tr>
              <tr><td>Recovered</td>
                <td className="num">{fmtMoney(b.amount_recovered, 2)}</td>
                <td className={`num${c.amount_recovered > b.amount_recovered ? " money" : ""}`}>
                  {fmtMoney(c.amount_recovered, 2)}</td></tr>
              {/* Fewer retries only counts as a win if the money did not go backwards --
                  otherwise a case we lost would show a green "0 retries" as a victory. */}
              <tr><td>Retries</td>
                <td className="num">{b.retries}</td>
                <td className={`num${c.retries < b.retries
                                     && c.amount_recovered >= b.amount_recovered
                                     ? " money" : ""}`}>{c.retries}</td></tr>
              <tr><td>Contacts</td>
                <td className="num">{b.contacts}</td>
                <td className="num">{c.contacts}</td></tr>
              <tr><td>Stop reason</td>
                <td className="num sub">{b.stop_reason || "-"}</td>
                <td className="num sub">{c.stop_reason || "-"}</td></tr>
            </tbody>
          </table>
          {b.amount_recovered > c.amount_recovered && (
            <div className="sub" style={{ marginTop: 8 }}>
              On this case the baseline collected more. Individual cases vary either way;
              the aggregate comparison is on the Overview.
            </div>
          )}
        </>
      )}

      <div className="trace-head" style={{ marginTop: 20, marginBottom: 10 }}>
        <h2>Agent trace</h2>
        <span className="spacer" style={{ flex: 1 }} />
        <label className="toggle">
          <input type="checkbox" checked={onlyDecisions}
                 onChange={(e) => setOnlyDecisions(e.target.checked)} />
          Actions &amp; policy only
        </label>
        <span className="count">{events.length} of {data.audit_events.length} steps</span>
      </div>
      <div className="tl">
        {events.map((e) => (
          <div className="ev" key={e.seq}>
            <div className="stage">
              {STAGE_LABEL[e.agent_decision] || e.agent_decision}
            </div>
            <div>
              <div className="body">{e.reason}</div>
              <div className="meta">
                {e.action && <Chip>{titleize(e.action)}</Chip>}
                {e.policy_result && <Chip kind={e.policy_result} />}
                {e.rules_fired && e.rules_fired.split(",").filter(Boolean).map((r) =>
                  <Chip key={r} kind="rule">{r}</Chip>)}
                {e.action_result && <span>{titleize(e.action_result)}</span>}
                {Number(e.amount_recovered) > 0 &&
                  <span className="money">+{fmtMoney(Number(e.amount_recovered), 2)}</span>}
              </div>
            </div>
          </div>
        ))}
        {!events.length && <div className="empty">No steps match that filter.</div>}
      </div>
      <div className="sub" style={{ marginTop: 12 }}>
        Stop reason: {c.stop_reason || "-"}
      </div>
      <div className="hint">
        <kbd>k</kbd> previous case &middot; <kbd>j</kbd> next case &middot; click the
        transaction id to copy it
      </div>
    </div>
  );
}
