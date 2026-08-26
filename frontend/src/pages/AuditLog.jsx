import React, { useCallback, useMemo, useEffect, useRef, useState } from "react";
import { api, fmtMoney, titleize } from "../services/api";
import { Chip } from "../components/Chip";
import { useSearchKeys, useTableNav } from "../hooks/useTableNav";

const STAGES = [
  "load_transaction", "score_recovery", "diagnose_root_cause",
  "calculate_expected_recovery", "select_intervention", "validate_policy",
  "execute_action", "monitor_outcome",
];

export function AuditLog({ onSelect }) {
  const [data, setData] = useState(null);
  const [decision, setDecision] = useState("");
  const [verdict, setVerdict] = useState("");
  const [q, setQ] = useState("");
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const tableRef = useRef(null);
  const searchRef = useRef(null);

  useEffect(() => {
    let live = true;
    setBusy(true);
    api.auditLog(300, decision || undefined, verdict || undefined)
      .then((d) => { if (live) { setData(d); setErr(null); } })
      .catch((e) => live && setErr(String(e)))
      .finally(() => live && setBusy(false));
    return () => { live = false; };
  }, [decision, verdict]);

  const clearSearch = useCallback(() => setQ(""), []);
  useTableNav(tableRef, onSelect);
  useSearchKeys(searchRef, clearSearch);

  const rows = data?.rows || [];
  const shown = useMemo(() => {
    const flat = (x) => String(x).toLowerCase().replace(/[_\s]+/g, " ");
    const term = flat(q.trim());
    if (!term) return rows;
    return rows.filter((r) =>
      flat([r.transaction_id, r.reason, r.action, r.rules_fired, r.agent_decision]
        .filter(Boolean).join(" ")).includes(term));
  }, [rows, q]);

  const active = (decision ? 1 : 0) + (verdict ? 1 : 0) + (q.trim() ? 1 : 0);
  function reset() { setDecision(""); setVerdict(""); setQ(""); searchRef.current?.focus(); }

  if (err) return <div className="err">{err}</div>;
  if (!data) return <div className="card"><div className="empty">Loading…</div></div>;

  return (
    <div className="card">
      <h2>Audit log</h2>
      <div className="sub">
        Every decision and policy verdict, append-only and hash-chained.{" "}
        {data.total.toLocaleString()} rows · chain{" "}
        <b style={{ color: data.chain_valid ? "var(--good)" : "var(--critical)" }}>
          {data.chain_valid ? "verified" : "BROKEN"}
        </b>
      </div>

      <div className="controls">
        <div className="field">
          <label htmlFor="a-stage">Stage</label>
          <select id="a-stage" className={decision ? "on" : ""} value={decision}
                  onChange={(e) => setDecision(e.target.value)}>
            <option value="">All</option>
            {STAGES.map((d) => <option key={d} value={d}>{titleize(d)}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="a-verdict">Policy verdict</label>
          <select id="a-verdict" className={verdict ? "on" : ""} value={verdict}
                  onChange={(e) => setVerdict(e.target.value)}>
            <option value="">All</option>
            {["approve", "modify", "reject"].map((v) =>
              <option key={v} value={v}>{titleize(v)}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="a-search">Search</label>
          <input id="a-search" ref={searchRef} type="search" autoComplete="off"
                 className={q.trim() ? "on" : ""} placeholder="reason, rule, transaction&hellip;"
                 value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <span className="spacer" />
        <span className="count">
          {busy ? "loading…" : `${shown.length} of ${rows.length} shown`}
        </span>
        {active > 0 && (
          <button className="act" onClick={reset}>
            Reset {active} filter{active === 1 ? "" : "s"}
          </button>
        )}
      </div>

      <div className={`scroll${busy ? " busy" : ""}`} ref={tableRef}>
        <table>
          <thead><tr>
            <th className="num">Seq</th><th>Transaction</th><th>Stage</th>
            <th>Reason</th><th>Action</th><th>Policy</th><th>Rules</th>
            <th className="num">Recovered</th>
          </tr></thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.seq} data-id={r.transaction_id} tabIndex={0} className="clickable"
                  onClick={() => onSelect(r.transaction_id)}>
                <td className="num mono">{r.seq}</td>
                <td className="mono">{r.transaction_id}</td>
                <td>{titleize(r.agent_decision)}</td>
                <td style={{ maxWidth: 380 }}>{r.reason}</td>
                <td>{titleize(r.action) || "-"}</td>
                <td>
                  {r.policy_result
                    ? <span role="button" tabIndex={-1} title={`Show only ${r.policy_result}`}
                            onClick={(e) => { e.stopPropagation(); setVerdict(r.policy_result); }}>
                        <Chip kind={r.policy_result} />
                      </span>
                    : "-"}
                </td>
                <td>{(r.rules_fired || "").split(",").filter(Boolean).map((x) =>
                  <Chip key={x} kind="rule">{x}</Chip>)}</td>
                <td className="num">
                  {Number(r.amount_recovered) > 0
                    ? <span className="money">{fmtMoney(Number(r.amount_recovered), 2)}</span> : "-"}
                </td>
              </tr>
            ))}
            {!shown.length && !busy && (
              <tr><td colSpan={8} className="empty">
                No audit rows match these filters.{active > 0 && <> Try <b>Reset</b>.</>}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="hint">
        Click a row to open that case &middot; <kbd>&uarr;</kbd><kbd>&darr;</kbd> move
        &middot; <kbd>Enter</kbd> open &middot; <kbd>/</kbd> search
      </div>
    </div>
  );
}
