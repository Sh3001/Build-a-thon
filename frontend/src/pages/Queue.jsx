import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, fmtMoney, fmtPct, titleize } from "../services/api";
import { Chip } from "../components/Chip";
import { loadPrefs, savePrefs, useSearchKeys, useTableNav } from "../hooks/useTableNav";

const COLS = [
  { key: "customer_id", label: "Customer" },
  { key: "transaction_id", label: "Transaction" },
  { key: "amount_usd", label: "Amount", num: true, sortable: true },
  { key: "failure_code", label: "Failure cause" },
  { key: "recovery_probability", label: "Recovery prob.", num: true, sortable: true },
  { key: "expected_recovery", label: "Expected recovery", num: true, sortable: true },
  { key: "recommended_action", label: "Recommended action" },
  { key: "status", label: "Status" },
];

const STATUSES = ["recovered", "escalated", "stopped", "exhausted"];
const PREFS = "recoverai.queue";
const DEFAULTS = { sort: "expected_recovery", dir: "desc", status: "", code: "" };

export function Queue({ onSelect, selected, metrics, preset, onRows }) {
  const saved = useMemo(() => loadPrefs(PREFS, DEFAULTS), []);
  const [rows, setRows] = useState([]);
  const [sort, setSort] = useState(saved.sort);
  const [dir, setDir] = useState(saved.dir);
  const [status, setStatus] = useState(saved.status);
  const [code, setCode] = useState(saved.code);
  const [q, setQ] = useState("");
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const tableRef = useRef(null);
  const searchRef = useRef(null);

  useEffect(() => {
    let live = true;
    setBusy(true);
    api.queue({ limit: 200, sort, direction: dir,
                ...(status && { status }), ...(code && { failure_code: code }) })
      .then((r) => { if (live) { setRows(r); setErr(null); } })
      .catch((e) => live && setErr(String(e)))
      .finally(() => live && setBusy(false));
    return () => { live = false; };
  }, [sort, dir, status, code]);

  useEffect(() => { savePrefs(PREFS, { sort, dir, status, code }); }, [sort, dir, status, code]);

  // A drill-down from the Overview replaces the current filters outright, so what you
  // land on always matches the tile you clicked.
  useEffect(() => {
    if (!preset) return;
    setStatus(preset.status ?? "");
    setCode(preset.code ?? "");
    setQ(preset.q ?? "");
  }, [preset]);

  // Sourced from metrics, not from `rows` -- deriving it from the filtered rows collapsed
  // the menu to the one cause already chosen, so you could never switch between causes.
  const codes = useMemo(
    () => Object.keys(metrics?.recoverai?.by_failure_code || {}).sort(), [metrics]);

  // Underscores are normalised on both sides, so "retry payment" finds `retry_payment`
  // whether it was typed by hand or handed over by an Overview drill-down.
  const shown = useMemo(() => {
    const flat = (x) => String(x).toLowerCase().replace(/[_\s]+/g, " ");
    const term = flat(q.trim());
    if (!term) return rows;
    return rows.filter((r) =>
      flat([r.transaction_id, r.customer_id, r.failure_code, r.recommended_action, r.status]
        .filter(Boolean).join(" ")).includes(term));
  }, [rows, q]);

  // Hand the visible ordering up so Agent Trace can offer prev/next.
  useEffect(() => { onRows?.(shown.map((r) => r.transaction_id)); }, [shown, onRows]);

  const clearSearch = useCallback(() => setQ(""), []);
  useTableNav(tableRef, onSelect);
  useSearchKeys(searchRef, clearSearch);

  function sortBy(key) {
    if (sort === key) setDir((d) => (d === "desc" ? "asc" : "desc"));
    else { setSort(key); setDir("desc"); }
  }
  function reset() {
    setStatus(""); setCode(""); setQ("");
    setSort(DEFAULTS.sort); setDir(DEFAULTS.dir);
    searchRef.current?.focus();
  }

  const active = (status ? 1 : 0) + (code ? 1 : 0) + (q.trim() ? 1 : 0);

  if (err) return <div className="err">{err}</div>;

  return (
    <div className="card">
      <h2>Recovery queue</h2>
      <div className="sub">
        Ranked by expected recovery = amount &times; P(recovery). Click a row to open its
        agent trace.
      </div>

      <div className="controls">
        <div className="field">
          <label htmlFor="q-status">Status</label>
          <select id="q-status" className={status ? "on" : ""} value={status}
                  onChange={(e) => setStatus(e.target.value)}>
            <option value="">All</option>
            {STATUSES.map((s) => <option key={s} value={s}>{titleize(s)}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="q-code">Failure cause</label>
          <select id="q-code" className={code ? "on" : ""} value={code}
                  onChange={(e) => setCode(e.target.value)}>
            <option value="">All</option>
            {codes.map((c) => <option key={c} value={c}>{titleize(c)}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="q-search">Search</label>
          <input id="q-search" ref={searchRef} type="search" autoComplete="off"
                 className={q.trim() ? "on" : ""} placeholder="transaction, customer&hellip;"
                 value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <span className="spacer" />
        <span className="count">
          {busy ? "loading…"
                : q.trim() ? `${shown.length} of ${rows.length} match`
                : `${rows.length} cases`}
        </span>
        {active > 0 && (
          <button className="act" onClick={reset}>
            Reset {active} filter{active === 1 ? "" : "s"}
          </button>
        )}
      </div>

      <div className={`scroll${busy ? " busy" : ""}`} ref={tableRef}>
        <table>
          <thead>
            <tr>
              {COLS.map((c) => (
                <th key={c.key}
                    className={`${c.num ? "num " : ""}${c.sortable ? "sortable" : ""}`}
                    data-dir={c.sortable && sort === c.key ? dir : undefined}
                    onClick={() => c.sortable && sortBy(c.key)}
                    title={c.sortable ? `Sort by ${c.label}` : undefined}
                    aria-sort={sort === c.key ? (dir === "asc" ? "ascending" : "descending")
                                              : "none"}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.transaction_id} data-id={r.transaction_id} tabIndex={0}
                  className={`clickable${selected === r.transaction_id ? " sel" : ""}`}
                  onClick={() => onSelect(r.transaction_id)}>
                <td className="mono">{r.customer_id}</td>
                <td className="mono">{r.transaction_id}</td>
                <td className="num">{fmtMoney(r.amount_usd, 2)}</td>
                <td>
                  <button className="linky" title={`Show only ${titleize(r.failure_code)}`}
                          onClick={(e) => { e.stopPropagation(); setCode(r.failure_code); }}>
                    {titleize(r.failure_code)}
                  </button>
                </td>
                <td className="num">{fmtPct(r.recovery_probability)}</td>
                <td className="num">{fmtMoney(r.expected_recovery, 2)}</td>
                <td>{titleize(r.recommended_action) || "-"}</td>
                <td>
                  <span role="button" tabIndex={-1} title={`Show only ${r.status}`}
                        onClick={(e) => { e.stopPropagation(); setStatus(r.status); }}>
                    <Chip kind={r.status} />
                  </span>
                </td>
              </tr>
            ))}
            {!shown.length && !busy && (
              <tr><td colSpan={COLS.length} className="empty">
                No cases match these filters.{active > 0 && <> Try <b>Reset</b>.</>}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="hint">
        Click a row for its agent trace &middot; <kbd>&uarr;</kbd><kbd>&darr;</kbd> move
        &middot; <kbd>Enter</kbd> open &middot; <kbd>/</kbd> search
        &middot; click a cause or status to filter by it
      </div>
    </div>
  );
}
