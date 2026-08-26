import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, fmtNum, titleize } from "../services/api";
import { Chip } from "../components/Chip";
import { useSearchKeys } from "../hooks/useTableNav";

export function DLQ() {
  const [data, setData] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const searchRef = useRef(null);

  const load = useCallback(async (all) => {
    setBusy(true);
    try {
      setData(await api.dlq(!all));
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { load(showAll); }, [load, showAll]);
  useSearchKeys(searchRef, useCallback(() => setQ(""), []));

  async function release(customer_id, channel) {
    setBusy(true);
    try {
      await api.dlqRelease(customer_id, channel);
      await load(showAll);
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  }

  if (err) return <div className="err">{err}</div>;
  if (!data) return <div className="card"><div className="empty">Loading…</div></div>;

  const s = data.stats;
  const term = q.trim().toLowerCase();
  const rows = (data.entries || []).filter((r) =>
    !term || `${r.customer_id} ${r.channel} ${r.last_error || ""}`.toLowerCase().includes(term));

  return (
    <>
      <div className="tiles">
        <div className="tile"><div className="k">Quarantined</div>
          <div className={`v${s.quarantined ? " critical" : ""}`}>{fmtNum(s.quarantined)}</div>
          <div className="s">customer + channel pairs held for review</div></div>
        <div className="tile"><div className="k">Pairs tracked</div>
          <div className="v">{fmtNum(s.tracked_pairs)}</div>
          <div className="s">any pair that has bounced at least once</div></div>
        <div className="tile"><div className="k">Delivery failures</div>
          <div className="v">{fmtNum(s.total_failures)}</div>
          <div className="s">hard bounces across the run</div></div>
        <div className="tile"><div className="k">Threshold</div>
          <div className="v">{fmtNum(s.threshold)}</div>
          <div className="s">consecutive failures before quarantine</div></div>
      </div>

      <div className="card">
        <h2>Dead letter queue</h2>
        <div className="sub">
          A channel that hard-bounces {s.threshold} times in a row for the same customer is
          quarantined, and <b>R-DLQ</b> refuses further contact on it. Retrying a dead
          address costs money per attempt, delivers nothing, and on a real provider damages
          sender reputation for every other customer. A successful delivery resets the
          counter, so a transient outage never quarantines anyone. Releasing a pair puts it
          back in service immediately.
        </div>

        <div className="controls">
          <div className="field">
            <label htmlFor="d-search">Search</label>
            <input id="d-search" ref={searchRef} type="search" autoComplete="off"
                   className={term ? "on" : ""} placeholder="customer, channel, error&hellip;"
                   value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <label className="toggle">
            <input type="checkbox" checked={showAll}
                   onChange={(e) => setShowAll(e.target.checked)} />
            Include pairs that recovered
          </label>
          <span className="spacer" />
          <span className="count">{busy ? "working…" : `${rows.length} shown`}</span>
        </div>

        <div className={`scroll${busy ? " busy" : ""}`}>
          <table>
            <thead><tr>
              <th>Customer</th><th>Channel</th><th className="num">Consecutive failures</th>
              <th>State</th><th>Last error</th><th></th>
            </tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.customer_id}:${r.channel}`}>
                  <td className="mono">{r.customer_id}</td>
                  <td>{titleize(r.channel)}</td>
                  <td className="num">{r.failures}</td>
                  <td>
                    {r.quarantined
                      ? <Chip kind="reject">quarantined</Chip>
                      : <Chip kind="approve">in service</Chip>}
                  </td>
                  <td style={{ maxWidth: 420 }} className="sub">{r.last_error || "-"}</td>
                  <td>
                    {r.quarantined && (
                      <button className="act" disabled={busy}
                              onClick={() => release(r.customer_id, r.channel)}>
                        Release
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {!rows.length && !busy && (
                <tr><td colSpan={6} className="empty">
                  {showAll ? "No delivery failures recorded in this run."
                           : "Nothing quarantined. Every channel is in service."}
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
