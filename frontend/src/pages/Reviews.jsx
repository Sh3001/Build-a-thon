import React, { useCallback, useEffect, useState } from "react";
import { api, fmtMoney, fmtNum, titleize } from "../services/api";
import { Chip } from "../components/Chip";

/**
 * The human-review queue.
 *
 * Ordered by value, because an operator's minute is the scarce resource. Approving does
 * not execute: it records an override and returns the verdict that permits the action,
 * which the executor still requires. That distinction is stated on the page because an
 * operator who believes the button charges the customer will use it differently from one
 * who knows it authorises a later attempt.
 */
export function Reviews({ onSelect }) {
  const [rows, setRows] = useState(null);
  const [stats, setStats] = useState(null);
  const [overrides, setOverrides] = useState([]);
  const [reasons, setReasons] = useState({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [notice, setNotice] = useState(null);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const [queue, s, o] = await Promise.all([
        api.reviews(200), api.reviewStats(), api.overrides(50),
      ]);
      setRows(queue);
      setStats(s);
      setOverrides(o.overrides || []);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function resolve(id, approve) {
    const reason = (reasons[id] || "").trim();
    if (reason.length < 3) {
      // The backend refuses a blank justification too. Checking here as well saves a
      // round trip; it is not the control.
      setNotice("A reason is required. An unexplained override is indistinguishable "
                + "from a mistake when someone reads the log a year from now.");
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      const out = approve ? await api.reviewApprove(id, reason)
                          : await api.reviewReject(id, reason);
      setNotice(out.note || `Recorded as ${out.decision}.`);
      setReasons((r) => ({ ...r, [id]: "" }));
      await load();
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  }

  if (err) return <div className="err">{err}</div>;
  if (!rows || !stats) return <div className="card"><div className="empty">Loading…</div></div>;

  return (
    <>
      <div className="tiles">
        <div className="tile"><div className="k">Awaiting a decision</div>
          <div className={`v${stats.pending ? " critical" : ""}`}>{fmtNum(stats.pending)}</div>
          <div className="s">cases the agent withheld rather than dropped</div></div>
        <div className="tile"><div className="k">Value held</div>
          <div className="v">{fmtMoney(stats.pending_value_usd)}</div>
          <div className="s">at risk while these sit unreviewed</div></div>
        <div className="tile"><div className="k">Resolved</div>
          <div className="v">{fmtNum((stats.by_status?.approved || 0)
                                     + (stats.by_status?.rejected || 0))}</div>
          <div className="s">{fmtNum(stats.by_status?.approved || 0)} approved · {}
            {fmtNum(stats.by_status?.rejected || 0)} rejected</div></div>
        <div className="tile"><div className="k">SLA</div>
          <div className="v">{fmtNum(stats.sla_hours)}h</div>
          <div className="s">after which a task expires unreviewed</div></div>
      </div>

      <div className="card">
        <h2>Human review queue</h2>
        <div className="sub">
          The policy engine returns four verdicts and only two of them permit execution.
          A <b>HUMAN_REVIEW</b> verdict withholds the action and puts the case here rather
          than dropping it, which is the difference between a safety limit and a revenue
          leak. <b>Approving records an authorisation; it does not execute.</b> The action
          still runs through the executor, which requires the verdict this produces, so an
          override reaches the rails through the same gate as everything else.
        </div>
        {notice && <div className="sub" style={{ marginTop: 8 }}><b>{notice}</b></div>}

        {rows.length === 0 ? (
          <div className="empty">Nothing awaiting review.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Case</th><th>Rule</th><th className="num">Amount</th>
                <th>Why it was withheld</th><th>Reason for your decision</th><th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.review_id}>
                  <td>
                    <button className="linky" onClick={() => onSelect && onSelect(r.transaction_id)}>
                      {r.transaction_id}
                    </button>
                    <div className="s">{new Date(r.created_at).toLocaleString()}</div>
                  </td>
                  <td><Chip>{r.rule_id}</Chip></td>
                  <td className="num">{fmtMoney(r.amount_usd, 2)}</td>
                  <td className="s">{r.reason}</td>
                  <td>
                    <input
                      type="text"
                      value={reasons[r.review_id] || ""}
                      placeholder="required: what did you check?"
                      onChange={(e) =>
                        setReasons((s) => ({ ...s, [r.review_id]: e.target.value }))}
                      style={{ width: "100%" }}
                    />
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button disabled={busy} onClick={() => resolve(r.review_id, true)}>
                      Approve
                    </button>{" "}
                    <button disabled={busy} onClick={() => resolve(r.review_id, false)}>
                      Reject
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h2>Override log</h2>
        <div className="sub">
          Append-only. Every override records who, when, from what verdict to what verdict,
          and why. Readable by the auditor role, which can approve nothing. Separating the
          person who evaluates the system from the person who authorises money to move is
          what makes this log worth keeping.
        </div>
        {overrides.length === 0 ? (
          <div className="empty">No overrides recorded.</div>
        ) : (
          <table>
            <thead>
              <tr><th>When</th><th>Who</th><th>Case</th><th>Decision</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {overrides.map((o) => (
                <tr key={o.override_id}>
                  <td className="s">{new Date(o.at).toLocaleString()}</td>
                  <td>{o.actor} <Chip>{titleize(o.actor_role)}</Chip></td>
                  <td>{o.transaction_id}</td>
                  <td>
                    <span className="s">{titleize(o.original_decision)} →</span>{" "}
                    <b>{titleize(o.new_decision)}</b>
                  </td>
                  <td className="s">{o.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
