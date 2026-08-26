import React from "react";
import { LineChart } from "../charts/LineChart";
import { BarList } from "../charts/BarList";
import { GroupedBars } from "../charts/GroupedBars";
import { fmtMoney, fmtNum, fmtPct, titleize } from "../services/api";

export function Analytics({ metrics: m }) {
  if (!m) return <div className="empty">Loading…</div>;
  const agent = m.recoverai, base = m.baseline, cmp = m.comparison, model = m.model || {};

  const series = [
    { label: "Baseline", color: "var(--baseline)",
      points: (base.recovery_over_time || []).map((p) => ({ y: p.cumulative_revenue })) },
    { label: "RecoverAI", color: "var(--series-1)",
      points: (agent.recovery_over_time || []).map((p) => ({ y: p.cumulative_revenue })) },
  ];

  const byCode = Object.entries(agent.by_failure_code || {})
    .map(([k, v]) => ({ label: titleize(k), value: v.revenue_recovered, rate: v.recovery_rate,
                        cases: v.cases }))
    .sort((a, b) => b.value - a.value);

  const rateGroups = Object.keys(agent.by_category || {}).map((c) => ({
    label: titleize(c),
    values: { baseline: base.by_category?.[c]?.recovery_rate ?? 0,
              agent: agent.by_category[c].recovery_rate },
  }));

  return (
    <>
      <div className="tiles">
        <div className="tile"><div className="k">Value recovery rate</div>
          <div className="v">{fmtPct(agent.value_recovery_rate)}</div>
          <div className="s">baseline {fmtPct(base.value_recovery_rate)}</div></div>
        <div className="tile"><div className="k">Avg recovery time</div>
          <div className="v">{agent.avg_recovery_hours?.toFixed(0)}h</div>
          <div className="s">baseline {base.avg_recovery_hours?.toFixed(0)}h · median {agent.median_recovery_hours?.toFixed(0)}h</div></div>
        <div className="tile"><div className="k">Retries used</div>
          <div className="v">{fmtNum(agent.total_retries)}</div>
          <div className="s">baseline {fmtNum(base.total_retries)}</div></div>
        <div className="tile"><div className="k">Model ROC-AUC</div>
          <div className="v">{model.roc_auc ?? "-"}</div>
          <div className="s">PR-AUC {model.pr_auc ?? "-"} · held-out test</div></div>
      </div>

      <div className="card">
        <h2>Cumulative revenue recovered over time</h2>
        <div className="sub">
          Days since the original payment failure. The baseline collects faster early -
          it retries everything at 24h - and then flattens; RecoverAI keeps recovering
          because it repairs instruments and times retries to the cause.
        </div>
        <LineChart series={series} format={(v) => fmtMoney(v)} />
      </div>

      <div className="grid2">
        <div className="card">
          <h2>Recovery rate by failure category</h2>
          <div className="sub">Share of cases recovered, same cases in both arms.</div>
          <GroupedBars groups={rateGroups}
            series={[{ key: "baseline", label: "Baseline", color: "var(--baseline)" },
                     { key: "agent", label: "RecoverAI", color: "var(--series-1)" }]}
            format={(v) => fmtPct(v)} />
        </div>
        <div className="card">
          <h2>Revenue recovered by failure type</h2>
          <div className="sub">RecoverAI, all fourteen failure codes.</div>
          <BarList rows={byCode} color="var(--series-2)" format={(v) => fmtMoney(v)}
            tip={(r) => [["recovery rate", fmtPct(r.rate)], ["cases", fmtNum(r.cases)]]} />
        </div>
      </div>

      <div className="card">
        <h2>Baseline vs RecoverAI</h2>
        <table>
          <thead><tr>
            <th>Metric</th><th className="num">Baseline</th>
            <th className="num">RecoverAI</th><th className="num">Delta</th>
          </tr></thead>
          <tbody>
            {/* `better` says which direction is an improvement: fewer retries and fewer
                unsafe actions are wins, so a naive "higher is greener" rule would paint
                extra cost as good and eliminating 399 unsafe actions as neutral. */}
            {[
              ["Revenue recovered", base.revenue_recovered, agent.revenue_recovered, fmtMoney, "up"],
              ["Cases recovered", base.cases_recovered, agent.cases_recovered, fmtNum, "up"],
              ["Recovery rate", base.recovery_rate, agent.recovery_rate, fmtPct, "up"],
              ["Retries", base.total_retries, agent.total_retries, fmtNum, "down"],
              ["Customer contacts", base.total_contacts, agent.total_contacts, fmtNum, "flat"],
              ["Unsafe risk actions", base.risk_actions_taken, agent.risk_actions_taken, fmtNum, "down"],
              ["Action cost", base.total_cost, agent.total_cost, (v) => fmtMoney(v, 2), "flat"],
            ].map(([label, b, a, f, better]) => {
              const d = a - b;
              const improved = better === "up" ? d > 0 : better === "down" ? d < 0 : null;
              const color = improved === null ? "var(--text-secondary)"
                          : improved ? "var(--good)" : "var(--text-muted)";
              return (
                <tr key={label}>
                  <td>{label}</td>
                  <td className="num">{f(b)}</td>
                  <td className="num">{f(a)}</td>
                  <td className="num" style={{ color }}>{(d >= 0 ? "+" : "\u2212") + f(Math.abs(d))}</td>
                </tr>
              );
            })}
            <tr>
              <td><b>Incremental recovered revenue</b></td>
              <td className="num">-</td>
              <td className="num"><b>{fmtMoney(cmp.incremental_recovered_revenue)}</b></td>
              <td className="num" style={{ color: "var(--good)" }}>{cmp.recovery_uplift_pct}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  );
}
