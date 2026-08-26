import React from "react";
import { StatTile } from "../components/StatTile";
import { GroupedBars } from "../charts/GroupedBars";
import { BarList } from "../charts/BarList";
import { fmtMoney, fmtNum, fmtPct, titleize } from "../services/api";

export function Overview({ overview: o, metrics: m, onDrill }) {
  if (!o || !m) return <div className="empty">Loading…</div>;
  const agent = m.recoverai, base = m.baseline, cmp = m.comparison;

  const control = m.control || {};
  const cats = Object.keys(agent.by_category || {});
  const groups = cats.map((c) => ({
    label: titleize(c),
    note: agent.by_category[c].cases,
    values: {
      control: control.by_category?.[c]?.revenue_recovered ?? 0,
      baseline: base.by_category?.[c]?.revenue_recovered ?? 0,
      agent: agent.by_category[c].revenue_recovered,
    },
  }));

  const actions = Object.entries(agent.by_action || {})
    .map(([k, v]) => ({ label: titleize(k), value: v.amount_recovered || 0, uses: v.uses || 0,
                        wins: v.wins || 0 }))
    .sort((a, b) => b.value - a.value);

  return (
    <>
      <div className="tiles">
        <StatTile label="Revenue at Risk" value={fmtMoney(o.revenue_at_risk)}
                  sub={`${fmtNum(o.cases_processed)} cases on the held-out test set`}
                  onClick={() => onDrill({})} />
        <StatTile label="Money Recovered (gross)" value={fmtMoney(o.revenue_recovered)}
                  sub={`${fmtNum(o.cases_recovered)} cases · ${fmtPct(o.recovery_rate)} recovery rate`}
                  onClick={() => onDrill({ status: "recovered" })} />
        <StatTile label="Money We Caused" value={fmtMoney(o.incremental_vs_control)} tone="good"
                  hint="Gross recovery minus what the untouched control arm collected on its own. This is the number that survives scrutiny."
                  sub={o.control_ci_low != null
                    ? `vs no-touch control · 90% CI ${fmtMoney(o.control_ci_low)}–${fmtMoney(o.control_ci_high)}`
                    : "incremental over doing nothing"} />
        <StatTile label="Would Have Arrived Anyway" value={fmtMoney(o.control_recovered)}
                  hint="The no-touch control arm. These cases self-cured with no intervention at all."
                  sub={`no-touch control · ${fmtPct(o.control_recovery_rate)} self-cure rate · `
                       + `${fmtPct(o.share_of_revenue_that_is_causal)} of our revenue is causal`} />
        <StatTile label="Incremental vs Baseline" value={fmtMoney(o.incremental_recovery_vs_baseline)}
                  tone="good"
                  sub={o.incremental_ci_low != null
                    ? `vs retry-every-24h · 90% CI ${fmtMoney(o.incremental_ci_low)}–${fmtMoney(o.incremental_ci_high)}`
                    : `${o.recovery_uplift_pct}% uplift over retry-every-24h`} />
        <StatTile label="Cases Escalated" value={fmtNum(o.cases_escalated)}
                  sub="handed to human review"
                  onClick={() => onDrill({ status: "escalated" })} />
        <StatTile label="Cases Stopped" value={fmtNum(o.cases_stopped)}
                  sub="closed without recovery"
                  onClick={() => onDrill({ status: "stopped" })} />
        <StatTile label="Unsafe Actions Prevented" value={fmtNum(o.unsafe_actions_prevented)}
                  tone="good" sub={`baseline took ${fmtNum(base.risk_actions_taken)} on risk cases; agent took ${agent.risk_actions_taken}`}
                  hint="Automated actions the policy engine refused on fraud or compliance holds." />
        <StatTile label="Action Cost" value={fmtMoney(o.total_cost, 2)}
                  sub={`net ${fmtMoney(o.net_recovered)} · ROI ${cmp.incremental_roi}x`} />
      </div>

      <div className="grid2" style={{ marginTop: 16 }}>
        <div className="card">
          <h2>Revenue recovered by failure category</h2>
          <div className="sub">
            Same cases, same simulated rails. <b>Control</b> is never touched - it shows
            what arrives on its own, and is the only reason the other two bars mean
            anything.
          </div>
          <GroupedBars
            groups={groups}
            series={[
              { key: "control", label: "Control", color: "var(--series-4)" },
              { key: "baseline", label: "Baseline", color: "var(--baseline)" },
              { key: "agent", label: "RecoverAI", color: "var(--series-1)" },
            ]}
            format={(v) => fmtMoney(v)}
          />
        </div>

        <div className="card">
          <h2>Revenue recovered by intervention</h2>
          <div className="sub">
            Credited to the action immediately preceding the payment. Reminders and
            payment-method updates show $0 by design - they do not collect, they
            raise the odds of the retry that follows, which takes the credit.
          </div>
          <BarList
            rows={actions}
            color="var(--series-3)"
            format={(v) => fmtMoney(v)}
            tip={(r) => [["wins", fmtNum(r.wins)], ["times used", fmtNum(r.uses)]]}
            onSelect={(r) => onDrill({ q: r.label })}
          />
          <div className="hint">Click a bar to see the cases that action was chosen for.</div>
        </div>
      </div>
    </>
  );
}
