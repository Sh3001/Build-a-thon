import React, { useState } from "react";
import { Tooltip, TipRow } from "../components/Tooltip";

/**
 * Baseline vs RecoverAI, one group per category. Two bars per group with a 2px surface
 * gap; identity is carried by the legend AND the direct label, never by color alone.
 */
export function GroupedBars({ groups, series, format = (v) => v }) {
  const [at, setAt] = useState(null);
  const max = Math.max(...groups.flatMap((g) => series.map((s) => g.values[s.key] ?? 0)), 1e-9);
  return (
    <>
      <div className="legend">
        {series.map((s) => (
          <span className="item" key={s.key}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      {groups.map((g) => (
        <div className="group" key={g.label}>
          <div className="glabel">{g.label}</div>
          <div className="pair">
            {series.map((s) => {
              const v = g.values[s.key] ?? 0;
              return (
                <div
                  className="barrow"
                  key={s.key}
                  style={{ gridTemplateColumns: "86px 1fr 100px", marginBottom: 0 }}
                  onMouseMove={(e) => setAt({ x: e.clientX, y: e.clientY, g, s, v })}
                  onMouseLeave={() => setAt(null)}
                >
                  <span className="label">{s.label}</span>
                  <span className="bartrack" style={{ height: 13 }}>
                    <span className="barfill"
                          style={{ width: `${Math.max((v / max) * 100, 1)}%`, background: s.color }} />
                  </span>
                  <span className="val">{format(v)}</span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
      {at && (
        <Tooltip at={at}>
          <div className="t">{at.g.label}</div>
          <TipRow label={at.s.label} value={format(at.v)} />
          {at.g.note && <TipRow label="cases" value={at.g.note} />}
        </Tooltip>
      )}
    </>
  );
}
