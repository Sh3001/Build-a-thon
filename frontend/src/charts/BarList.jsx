import React, { useState } from "react";
import { Tooltip, TipRow } from "../components/Tooltip";

/**
 * Horizontal magnitude bars with direct value labels.
 * The value label is not decoration: three light-mode series sit below 3:1 against the
 * surface, so the validated palette's relief rule requires a visible label.
 */
export function BarList({ rows, color = "var(--series-1)", format = (v) => v, tip, onSelect }) {
  const [at, setAt] = useState(null);
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  return (
    <>
      {rows.map((r) => (
        <div
          className={`barrow${onSelect ? " tappable" : ""}`}
          key={r.label}
          role={onSelect ? "button" : undefined}
          tabIndex={onSelect ? 0 : undefined}
          title={onSelect ? `Show ${r.label} cases` : undefined}
          onClick={() => onSelect?.(r)}
          onKeyDown={(e) => {
            if (onSelect && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onSelect(r); }
          }}
          onMouseMove={(e) => tip && setAt({ x: e.clientX, y: e.clientY, row: r })}
          onMouseLeave={() => setAt(null)}
        >
          <span className="label" title={r.label}>{r.label}</span>
          <span className="bartrack">
            <span
              className="barfill"
              style={{ width: `${Math.max((r.value / max) * 100, 1.5)}%`,
                       background: r.color || color }}
            />
          </span>
          <span className="val">{format(r.value)}</span>
        </div>
      ))}
      {at && tip && (
        <Tooltip at={at}>
          <div className="t">{at.row.label}</div>
          {tip(at.row).map(([k, v]) => <TipRow key={k} label={k} value={v} />)}
        </Tooltip>
      )}
    </>
  );
}
