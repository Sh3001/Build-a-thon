import React, { useRef, useState } from "react";
import { Tooltip, TipRow } from "../components/Tooltip";

/**
 * Cumulative recovery over time. 2px strokes, a crosshair on hover, and one direct
 * end-label per series so identity never depends on color alone.
 */
export function LineChart({ series, height = 250, format = (v) => v, xLabel = "day" }) {
  const [hover, setHover] = useState(null);
  const ref = useRef(null);
  const W = 720, H = height, P = { t: 14, r: 92, b: 30, l: 62 };

  const n = Math.max(...series.map((s) => s.points.length));
  const maxY = Math.max(...series.flatMap((s) => s.points.map((p) => p.y)), 1e-9);
  const x = (i) => P.l + (i / Math.max(n - 1, 1)) * (W - P.l - P.r);
  const y = (v) => H - P.b - (v / maxY) * (H - P.t - P.b);

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => maxY * f);
  const xticks = [0, Math.floor((n - 1) / 3), Math.floor((2 * (n - 1)) / 3), n - 1];

  function onMove(e) {
    const box = ref.current.getBoundingClientRect();
    const rel = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((rel - P.l) / (W - P.l - P.r)) * (n - 1));
    if (i >= 0 && i < n) setHover({ i, x: e.clientX, y: e.clientY });
  }

  return (
    <>
      <div className="legend">
        {series.map((s) => (
          <span className="item" key={s.label}>
            <span className="swatch" style={{ background: s.color }} />{s.label}
          </span>
        ))}
      </div>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width="100%" height={H}
           onMouseMove={onMove} onMouseLeave={() => setHover(null)}
           role="img" aria-label="Cumulative revenue recovered over time">
        {ticks.map((t, i) => (
          <g key={i}>
            <line className="grid" x1={P.l} x2={W - P.r} y1={y(t)} y2={y(t)} />
            <text className="axis" x={P.l - 8} y={y(t) + 3.5} textAnchor="end">{format(t)}</text>
          </g>
        ))}
        {xticks.map((i) => (
          <text className="axis" key={i} x={x(i)} y={H - P.b + 15} textAnchor="middle">
            {xLabel} {i}
          </text>
        ))}
        {hover && (
          <line className="grid" x1={x(hover.i)} x2={x(hover.i)} y1={P.t} y2={H - P.b}
                stroke="var(--border-strong)" />
        )}
        {series.map((s) => (
          <g key={s.label}>
            <path
              d={s.points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.y)}`).join(" ")}
              fill="none" stroke={s.color} strokeWidth="2"
              strokeLinejoin="round" strokeLinecap="round"
            />
            {hover && s.points[hover.i] && (
              <circle cx={x(hover.i)} cy={y(s.points[hover.i].y)} r="4.5"
                      fill={s.color} stroke="var(--surface-1)" strokeWidth="2" />
            )}
            <text x={W - P.r + 8} y={y(s.points[s.points.length - 1].y) + 3.5}
                  fill="var(--text-secondary)" fontSize="11">{s.label}</text>
          </g>
        ))}
      </svg>
      {hover && (
        <Tooltip at={hover}>
          <div className="t">{xLabel} {hover.i}</div>
          {series.map((s) => (
            <TipRow key={s.label} label={s.label}
                    value={format(s.points[hover.i]?.y ?? 0)} />
          ))}
        </Tooltip>
      )}
    </>
  );
}
