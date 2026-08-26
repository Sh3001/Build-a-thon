import React from "react";

/** Follows the cursor; never intercepts pointer events. */
export function Tooltip({ at, children }) {
  if (!at) return null;
  const style = {
    left: Math.min(at.x + 14, window.innerWidth - 280),
    top: Math.max(at.y - 12, 8),
  };
  return <div className="tip" style={style}>{children}</div>;
}

export function TipRow({ label, value }) {
  return <div className="r"><span>{label}</span><b>{value}</b></div>;
}
