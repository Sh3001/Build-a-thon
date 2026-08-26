import React from "react";

/**
 * A tile is a plain div until it is given `onClick`, at which point it becomes a real
 * <button> -- so the drill-downs are keyboard-reachable and announce themselves, rather
 * than being a div with a click handler bolted on.
 */
export function StatTile({ label, value, sub, tone, onClick, hint }) {
  const body = (
    <>
      <div className="k">{label}</div>
      <div className={`v${tone ? " " + tone : ""}`}>{value}</div>
      {sub && <div className="s">{sub}</div>}
      {onClick && <div className="drill">View cases &rarr;</div>}
    </>
  );
  if (!onClick) return <div className="tile" title={hint}>{body}</div>;
  return (
    <button type="button" className="tile tappable" onClick={onClick}
            title={hint || `Show these cases in the recovery queue`}>
      {body}
    </button>
  );
}
