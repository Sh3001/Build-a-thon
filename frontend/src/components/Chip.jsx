import React from "react";
import { titleize } from "../services/api";

export function Chip({ kind, children }) {
  return <span className={`chip${kind ? " " + kind : ""}`}>{children ?? titleize(kind)}</span>;
}
