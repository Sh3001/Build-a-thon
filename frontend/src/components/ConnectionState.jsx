import React, { useEffect, useState } from "react";
import { api, fmtNum } from "../services/api";

export function ConnectionState({ status, onQueue }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    if (!status?.source_configured) return;
    const controller = new AbortController();
    setError(null);
    api.sourceReport({ signal: controller.signal }).then(setData).catch(e => {
      if (e.name !== "AbortError") setError(e.message);
    });
    return () => controller.abort();
  }, [status?.source_configured, revision]);
  const uci = status?.source_kind === "uci";
  return <section className="card" aria-live="polite">
    <h2>{uci ? "UCI credit-card repayment dataset" : status?.source_configured ? "Connected payment source" : "Connect your payment data"}</h2>
    <p>{status?.message || "Checking connection settings…"}</p>
    {!status?.source_configured && <p>When your database is ready, configure its connection and
      failed-payment query on the server, then restart the application. Connection credentials
      stay on the server. Recovery history needs separate, writable application storage.</p>}
    {status?.source_configured && <>
      {uci && <p>Real historical accounts, not live failed payments. <a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients">Yeh (2009), UCI</a> · CC BY 4.0.</p>}
      <p>Source data is read-only. Recovery outcomes and experiment comparisons are unavailable
        until a payment provider and recovery storage are connected.</p>
      {error && <p role="alert" className="err">{error}</p>}
      {data && <p>{fmtNum(data.fetched)} records fetched · {fmtNum(data.accepted)} accepted · {fmtNum(data.rejected)} rejected.
        {" "}{uci ? `${fmtNum(data.excluded)} accounts excluded because they have no recorded delay or no positive bill balance.` : "This is a bounded snapshot, not a total across the entire database."}</p>}
      {uci && data?.note && <p>{data.note}</p>}
      {data?.model_monitoring && <p>Model monitoring: {data.model_monitoring.status.replaceAll("_", " ")}
        {data.model_monitoring.severity && ` · drift severity: ${data.model_monitoring.severity}`}</p>}
      <button onClick={() => setRevision(r => r + 1)}>Refresh connection</button>{" "}
      <button onClick={onQueue}>{uci ? "View delinquent accounts" : "View failed payments"}</button>
    </>}
  </section>;
}
