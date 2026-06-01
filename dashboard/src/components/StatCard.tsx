import type { ReactNode } from "react";

// A Signal stat card — mono label + big mono number + optional caption. NO delta,
// sparkline, or trend element: those need a time-series/metrics endpoint we don't
// have, and inventing them would fabricate analytics (spec non-goal). Values are
// current counts read from existing endpoints only.
export function StatCard({
  label,
  value,
  cap,
}: {
  label: string;
  value: ReactNode;
  cap?: string;
}) {
  return (
    <div className="card stat-card">
      <div className="lab">{label}</div>
      <div className="stat-num">{value}</div>
      {cap ? <div className="cap">{cap}</div> : null}
    </div>
  );
}
