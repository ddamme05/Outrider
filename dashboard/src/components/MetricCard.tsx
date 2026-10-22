// Render a metric, its period delta, and a trend sparkline. Callers supply the
// values (DECISIONS#039); this component owns presentation.
import type { ReactNode } from "react";

import type { DeltaInfo } from "../lib/metrics";
import { Sparkline, type SparkVariant } from "./Sparkline";

export function MetricCard({
  label,
  value,
  cap,
  delta,
  spark,
  sparkVariant,
  sparkIncomplete,
}: {
  label: string;
  value: ReactNode;
  cap?: ReactNode;
  delta: DeltaInfo;
  spark: number[];
  sparkVariant: SparkVariant;
  sparkIncomplete?: boolean[];
}) {
  return (
    <div className="card stat-card metric-card">
      <div className="lab">{label}</div>
      <div className="stat-num">{value}</div>
      {cap ? <div className="cap">{cap}</div> : null}
      <div className={`delta ${delta.cls}`}>
        <span aria-hidden="true">{delta.glyph}</span>
        <span className="num">{delta.label}</span>
      </div>
      <Sparkline
        values={spark}
        variant={sparkVariant}
        label={`${label} trend`}
        incomplete={sparkIncomplete}
      />
    </div>
  );
}
