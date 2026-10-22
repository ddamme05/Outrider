// Derive analytics labels, deltas, and series statistics from API aggregates.
// These pure helpers preserve missing-data and zero-value distinctions
// (DECISIONS#039); components own rendering.
import type { components } from "../api/schema";

export type MetricsResponse = components["schemas"]["DashboardMetricsResponse"];
export type MetricBucket = components["schemas"]["MetricBucket"];
export type PeriodTotals = components["schemas"]["PeriodTotals"];
export type ReplayMetricsResponse = components["schemas"]["ReplayMetricsResponse"];

// No verdicts means an undefined rate, distinct from a measured 0% equivalence
// rate. The denominator includes only reviews with verdicts (DECISIONS#039).
export function replayRate(equivalent: number, total: number): number | null {
  return total > 0 ? (equivalent / total) * 100 : null;
}

// Compare replay rates in percentage points. A real prior 0% remains a valid
// baseline; null means no verdict data. Higher equivalence uses the positive style.
export function replayDeltaInfo(curRate: number | null, prevRate: number | null): DeltaInfo {
  if (curRate === null) return { cls: "flat", glyph: "—", label: "vs prev" }; // no current verdicts
  if (prevRate === null) return { cls: "flat", glyph: "—", label: "new" }; // no PRIOR verdicts
  const percentagePointChange = curRate - prevRate; // percentage points
  if (Math.abs(percentagePointChange) < 0.05) return { cls: "flat", glyph: "—", label: "vs prev" };
  const isIncrease = percentagePointChange > 0;
  return { cls: isIncrease ? "up" : "down", glyph: isIncrease ? "▲" : "▼", label: `${Math.abs(percentagePointChange).toFixed(1)}pp` };
}

// Policy/schema display order, including categories with zero findings.
export const SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"] as const;
export const TIER_ORDER = ["observed", "inferred", "judged"] as const;

// Polarity determines whether an increase is positive, negative, or neutral.
export type DeltaPolarity = "up-good" | "up-bad" | "neutral";

export interface DeltaInfo {
  /** Direction/polarity class used by the metric card. */
  cls: "up" | "down" | "up-bad" | "down-good" | "flat";
  /** ▲ / ▼ / — */
  glyph: string;
  /** Relative percentage, percentage-point change, or a missing/unchanged label. */
  label: string;
}

// Compare period totals. Equal totals are flat; a non-positive prior baseline
// uses "new". Positive baselines produce a relative percentage (DECISIONS#039).
export function deltaInfo(
  current: number,
  previous: number,
  polarity: DeltaPolarity,
): DeltaInfo {
  if (current === previous) {
    return { cls: "flat", glyph: "—", label: "vs prev" };
  }
  const isIncrease = current > previous;
  let label: string;
  if (previous > 0) {
    const changePercent = (Math.abs(current - previous) / previous) * 100;
    label = `${changePercent < 10 ? changePercent.toFixed(1) : changePercent.toFixed(0)}%`;
  } else {
    // No positive baseline for a relative change. API count/cost totals are nonnegative.
    label = "new";
  }
  // Neutral metrics still show direction and magnitude, with no good/bad color.
  let cls: DeltaInfo["cls"];
  if (polarity === "neutral") {
    cls = "flat";
  } else if (polarity === "up-good") {
    cls = isIncrease ? "up" : "down";
  } else {
    cls = isIncrease ? "up-bad" : "down-good";
  }
  return { cls, glyph: isIncrease ? "▲" : "▼", label };
}

// Use UTC fields so axis labels match the API's UTC bucket boundaries.
const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
export function formatBucketLabel(iso: string, granularity: string): string {
  const bucketDate = new Date(iso);
  if (granularity === "hour") {
    return `${String(bucketDate.getUTCHours()).padStart(2, "0")}:00`;
  }
  return `${MONTHS[bucketDate.getUTCMonth()]} ${bucketDate.getUTCDate()}`;
}

// Keep regularly spaced ticks plus the last label, which can add one beyond max.
// Return the original array when no thinning is needed.
export function thinLabels(labels: string[], max: number): (string | null)[] {
  const labelCount = labels.length;
  if (labelCount <= max) return labels;
  const step = Math.ceil(labelCount / max);
  return labels.map((tickLabel, index) => (index % step === 0 || index === labelCount - 1 ? tickLabel : null));
}

export interface SeriesStats {
  total: number;
  avg: number;
  peak: number;
  peakIndex: number;
}

// Summarize a bucket series, keeping the first peak when values tie.
export function seriesStats(values: number[]): SeriesStats {
  if (values.length === 0) return { total: 0, avg: 0, peak: 0, peakIndex: -1 };
  let total = 0;
  let peak = -Infinity;
  let peakIndex = 0;
  values.forEach((value, index) => {
    total += value;
    if (value > peak) {
      peak = value;
      peakIndex = index;
    }
  });
  // A non-positive peak has no meaningful peak-day attribution (DECISIONS#039).
  return { total, avg: total / values.length, peak, peakIndex: peak > 0 ? peakIndex : -1 };
}
