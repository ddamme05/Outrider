// Pure helpers for the Signal Overview analytics (DECISIONS#039). The /api/metrics
// endpoint returns honest aggregates; these functions DERIVE the operator-facing
// display values (deltas, per-day series, labels) from them — no fabrication, no
// interpolation. Kept pure + separate from the components so they unit-test
// directly (mirrors lib/hitl.ts).
import type { components } from "../api/schema";

export type MetricsResponse = components["schemas"]["DashboardMetricsResponse"];
export type MetricBucket = components["schemas"]["MetricBucket"];
export type PeriodTotals = components["schemas"]["PeriodTotals"];
export type ReplayMetricsResponse = components["schemas"]["ReplayMetricsResponse"];

// Replay-equivalence rate as a percentage, or null when the window has NO verdicts.
// Honest-zeros sibling of seriesStats: a window with zero verdicted reviews has no
// DEFINED rate, so we return null and the card renders "—" — never 0%, which would
// wrongly imply "every replay diverged" (the denominator is verdicted reviews only,
// DECISIONS#039). The frontend derives the %, never the server (no /0 server-side).
export function replayRate(equivalent: number, total: number): number | null {
  return total > 0 ? (equivalent / total) * 100 : null;
}

// Canonical display order — matches the policy severity enum (5 values) and the
// evidence-tier enum (3 values). The endpoint zero-fills every key, so a key may
// be present with count 0; render it honestly rather than dropping it.
export const SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"] as const;
export const TIER_ORDER = ["observed", "inferred", "judged"] as const;

// A metric's delta polarity: does "up" mean better, worse, or neither? The mockup
// encodes this in the delta CSS class (up/down are green/red; up-bad/down-good
// invert; flat is grey) — direction alone does NOT imply color.
export type DeltaPolarity = "up-good" | "up-bad" | "neutral";

export interface DeltaInfo {
  /** the mockup delta class: up | down | up-bad | down-good | flat */
  cls: "up" | "down" | "up-bad" | "down-good" | "flat";
  /** ▲ / ▼ / — */
  glyph: string;
  /** "21.6%" when the prior window is non-empty; "new"/"vs prev" otherwise */
  label: string;
}

// Period-over-period delta. The endpoint returns current + previous totals and the
// FRONTEND computes the %, so an empty prior window (previous === 0) is honest
// facts, never a server-side divide-by-zero (DECISIONS#039): we show "new"/"vs
// prev" instead of a fabricated ratio.
export function deltaInfo(
  current: number,
  previous: number,
  polarity: DeltaPolarity,
): DeltaInfo {
  if (current === previous) {
    return { cls: "flat", glyph: "—", label: "vs prev" };
  }
  const up = current > previous;
  let label: string;
  if (previous > 0) {
    const pct = (Math.abs(current - previous) / previous) * 100;
    label = `${pct < 10 ? pct.toFixed(1) : pct.toFixed(0)}%`;
  } else {
    // previous === 0 and current !== previous: counts are ≥ 0, so current > 0 — no finite
    // ratio, render honest "new" rather than ∞%. (A decrease can't reach here.)
    label = "new";
  }
  // `neutral` (e.g. Findings) still shows direction + magnitude — more/fewer findings is real
  // signal — but in grey, because up/down isn't inherently good or bad. up-good/up-bad colour it.
  let cls: DeltaInfo["cls"];
  if (polarity === "neutral") {
    cls = "flat";
  } else if (polarity === "up-good") {
    cls = up ? "up" : "down";
  } else {
    cls = up ? "up-bad" : "down-good";
  }
  return { cls, glyph: up ? "▲" : "▼", label };
}

// Bucket timestamp → short axis label. Buckets are UTC (the endpoint forces
// date_trunc(..,'UTC')), so read UTC fields to match the bucket boundary exactly.
const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
export function formatBucketLabel(iso: string, granularity: string): string {
  const d = new Date(iso);
  if (granularity === "hour") {
    return `${String(d.getUTCHours()).padStart(2, "0")}:00`;
  }
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

// Thin a label array down to at most `max` evenly-spaced ticks (first + last
// always kept) — 24h/30d windows have too many buckets to label every one.
export function thinLabels(labels: string[], max: number): (string | null)[] {
  const n = labels.length;
  if (n <= max) return labels;
  const step = Math.ceil(n / max);
  return labels.map((l, i) => (i % step === 0 || i === n - 1 ? l : null));
}

export interface SeriesStats {
  total: number;
  avg: number;
  peak: number;
  peakIndex: number;
}

// Total / average / peak over a per-day series — the hero chart's legend line.
// All client-derived from the bucket array; no dedicated endpoint field.
export function seriesStats(values: number[]): SeriesStats {
  if (values.length === 0) return { total: 0, avg: 0, peak: 0, peakIndex: -1 };
  let total = 0;
  let peak = -Infinity;
  let peakIndex = 0;
  values.forEach((v, i) => {
    total += v;
    if (v > peak) {
      peak = v;
      peakIndex = i;
    }
  });
  // An all-zero (honest-empty) window has no meaningful peak DAY — report peakIndex -1 so the
  // chart legend doesn't fabricate "peak $0.00 on <first bucket>" (DECISIONS#039 honest-zeros).
  return { total, avg: total / values.length, peak, peakIndex: peak > 0 ? peakIndex : -1 };
}
