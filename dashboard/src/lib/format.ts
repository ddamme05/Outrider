import type { components } from "../api/schema";

// Unpriced calls make the reported review cost a lower bound.
export function formatReviewCost(
  metrics: Pick<components["schemas"]["ReviewMetricsView"], "total_cost_usd" | "cost_complete">,
): string {
  return `${metrics.cost_complete === false ? "\u2265" : ""}$${metrics.total_cost_usd.toFixed(2)}`;
}

// Shared work-duration label: milliseconds below 1s, tenths below 10s,
// rounded seconds thereafter.
export function formatDurationMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
}

// Reject missing/invalid endpoints and negative spans (including clock skew).
export function spanMs(start: string | null | undefined, end: string | null | undefined): number | null {
  if (!start || !end) return null;
  const durationMs = new Date(end).getTime() - new Date(start).getTime();
  return Number.isFinite(durationMs) && durationMs >= 0 ? durationMs : null;
}

// Sum merged intervals: concurrent workers must not double-count wall time,
// and separate analyze passes must exclude intervening trace gaps. Drop invalid
// spans and return null when no valid intervals remain.
export function unionDurationMs(
  spans: ReadonlyArray<{ start: string | null | undefined; end: string | null | undefined }>,
): number | null {
  const intervals: Array<[number, number]> = [];
  for (const { start, end } of spans) {
    if (!start || !end) continue;
    const startMs = new Date(start).getTime();
    const endMs = new Date(end).getTime();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs < startMs) continue;
    intervals.push([startMs, endMs]);
  }
  if (intervals.length === 0) return null;
  intervals.sort((leftInterval, rightInterval) => leftInterval[0] - rightInterval[0]);
  let totalMs = 0;
  let mergedInterval: [number, number] | null = null;
  for (const [intervalStartMs, intervalEndMs] of intervals) {
    if (mergedInterval === null) {
      mergedInterval = [intervalStartMs, intervalEndMs];
    } else if (intervalStartMs <= mergedInterval[1]) {
      if (intervalEndMs > mergedInterval[1]) mergedInterval[1] = intervalEndMs;
    } else {
      totalMs += mergedInterval[1] - mergedInterval[0];
      mergedInterval = [intervalStartMs, intervalEndMs];
    }
  }
  if (mergedInterval !== null) totalMs += mergedInterval[1] - mergedInterval[0];
  return totalMs;
}

// Show a rounded minute/hour/day countdown through 365 days. Hide absent,
// invalid, and farther-future expiries, including the demo pending-review dates.
export function expiresLabel(expiresAt: string | null): string | null {
  if (!expiresAt) return null;
  const remainingMs = new Date(expiresAt).getTime() - Date.now();
  if (Number.isNaN(remainingMs)) return null;
  if (remainingMs <= 0) return "expired";
  const minutes = Math.round(remainingMs / 60000);
  if (minutes < 60) return `expires in ${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `expires in ${hours}h`;
  const days = Math.round(hours / 24);
  if (days <= 365) return `expires in ${days}d`;
  return null;
}
