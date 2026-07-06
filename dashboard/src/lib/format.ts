// Human duration from a millisecond span: "Xms" under 1s, "X.Xs" under 10s, "Xs"
// above. The single home for work-duration formatting — shared by the pipeline node
// cards and the replay-timeline phase headers so the same span never renders two ways.
export function formatDurationMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
}

// Millisecond span between two ISO timestamps, or null when either is absent/unparseable
// or the span is negative (clock skew). Shared so the validity guard lives in one place.
export function spanMs(start: string | null | undefined, end: string | null | undefined): number | null {
  if (!start || !end) return null;
  const ms = new Date(end).getTime() - new Date(start).getTime();
  return Number.isFinite(ms) && ms >= 0 ? ms : null;
}

// Total wall-time covered by a set of possibly-overlapping intervals: the UNION of the
// intervals, not the sum of their spans. Parallel analyze fans out into concurrent worker
// phases whose spans overlap — summing them would multi-count wall-time (up to the
// concurrency factor). Merging collapses that overlap. And because a node's phases can
// straddle multiple analyze⇄trace passes with trace time BETWEEN them, merging (rather than
// a single earliest→latest span) correctly excludes the trace gap: non-overlapping clusters
// stay separate and only their own durations are summed. Endpoints are validated with the
// same guard as spanMs (absent/unparseable/negative dropped). Returns null when none valid.
export function unionDurationMs(
  spans: ReadonlyArray<{ start: string | null | undefined; end: string | null | undefined }>,
): number | null {
  const intervals: Array<[number, number]> = [];
  for (const { start, end } of spans) {
    if (!start || !end) continue;
    const a = new Date(start).getTime();
    const b = new Date(end).getTime();
    if (!Number.isFinite(a) || !Number.isFinite(b) || b < a) continue;
    intervals.push([a, b]);
  }
  if (intervals.length === 0) return null;
  intervals.sort((x, y) => x[0] - y[0]);
  let total = 0;
  let cur: [number, number] | null = null;
  for (const [s, e] of intervals) {
    if (cur === null) {
      cur = [s, e];
    } else if (s <= cur[1]) {
      if (e > cur[1]) cur[1] = e; // overlapping/contiguous — extend the merged interval
    } else {
      total += cur[1] - cur[0]; // disjoint (e.g. a trace gap) — bank and restart
      cur = [s, e];
    }
  }
  if (cur !== null) total += cur[1] - cur[0];
  return total;
}

// Human "expires in Xm/Xh/Xd" from an ISO timestamp. Returns null when absent or
// unparseable, "expired" when already past, and null again when more than a year out —
// no countdown is meaningful for a far-future timeout (e.g. the demo's pinned-pending
// HITL reviews, which carry a ~100-year expiry so they never read "expired").
export function expiresLabel(expiresAt: string | null): string | null {
  if (!expiresAt) return null;
  const ms = new Date(expiresAt).getTime() - Date.now();
  if (Number.isNaN(ms)) return null;
  if (ms <= 0) return "expired";
  const mins = Math.round(ms / 60000);
  if (mins < 60) return `expires in ${mins}m`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `expires in ${hours}h`;
  const days = Math.round(hours / 24);
  if (days <= 365) return `expires in ${days}d`;
  return null;
}
