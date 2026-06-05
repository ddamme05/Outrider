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

// Human "expires in Xm/Xh" from an ISO timestamp. Returns null when absent or
// unparseable; "expired" when already past.
export function expiresLabel(expiresAt: string | null): string | null {
  if (!expiresAt) return null;
  const ms = new Date(expiresAt).getTime() - Date.now();
  if (Number.isNaN(ms)) return null;
  if (ms <= 0) return "expired";
  const mins = Math.round(ms / 60000);
  return mins < 60 ? `expires in ${mins}m` : `expires in ${Math.round(mins / 60)}h`;
}
