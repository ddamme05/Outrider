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
