// Unknown review statuses keep their raw label and use the completed pill style.
const PILL_CLASS: Record<string, string> = {
  running: "status-running",
  awaiting_approval: "status-awaiting",
  awaiting_approval_expired: "status-expired",
  completed: "status-completed",
  failed: "status-failed",
  skipped: "status-skipped",
};

export function StatusPill({ status }: { status: string }) {
  const variant = PILL_CLASS[status] ?? "status-completed";
  return (
    <span className={`pill ${variant}`}>
      <span className="dot" aria-hidden="true" />
      {status}
    </span>
  );
}
