import { type AuditEvent, eventFamily, eventNode, summarizeEvent } from "../lib/auditEvent";

// The audit-feed flat table: the review's event stream, ordered as returned (by
// sequence_number). Rendered as a labeled table — a 1-based per-review index (#),
// event type, node, detail, time. The "#" is a display index (the Nth event in
// THIS review), not the raw global audit sequence_number — that's confusing across
// reviews (a review's events can start at e.g. 108); the raw value is kept on hover
// for traceability. This is the literal audit record, surfaced (FUP-133).
export function AuditFeed({ events }: { events: AuditEvent[] }) {
  if (events.length === 0) {
    return <p style={{ color: "var(--muted)" }}>No audit events recorded for this review.</p>;
  }
  return (
    <div className="audit-feed">
      <div className="afhead" aria-hidden="true">
        <span>#</span>
        <span>Event</span>
        <span>Node</span>
        <span>Detail</span>
        <span>Time</span>
      </div>
      {events.map((e, i) => {
        const node = eventNode(e);
        const summary = summarizeEvent(e);
        return (
          <div
            className={`afrow ev-c-${eventFamily(e.event_type)}`}
            key={`${e.sequence_number ?? i}-${e.event_type}-${i}`}
          >
            <span className="af-seq mono" title={`audit sequence ${e.sequence_number ?? i}`}>
              {i + 1}
            </span>
            <span className="af-type mono">{e.event_type}</span>
            <span className="af-node mono">{node ?? ""}</span>
            <span className="af-summary">{summary}</span>
            <span className="af-ts mono">
              {e.timestamp ? e.timestamp.slice(0, 19).replace("T", " ") : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}
