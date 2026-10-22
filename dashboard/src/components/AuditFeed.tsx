import { type AuditEvent, eventFamily, eventNode, summarizeEvent } from "../lib/auditEvent";

// Keep the supplied audit order (FUP-133). Number rows within this review;
// preserve the global sequence number on hover for traceability.
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
      {events.map((auditEvent, eventIndex) => {
        const node = eventNode(auditEvent);
        const summary = summarizeEvent(auditEvent);
        return (
          <div
            className={`afrow ev-c-${eventFamily(auditEvent.event_type)}`}
            key={`${auditEvent.sequence_number ?? eventIndex}-${auditEvent.event_type}-${eventIndex}`}
          >
            <span
              className="af-seq mono"
              title={`audit sequence ${auditEvent.sequence_number ?? eventIndex}`}
            >
              {eventIndex + 1}
            </span>
            <span className="af-type mono">{auditEvent.event_type}</span>
            <span className="af-node mono">{node ?? ""}</span>
            <span className="af-summary">{summary}</span>
            <span className="af-ts mono">
              {auditEvent.timestamp ? auditEvent.timestamp.slice(0, 19).replace("T", " ") : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}
