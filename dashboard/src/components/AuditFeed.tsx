import type { components } from "../api/schema";

type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];

// A terse one-line summary per event type for the feed. Only fields confirmed to
// exist on each narrowed type are read; unhandled types fall through to "" (the
// type + node badge still render). Display-only — values shown as audited.
function summarize(e: AuditEvent): string {
  switch (e.event_type) {
    case "llm_call":
      return `${e.model} · $${e.cost_usd.toFixed(2)} · ${e.input_tokens}+${e.output_tokens} tok`;
    case "finding":
      return `${e.finding_type} · ${e.severity} · ${e.file_path}`;
    case "review_phase":
      return e.marker;
    case "file_examination":
      return e.file_path;
    case "trace_decision":
      return e.resolution_status;
    default:
      return "";
  }
}

function nodeOf(e: AuditEvent): string | null {
  return "node_id" in e && typeof e.node_id === "string" ? e.node_id : null;
}

// The audit-feed tab: the review's event stream, ordered as returned (by
// sequence_number). One row per event — sequence, type, node, summary, timestamp.
// This is the literal audit record, surfaced (FUP-133).
export function AuditFeed({ events }: { events: AuditEvent[] }) {
  if (events.length === 0) {
    return <p style={{ color: "var(--text-2)" }}>No audit events recorded for this review.</p>;
  }
  return (
    <div className="audit-feed">
      {events.map((e, i) => {
        const node = nodeOf(e);
        const summary = summarize(e);
        return (
          <div className="afrow" key={`${e.sequence_number ?? i}-${e.event_type}-${i}`}>
            <span className="af-seq mono">{e.sequence_number ?? i}</span>
            <span className="af-type mono">{e.event_type}</span>
            <span className="af-node mono">{node ?? ""}</span>
            <span className="af-summary">{summary}</span>
            <span className="af-ts mono">{e.timestamp ? e.timestamp.slice(0, 19).replace("T", " ") : ""}</span>
          </div>
        );
      })}
    </div>
  );
}
