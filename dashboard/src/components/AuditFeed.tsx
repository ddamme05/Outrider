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

// event_type → color family (the --ev-* palette). Groups the 16 discriminated
// event types into 9 families so the feed reads at a glance; unmapped types fall
// back to the neutral "framing" family rather than going uncolored.
const EV_FAMILY: Record<string, string> = {
  llm_call: "model",
  finding: "finding",
  analyze_response_rejected: "rejected",
  finding_proposal_rejected: "rejected",
  file_examination: "file",
  trace_decision: "trace",
  hitl_request: "human",
  hitl_decision: "human",
  publish: "output",
  publish_attempt: "output",
  publish_eligibility: "output",
  publish_routing: "output",
  review_phase: "framing",
  agent_transition: "framing",
  analyze_completed: "complete",
  synthesize_completed: "complete",
};

// The audit-feed tab: the review's event stream, ordered as returned (by
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
        const node = nodeOf(e);
        const summary = summarize(e);
        return (
          <div
            className={`afrow ev-c-${EV_FAMILY[e.event_type] ?? "framing"}`}
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
