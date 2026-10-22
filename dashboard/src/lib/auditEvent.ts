import type { components } from "../api/schema";
import { prettyModel } from "./modelLabel";

// The audit table and replay timeline share the API's event union.
export type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];

// Display recorded values; callers handle the event type and optional node display.
export function summarizeEvent(event: AuditEvent): string {
  switch (event.event_type) {
    case "llm_call": {
      // A missing price is unpriced; zero remains a recorded price.
      const costLabel = event.cost_usd == null ? "unpriced" : `$${event.cost_usd.toFixed(2)}`;
      return `${prettyModel(event.model)} · ${costLabel} · ${event.input_tokens}+${event.output_tokens} tok`;
    }
    case "finding":
      return `${event.finding_type} · ${event.severity} · ${event.file_path}`;
    case "review_phase":
      return event.marker;
    case "file_examination":
      return event.file_path;
    case "trace_decision":
      return event.resolution_status;
    case "agent_transition":
      return `${event.from_node} → ${event.to_node}`;
    case "cache_lookup":
      return `${event.outcome} · ${event.file_path}`;
    case "cache_serve":
      return `served · ${event.served_finding_count} finding(s) · ${event.file_path}`;
    case "scope_exclusion":
      return `${event.applied ? "applied" : "shadow"} · ${event.entries.length} scope(s) · ${event.file_path}`;
    case "observed_skip_shadow":
      // Show recorded coverage counts and whether the model skip was enforced.
      return `OBSERVED shadow · ${event.blockers.length}/${event.changed_regions.length} changed regions unbacked by a query${event.skip_enforced ? " · LLM skipped" : ""} · ${event.file_path}`;
    case "analyze_completed":
      return `${event.n_findings_emitted} finding(s)${event.n_findings_dropped_over_cap ? ` · ${event.n_findings_dropped_over_cap} over-cap dropped` : ""}`;
    case "synthesize_completed":
      return `${event.n_findings} finding(s)${event.n_findings_dropped_over_cap ? ` · ${event.n_findings_dropped_over_cap} over-cap dropped` : ""}`;
    default:
      return "";
  }
}

export function eventNode(event: AuditEvent): string | null {
  return "node_id" in event && typeof event.node_id === "string" ? event.node_id : null;
}

// Feed colors group related events; missing mappings use neutral "framing".
export const EV_FAMILY: Record<string, string> = {
  llm_call: "model",
  finding: "finding",
  analyze_response_rejected: "rejected",
  finding_proposal_rejected: "rejected",
  file_examination: "file",
  cache_lookup: "file",
  cache_serve: "file",
  scope_exclusion: "file",
  observed_skip_shadow: "file",
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

export function eventFamily(eventType: string): string {
  return EV_FAMILY[eventType] ?? "framing";
}

// Mark rows hidden by the replay diagnostic filter. This classifies visibility,
// not whether an event affected the review.
export function isDiagnosticEvent(event: AuditEvent): boolean {
  switch (event.event_type) {
    case "observed_skip_shadow":
      // Keep coverage diagnostics behind the toggle even when a skip was enforced.
      return true;
    case "cache_lookup":
      // Keep would_hit visible; shadow lookups can still call the model.
      return event.outcome === "miss";
    case "scope_exclusion":
      // Applied filtering stays visible; shadow exclusions are diagnostic.
      return !event.applied;
    default:
      return false;
  }
}
