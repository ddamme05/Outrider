import type { components } from "../api/schema";

// The typed audit-event union, as returned by both /events and /replay-timeline (the
// same `AuditEvent` schema union). Shared by AuditFeed (flat table) + ReplayFeed.
export type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];

// A terse one-line summary per event type. Only fields confirmed on each narrowed type
// are read; unhandled types fall through to "" (the type + node badge still render).
// Display-only — values shown as audited, no recomputation (DECISIONS#014/#016 metadata).
export function summarizeEvent(e: AuditEvent): string {
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
    case "agent_transition":
      return `${e.from_node} → ${e.to_node}`;
    case "cache_lookup":
      return `${e.outcome} · ${e.file_path}`;
    case "cache_serve":
      return `served · ${e.served_finding_count} finding(s) · ${e.file_path}`;
    case "scope_exclusion":
      return `${e.applied ? "applied" : "shadow"} · ${e.entries.length} scope(s) · ${e.file_path}`;
    case "observed_skip_shadow":
      // OBSERVED-producer shadow telemetry: how many of the file's changed regions had NO
      // eligible query match backing a module-level OBSERVED claim (so the model's OBSERVED
      // citations there stay unproven). `not_eligible · 38/38` read as cryptic — spell it out.
      return `OBSERVED shadow · ${e.blockers.length}/${e.changed_regions.length} changed regions unbacked by a query${e.skip_enforced ? " · LLM skipped" : ""} · ${e.file_path}`;
    case "analyze_completed":
      return `${e.n_findings_emitted} finding(s)${e.n_findings_dropped_over_cap ? ` · ${e.n_findings_dropped_over_cap} over-cap dropped` : ""}`;
    case "synthesize_completed":
      return `${e.n_findings} finding(s)${e.n_findings_dropped_over_cap ? ` · ${e.n_findings_dropped_over_cap} over-cap dropped` : ""}`;
    default:
      return "";
  }
}

export function eventNode(e: AuditEvent): string | null {
  return "node_id" in e && typeof e.node_id === "string" ? e.node_id : null;
}

// event_type → color family (the --ev-* palette). Groups the discriminated event types
// into families so the feed/timeline reads at a glance; unmapped types fall back to the
// neutral "framing" family rather than going uncolored.
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

// Whether an event is pure telemetry (no causal effect on the review) vs signal. The analyze
// fan-out emits high-volume shadow/telemetry once per file; the feed hides these by default
// behind an explicit count + toggle so review-relevant rows stay readable. Payload-aware, NOT
// by-type: a cache `would_hit` and an `applied` scope exclusion ARE causal (in serve/enforcing
// mode) and must stay visible — only their misses/shadow counterparts are noise.
export function isDiagnosticEvent(e: AuditEvent): boolean {
  switch (e.event_type) {
    case "observed_skip_shadow":
      // The OBSERVED-producer shadow — always telemetry (never serves/enforces anything).
      return true;
    case "cache_lookup":
      // outcome is "would_hit" | "miss". A "would_hit" is causal under SERVE mode (the model
      // is skipped and a `cache_serve` follows); a "miss" always means "the model ran" — noise.
      return e.outcome === "miss";
    case "scope_exclusion":
      // applied=true actually dropped scopes from the prompt (causal); the shadow classifier
      // (applied=false) changes nothing.
      return !e.applied;
    default:
      return false;
  }
}
