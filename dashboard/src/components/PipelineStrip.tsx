import type { components } from "../api/schema";
import { formatDurationMs, unionDurationMs } from "../lib/format";
import { prettyModel } from "../lib/modelLabel";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
// One reconstructed phase from the server's replay-VERIFIED grouping. A node can own MANY
// phases: analyze fans out per file (a plan phase + one worker phase per concurrent file +
// an aggregate phase, keyed by phase_key) and repeats across analyze⇄trace rounds. Per-node
// stats aggregate across all of a node's phases — cost sums; duration takes the interval
// union (concurrent worker spans overlap, so summing would multi-count wall-time).
type Phase = NonNullable<TimelineData["phases"]>[number];
type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];
type LLMCall = Extract<AuditEvent, { event_type: "llm_call" }>;
type TraceDec = Extract<AuditEvent, { event_type: "trace_decision" }>;

// The 7-node graph as the mockup's per-node cards. Per-node model/cost/timing/files/
// resolved come from the server's replay-VERIFIED reconstruction (`reconstruct().phases`,
// exposed by /replay-timeline only when the verdict is replay-equivalent — the FUP-125
// gate). The frontend does NOT re-group raw `review_phase` events: there is exactly one
// reconstruction surface (the server), rendered two ways (this strip + the timeline tab).
// NOTHING is fabricated: when `phases` is null (non-equivalent verdict or the timeline
// hasn't loaded), per-node stats fail closed to "—" — node states still derive from review
// status (a backed coarse inference), but no timing/cost/passed/posted is claimed.
const NODES = ["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"] as const;
type NodeName = (typeof NODES)[number];
type NodeState = "done" | "paused" | "skipped" | "pending" | "";

const STATIC_MODEL: Partial<Record<NodeName, string>> = {
  intake: "no LLM",
  hitl: "gate",
  publish: "no LLM",
};


export function PipelineStrip({
  status,
  phases,
  gatedCount,
  policyVersion,
}: {
  status: string;
  // The server's replay-verified phase grouping, or null when it isn't trustworthy:
  // a non-equivalent verdict (the FUP-125 gate suppresses it) or the timeline query
  // hasn't loaded. Null → per-node stats fail closed to "—"; node states stay
  // status-backed. This is the single reconstruction surface (no client re-grouping).
  phases: Phase[] | null;
  // Authoritative gated count from the review's findings_requiring_approval snapshot.
  // null = no snapshot (gate set unknown) — distinct from 0; never rendered as "0".
  gatedCount: number | null;
  policyVersion?: string | null;
}) {
  const phasesLoaded = phases !== null;
  const awaiting = status.startsWith("awaiting_approval");

  // All of a node's phases (analyze owns many: plan + per-file workers + aggregate, ×rounds).
  const phasesFor = (node: NodeName): Phase[] =>
    (phases ?? []).filter((p) => p.node_id === node);
  // The per-operation events recorded inside a node's phases, by type.
  const eventsIn = (node: NodeName): AuditEvent[] => phasesFor(node).flatMap((p) => p.events);

  const ended = (node: NodeName): boolean => phasesFor(node).some((p) => p.end != null);

  // Total wall-time the node held across its phases — the UNION of phase intervals, not the
  // sum. Parallel analyze overlaps concurrent worker phases (summing multi-counts), and its
  // phases straddle multiple analyze⇄trace passes (a single earliest→latest span would swallow
  // the trace gap between them). unionDurationMs handles both — see its docstring in format.ts.
  const durationMs = (node: NodeName): number | null =>
    unionDurationMs(
      phasesFor(node).map((p) => ({ start: p.start?.timestamp, end: p.end?.timestamp })),
    );

  const llmFor = (node: NodeName): { model: string; cost: number } | null => {
    const calls = eventsIn(node).filter((e): e is LLMCall => e.event_type === "llm_call");
    if (calls.length === 0) return null;
    // A node can call MORE than one model in a single review: analyze routes DEEP-tier files
    // to Sonnet and STANDARD-tier files to Haiku (tiered routing, DECISIONS.md#041), so a
    // mixed-tier review legitimately uses both. Show every DISTINCT model, sorted for a stable
    // label — fan-out completion order is nondeterministic, so keying off the last (or first)
    // call would flip the label run to run. Cost still sums across all calls.
    const models = [...new Set(calls.map((c) => prettyModel(c.model)))].sort();
    return {
      model: models.join(" + "),
      cost: calls.reduce((s, c) => s + c.cost_usd, 0),
    };
  };

  const stateOf = (node: NodeName): NodeState => {
    if (node === "hitl") {
      if (awaiting) return "paused";
      if (ended(node) || status === "completed") return "done";
      if (status === "failed") return "skipped";
      return "pending";
    }
    if (node === "publish") {
      if (ended(node) || status === "completed") return "done";
      if (status === "failed") return "skipped";
      return "pending";
    }
    // pre-hitl nodes: done if observed-ended, or status implies it (completed; or
    // awaiting → the graph already passed these to reach the gate). Status-backed.
    if (ended(node) || status === "completed" || awaiting) return "done";
    return "";
  };

  const modelOf = (node: NodeName, state: NodeState): string => {
    if (node in STATIC_MODEL) return STATIC_MODEL[node]!;
    // LLM model needs the verified reconstruction; without it we don't claim one.
    const m = phasesLoaded ? llmFor(node)?.model : undefined;
    if (m) return m;
    return state === "done" ? "—" : "";
  };

  const statOf = (node: NodeName, state: NodeState): string => {
    // The gate count is backed by the review's gated set, not the reconstruction;
    // null snapshot → just "paused", never a fabricated "0 findings".
    if (node === "hitl" && state === "paused") {
      return gatedCount === null ? "paused" : `paused · ${gatedCount} findings`;
    }
    if (state === "skipped") return "skipped";
    if (state === "pending") return "pending";
    // Per-node timing/cost/files/resolved require the verified reconstruction. Without
    // it we show "—" — never an unbacked "passed"/"posted"/timing/cost.
    if (!phasesLoaded) return "—";
    const dur = durationMs(node);
    const durStr = dur === null ? null : formatDurationMs(dur);
    if (node === "hitl" || node === "publish") return durStr ?? "—";
    if (state !== "done") return "—";
    const parts: string[] = [];
    if (durStr) parts.push(durStr);
    if (node === "intake") {
      const n = eventsIn("intake").filter((e) => e.event_type === "file_examination").length;
      if (n > 0) parts.push(`${n} files`);
    } else {
      const cost = llmFor(node)?.cost;
      if (cost !== undefined) parts.push(`$${cost.toFixed(2)}`);
      if (node === "trace") {
        const traces = eventsIn("trace").filter(
          (e): e is TraceDec => e.event_type === "trace_decision",
        );
        if (traces.length > 0) {
          const resolved = traces.filter((t) => t.resolution_status === "resolved").length;
          parts.push(`${resolved}/${traces.length} resolved`);
        }
      }
    }
    return parts.length > 0 ? parts.join(" · ") : "—";
  };

  return (
    <div className="panel">
      <div className="panel-h">
        <h2>Pipeline</h2>
        <div className="sub">7 nodes · analyze ⇄ trace ≤ 2{awaiting ? " · paused at hitl" : ""}</div>
      </div>
      <div className="panel-b">
        <div className="pipe" role="img" aria-label={`Pipeline state for a ${status} review`}>
          {NODES.map((node) => {
            const state = stateOf(node);
            return (
              <div className={`pnode ${state}`} key={node}>
                <span className="pn-name">
                  {state === "done" ? (
                    <span className="pn-check" aria-hidden="true">
                      ✓
                    </span>
                  ) : state === "paused" ? (
                    <span aria-hidden="true">⏸</span>
                  ) : null}
                  {node}
                </span>
                <span className="pn-model">{modelOf(node, state)}</span>
                <span className="pn-stat">{statOf(node, state)}</span>
              </div>
            );
          })}
        </div>
        <div className="pipe-note">
          {awaiting ? (
            gatedCount === null ? (
              <>HITL gate engaged — human approval required before publish. </>
            ) : (
              <>
                HITL gate engaged: {gatedCount} critical/high finding
                {gatedCount === 1 ? "" : "s"} require human approval before publish.{" "}
              </>
            )
          ) : phasesLoaded ? (
            <>Per-node model, cost and timing are from the replay-verified reconstruction. </>
          ) : (
            <>
              Node states reflect review status; per-node model, cost and timing load from the
              replay-verified timeline.{" "}
            </>
          )}
          <b>Severity is set by policy{policyVersion ? ` ${policyVersion}` : ""}, not the model.</b>
        </div>
      </div>
    </div>
  );
}
