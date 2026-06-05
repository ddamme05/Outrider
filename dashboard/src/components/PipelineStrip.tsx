import type { components } from "../api/schema";
import { formatDurationMs, spanMs } from "../lib/format";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
// One reconstructed phase from the server's replay-VERIFIED grouping. A node can own
// more than one phase (analyze runs once per analyze⇄trace round), so per-node stats
// aggregate across all of a node's phases.
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

function prettyModel(model: string): string {
  const l = model.toLowerCase();
  if (l.includes("haiku")) return "Haiku";
  if (l.includes("sonnet")) return "Sonnet";
  if (l.includes("opus")) return "Opus";
  return model;
}

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

  // All of a node's phases (analyze may own >1 across analyze⇄trace rounds).
  const phasesFor = (node: NodeName): Phase[] =>
    (phases ?? []).filter((p) => p.node_id === node);
  // The per-operation events recorded inside a node's phases, by type.
  const eventsIn = (node: NodeName): AuditEvent[] => phasesFor(node).flatMap((p) => p.events);

  const ended = (node: NodeName): boolean => phasesFor(node).some((p) => p.end != null);

  // Total wall-time the node held across its phases (sum of each closed phase's span).
  const durationMs = (node: NodeName): number | null => {
    let total = 0;
    let any = false;
    for (const p of phasesFor(node)) {
      const ms = spanMs(p.start?.timestamp, p.end?.timestamp);
      if (ms !== null) {
        total += ms;
        any = true;
      }
    }
    return any ? total : null;
  };

  const llmFor = (node: NodeName): { model: string; cost: number } | null => {
    const calls = eventsIn(node).filter((e): e is LLMCall => e.event_type === "llm_call");
    const last = calls[calls.length - 1];
    if (!last) return null;
    return {
      model: prettyModel(last.model),
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
