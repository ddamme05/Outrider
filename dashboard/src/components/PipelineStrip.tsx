import type { components } from "../api/schema";

type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];
type LLMCall = Extract<AuditEvent, { event_type: "llm_call" }>;
type Phase = Extract<AuditEvent, { event_type: "review_phase" }>;
type FileExam = Extract<AuditEvent, { event_type: "file_examination" }>;
type TraceDec = Extract<AuditEvent, { event_type: "trace_decision" }>;

// The 7-node graph as the mockup's per-node cards. Per-node model/cost come from
// LLMCallEvent, wall-time from review_phase start/end markers, intake file count
// from file_examination, trace resolved/total from trace_decision — all from the
// audit stream the detail view already fetches. NOTHING is fabricated: a node
// whose datum isn't backed renders nothing for that datum (the card still shows
// what IS known); a node that hasn't run shows "pending"/"—".
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

function fmtDur(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
}

export function PipelineStrip({
  status,
  events,
  gatedCount,
  policyVersion,
}: {
  status: string;
  events: AuditEvent[];
  gatedCount: number;
  policyVersion?: string | null;
}) {
  const llm = events.filter((e): e is LLMCall => e.event_type === "llm_call");
  const phases = events.filter((e): e is Phase => e.event_type === "review_phase");
  const files = events.filter((e): e is FileExam => e.event_type === "file_examination");
  const traces = events.filter((e): e is TraceDec => e.event_type === "trace_decision");
  const awaiting = status.startsWith("awaiting_approval");

  const ended = (node: NodeName): boolean =>
    phases.some((p) => p.node_id === node && p.marker === "end");

  const durationMs = (node: NodeName): number | null => {
    const ns = phases.filter((p) => p.node_id === node);
    const start = ns.find((p) => p.marker === "start")?.timestamp;
    const end = ns.find((p) => p.marker === "end")?.timestamp;
    if (!start || !end) return null;
    const ms = new Date(end).getTime() - new Date(start).getTime();
    return Number.isFinite(ms) && ms >= 0 ? ms : null;
  };

  const llmFor = (node: NodeName): { model: string; cost: number } | null => {
    const calls = llm.filter((c) => c.node_id === node);
    const last = calls[calls.length - 1];
    if (!last) return null;
    return {
      model: prettyModel(last.model),
      cost: calls.reduce((s, c) => s + c.cost_usd, 0),
    };
  };

  const stateOf = (node: NodeName): NodeState => {
    const done = ended(node) || status === "completed";
    if (node === "hitl") {
      if (awaiting) return "paused";
      if (done) return "done";
      if (status === "failed") return "skipped";
      return "pending";
    }
    if (node === "publish") {
      if (ended(node) || status === "completed") return "done";
      if (status === "failed") return "skipped";
      return "pending";
    }
    return done ? "done" : "";
  };

  const modelOf = (node: NodeName, state: NodeState): string => {
    if (node in STATIC_MODEL) return STATIC_MODEL[node]!;
    const m = llmFor(node)?.model;
    if (m) return m;
    return state === "done" ? "—" : "";
  };

  const statOf = (node: NodeName, state: NodeState): string => {
    const dur = durationMs(node);
    const durStr = dur === null ? null : fmtDur(dur);
    if (node === "hitl") {
      if (state === "paused") return `paused · ${gatedCount} findings`;
      if (state === "done") return durStr ?? "passed";
      if (state === "skipped") return "skipped";
      return "pending";
    }
    if (node === "publish") {
      if (state === "done") return durStr ?? "posted";
      if (state === "skipped") return "skipped";
      return "pending";
    }
    if (state !== "done") return "—";
    const parts: string[] = [];
    if (durStr) parts.push(durStr);
    if (node === "intake") {
      const n = files.filter((f) => f.node_id === "intake").length;
      if (n > 0) parts.push(`${n} files`);
    } else {
      const cost = llmFor(node)?.cost;
      if (cost !== undefined) parts.push(`$${cost.toFixed(2)}`);
      if (node === "trace" && traces.length > 0) {
        const resolved = traces.filter((t) => t.resolution_status === "resolved").length;
        parts.push(`${resolved}/${traces.length} resolved`);
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
            <>
              HITL gate engaged: {gatedCount} critical/high finding
              {gatedCount === 1 ? "" : "s"} require human approval before publish.{" "}
            </>
          ) : (
            <>Per-node model, cost and timing are reconstructed from the audit stream. </>
          )}
          <b>Severity is set by policy{policyVersion ? ` ${policyVersion}` : ""}, not the model.</b>
        </div>
      </div>
    </div>
  );
}
