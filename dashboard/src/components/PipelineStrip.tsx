import type { components } from "../api/schema";
import { formatDurationMs, unionDurationMs } from "../lib/format";
import { prettyModel } from "../lib/modelLabel";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
// A node can own several phases across parallel workers and analyze/trace rounds.
type Phase = NonNullable<TimelineData["phases"]>[number];
type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];
type LLMCall = Extract<AuditEvent, { event_type: "llm_call" }>;
type TraceDec = Extract<AuditEvent, { event_type: "trace_decision" }>;

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
  // Only replay-verified phases support node metrics. Null leaves metrics unknown.
  phases: Phase[] | null;
  // The approval snapshot supplies this count; null means unknown, not zero.
  gatedCount: number | null;
  policyVersion?: string | null;
}) {
  const phasesLoaded = phases !== null;
  const awaiting = status.startsWith("awaiting_approval");

  const phasesFor = (node: NodeName): Phase[] =>
    (phases ?? []).filter((phase) => phase.node_id === node);
  const eventsIn = (node: NodeName): AuditEvent[] => phasesFor(node).flatMap((phase) => phase.events);

  const ended = (node: NodeName): boolean => phasesFor(node).some((phase) => phase.end != null);

  // Union phase intervals: parallel workers overlap and trace gaps must stay excluded.
  const durationMs = (node: NodeName): number | null =>
    unionDurationMs(
      phasesFor(node).map((phase) => ({ start: phase.start?.timestamp, end: phase.end?.timestamp })),
    );

  const llmFor = (node: NodeName): { model: string; cost: number; costComplete: boolean } | null => {
    const calls = eventsIn(node).filter((event): event is LLMCall => event.event_type === "llm_call");
    if (calls.length === 0) return null;
    // Mixed-model workers finish in any order; sorted distinct names keep the label stable.
    const models = [...new Set(calls.map((call) => prettyModel(call.model)))].sort();
    // Unpriced calls make the sum a lower bound; null must not become a free call.
    const pricedCosts = calls
      .map((call) => call.cost_usd)
      .filter((cost): cost is number => cost != null);
    return {
      model: models.join(" + "),
      cost: pricedCosts.reduce((totalCost, cost) => totalCost + cost, 0),
      costComplete: pricedCosts.length === calls.length,
    };
  };

  const stateOf = (node: NodeName): NodeState => {
    if (node === "hitl" || node === "publish") {
      if (node === "hitl" && awaiting) return "paused";
      if (ended(node) || status === "completed") return "done";
      if (status === "failed") return "skipped";
      return "pending";
    }
    // Reaching the approval gate implies the earlier nodes finished.
    if (ended(node) || status === "completed" || awaiting) return "done";
    return "";
  };

  const modelOf = (node: NodeName, state: NodeState): string => {
    if (node in STATIC_MODEL) return STATIC_MODEL[node]!;
    // LLM model needs the verified reconstruction; without it we don't claim one.
    const modelLabel = phasesLoaded ? llmFor(node)?.model : undefined;
    if (modelLabel) return modelLabel;
    return state === "done" ? "—" : "";
  };

  const statOf = (node: NodeName, state: NodeState): string => {
    if (node === "hitl" && state === "paused") {
      return gatedCount === null ? "paused" : `paused · ${gatedCount} findings`;
    }
    if (state === "skipped") return "skipped";
    if (state === "pending") return "pending";
    // Review status alone cannot establish timing, cost, or event counts.
    if (!phasesLoaded) return "—";
    const duration = durationMs(node);
    const durationLabel = duration === null ? null : formatDurationMs(duration);
    if (node === "hitl" || node === "publish") return durationLabel ?? "—";
    if (state !== "done") return "—";
    const parts: string[] = [];
    if (durationLabel) parts.push(durationLabel);
    if (node === "intake") {
      const fileCount = eventsIn("intake").filter(
        (event) => event.event_type === "file_examination",
      ).length;
      if (fileCount > 0) parts.push(`${fileCount} files`);
    } else {
      const llmSummary = llmFor(node);
      if (llmSummary) parts.push(`${llmSummary.costComplete ? "" : "\u2265"}$${llmSummary.cost.toFixed(2)}`);
      if (node === "trace") {
        const traceDecisions = eventsIn("trace").filter(
          (event): event is TraceDec => event.event_type === "trace_decision",
        );
        if (traceDecisions.length > 0) {
          const resolvedCount = traceDecisions.filter(
            (traceDecision) => traceDecision.resolution_status === "resolved",
          ).length;
          parts.push(`${resolvedCount}/${traceDecisions.length} resolved`);
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
            const nodeState = stateOf(node);
            return (
              <div className={`pnode ${nodeState}`} key={node}>
                <span className="pn-name">
                  {nodeState === "done" ? (
                    <span className="pn-check" aria-hidden="true">
                      ✓
                    </span>
                  ) : nodeState === "paused" ? (
                    <span aria-hidden="true">⏸</span>
                  ) : null}
                  {node}
                </span>
                <span className="pn-model">{modelOf(node, nodeState)}</span>
                <span className="pn-stat">{statOf(node, nodeState)}</span>
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
