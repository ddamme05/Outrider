// The 7-node graph as a slim line. State is COARSE — derived from the review's
// `status` only; per-node event-level detail lives in the Audit feed tab + the
// Per-node details grid (both event-backed, FUP-133). We mark a node
// "done"/"paused" only where `status` guarantees it; everything else stays
// neutral. No trace round-count badge (the detail contract doesn't carry it —
// see the mockup's "⇄ ×1", which is fabricated and deliberately omitted here).
const NODES = ["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"] as const;

type NodeState = "done" | "paused" | "pending" | "";

function nodeStates(status: string): Record<string, NodeState> {
  const blank = Object.fromEntries(NODES.map((n) => [n, ""])) as Record<string, NodeState>;
  if (status === "completed") {
    return Object.fromEntries(NODES.map((n) => [n, "done"])) as Record<string, NodeState>;
  }
  if (status === "awaiting_approval" || status === "awaiting_approval_expired") {
    return {
      ...blank,
      intake: "done",
      triage: "done",
      analyze: "done",
      trace: "done",
      synthesize: "done",
      hitl: "paused",
      publish: "pending",
    };
  }
  // running / failed / skipped / unknown: we can't know the live node, so neutral.
  return blank;
}

export function PipelineStrip({ status }: { status: string }) {
  const states = nodeStates(status);
  return (
    <>
      <div className="pipeline-line" role="img" aria-label={`Pipeline state for a ${status} review`}>
        {NODES.map((node, i) => (
          <span key={node} style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
            <span className={`node ${states[node]}`}>
              <span className="nd" aria-hidden="true" />
              {node}
              {states[node] === "paused" ? " PAUSED" : ""}
            </span>
            {i < NODES.length - 1 ? (
              <span className="arr" aria-hidden="true">
                →
              </span>
            ) : null}
          </span>
        ))}
      </div>
      <p className="pipeline-caption">
        Reflects review status, not live per-node state — see the Audit feed tab and
        Per-node details for the event-level breakdown.
      </p>
    </>
  );
}
