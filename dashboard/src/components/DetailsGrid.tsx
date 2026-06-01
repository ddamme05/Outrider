import type { components } from "../api/schema";

type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];
type LLMCallEvent = Extract<AuditEvent, { event_type: "llm_call" }>;

interface NodeAgg {
  node: string;
  model: string;
  cost: number;
  input: number;
  output: number;
  calls: number;
}

// Per-node breakdown derived from the audit stream (FUP-133). Everything here is
// event-backed: cost/tokens/model/calls from LLMCallEvent, files-examined and
// trace-decision counts from their events. Triage per-file tiers are deliberately
// NOT shown — no audit event carries them (would require a new event type), so we
// omit rather than fabricate.
export function DetailsGrid({ events }: { events: AuditEvent[] }) {
  const llm = events.filter((e): e is LLMCallEvent => e.event_type === "llm_call");
  if (llm.length === 0) {
    return <p className="details-empty">No per-node LLM activity recorded for this review.</p>;
  }

  const byNode = new Map<string, NodeAgg>();
  for (const e of llm) {
    const agg = byNode.get(e.node_id) ?? {
      node: e.node_id,
      model: e.model,
      cost: 0,
      input: 0,
      output: 0,
      calls: 0,
    };
    agg.cost += e.cost_usd;
    agg.input += e.input_tokens;
    agg.output += e.output_tokens;
    agg.calls += 1;
    agg.model = e.model;
    byNode.set(e.node_id, agg);
  }
  const rows = [...byNode.values()];
  const filesExamined = events.filter((e) => e.event_type === "file_examination").length;
  const traceDecisions = events.filter((e) => e.event_type === "trace_decision").length;

  return (
    <div className="details-grid-wrap">
      <table className="details-grid">
        <thead>
          <tr>
            <th>Node</th>
            <th>Model</th>
            <th>Calls</th>
            <th>Tokens (in / out)</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.node}>
              <td className="mono">{r.node}</td>
              <td className="mono">{r.model}</td>
              <td className="mono num">{r.calls}</td>
              <td className="mono num">
                {r.input} / {r.output}
              </td>
              <td className="mono num">${r.cost.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="details-foot">
        <span className="mono">{filesExamined}</span> files examined ·{" "}
        <span className="mono">{traceDecisions}</span> trace decisions. Triage per-file tiers
        aren't carried in the audit stream, so they're not shown here.
      </p>
    </div>
  );
}
