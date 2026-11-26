import type { components } from "../api/schema";
import { LLMContentPanel } from "./ReplayFeed";

type Timeline = components["schemas"]["ReplayTimelineResponse"];

export function OriginalFileAnalysis({ data, cacheKey }: { data: Timeline; cacheKey: string }) {
  // Concurrent file calls can finish in any order. Use the cache lookup recorded
  // in this review to identify the worker that produced the original result.
  const lookup = data.events.find(
    (event) => event.event_type === "cache_lookup" && event.cache_key === cacheKey,
  );
  if (!data.replay_equivalent || lookup?.event_type !== "cache_lookup" || !lookup.phase_key) {
    return <p className="queue-notice">The original file analysis could not be identified in the verified timeline.</p>;
  }
  const calls = data.events.filter(
    (event) => event.event_type === "llm_call" && event.node_id === "analyze" && event.phase_key === lookup.phase_key,
  );
  return (
    <section className="panel">
      <div className="panel-h"><h2>Original analysis of {lookup.file_path}</h2></div>
      <div className="panel-b">
        {calls.length === 0 ? <p>No model call was recorded for this file analysis.</p> : null}
        {calls.map((call) => {
          const content = data.llm_exchanges.find((entry) => entry.event_id === call.event_id);
          return (
            <details key={call.event_id}>
              <summary>Model exchange for {lookup.file_path}</summary>
              {content ? <LLMContentPanel content={content} /> : <p>No saved exchange is available for this call.</p>}
            </details>
          );
        })}
      </div>
    </section>
  );
}
