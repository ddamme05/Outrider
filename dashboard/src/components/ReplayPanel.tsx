import type { components } from "../api/schema";

type ReplayVerdict = components["schemas"]["ReplayVerdict"];

// The one fully contract-backed rich surface: every field shown comes straight
// from ReplayVerdict. `mode`/counts are null only when reconstruct() itself
// raised (corrupt row / is_eval drift) — then `reason` carries why.
export function ReplayPanel({ verdict }: { verdict: ReplayVerdict }) {
  const { replay_equivalent, mode, event_count, finding_count, orphan_finding_count, reason } =
    verdict;

  const meta: string[] = [];
  if (event_count !== null) meta.push(`${event_count} events`);
  if (finding_count !== null) meta.push(`${finding_count} findings`);
  if (orphan_finding_count) meta.push(`${orphan_finding_count} orphaned`);
  if (mode !== null) meta.push(mode);

  return (
    <div className="card replay-pop">
      <span className={`replay-result ${replay_equivalent ? "" : "fail"}`}>
        <span aria-hidden="true">{replay_equivalent ? "✓" : "✗"}</span>
        {replay_equivalent ? "replay-equivalent" : "not replay-equivalent"}
        {meta.length > 0 ? <span className="meta">{meta.join(" · ")}</span> : null}
      </span>
      {!replay_equivalent && reason ? <div className="replay-reason mono">{reason}</div> : null}
      <div className="replay-note">
        {mode === "metadata_only"
          ? "Metadata-only replay: the event sequence is permanent; content fields are redacted past the retention window. Severity is recomputed from the versioned policy — never model-set."
          : "Full reconstruction (LLM exchanges + finding content) within the retention window. Severity is recomputed from the versioned policy — never model-set."}
      </div>
    </div>
  );
}
