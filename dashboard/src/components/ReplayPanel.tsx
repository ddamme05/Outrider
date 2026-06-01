import type { components } from "../api/schema";

type ReplayVerdict = components["schemas"]["ReplayVerdict"];

const SEVERITY_NOTE = " Severity is recomputed from the versioned policy — never model-set.";

// One note per ReplayMode (full | metadata_only | mixed), plus null when
// reconstruct() itself failed. Never collapse mixed/null into "full".
function modeNote(mode: string | null): string {
  switch (mode) {
    case "full":
      return (
        "Full reconstruction (LLM exchanges + finding content) within the retention window." +
        SEVERITY_NOTE
      );
    case "metadata_only":
      return (
        "Metadata-only replay: the event sequence is permanent; content fields are redacted past " +
        "the retention window." +
        SEVERITY_NOTE
      );
    case "mixed":
      return (
        "Mixed reconstruction: some content survives and some was purged at its retention TTL — " +
        "each item is labeled, never silently hybridized." +
        SEVERITY_NOTE
      );
    default:
      return "Reconstruction did not complete — no mode could be determined (corrupt row, payload, or is_eval drift). See the reason above.";
  }
}

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
        {modeNote(mode)}
      </div>
    </div>
  );
}
