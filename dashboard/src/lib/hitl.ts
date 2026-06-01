import type { components } from "../api/schema";

type FindingView = components["schemas"]["FindingView"];
type PerFindingDecisionPayload = components["schemas"]["PerFindingDecisionPayload"];

export type Outcome = "approve" | "reject" | "suppress" | "severity_override";

export const SEVERITIES = ["critical", "high", "medium", "low", "info"] as const;

// After a 202 the graph resumes in the background and the status flips
// awaiting → running. If it hasn't advanced within this window, the resume may
// have failed (e.g. the natural-key-conflict stuck case) — the UI stops showing
// a bare "submitted" and offers refresh + re-submit. See FUP-135.
export const RESUME_WINDOW_MS = 12000;

// A review only accepts decisions while it sits at the HITL gate. The detail
// view can also show old completed/running/failed reviews that carried gated
// findings — those must be READ-ONLY, not controls that 409.
export function isActionable(status: string): boolean {
  return status === "awaiting_approval" || status === "awaiting_approval_expired";
}

// The gated set is the AUTHORITATIVE `ReviewDetail.findings_requiring_approval`
// from the server (FUP-134) — membership by finding_id, not inferred from
// severity. The decide endpoint enforces the exact same set.

// In-progress reviewer state for one finding. Final shape is built at submit.
export interface DecisionDraft {
  outcome: Outcome | null;
  reason: string;
  overrideSeverity: string | null;
}

export const EMPTY_DRAFT: DecisionDraft = { outcome: null, reason: "", overrideSeverity: null };

// Mirrors the server payload validator (PerFindingDecisionPayload): non-approve
// needs a non-blank reason; severity_override needs an override that differs
// from the finding's current (baseline) severity. Server stays authoritative;
// this only gates the submit button to avoid pointless 422 round-trips.
export function isDraftValid(draft: DecisionDraft, finding: FindingView): boolean {
  if (draft.outcome === null) return false;
  if (draft.outcome !== "approve" && draft.reason.trim() === "") return false;
  if (draft.outcome === "severity_override") {
    if (!draft.overrideSeverity) return false;
    if (draft.overrideSeverity === finding.severity.toLowerCase()) return false;
  }
  return true;
}

// Build one payload entry. approve still sends reason: "" (the field is required
// in the payload; only non-approve needs it non-blank). original_severity and
// reviewer_id are NOT sent — the server derives/sets them.
export function toPayload(draft: DecisionDraft, finding: FindingView): PerFindingDecisionPayload {
  const entry: PerFindingDecisionPayload = {
    finding_id: finding.finding_id,
    outcome: draft.outcome as Outcome,
    reason: draft.outcome === "approve" ? "" : draft.reason.trim(),
  };
  if (draft.outcome === "severity_override" && draft.overrideSeverity) {
    return { ...entry, override_severity: draft.overrideSeverity as PerFindingDecisionPayload["override_severity"] };
  }
  return entry;
}

// We can't read the HTTP status off an openapi-react-query mutation error (it
// throws only the parsed body), so classify on the body shape. The mismatch 422
// carries {detail:{missing,extras}}; everything else (409 conflict, etc.) gets
// the honest generic message.
export function decideErrorMessage(error: unknown): string {
  const detail = (error as { detail?: unknown } | null)?.detail;
  if (detail && typeof detail === "object" && ("missing" in detail || "extras" in detail)) {
    return "Your decision set no longer matches the review's gated findings — refresh and try again.";
  }
  return "Couldn't submit — the review may already be decided or its status changed. Refresh and try again.";
}
