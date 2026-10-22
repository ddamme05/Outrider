import type { components } from "../api/schema";

type FindingView = components["schemas"]["FindingView"];
type PerFindingDecisionPayload = components["schemas"]["PerFindingDecisionPayload"];

export type Outcome = "approve" | "reject" | "suppress" | "severity_override";

export const SEVERITIES = ["critical", "high", "medium", "low", "info"] as const;

// After an accepted submit, a still-pending review offers a same-payload retry
// once this resume window passes. Background resume may have failed (FUP-135).
export const RESUME_WINDOW_MS = 12000;

// Controls are actionable only at the current or expired approval gate.
// Old gated findings on running/completed/failed reviews stay read-only.
export function isActionable(status: string): boolean {
  return status === "awaiting_approval" || status === "awaiting_approval_expired";
}

// ReviewDetail uses the server approval snapshot for membership by finding_id
// (FUP-134); severity alone cannot identify the exact gated set.

// In-progress reviewer state for one finding. Final shape is built at submit.
export interface DecisionDraft {
  outcome: Outcome | null;
  reason: string;
  overrideSeverity: string | null;
}

export const EMPTY_DRAFT: DecisionDraft = { outcome: null, reason: "", overrideSeverity: null };

// Client-side submit checks cover a non-blank reason for non-approve outcomes
// and a changed severity for overrides. The server validates the full payload.
export function isDraftValid(draft: DecisionDraft, finding: FindingView): boolean {
  if (draft.outcome === null) return false;
  if (draft.outcome !== "approve" && draft.reason.trim() === "") return false;
  if (draft.outcome === "severity_override") {
    if (!draft.overrideSeverity) return false;
    if (draft.overrideSeverity === finding.severity.toLowerCase()) return false;
  }
  return true;
}

// Approve sends the required reason field as empty text. The server supplies
// original severity and reviewer identity; neither belongs in this payload.
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

// Mutation errors expose the parsed body, not the HTTP status. Recognize the
// gated-set mismatch shape; use a generic refresh message for other failures.
export function decideErrorMessage(error: unknown): string {
  const detail = (error as { detail?: unknown } | null)?.detail;
  if (detail && typeof detail === "object" && ("missing" in detail || "extras" in detail)) {
    return "Your decision set no longer matches the review's gated findings — refresh and try again.";
  }
  return "Couldn't submit — the review may already be decided or its status changed. Refresh and try again.";
}
