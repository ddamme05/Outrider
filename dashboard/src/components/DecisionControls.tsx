import type { components } from "../api/schema";
import { type DecisionDraft, type Outcome, SEVERITIES } from "../lib/hitl";

type FindingView = components["schemas"]["FindingView"];

const OUTCOMES: Outcome[] = ["approve", "reject", "suppress", "severity_override"];
const SEL_CLASS: Record<Outcome, string> = {
  approve: "sel-approve",
  reject: "sel-reject",
  suppress: "sel-suppress",
  severity_override: "sel-override",
};

// Controlled per-finding HITL controls. Pure — owns no state; the detail view
// holds the decisions map so the submit bar can gate on all gated findings.
export function DecisionControls({
  finding,
  draft,
  disabled,
  onChange,
}: {
  finding: FindingView;
  draft: DecisionDraft;
  disabled: boolean;
  onChange: (next: DecisionDraft) => void;
}) {
  const baseline = finding.severity.toLowerCase();
  const showReason = draft.outcome !== null && draft.outcome !== "approve";
  const showOverride = draft.outcome === "severity_override";
  const reasonLen = draft.reason.length;

  return (
    <div className="hitl">
      <div className="h-label">
        ★ Decision required <span className="req">· gates the PR</span>
      </div>

      <div className="outcome-row">
        {OUTCOMES.map((o) => (
          <button
            key={o}
            type="button"
            disabled={disabled}
            className={`outcome-btn ${draft.outcome === o ? SEL_CLASS[o] : ""}`}
            aria-pressed={draft.outcome === o}
            onClick={() =>
              onChange({
                outcome: o,
                // Clear override fields when leaving severity_override.
                reason: draft.reason,
                overrideSeverity: o === "severity_override" ? draft.overrideSeverity : null,
              })
            }
          >
            {o}
          </button>
        ))}
      </div>

      {showReason ? (
        <div className="reason-wrap show">
          <label htmlFor={`reason-${finding.finding_id}`}>
            Reason <span className="muted">(required, ≤500 chars)</span>
          </label>
          <textarea
            id={`reason-${finding.finding_id}`}
            maxLength={500}
            disabled={disabled}
            value={draft.reason}
            placeholder="Why this decision?"
            onChange={(e) => onChange({ ...draft, reason: e.target.value })}
          />
          <div className={`counter ${reasonLen > 480 ? "warn" : ""}`}>{reasonLen} / 500</div>
        </div>
      ) : null}

      {showOverride ? (
        <div className="override-wrap show">
          <div className="field">
            <label>Original severity (read-only)</label>
            <div className="ro">{baseline}</div>
          </div>
          <div className="field">
            <label htmlFor={`override-${finding.finding_id}`}>Override to</label>
            <select
              id={`override-${finding.finding_id}`}
              aria-label="Override severity"
              disabled={disabled}
              value={draft.overrideSeverity ?? ""}
              onChange={(e) =>
                onChange({ ...draft, overrideSeverity: e.target.value === "" ? null : e.target.value })
              }
            >
              <option value="">select…</option>
              {SEVERITIES.map((s) => (
                <option key={s} value={s} disabled={s === baseline}>
                  {s}
                  {s === baseline ? " (current)" : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : null}
    </div>
  );
}
