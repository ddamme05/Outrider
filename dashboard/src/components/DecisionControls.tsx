import type { components } from "../api/schema";
import { type DecisionDraft, type Outcome, SEVERITIES } from "../lib/hitl";

type FindingView = components["schemas"]["FindingView"];

const OUTCOMES: Outcome[] = ["approve", "reject", "suppress", "severity_override"];
const OUTCOME_CLASS: Record<Outcome, string> = {
  approve: "sel-approve",
  reject: "sel-reject",
  suppress: "sel-suppress",
  severity_override: "sel-override",
};

// The detail view owns the draft and validates all gated findings before submit.
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
  const baselineSeverity = finding.severity.toLowerCase();
  const showReason = draft.outcome !== null && draft.outcome !== "approve";
  const showSeverityOverride = draft.outcome === "severity_override";
  const reasonLength = draft.reason.length;

  return (
    <div className="hitl">
      <div className="h-label">
        ★ Decision required <span className="req">· gates the PR</span>
      </div>

      <div className="outcome-row">
        {OUTCOMES.map((outcome) => (
          <button
            key={outcome}
            type="button"
            disabled={disabled}
            className={`outcome-btn ${draft.outcome === outcome ? OUTCOME_CLASS[outcome] : ""}`}
            aria-pressed={draft.outcome === outcome}
            onClick={() =>
              onChange({
                outcome: outcome,
                // Keep the reason; only severity_override retains a selected severity.
                reason: draft.reason,
                overrideSeverity: outcome === "severity_override" ? draft.overrideSeverity : null,
              })
            }
          >
            {outcome}
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
            onChange={(changeEvent) => onChange({ ...draft, reason: changeEvent.target.value })}
          />
          <div className={`counter ${reasonLength > 480 ? "warn" : ""}`}>{reasonLength} / 500</div>
        </div>
      ) : null}

      {showSeverityOverride ? (
        <div className="override-wrap show">
          <div className="field">
            <label>Original severity (read-only)</label>
            <div className="ro">{baselineSeverity}</div>
          </div>
          <div className="field">
            <label htmlFor={`override-${finding.finding_id}`}>Override to</label>
            <select
              id={`override-${finding.finding_id}`}
              aria-label="Override severity"
              disabled={disabled}
              value={draft.overrideSeverity ?? ""}
              onChange={(changeEvent) =>
                onChange({
                  ...draft,
                  overrideSeverity: changeEvent.target.value === "" ? null : changeEvent.target.value,
                })
              }
            >
              <option value="">select…</option>
              {SEVERITIES.map((severity) => (
                <option key={severity} value={severity} disabled={severity === baselineSeverity}>
                  {severity}
                  {severity === baselineSeverity ? " (current)" : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : null}
    </div>
  );
}
