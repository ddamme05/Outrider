import type { components } from "../api/schema";
import { DecisionControls } from "./DecisionControls";
import type { DecisionDraft } from "../lib/hitl";

type FindingView = components["schemas"]["FindingView"];

function loc(f: FindingView): string {
  return f.line_start === f.line_end
    ? `${f.file_path}:${f.line_start}`
    : `${f.file_path}:${f.line_start}-${f.line_end}`;
}

// One finding card. When `decision`/`onChange` are passed (review is at the HITL
// gate AND this finding is gated), the decision controls render at the bottom;
// otherwise the card is read-only. severity/tier values are lowercase per the
// policy enums, so they map straight to the sev-*/tier-* classes.
export function FindingCard({
  finding,
  wasGated,
  decision,
  disabled,
  onDecisionChange,
}: {
  finding: FindingView;
  // True iff this finding is in the review's authoritative HITL gated set.
  wasGated?: boolean;
  decision?: DecisionDraft;
  disabled?: boolean;
  onDecisionChange?: (next: DecisionDraft) => void;
}) {
  const sev = finding.severity.toLowerCase();
  const tier = finding.evidence_tier.toLowerCase();

  return (
    <div className="card finding">
      <div className="f-head">
        <span className={`pill sev-pill sev-${sev}`}>
          <span className="dot" aria-hidden="true" />
          {finding.severity}
        </span>
        <span className="f-type">{finding.finding_type}</span>
        <span className="f-loc mono">{loc(finding)}</span>
      </div>

      {finding.content_redacted ? (
        <div className="f-desc redacted">
          Content redacted
          {finding.redaction_sweep_at
            ? ` in the findings retention sweep on ${finding.redaction_sweep_at.slice(0, 10)}`
            : ""}
          . The finding's metadata (type, severity, location, proof) is permanent;
          its title/description/evidence/fix were purged per the retention policy.
        </div>
      ) : (
        <div className="f-desc">{finding.description ?? finding.title ?? "—"}</div>
      )}

      <div className="f-tags">
        <span className={`tier tier-${tier}`}>
          <span className="dot" aria-hidden="true" />
          {tier}
        </span>
        <span className="chip quiet">{finding.dimension}</span>
        {finding.publish_destination ? (
          <span className="dest">→ {finding.publish_destination}</span>
        ) : null}
        {finding.eligibility ? (
          <span className="dest">
            {finding.eligibility}
            {finding.eligibility_reason ? ` · ${finding.eligibility_reason}` : ""}
          </span>
        ) : null}
      </div>

      {!finding.content_redacted &&
      (finding.evidence || finding.query_match_id || finding.trace_path || finding.suggested_fix) ? (
        <details className="evidence">
          <summary>▸ Evidence</summary>

          {finding.evidence ? <pre className="evidence-pre">{finding.evidence}</pre> : null}

          {finding.query_match_id ? (
            <div className="proof">
              <div className="plabel">Proof · tree-sitter query match</div>
              <span className="pval mono">query_match_id = {finding.query_match_id}</span>
            </div>
          ) : null}

          {finding.trace_path && finding.trace_path.length > 0 ? (
            <div className="proof">
              <div className="plabel">Proof · ast_facts trace path</div>
              <div className="trace-chain">
                {finding.trace_path.map((seg, i) => (
                  <span key={`${seg}-${i}`} style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                    <span className="seg">{seg}</span>
                    {i < finding.trace_path!.length - 1 ? (
                      <span className="arr" aria-hidden="true">
                        →
                      </span>
                    ) : null}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {finding.suggested_fix ? (
            <div className="fix-note">
              <span className="flabel">Suggested fix · </span>
              {finding.suggested_fix}
            </div>
          ) : null}
        </details>
      ) : null}

      {decision && onDecisionChange ? (
        <DecisionControls
          finding={finding}
          draft={decision}
          disabled={disabled ?? false}
          onChange={onDecisionChange}
        />
      ) : wasGated ? (
        // Gated finding on a non-actionable review (e.g. an old completed one) —
        // show why there are no controls rather than rendering inert buttons.
        <div className="not-gated">
          <span className="ng">gated the PR · decided at review time</span>
        </div>
      ) : null}
    </div>
  );
}
