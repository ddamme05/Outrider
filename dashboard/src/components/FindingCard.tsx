import type { components } from "../api/schema";
import { DecisionControls } from "./DecisionControls";
import type { DecisionDraft } from "../lib/hitl";

type FindingView = components["schemas"]["FindingView"];

function loc(f: FindingView): string {
  return f.line_start === f.line_end
    ? `${f.file_path}:${f.line_start}`
    : `${f.file_path}:${f.line_start}-${f.line_end}`;
}

// The proof box: tier-keyed, always visible (mockup .f-proof). OBSERVED shows the
// query_match_id, INFERRED the trace_path chain, JUDGED the model-interpretation
// note. Proof metadata is permanent — it renders even when content is redacted.
function ProofBox({ finding, tier }: { finding: FindingView; tier: string }) {
  if (tier === "observed" && finding.query_match_id) {
    return (
      <div className="f-proof">
        <span className="pk">proof · observed → query_match_id</span>
        <code>{finding.query_match_id}</code>
      </div>
    );
  }
  if (tier === "inferred" && finding.trace_path && finding.trace_path.length > 0) {
    return (
      <div className="f-proof inferred">
        <span className="pk">proof · inferred → trace_path</span>
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
    );
  }
  if (tier === "judged") {
    return (
      <div className="f-proof judged">
        <span className="pk">proof · judged → model interpretation</span>
        No structural match — the model&rsquo;s judgment. Confidence is derived from the tier, not
        model-set.
      </div>
    );
  }
  return null;
}

// One finding card (mockup-faithful). When `decision`/`onDecisionChange` are passed
// (review is at the HITL gate AND this finding is gated), the decision controls
// render at the bottom; otherwise read-only. severity/tier/dimension values are
// lowercase per the policy enums, so they map straight to the sev-*/tier-*/dim-*
// classes. Gated findings carry the sev-high left edge (.gated).
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
  const dim = finding.dimension;

  return (
    <div className={`finding ${wasGated ? "gated" : ""}`}>
      <div className="f-head">
        <span className={`pill sev-pill sev-${sev}`}>
          <span className="dot" aria-hidden="true" />
          {finding.severity}
        </span>
        <span className={`tier tier-${tier}`}>
          <span className="dot" aria-hidden="true" />
          {tier}
        </span>
        {finding.publish_destination ? (
          <span className="dest">{finding.publish_destination}</span>
        ) : null}
        <span className="ft-tag">
          {finding.finding_type} ·{" "}
          <span className={`dim-dot dim-${dim}`} aria-hidden="true" />
          <span className={`dim-w-${dim}`}>{dim}</span>
        </span>
        <span className="f-loc mono">{loc(finding)}</span>
      </div>

      <div className="f-body">
        {finding.content_redacted ? (
          <div className="f-desc redacted">
            Content redacted
            {finding.redaction_sweep_at
              ? ` in the findings retention sweep on ${finding.redaction_sweep_at.slice(0, 10)}`
              : ""}
            . The finding&rsquo;s metadata (type, severity, location, proof) is permanent; its
            title/description/evidence/fix were purged per the retention policy.
          </div>
        ) : (
          <div className="f-desc">{finding.description ?? finding.title ?? "—"}</div>
        )}

        <ProofBox finding={finding} tier={tier} />

        {!finding.content_redacted && finding.evidence ? (
          <div className="f-evidence">{finding.evidence}</div>
        ) : null}

        {!finding.content_redacted && finding.suggested_fix ? (
          <div className="f-fix">
            <b>Suggested fix:</b> {finding.suggested_fix}
          </div>
        ) : null}

        {finding.eligibility ? (
          <div className="f-elig">
            {finding.eligibility}
            {finding.eligibility_reason ? ` · ${finding.eligibility_reason}` : ""}
          </div>
        ) : null}
      </div>

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
