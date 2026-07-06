import type { components } from "../api/schema";
import { DecisionControls } from "./DecisionControls";
import { CodeBlock } from "./CodeBlock";
import type { DecisionDraft } from "../lib/hitl";
import {
  destLabel,
  dimensionLabel,
  eligibilityPhrase,
  eligibilityReasonPhrase,
  hitlOutcomeLabel,
  severityLabel,
  typeLabel,
} from "../lib/findingSections";

type FindingView = components["schemas"]["FindingView"];

function loc(f: FindingView): string {
  return f.line_start === f.line_end
    ? `${f.file_path}:${f.line_start}`
    : `${f.file_path}:${f.line_start}-${f.line_end}`;
}

// publish_destination → tag variant + dot color (mockup .dest-inline/.dest-review/
// .dest-dashboard). inline_comment posts on the diff (info), review_body rides the
// review summary (medium), dashboard_only never reaches GitHub (faint/dashed).
// Keys are the LOWERCASE wire values the `PublishDestination` StrEnum serializes
// (`inline_comment`/`review_body`/`dashboard_only`) — the API never sends uppercase;
// the label is HUMANIZED for display (destLabel: "Inline comment"), not the raw enum.
const DEST_CLASS: Record<string, string> = {
  inline_comment: "dest-inline",
  review_body: "dest-review",
  dashboard_only: "dest-dashboard",
};
const DEST_DOT: Record<string, string> = {
  inline_comment: "var(--sev-info)",
  review_body: "var(--sev-medium)",
  dashboard_only: "var(--sev-low)",
};

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
// render at the bottom; otherwise read-only. The head badges display HUMANIZED
// labels from lib/findingSections (the TS mirror of the Python presentation layer);
// the raw lowercase wire values still key the sev-*/tier-*/dim-*/dest-* CSS classes.
// Gated findings carry the sev-high left edge (.gated).
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
          {severityLabel(finding.severity)}
        </span>
        {/* tier acronym in the badge (OBSERVED/INFERRED/JUDGED); the full phrase
            lives in the proof box below. */}
        <span className={`tier tier-${tier}`}>
          <span className="dot" aria-hidden="true" />
          {tier.toUpperCase()}
        </span>
        {finding.publish_destination ? (
          <span className={`dest ${DEST_CLASS[finding.publish_destination] ?? ""}`}>
            <span
              className="tdot"
              aria-hidden="true"
              style={{ background: DEST_DOT[finding.publish_destination] ?? "var(--muted)" }}
            />
            {destLabel(finding.publish_destination)}
          </span>
        ) : null}
        <span className="ft-tag">
          {typeLabel(finding.finding_type)} ·{" "}
          <span className={`dim-dot dim-${dim}`} aria-hidden="true" />
          <span className={`dim-w-${dim}`}>{dimensionLabel(dim)}</span>
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
          <>
            {/* Lead with the title (the one-line summary), description below — matching
                the GitHub/Slack renderers, which both lead with the title. */}
            {finding.title ? <div className="f-title">{finding.title}</div> : null}
            {finding.description ? <div className="f-desc">{finding.description}</div> : null}
            {!finding.title && !finding.description ? <div className="f-desc">—</div> : null}
          </>
        )}

        <ProofBox finding={finding} tier={tier} />

        {!finding.content_redacted && finding.evidence ? (
          <CodeBlock code={finding.evidence} filePath={finding.file_path} />
        ) : null}

        {!finding.content_redacted && finding.suggested_fix ? (
          <div className="f-fix">
            <b>Suggested fix:</b> {finding.suggested_fix}
          </div>
        ) : null}

        {finding.eligibility ? (
          <div className="f-elig">
            {eligibilityPhrase(finding.eligibility)}
            {finding.eligibility_reason
              ? ` · ${eligibilityReasonPhrase(finding.eligibility_reason)}`
              : ""}
          </div>
        ) : null}

        {/* override provenance: present-or-absent as a unit (a finding never decided
            on has hitl_decision === null). The severity pair is non-null ONLY under
            outcome "severity_override" — guard before rendering the arrow. */}
        {finding.hitl_decision ? (
          <div className="f-prov">
            <span className="prov-k">HITL · {hitlOutcomeLabel(finding.hitl_decision.outcome)}</span>
            {finding.hitl_decision.outcome === "severity_override" &&
            finding.hitl_decision.original_severity &&
            finding.hitl_decision.override_severity ? (
              <span className="prov-sev mono">
                {severityLabel(finding.hitl_decision.original_severity)} →{" "}
                {severityLabel(finding.hitl_decision.override_severity)}
              </span>
            ) : null}
            <span className="prov-by">by {finding.hitl_decision.reviewer_id}</span>
            {finding.hitl_decision.reason ? (
              <span className="prov-reason"> · {finding.hitl_decision.reason}</span>
            ) : null}
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
