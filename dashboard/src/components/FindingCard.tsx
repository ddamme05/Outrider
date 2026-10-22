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
  tierPhrase,
  typeLabel,
} from "../lib/findingSections";
import { renderInlineCode } from "../lib/renderInlineCode";

type FindingView = components["schemas"]["FindingView"];

function findingLocation(finding: FindingView): string {
  return finding.line_start === finding.line_end
    ? `${finding.file_path}:${finding.line_start}`
    : `${finding.file_path}:${finding.line_start}-${finding.line_end}`;
}

// Wire destinations select badge styles; labels are humanized separately.
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

// Proof metadata survives content redaction.
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
          {finding.trace_path.map((segment, segmentIndex) => (
            <span key={`${segment}-${segmentIndex}`} style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
              <span className="seg">{segment}</span>
              {segmentIndex < finding.trace_path!.length - 1 ? (
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
  const severityClass = finding.severity.toLowerCase();
  const evidenceTier = finding.evidence_tier.toLowerCase();
  const dimension = finding.dimension;

  return (
    <div className={`finding ${wasGated ? "gated" : ""}`}>
      <div className="f-head">
        <span className={`pill sev-pill sev-${severityClass}`}>
          <span className="dot" aria-hidden="true" />
          {severityLabel(finding.severity)}
        </span>
        <span className={`tier tier-${evidenceTier}`} title={tierPhrase(finding.evidence_tier)}>
          <span className="dot" aria-hidden="true" />
          {evidenceTier.toUpperCase()}
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
          <span className={`dim-dot dim-${dimension}`} aria-hidden="true" />
          <span className={`dim-w-${dimension}`}>{dimensionLabel(dimension)}</span>
        </span>
        <span className="f-loc mono">{findingLocation(finding)}</span>
      </div>

      <FindingBody finding={finding} evidenceTier={evidenceTier} />

      {decision && onDecisionChange ? (
        <DecisionControls
          finding={finding}
          draft={decision}
          disabled={disabled ?? false}
          onChange={onDecisionChange}
        />
      ) : wasGated ? (
        // Preserve the gate history after the decision controls are gone.
        <div className="not-gated">
          <span className="ng">gated the PR · decided at review time</span>
        </div>
      ) : null}
    </div>
  );
}

function FindingBody({ finding, evidenceTier }: { finding: FindingView; evidenceTier: string }) {
  return (
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
          {finding.title ? (
            <div className="f-title">{renderInlineCode(finding.title)}</div>
          ) : null}
          {finding.description ? (
            <div className="f-desc">{renderInlineCode(finding.description)}</div>
          ) : null}
          {!finding.title && !finding.description ? <div className="f-desc">—</div> : null}
        </>
      )}

      <ProofBox finding={finding} tier={evidenceTier} />

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

      {/* Only an override decision can supply a severity change. */}
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
  );
}
