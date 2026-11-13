import { useState } from "react";
import { Link } from "react-router";

import { $api } from "../api/client";
import type { components } from "../api/schema";

type CandidateView = components["schemas"]["CandidateView"];
type Assessment = components["schemas"]["CandidateAssessmentRef"];

const HEAD_LABELS: Record<NonNullable<Assessment["head_assessment"]>, string> = {
  supports: "supported",
  refutes: "refuted",
  uncertain: "uncertain",
};

const IMPACT_LABELS: Record<NonNullable<Assessment["impact_assessment"]>, string> = {
  supports: "supported",
  refutes: "refuted",
  uncertain: "unproved",
  not_claimed: "no broader impact claimed",
};

const CHANGE_LABELS: Record<NonNullable<Assessment["change_assessment"]>, string> = {
  introduced_or_worsened: "introduced or worsened by this change",
  fix_attempt: "this PR attempts a fix",
  unrelated_preexisting: "existing concern unrelated to this change",
  uncertain: "uncertain",
};

const REPAIR_LABELS: Record<NonNullable<Assessment["repair_assessment"]>, string> = {
  supports: "supported",
  refutes: "refuted",
  uncertain: "uncertain",
  not_proposed: "no repair proposed",
};

const DRAFT_LABELS: Record<CandidateView["draft_disposition"], string> = {
  included_in_draft: "Included in the draft",
  selected_in_analysis: "Selected during analysis",
  duplicate: "Another warning for this location and category was selected",
  held_by_limit: "Held back by the finding limit",
  needs_classification: "Needs classification",
  unqualified: "Did not pass validation",
  malformed: "Could not read this warning",
  not_recorded: "Draft selection was not recorded",
};

function SavedCandidate({ item, reviewId }: { item: CandidateView; reviewId: string }) {
  const original = item.original;
  const assessment = item.assessment;
  return (
    <article className="finding">
      <h3>{original?.title ?? "Saved model warning"}</h3>
      <p>
        {item.candidate.file_path}
        {original ? `:${original.line_start}–${original.line_end}` : ""}
        {item.candidate.source_side === "base" ? " · Removed BASE (absent from HEAD)" : ""}
        {" · "}{DRAFT_LABELS[item.draft_disposition]}
      </p>
      {item.content_redacted ? (
        <p className="muted">Original content is no longer available.</p>
      ) : original ? (
        <>
          <p>{original.description}</p>
          <p className="muted">
            Reported category: {original.finding_type?.replaceAll("_", " ") ?? "unclassified"}
            {item.candidate.severity ? ` · original priority: ${item.candidate.severity}` : ""}
          </p>
          <pre>{original.evidence}</pre>
        </>
      ) : item.raw_preview ? (
        <details>
          <summary>Inspect the original entry</summary>
          <pre>{item.raw_preview}</pre>
          {item.preview_truncated ? <p>Preview shortened.</p> : null}
        </details>
      ) : null}
      {assessment ? (
        <div>
          <p>Follow-up check: {assessment.status} · {assessment.outcome.replaceAll("_", " ")}</p>
          {assessment.head_assessment && assessment.change_assessment ? (
            <p>
              Local bug: {HEAD_LABELS[assessment.head_assessment]}
              {assessment.impact_assessment ? (
                <> · Claimed impact: {IMPACT_LABELS[assessment.impact_assessment]}</>
              ) : null}
              {" · "}PR relation: {CHANGE_LABELS[assessment.change_assessment]}.
            </p>
          ) : null}
          <p>
            Proposed repair: {assessment.repair_assessment
              ? REPAIR_LABELS[assessment.repair_assessment]
              : "not assessed"}.
          </p>
          {item.assessment_reason ? <p>{item.assessment_reason}</p> : null}
          {item.assessment_content_redacted ? (
            <p className="muted">The check's explanation is no longer available.</p>
          ) : null}
          {assessment.suggested_finding_type ? (
            <p>
              Category suggestion awaiting review: {assessment.suggested_finding_type.replaceAll("_", " ")}
              {assessment.suggested_severity ? ` (${assessment.suggested_severity})` : ""}.
            </p>
          ) : null}
        </div>
      ) : <p className="muted">No follow-up check recorded.</p>}
      <Link to={`/reviews/${reviewId}/replay`}>Inspect the saved review history</Link>
    </article>
  );
}

export function CandidatePanel({ reviewId }: { reviewId: string }) {
  const [expanded, setExpanded] = useState(false);
  const [offset, setOffset] = useState(0);
  const limit = 20;
  const query = $api.useQuery(
    "get", "/api/reviews/{review_id}/candidates",
    { params: { path: { review_id: reviewId }, query: { limit, offset } } },
    { enabled: expanded },
  );
  const data = query.data;
  return (
    <section className="panel">
      <div className="panel-h">
        <h2>Saved model warnings</h2>
        <button className="btn" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
          {expanded ? "Hide saved warnings" : "Show saved warnings"}
        </button>
      </div>
      {expanded ? (
        <div className="panel-b">
          <p>
            Original warnings stay visible when they are held back or need classification.
            Follow-up checks are assessments for review; they do not rewrite the original claim.
          </p>
          {query.isLoading ? <p>Loading saved warnings…</p> : null}
          {query.error ? <p className="error">Could not load saved warnings.</p> : null}
          {data?.total === 0 ? <p>No saved individual warnings are available for this review.</p> : null}
          {data?.candidates.map((item) => (
            <SavedCandidate key={item.candidate.candidate_id} item={item} reviewId={reviewId} />
          ))}
          {data ? (
            <div>
              <p>{data.total} saved warnings · page {Math.floor(offset / limit) + 1}</p>
              <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
                Previous
              </button>
              <button className="btn" disabled={offset + limit >= data.total} onClick={() => setOffset(offset + limit)}>
                Next
              </button>
              <button className="btn" disabled={query.isFetching} onClick={() => void query.refetch()}>
                Refresh saved warnings
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
