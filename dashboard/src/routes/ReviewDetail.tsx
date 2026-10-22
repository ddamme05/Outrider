import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import type { components } from "../api/schema";
import { FindingCard } from "../components/FindingCard";
import { PipelineStrip } from "../components/PipelineStrip";
import { PolicyModal } from "../components/PolicyModal";
import { ReplayFeed } from "../components/ReplayFeed";
import { ReplayInfoModal } from "../components/ReplayInfoModal";
import { StatusPill } from "../components/StatusPill";
import { useDemoStatus } from "../lib/demo";
import { expiresLabel, formatReviewCost } from "../lib/format";
import { SEVERITY_ORDER } from "../lib/metrics";
import {
  type DecisionDraft,
  EMPTY_DRAFT,
  RESUME_WINDOW_MS,
  decideErrorMessage,
  isActionable,
  isDraftValid,
  toPayload,
} from "../lib/hitl";

// Null means synthesis has not recorded a duration yet.
function fmtDuration(seconds: number | null): string {
  if (seconds === null) return "pending";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m${Math.round(seconds % 60)}s`;
}

export function ReviewDetail() {
  const { reviewId } = useParams<{ reviewId: string }>();

  const enabled = typeof reviewId === "string" && reviewId.length > 0;
  const pathParams = { params: { path: { review_id: reviewId ?? "" } } };

  const detailQuery = $api.useQuery("get", "/api/reviews/{review_id}", pathParams, {
    enabled,
    refetchInterval: 2000,
  });
  const findingsQuery = $api.useQuery("get", "/api/reviews/{review_id}/findings", pathParams, {
    enabled,
  });
  // The server verifies and groups the timeline used by the verdict, pipeline, and feed.
  const timelineQuery = $api.useQuery("get", "/api/reviews/{review_id}/replay-timeline", pathParams, {
    enabled,
    refetchInterval: 2000,
  });
  // The raw count includes the projected verdict event omitted from the timeline.
  const eventsQuery = $api.useQuery("get", "/api/reviews/{review_id}/events", pathParams, {
    enabled,
    refetchInterval: 2000,
  });

  // Hooks must run unconditionally, before the early returns below.
  const queryClient = useQueryClient();
  const [showPolicy, setShowPolicy] = useState(false);
  const [showReplayInfo, setShowReplayInfo] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, DecisionDraft>>({});
  const [submitted, setSubmitted] = useState(false);
  const decisionMutation = $api.useMutation("post", "/reviews/{review_id}/decide");

  const [submitSeq, setSubmitSeq] = useState(0);
  const [stuck, setStuck] = useState(false);
  // Retries must resend the accepted payload: changed decisions conflict with its
  // audit record. Editing stays locked until a reload gives us fresh state.
  const [submittedPayload, setSubmittedPayload] = useState<ReturnType<typeof toPayload>[] | null>(
    null,
  );

  // An accepted submit starts a new window for detecting a stuck background resume.
  useEffect(() => {
    if (submitSeq === 0) return;
    setStuck(false);
    const resumeTimer = setTimeout(() => setStuck(true), RESUME_WINDOW_MS);
    return () => clearTimeout(resumeTimer);
  }, [submitSeq]);

  // Copy before sorting so cached data stays untouched; ties keep server order.
  const sortedFindings = useMemo(() => {
    const rank = (severity: string) => {
      const severityRank = SEVERITY_ORDER.indexOf(severity as (typeof SEVERITY_ORDER)[number]);
      return severityRank === -1 ? SEVERITY_ORDER.length : severityRank;
    };
    return [...(findingsQuery.data?.findings ?? [])].sort(
      (leftFinding, rightFinding) => rank(leftFinding.severity) - rank(rightFinding.severity),
    );
  }, [findingsQuery.data]);
  const actionable = isActionable(detailQuery.data?.status ?? "");
  // Demo and unresolved deployments lock controls without changing a finding's
  // gated label. Only confirmed production can submit decisions.
  const demoStatus = useDemoStatus();
  const isDemo = demoStatus === "demo";
  const productionConfirmed = demoStatus === "production";
  // A metadata outage needs an explicit message while controls stay locked.
  const demoChecking = demoStatus === "loading";
  // The server's approval snapshot determines gating, not finding severity.
  const gatedFindingIds = useMemo(
    () => new Set(detailQuery.data?.findings_requiring_approval ?? []),
    [detailQuery.data],
  );
  const gatedFindings = useMemo(
    () => (actionable ? sortedFindings.filter((finding) => gatedFindingIds.has(finding.finding_id)) : []),
    [actionable, sortedFindings, gatedFindingIds],
  );

  if (!enabled) {
    return <p className="error">No review id in the URL.</p>;
  }
  // Background failures keep the last loaded state and show a banner.
  if (detailQuery.isLoading) {
    return <p>Loading…</p>;
  }
  if (detailQuery.error && !detailQuery.data) {
    // The parsed error body doesn't tell us whether this was a 404 or a server failure.
    return (
      <p className="error">Couldn't load this review — it may not exist, or the request failed.</p>
    );
  }

  const review = detailQuery.data;
  if (!review) {
    return <p className="error">Review not found.</p>;
  }

  const expires = review.status.startsWith("awaiting_approval") ? expiresLabel(review.expires_at) : null;
  // Unknown counts stay visibly unknown; unavailable data does not mean zero.
  const eventsLoaded = eventsQuery.data !== undefined;
  const eventCountLabel = eventsLoaded ? String(eventsQuery.data.total) : eventsQuery.error ? "—" : "…";
  const findingsLoaded = findingsQuery.data !== undefined;
  const findingCountLabel = findingsLoaded
    ? String(sortedFindings.length)
    : findingsQuery.error
      ? "—"
      : "…";
  const approvalSnapshot = review.findings_requiring_approval;
  // Keep the original gated count after completion. Null means no approval
  // snapshot; an empty snapshot means zero gated findings.
  const gatedCount = approvalSnapshot == null ? null : approvalSnapshot.length;
  const gatedCountLabel = approvalSnapshot == null ? "—" : String(approvalSnapshot.length);

  const getDraft = (findingId: string): DecisionDraft => drafts[findingId] ?? EMPTY_DRAFT;
  const setDraft = (findingId: string, nextDraft: DecisionDraft) =>
    setDrafts((currentDrafts) => ({ ...currentDrafts, [findingId]: nextDraft }));
  const decidedCount = gatedFindings.filter(
    (finding) => isDraftValid(getDraft(finding.finding_id), finding),
  ).length;
  const allGatedValid = gatedFindings.length > 0 && decidedCount === gatedFindings.length;
  const resumed = submitted && !actionable;
  const canSubmit =
    productionConfirmed && actionable && allGatedValid && !decisionMutation.isPending && (!submitted || stuck);
  // Reserve bottom space whenever the fixed decision bar is visible.
  const showHitlBar = submitted || (actionable && gatedFindings.length > 0);

  const onSubmit = () => {
    // Reuse accepted decisions on retries; otherwise build them from the drafts.
    const decisions = submittedPayload ?? gatedFindings.map(
      (finding) => toPayload(getDraft(finding.finding_id), finding),
    );
    decisionMutation.mutate(
      { params: { path: { review_id: reviewId } }, body: { decisions } },
      {
        onSuccess: () => {
          setSubmitted(true);
          setSubmittedPayload(decisions);
          setSubmitSeq((sequence) => sequence + 1);
          // The 202 resumes work in the background; polling tracks its progress.
          void queryClient.invalidateQueries();
        },
      },
    );
  };

  return (
    <section className={showHitlBar ? "has-sticky-bar" : undefined}>
      <Link to="/reviews" className="backlink">
        ← Reviews
      </Link>

      {detailQuery.error && detailQuery.data ? (
        <p className="queue-notice" role="alert">
          Couldn't refresh — showing the last loaded state. It may be out of date.
        </p>
      ) : null}

      <div className="panel">
        <div className="panel-b">
          <div className="rd-head">
            <div className="rd-title-block">
              <div className="rd-title-row">
                <h1 className="rd-title">
                  {review.repo_full_name ?? `repo ${review.repo_id}`}{" "}
                  <span className="prnum">#{review.pr_number}</span>
                </h1>
                {timelineQuery.data ? (
                  <button
                    type="button"
                    className={`pill verdict-pill${
                      timelineQuery.data.replay_equivalent ? "" : " status-expired"
                    }`}
                    aria-haspopup="dialog"
                    aria-expanded={showReplayInfo}
                    onClick={() => setShowReplayInfo(true)}
                    aria-label="What replay-equivalent means"
                  >
                    {timelineQuery.data.replay_equivalent ? (
                      <>
                        replay-equivalent <b style={{ color: "var(--pos)" }}>✓</b>
                      </>
                    ) : (
                      <>
                        not replay-equivalent <b style={{ color: "var(--neg)" }}>✗</b>
                      </>
                    )}
                  </button>
                ) : timelineQuery.error ? (
                  // A missing verdict must not imply a successful replay check.
                  <span className="pill status-expired" aria-label="replay verdict">
                    replay verdict unavailable
                  </span>
                ) : null}
              </div>
              {review.pr_title ? <div className="rd-subtitle">{review.pr_title}</div> : null}
              <div className="rd-meta">
                <span>
                  head <span className="mono">{review.head_sha.slice(0, 9)}</span>
                </span>
                {review.policy_version ? (
                  <button
                    type="button"
                    className="chip policy mono"
                    aria-haspopup="dialog"
                    aria-expanded={showPolicy}
                    onClick={() => setShowPolicy(true)}
                  >
                    policy {review.policy_version}
                  </button>
                ) : null}
                <StatusPill status={review.status} />
                {expires ? (
                  <span className="pill status-expired">
                    <span className="dot" aria-hidden="true" />
                    {expires}
                  </span>
                ) : null}
                {review.is_eval ? <span className="eval-tag mono">is_eval</span> : null}
              </div>
            </div>
            <div className="rd-actions">
              <Link
                to={`/reviews/${reviewId}/replay`}
                className="btn primary"
                aria-label="Open the real-time replay reconstruction for this review"
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path
                    d="M3 8a5 5 0 1 1 1.5 3.5"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                  />
                  <path
                    d="M2 5v3h3"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                Replay · reconstruct
              </Link>
            </div>
          </div>
        </div>
      </div>

      {showPolicy && review.policy_version ? (
        <PolicyModal version={review.policy_version} onClose={() => setShowPolicy(false)} />
      ) : null}

      {showReplayInfo ? <ReplayInfoModal onClose={() => setShowReplayInfo(false)} /> : null}

      <ReviewMetrics
        metrics={review.metrics}
        policyVersion={review.policy_version}
        findingCountLabel={findingCountLabel}
        eventCountLabel={eventCountLabel}
      />

      <PipelineStrip
        status={review.status}
        phases={timelineQuery.data?.phases ?? null}
        gatedCount={gatedCount}
        policyVersion={review.policy_version}
      />

      <div className="panel">
        <div className="panel-h">
          <h2>Findings</h2>
          <div className="sub">
            {findingCountLabel} findings · {gatedCountLabel} gated
          </div>
        </div>
        <div className="panel-b">
          {findingsQuery.isLoading ? (
            <p>Loading findings…</p>
          ) : findingsQuery.error ? (
            <p className="error">Failed to load findings.</p>
          ) : sortedFindings.length === 0 ? (
            <p style={{ color: "var(--muted)" }}>No findings recorded for this review.</p>
          ) : (
            sortedFindings.map((finding) => {
              const wasGated = gatedFindingIds.has(finding.finding_id);
              const decidable = actionable && wasGated;
              return (
                <FindingCard
                  key={finding.finding_id}
                  finding={finding}
                  wasGated={wasGated}
                  decision={decidable ? getDraft(finding.finding_id) : undefined}
                  // Retrying a stuck submit must preserve the accepted decisions.
                  disabled={!productionConfirmed || decisionMutation.isPending || submitted}
                  onDecisionChange={decidable ? (nextDraft) => setDraft(finding.finding_id, nextDraft) : undefined}
                />
              );
            })
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-h">
          <h2>Audit feed</h2>
          <div className="sub">phase-grouped · {eventCountLabel} events</div>
        </div>
        {timelineQuery.isLoading ? (
          <div className="panel-b">
            <p>Loading the audit feed…</p>
          </div>
        ) : timelineQuery.error ? (
          <div className="panel-b">
            <p className="error">Failed to load the audit feed.</p>
          </div>
        ) : timelineQuery.data ? (
          <ReplayFeed data={timelineQuery.data} flat />
        ) : null}
      </div>

      {showHitlBar ? (
        <div className="hitl-sticky" role="region" aria-label="Pending decisions">
          <span className="hs-tick" aria-hidden="true" />
          <span className="status-text">
            {isDemo ? (
              <b>Read-only demo — decisions are disabled on this deployment.</b>
            ) : demoChecking ? (
              <b>Checking deployment status — decisions are unavailable until it resolves.</b>
            ) : resumed ? (
              <b>Decision submitted — review is now {review.status}.</b>
            ) : submitted && stuck ? (
              <b>Resume hasn't completed yet — refresh to check, or re-submit.</b>
            ) : submitted ? (
              <b>Decision submitted — resuming the review…</b>
            ) : (
              <>
                <b>
                  {decidedCount} / {gatedFindings.length}
                </b>{" "}
                gated findings decided
              </>
            )}
          </span>
          {gatedFindings.length > 0 && !submitted && productionConfirmed ? (
            <span className="hs-pips" aria-hidden="true">
              {gatedFindings.map((finding) => (
                <span
                  key={finding.finding_id}
                  className={`hs-pip ${isDraftValid(getDraft(finding.finding_id), finding) ? "done" : ""}`}
                />
              ))}
            </span>
          ) : null}
          {decisionMutation.error ? (
            <span className="error" role="alert">
              {decideErrorMessage(decisionMutation.error)}
            </span>
          ) : null}
          <span className="hs-right">
            {submitted && !resumed ? (
              <button className="btn" onClick={() => void detailQuery.refetch()}>
                Refresh status
              </button>
            ) : null}
            {!resumed && productionConfirmed ? (
              <button className="btn primary" disabled={!canSubmit} onClick={onSubmit}>
                {decisionMutation.isPending
                  ? "Submitting…"
                  : submitted && stuck
                    ? "Re-submit"
                    : "Submit decision"}
              </button>
            ) : null}
          </span>
        </div>
      ) : null}
    </section>
  );
}

function ReviewMetrics({
  metrics,
  policyVersion,
  findingCountLabel,
  eventCountLabel,
}: {
  metrics: components["schemas"]["ReviewMetricsView"];
  policyVersion: string | null;
  findingCountLabel: string;
  eventCountLabel: string;
}) {
  return (
    <div className="metrics-strip">
      <div className="ms-card">
        <div className="lab">Total cost</div>
        <div className="ms-val">
          {formatReviewCost(metrics)}
          {metrics.cost_complete !== false ? null : (
            <span style={{ color: "var(--faint)", fontSize: 12 }}>
              {" "}
              incomplete — {metrics.unpriced_call_count} unpriced call(s)
            </span>
          )}
        </div>
        <div className="ms-sub">
          {policyVersion ? `policy ${policyVersion} · ` : ""}
          {findingCountLabel} findings
        </div>
      </div>
      <div className="ms-card">
        <div className="lab">Tokens</div>
        <div className="ms-val">
          {metrics.total_input_tokens.toLocaleString()}
          <span style={{ color: "var(--faint)", fontSize: 12 }}> in</span>
        </div>
        <div className="ms-sub">
          <span className="mono">{metrics.total_output_tokens.toLocaleString()}</span> out ·{" "}
          {metrics.llm_calls_made} LLM calls
        </div>
      </div>
      <div className="ms-card">
        <div className="lab">Duration · Events</div>
        <div className="ms-val">{fmtDuration(metrics.wall_clock_seconds)}</div>
        <div className="ms-sub">
          <span className="mono">{eventCountLabel}</span> audit events
        </div>
      </div>
      {/* Null file counts mean synthesis has not recorded them yet. */}
      <div className="ms-card">
        <div className="lab">Files</div>
        <div className="ms-val">
          {metrics.files_examined ?? "—"}
          <span style={{ color: "var(--faint)", fontSize: 12 }}> examined</span>
        </div>
        <div className="ms-sub">
          <span className="mono">{metrics.files_traced_beyond_diff ?? "—"}</span> traced beyond diff
        </div>
      </div>
    </div>
  );
}
