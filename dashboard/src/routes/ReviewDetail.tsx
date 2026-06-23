import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import { FindingCard } from "../components/FindingCard";
import { PipelineStrip } from "../components/PipelineStrip";
import { PolicyModal } from "../components/PolicyModal";
import { ReplayFeed } from "../components/ReplayFeed";
import { ReplayInfoModal } from "../components/ReplayInfoModal";
import { StatusPill } from "../components/StatusPill";
import { expiresLabel } from "../lib/format";
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

// Duration for the metrics strip. wall_clock_seconds is null until the review
// emits a SynthesizeCompletedEvent — render "pending", never a misleading zero.
function fmtDuration(seconds: number | null): string {
  if (seconds === null) return "pending";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m${Math.round(seconds % 60)}s`;
}

export function ReviewDetail() {
  // The route param is `reviewId` (router.tsx); the API path param is
  // `review_id`. useParams types it as possibly-undefined, so guard before
  // issuing requests rather than sending `review_id: undefined`.
  const { reviewId } = useParams<{ reviewId: string }>();

  const enabled = typeof reviewId === "string" && reviewId.length > 0;
  const pathParams = { params: { path: { review_id: reviewId ?? "" } } };

  const detail = $api.useQuery("get", "/api/reviews/{review_id}", pathParams, {
    enabled,
    refetchInterval: 2000,
  });
  const findings = $api.useQuery("get", "/api/reviews/{review_id}/findings", pathParams, {
    enabled,
  });
  // The grouped, replay-verified timeline (ROADMAP feature 6) — the verdict header + the
  // playable phase timeline. Polled so a running review's timeline fills in. Subsumes the
  // standalone replay-verdict panel (one verdict surface, not three).
  const timeline = $api.useQuery("get", "/api/reviews/{review_id}/replay-timeline", pathParams, {
    enabled,
    refetchInterval: 2000,
  });
  // Retained ONLY for the metrics-strip raw event COUNT (events.data.total) — the authoritative
  // count, which includes the projected ReplayVerdictEvent the timeline omits. The pipeline cards +
  // the timeline now read the replay-verified `.phases`/`.events` off /replay-timeline (the
  // FUP-125 closure), so events.data.events itself is no longer consumed. Polled with the detail
  // so a running review's count climbs live.
  const events = $api.useQuery("get", "/api/reviews/{review_id}/events", pathParams, {
    enabled,
    refetchInterval: 2000,
  });

  // Hooks must run unconditionally, before the early returns below.
  const queryClient = useQueryClient();
  const [showPolicy, setShowPolicy] = useState(false);
  const [showReplayInfo, setShowReplayInfo] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, DecisionDraft>>({});
  const [submitted, setSubmitted] = useState(false);
  const decide = $api.useMutation("post", "/reviews/{review_id}/decide");

  const [submitSeq, setSubmitSeq] = useState(0);
  const [stuck, setStuck] = useState(false);
  // The exact payload accepted by the first 202. A stuck re-submit MUST resend
  // this verbatim: a divergent payload hits the backend's HITLDecisionEvent
  // natural-key conflict (different decisions_content_hash) and wedges the review
  // pending sweep. Identical content is idempotent and safe. Editing only re-opens
  // after the page reloads (fresh state) — see FUP-135.
  const [submittedPayload, setSubmittedPayload] = useState<ReturnType<typeof toPayload>[] | null>(
    null,
  );

  // After a submit, arm a window: if the status hasn't advanced off the gate by
  // then, mark the resume "stuck" so the UI re-enables submit instead of showing
  // a permanent "submitted" (FUP-135). Re-armed on each submit via submitSeq.
  useEffect(() => {
    if (submitSeq === 0) return;
    setStuck(false);
    const t = setTimeout(() => setStuck(true), RESUME_WINDOW_MS);
    return () => clearTimeout(t);
  }, [submitSeq]);

  // Findings sorted most-severe-first (critical → info) so the reviewer leads with the
  // findings that gate the PR. The server returns them in production/dedup order; the
  // display order is a UI concern. Unknown severities sort last; ties keep server order
  // (Array.sort is stable). Copy first — never mutate the cached query data.
  const allFindings = useMemo(() => {
    const rank = (s: string) => {
      const i = SEVERITY_ORDER.indexOf(s as (typeof SEVERITY_ORDER)[number]);
      return i === -1 ? SEVERITY_ORDER.length : i;
    };
    return [...(findings.data?.findings ?? [])].sort((a, b) => rank(a.severity) - rank(b.severity));
  }, [findings.data]);
  const actionable = isActionable(detail.data?.status ?? "");
  // Authoritative gated set from the server (ReviewDetail.findings_requiring_approval),
  // by finding_id — never inferred from severity. The decide endpoint enforces
  // the identical set.
  const gatedSet = useMemo(
    () => new Set(detail.data?.findings_requiring_approval ?? []),
    [detail.data],
  );
  const gated = useMemo(
    () => (actionable ? allFindings.filter((f) => gatedSet.has(f.finding_id)) : []),
    [actionable, allFindings, gatedSet],
  );

  if (!enabled) {
    return <p className="error">No review id in the URL.</p>;
  }
  // Only hard-fail to the error screen on the INITIAL load (no data yet). A
  // failed background refetch (e.g. a poll after submit) keeps the last-known
  // data — don't blow the page away; a small banner surfaces it instead.
  if (detail.isLoading) {
    return <p>Loading…</p>;
  }
  if (detail.error && !detail.data) {
    // openapi-react-query throws only the parsed error body, not the Response,
    // so we can't read the status here to distinguish 404 from 5xx — keep the
    // message honest rather than claiming a cause we can't confirm.
    return (
      <p className="error">Couldn't load this review — it may not exist, or the request failed.</p>
    );
  }

  const d = detail.data;
  if (!d) {
    return <p className="error">Review not found.</p>;
  }

  const expires = d.status.startsWith("awaiting_approval") ? expiresLabel(d.expires_at) : null;
  // The audit stream backs the event count, the pipeline node stats, and the feed.
  // It fails CLOSED: until it loads we never assert a count or per-node stat — "0"
  // would be a fabricated audit fact when we simply haven't fetched the stream.
  const eventsLoaded = events.data !== undefined;
  const eventCountLabel = eventsLoaded ? String(events.data.total) : events.error ? "—" : "…";
  // Findings count fails CLOSED the same way: "…"/"—" until /findings loads, never
  // a fabricated "0 findings". The displayed GATED count is the server-authoritative
  // snapshot (findings_requiring_approval, from detail) — independent of the
  // /findings load AND of actionability, so a completed review still shows how many
  // findings gated the PR. null = gate not yet determined → "—". (`gated` below stays
  // the currently-decidable set, used only for the live decision controls.)
  const findingsLoaded = findings.data !== undefined;
  const findingCountLabel = findingsLoaded
    ? String(allFindings.length)
    : findings.error
      ? "—"
      : "…";
  const reqApproval = d.findings_requiring_approval;
  // null (no HITL-request snapshot) is distinct from [] (snapshot, nothing gated):
  // keep it null through to the display so neither the header nor the pipeline
  // renders a fabricated "0" when the gate set is simply unknown.
  const gatedCount = reqApproval == null ? null : reqApproval.length;
  const gatedCountLabel = reqApproval == null ? "—" : String(reqApproval.length);
  const m = d.metrics;

  const getDraft = (id: string): DecisionDraft => drafts[id] ?? EMPTY_DRAFT;
  const setDraft = (id: string, next: DecisionDraft) =>
    setDrafts((prev) => ({ ...prev, [id]: next }));
  const decidedCount = gated.filter((f) => isDraftValid(getDraft(f.finding_id), f)).length;
  const allGatedValid = gated.length > 0 && decidedCount === gated.length;
  // The review advanced off the gate after our submit — the resume took.
  const resumed = submitted && !actionable;
  // Submit is re-enabled if never submitted, OR the resume looks stuck.
  const canSubmit = actionable && allGatedValid && !decide.isPending && (!submitted || stuck);
  // The fixed decision bar shows whenever decisions are pending or a submit is in flight; the
  // section reserves bottom space (.has-sticky-bar) so the last content clears the fixed bar.
  const showHitlBar = submitted || (actionable && gated.length > 0);

  const onSubmit = () => {
    // First submit builds from the drafts; a stuck re-submit resends the exact
    // accepted payload (snapshotted on the first 202) — never a divergent one.
    const decisions = submittedPayload ?? gated.map((f) => toPayload(getDraft(f.finding_id), f));
    decide.mutate(
      { params: { path: { review_id: reviewId } }, body: { decisions } },
      {
        onSuccess: () => {
          setSubmitted(true);
          setSubmittedPayload(decisions); // snapshot the accepted payload
          setSubmitSeq((n) => n + 1); // (re-)arm the stuck-detection window
          // Decide returns 202 and resumes the graph in the background; let the
          // 2s polls surface the awaiting → running → completed transition.
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

      {detail.error && detail.data ? (
        <p className="queue-notice" role="alert">
          Couldn't refresh — showing the last loaded state. It may be out of date.
        </p>
      ) : null}

      {/* hero panel — subject identity + the "Replay · reconstruct" action + the replay
          verdict (no author/repo-slug: not exposed by the API) */}
      <div className="panel">
        <div className="panel-b">
          <div className="rd-head">
            <div className="rd-title-block">
              <div className="rd-title-row">
                <h1 className="rd-title">
                  {d.repo_full_name ?? `repo ${d.repo_id}`}{" "}
                  <span className="prnum">#{d.pr_number}</span>
                </h1>
                {/* The replay verdict sits by the title and is clickable (like the
                    policy chip) to explain what replay-equivalence means. */}
                {timeline.data ? (
                  <button
                    type="button"
                    className={`pill verdict-pill${
                      timeline.data.replay_equivalent ? "" : " status-expired"
                    }`}
                    aria-haspopup="dialog"
                    aria-expanded={showReplayInfo}
                    onClick={() => setShowReplayInfo(true)}
                    aria-label="What replay-equivalent means"
                  >
                    {timeline.data.replay_equivalent ? (
                      <>
                        replay-equivalent <b style={{ color: "var(--pos)" }}>✓</b>
                      </>
                    ) : (
                      <>
                        not replay-equivalent <b style={{ color: "var(--neg)" }}>✗</b>
                      </>
                    )}
                  </button>
                ) : timeline.error ? (
                  // Fail loud, never silently omit: a failed verdict load reads as an
                  // explicit "unavailable", not an absent (and so implicitly-fine) badge.
                  <span className="pill status-expired" aria-label="replay verdict">
                    replay verdict unavailable
                  </span>
                ) : null}
              </div>
              {d.pr_title ? <div className="rd-subtitle">{d.pr_title}</div> : null}
              <div className="rd-meta">
                <span>
                  head <span className="mono">{d.head_sha.slice(0, 9)}</span>
                </span>
                {d.policy_version ? (
                  <button
                    type="button"
                    className="chip policy mono"
                    aria-haspopup="dialog"
                    aria-expanded={showPolicy}
                    onClick={() => setShowPolicy(true)}
                  >
                    policy {d.policy_version}
                  </button>
                ) : null}
                <StatusPill status={d.status} />
                {expires ? (
                  <span className="pill status-expired">
                    <span className="dot" aria-hidden="true" />
                    {expires}
                  </span>
                ) : null}
                {d.is_eval ? <span className="eval-tag mono">is_eval</span> : null}
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

      {showPolicy && d.policy_version ? (
        <PolicyModal version={d.policy_version} onClose={() => setShowPolicy(false)} />
      ) : null}

      {showReplayInfo ? <ReplayInfoModal onClose={() => setShowReplayInfo(false)} /> : null}

      {/* this-review metrics strip — 4 cards, all server-backed */}
      <div className="metrics-strip">
        <div className="ms-card">
          <div className="lab">Total cost</div>
          <div className="ms-val">${m.total_cost_usd.toFixed(2)}</div>
          <div className="ms-sub">
            {d.policy_version ? `policy ${d.policy_version} · ` : ""}
            {findingCountLabel} findings
          </div>
        </div>
        <div className="ms-card">
          <div className="lab">Tokens</div>
          <div className="ms-val">
            {m.total_input_tokens.toLocaleString()}
            <span style={{ color: "var(--faint)", fontSize: 12 }}> in</span>
          </div>
          <div className="ms-sub">
            <span className="mono">{m.total_output_tokens.toLocaleString()}</span> out ·{" "}
            {m.llm_calls_made} LLM calls
          </div>
        </div>
        <div className="ms-card">
          <div className="lab">Duration · Events</div>
          <div className="ms-val">{fmtDuration(m.wall_clock_seconds)}</div>
          <div className="ms-sub">
            <span className="mono">{eventCountLabel}</span> audit events
          </div>
        </div>
        {/* the adaptive analyze⇄trace loop's headline output: how many files were
            examined and how many were pulled in beyond the diff. Both null until
            SynthesizeCompletedEvent — render "—", never a misleading 0. */}
        <div className="ms-card">
          <div className="lab">Files</div>
          <div className="ms-val">
            {m.files_examined ?? "—"}
            <span style={{ color: "var(--faint)", fontSize: 12 }}> examined</span>
          </div>
          <div className="ms-sub">
            <span className="mono">{m.files_traced_beyond_diff ?? "—"}</span> traced beyond diff
          </div>
        </div>
      </div>

      {/* pipeline — per-node cards from the server's replay-verified reconstruction
          (the same verified phases the audit feed below + the Replay page render). */}
      <PipelineStrip
        status={d.status}
        phases={timeline.data?.phases ?? null}
        gatedCount={gatedCount}
        policyVersion={d.policy_version}
      />

      {/* findings */}
      <div className="panel">
        <div className="panel-h">
          <h2>Findings</h2>
          <div className="sub">
            {findingCountLabel} findings · {gatedCountLabel} gated
          </div>
        </div>
        <div className="panel-b">
          {findings.isLoading ? (
            <p>Loading findings…</p>
          ) : findings.error ? (
            <p className="error">Failed to load findings.</p>
          ) : allFindings.length === 0 ? (
            <p style={{ color: "var(--muted)" }}>No findings recorded for this review.</p>
          ) : (
            allFindings.map((f) => {
              const wasGated = gatedSet.has(f.finding_id);
              const decidable = actionable && wasGated;
              return (
                <FindingCard
                  key={f.finding_id}
                  finding={f}
                  wasGated={wasGated}
                  decision={decidable ? getDraft(f.finding_id) : undefined}
                  // Lock controls while submitting and once submitted — including
                  // the stuck state: a re-submit must resend the SAME payload
                  // (divergent content wedges the audit row), so editing stays
                  // closed until the page reloads with fresh state.
                  disabled={decide.isPending || submitted}
                  onDecisionChange={decidable ? (next) => setDraft(f.finding_id, next) : undefined}
                />
              );
            })
          )}
        </div>
      </div>

      {/* audit feed — phase-grouped, replay-verified (static; the playable real-time
          reconstruction lives on the Replay · reconstruct page reached from the hero button). */}
      <div className="panel">
        <div className="panel-h">
          <h2>Audit feed</h2>
          <div className="sub">phase-grouped · {eventCountLabel} events</div>
        </div>
        {timeline.isLoading ? (
          <div className="panel-b">
            <p>Loading the audit feed…</p>
          </div>
        ) : timeline.error ? (
          <div className="panel-b">
            <p className="error">Failed to load the audit feed.</p>
          </div>
        ) : timeline.data ? (
          <ReplayFeed data={timeline.data} flat />
        ) : null}
      </div>

      {showHitlBar ? (
        <div className="hitl-sticky" role="region" aria-label="Pending decisions">
          <span className="hs-tick" aria-hidden="true" />
          <span className="status-text">
            {resumed ? (
              <b>Decision submitted — review is now {d.status}.</b>
            ) : submitted && stuck ? (
              <b>Resume hasn't completed yet — refresh to check, or re-submit.</b>
            ) : submitted ? (
              <b>Decision submitted — resuming the review…</b>
            ) : (
              <>
                <b>
                  {decidedCount} / {gated.length}
                </b>{" "}
                gated findings decided
              </>
            )}
          </span>
          {gated.length > 0 && !submitted ? (
            <span className="hs-pips" aria-hidden="true">
              {gated.map((f) => (
                <span
                  key={f.finding_id}
                  className={`hs-pip ${isDraftValid(getDraft(f.finding_id), f) ? "done" : ""}`}
                />
              ))}
            </span>
          ) : null}
          {decide.error ? (
            <span className="error" role="alert">
              {decideErrorMessage(decide.error)}
            </span>
          ) : null}
          <span className="hs-right">
            {submitted && !resumed ? (
              <button className="btn" onClick={() => void detail.refetch()}>
                Refresh status
              </button>
            ) : null}
            {!resumed ? (
              <button className="btn primary" disabled={!canSubmit} onClick={onSubmit}>
                {decide.isPending
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
