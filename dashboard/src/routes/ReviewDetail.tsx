import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import { AuditFeed } from "../components/AuditFeed";
import { FindingCard } from "../components/FindingCard";
import { PipelineStrip } from "../components/PipelineStrip";
import { PolicyModal } from "../components/PolicyModal";
import { ReplayPanel } from "../components/ReplayPanel";
import { StatusPill } from "../components/StatusPill";
import { expiresLabel } from "../lib/format";
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
  const replay = $api.useQuery("get", "/api/reviews/{review_id}/replay", pathParams, { enabled });
  // The audit-event stream powers the pipeline node cards AND the Audit-feed tab
  // (FUP-133). Polled with the detail so a running review's feed fills in.
  const events = $api.useQuery("get", "/api/reviews/{review_id}/events", pathParams, {
    enabled,
    refetchInterval: 2000,
  });

  // Hooks must run unconditionally, before the early returns below.
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<"findings" | "audit">("findings");
  const [showPolicy, setShowPolicy] = useState(false);
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

  const allFindings = useMemo(() => findings.data?.findings ?? [], [findings.data]);
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
  const eventCount = events.data?.total ?? 0;
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
    <section>
      <Link to="/reviews" className="backlink">
        ← Reviews
      </Link>

      {detail.error && detail.data ? (
        <p className="queue-notice" role="alert">
          Couldn't refresh — showing the last loaded state. It may be out of date.
        </p>
      ) : null}

      {/* hero panel — subject identity + replay verdict (no author/repo-slug: not
          exposed by the API; no "reconstruct" button: that screen isn't built) */}
      <div className="panel">
        <div className="panel-b">
          <div className="rd-head">
            <div className="rd-title-block">
              <h1 className="rd-title">
                repo {d.repo_id} <span className="prnum">#{d.pr_number}</span>
              </h1>
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
              {replay.data ? (
                <span className="pill" aria-label="replay verdict">
                  {replay.data.replay_equivalent ? (
                    <>
                      replay-equivalent <b style={{ color: "var(--pos)" }}>✓</b>
                    </>
                  ) : (
                    <>
                      not replay-equivalent <b style={{ color: "var(--neg)" }}>✗</b>
                    </>
                  )}
                </span>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {showPolicy && d.policy_version ? (
        <PolicyModal version={d.policy_version} onClose={() => setShowPolicy(false)} />
      ) : null}

      {/* this-review metrics strip — 3 cards, all server-backed */}
      <div className="metrics-strip">
        <div className="ms-card">
          <div className="lab">Total cost</div>
          <div className="ms-val">${m.total_cost_usd.toFixed(2)}</div>
          <div className="ms-sub">
            {d.policy_version ? `policy ${d.policy_version} · ` : ""}
            {allFindings.length} findings
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
            <span className="mono">{eventCount}</span> audit events
          </div>
        </div>
      </div>

      {/* pipeline — per-node cards from the audit stream */}
      <PipelineStrip
        status={d.status}
        events={events.data?.events ?? []}
        gatedCount={gated.length}
        policyVersion={d.policy_version}
      />

      {/* replay verdict detail (the live reconstruction screen isn't built; this
          is the honest verdict surface for the replay equivalence check) */}
      {replay.isLoading ? (
        <p className="replay-status">Checking replay equivalence…</p>
      ) : replay.error ? (
        <p className="replay-status error">Replay verdict unavailable — couldn't load it.</p>
      ) : replay.data ? (
        <ReplayPanel verdict={replay.data} />
      ) : null}

      {/* findings / audit feed */}
      <div className="tabs" role="tablist">
        <button
          className={`tab ${tab === "findings" ? "active" : ""}`}
          role="tab"
          aria-selected={tab === "findings"}
          onClick={() => setTab("findings")}
        >
          Findings <span className="muted">({allFindings.length})</span>
        </button>
        <button
          className={`tab ${tab === "audit" ? "active" : ""}`}
          role="tab"
          aria-selected={tab === "audit"}
          onClick={() => setTab("audit")}
        >
          Audit feed <span className="muted">({eventCount})</span>
        </button>
      </div>

      <div className="panel">
        <div className="panel-h">
          <h2>{tab === "findings" ? "Findings" : "Audit feed"}</h2>
          <div className="sub">
            {tab === "findings"
              ? `${allFindings.length} findings · ${gated.length} gated`
              : `${eventCount} events`}
          </div>
        </div>
        <div className="panel-b">
          {tab === "findings" ? (
            findings.isLoading ? (
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
            )
          ) : events.isLoading ? (
            <p>Loading audit feed…</p>
          ) : events.error ? (
            <p className="error">Failed to load the audit feed.</p>
          ) : (
            <AuditFeed events={events.data?.events ?? []} />
          )}
        </div>
      </div>

      {submitted || (actionable && gated.length > 0) ? (
        <div className="submit-bar">
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
          <span style={{ flex: 1 }} />
          {submitted && !resumed ? (
            <button className="btn" onClick={() => void detail.refetch()}>
              Refresh status
            </button>
          ) : null}
          {!resumed ? (
            <button className="btn primary" disabled={!canSubmit} onClick={onSubmit}>
              {decide.isPending ? "Submitting…" : submitted && stuck ? "Re-submit" : "Submit decision"}
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
