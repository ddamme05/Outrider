import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import { AuditFeed } from "../components/AuditFeed";
import { DetailsGrid } from "../components/DetailsGrid";
import { FindingCard } from "../components/FindingCard";
import { PipelineStrip } from "../components/PipelineStrip";
import { PolicyTable } from "../components/PolicyTable";
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

function metric(value: number | null, suffix = ""): string {
  // The metrics contract: file/wall-clock fields are null until the review emits
  // a SynthesizeCompletedEvent — render "pending", never a misleading zero.
  return value === null ? "pending" : `${value}${suffix}`;
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
  // The audit-event stream powers both the Audit-feed tab and the per-node Details
  // grid (FUP-133). Polled with the detail so a running review's feed fills in.
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

      <div className="hero-head">
        <h1>
          repo {d.repo_id} <span className="prnum">#{d.pr_number}</span>
        </h1>
        <div className="hero-strip">
          <span className="sha">{d.head_sha.slice(0, 9)}</span>
          <span className="sep" aria-hidden="true">
            ·
          </span>
          <span>
            cost <span className="b mono">${d.metrics.total_cost_usd.toFixed(2)}</span>
          </span>
          {d.policy_version ? (
            <>
              <span className="sep" aria-hidden="true">
                ·
              </span>
              <button
                className="chip policy mono"
                aria-expanded={showPolicy}
                onClick={() => setShowPolicy((v) => !v)}
              >
                policy {d.policy_version} ▸
              </button>
            </>
          ) : null}
          <span className="right">
            <StatusPill status={d.status} />
            {expires ? (
              <span className="pill status-expired" style={{ fontSize: 11 }}>
                <span className="dot" aria-hidden="true" />
                {expires}
              </span>
            ) : null}
            {d.is_eval ? <span className="eval-tag mono">is_eval</span> : null}
          </span>
        </div>
      </div>

      {showPolicy && d.policy_version ? <PolicyTable version={d.policy_version} /> : null}

      <PipelineStrip status={d.status} />

      <div className="detail-meta-grid">
        <div className="dmg-item">
          <div className="k">Cost</div>
          <div className="v">
            <b>${d.metrics.total_cost_usd.toFixed(2)}</b>
          </div>
        </div>
        <div className="dmg-item">
          <div className="k">LLM calls</div>
          <div className="v mono">{d.metrics.llm_calls_made}</div>
        </div>
        <div className="dmg-item">
          <div className="k">Tokens</div>
          <div className="v mono">
            {d.metrics.total_input_tokens} in · {d.metrics.total_output_tokens} out
          </div>
        </div>
        <div className="dmg-item">
          <div className="k">Files examined</div>
          <div className="v mono">{metric(d.metrics.files_examined)}</div>
        </div>
        <div className="dmg-item">
          <div className="k">Traced beyond diff</div>
          <div className="v mono">{metric(d.metrics.files_traced_beyond_diff)}</div>
        </div>
        <div className="dmg-item">
          <div className="k">Wall clock</div>
          <div className="v mono">{metric(d.metrics.wall_clock_seconds, "s")}</div>
        </div>
      </div>

      {events.data && events.data.events.length > 0 ? (
        <details className="details-disc">
          <summary>▸ Per-node details</summary>
          <DetailsGrid events={events.data.events} />
        </details>
      ) : null}

      {replay.isLoading ? (
        <p className="replay-status">Checking replay equivalence…</p>
      ) : replay.error ? (
        <p className="replay-status error">Replay verdict unavailable — couldn't load it.</p>
      ) : replay.data ? (
        <ReplayPanel verdict={replay.data} />
      ) : null}

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
          Audit feed <span className="muted">({events.data?.total ?? 0})</span>
        </button>
      </div>

      {tab === "findings" ? (
        findings.isLoading ? (
          <p>Loading findings…</p>
        ) : findings.error ? (
          <p className="error">Failed to load findings.</p>
        ) : allFindings.length === 0 ? (
          <p style={{ color: "var(--text-2)" }}>No findings recorded for this review.</p>
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
                // Lock controls while submitting and once submitted — including the
                // stuck state: a re-submit must resend the SAME payload (divergent
                // content wedges the audit row), so editing stays closed until the
                // page reloads with fresh state.
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

      {submitted || (actionable && gated.length > 0) ? (
        <div className="submit-bar">
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
