import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import { FindingCard } from "../components/FindingCard";
import { PipelineStrip } from "../components/PipelineStrip";
import { ReplayPanel } from "../components/ReplayPanel";
import { StatusPill } from "../components/StatusPill";

function metric(value: number | null, suffix = ""): string {
  // The metrics contract: file/wall-clock fields are null until the review emits
  // a SynthesizeCompletedEvent — render "pending", never a misleading zero.
  return value === null ? "pending" : `${value}${suffix}`;
}

function expiresLabel(expiresAt: string | null): string | null {
  if (!expiresAt) return null;
  const ms = new Date(expiresAt).getTime() - Date.now();
  if (Number.isNaN(ms)) return null;
  if (ms <= 0) return "expired";
  const mins = Math.round(ms / 60000);
  return mins < 60 ? `expires in ${mins}m` : `expires in ${Math.round(mins / 60)}h`;
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

  if (!enabled) {
    return <p className="error">No review id in the URL.</p>;
  }
  if (detail.isLoading) {
    return <p>Loading…</p>;
  }
  if (detail.error) {
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
  const findingList = findings.data?.findings ?? [];

  return (
    <section>
      <Link to="/reviews" className="backlink">
        ← Reviews
      </Link>

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
              <span className="mono">policy {d.policy_version}</span>
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

      {replay.data ? <ReplayPanel verdict={replay.data} /> : null}

      <div className="tabs" role="tablist">
        <button className="tab active" role="tab" aria-selected="true">
          Findings <span className="muted">({findingList.length})</span>
        </button>
      </div>

      {findings.isLoading ? (
        <p>Loading findings…</p>
      ) : findings.error ? (
        <p className="error">Failed to load findings.</p>
      ) : findingList.length === 0 ? (
        <p style={{ color: "var(--text-2)" }}>No findings recorded for this review.</p>
      ) : (
        findingList.map((f) => <FindingCard key={f.finding_id} finding={f} />)
      )}
    </section>
  );
}
