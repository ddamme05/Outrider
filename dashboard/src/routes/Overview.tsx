import { useState } from "react";
import { Link } from "react-router";

import { $api } from "../api/client";
import type { components } from "../api/schema";
import { HeroChart } from "../components/HeroChart";
import { MetricCard } from "../components/MetricCard";
import { RangeSeg, type MetricsWindow } from "../components/RangeSeg";
import { SeverityBar } from "../components/SeverityBar";
import { StatusPill } from "../components/StatusPill";
import { TierRows } from "../components/TierRows";
import { formatReviewCost } from "../lib/format";
import { deltaInfo, replayDeltaInfo, replayRate, type ReplayMetricsResponse } from "../lib/metrics";
import { useFilters } from "../state/filters";

type ReviewListItem = components["schemas"]["ReviewListItem"];

// Windowed review analytics followed by the server's approval queue.
export function Overview() {
  const includeEval = useFilters((filters) => filters.includeEval);
  const [window, setWindow] = useState<MetricsWindow>("7d");
  const pollingOptions = { refetchInterval: 2000 } as const;

  const metricsQuery = $api.useQuery(
    "get",
    "/api/metrics",
    { params: { query: { window, include_eval: includeEval } } },
    pollingOptions,
  );
  // Replay verdicts cover production reviews only, independent of the eval filter.
  const replayQuery = $api.useQuery(
    "get",
    "/api/metrics/replay",
    { params: { query: { window } } },
    pollingOptions,
  );

  const awaitingQuery = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: includeEval, status: "awaiting_approval", limit: 200 } } },
    pollingOptions,
  );
  const expiredQuery = $api.useQuery(
    "get",
    "/api/reviews",
    {
      params: {
        query: { include_eval: includeEval, status: "awaiting_approval_expired", limit: 200 },
      },
    },
    pollingOptions,
  );

  // Both queues must load before an empty result means no decisions are waiting.
  // Background failures retain the last known data.
  const approvalQueueLoaded = awaitingQuery.data !== undefined && expiredQuery.data !== undefined;
  const approvalQueueError = Boolean(awaitingQuery.error || expiredQuery.error);
  const approvalReviews: ReviewListItem[] = [
    ...(awaitingQuery.data?.reviews ?? []),
    ...(expiredQuery.data?.reviews ?? []),
  ].sort((leftReview, rightReview) => rightReview.created_at.localeCompare(leftReview.created_at));
  const approvalTotal = (awaitingQuery.data?.total ?? 0) + (expiredQuery.data?.total ?? 0);

  const metricsData = metricsQuery.data;

  return (
    <section>
      <div className="ov-head">
        <RangeSeg value={window} onChange={setWindow} />
      </div>

      {metricsQuery.error && !metricsData ? (
        <div className="panel">
          <div className="panel-b">
            <p className="error">Couldn&rsquo;t load metrics — retrying.</p>
          </div>
        </div>
      ) : !metricsData ? (
        <div className="panel">
          <div className="panel-b">
            <p style={{ color: "var(--muted)" }}>Loading metrics…</p>
          </div>
        </div>
      ) : (
        <Analytics data={metricsData} replay={replayQuery.data} stale={Boolean(metricsQuery.error)} />
      )}

      <ApprovalQueue
        rows={approvalReviews}
        total={approvalTotal}
        resolved={approvalQueueLoaded}
        error={approvalQueueError}
      />
    </section>
  );
}

function ApprovalQueue({
  rows,
  total,
  resolved,
  error,
}: {
  rows: ReviewListItem[];
  total: number;
  resolved: boolean;
  error: boolean;
}) {
  return (
    <div className="panel">
      <div className="panel-h">
        <h2>Needs decision</h2>
        <div className="sub">{resolved ? `${total} awaiting` : "…"}</div>
        <div className="right">
          <span className="pill">HITL gate · policy-set severity</span>
        </div>
      </div>
      {error && !resolved ? (
        <div className="panel-b">
          <p className="error">
            Couldn&rsquo;t load the approval queue — retrying. Reviews may be awaiting a decision.
          </p>
        </div>
      ) : !resolved ? (
        <div className="panel-b">
          <p style={{ color: "var(--muted)" }}>Loading…</p>
        </div>
      ) : rows.length === 0 ? (
        <div className="panel-b">
          <p style={{ color: "var(--muted)" }}>Nothing is awaiting your decision.</p>
        </div>
      ) : (
        <>
          {total > rows.length ? (
            <div className="panel-b" style={{ paddingBottom: 0 }}>
              <p className="queue-notice">
                Showing {rows.length} of {total} reviews awaiting a decision — the oldest may be
                beyond this page.
              </p>
            </div>
          ) : null}
          <div role="list">
            {rows.map((review) => (
              <Link to={`/reviews/${review.id}`} className="nd-item" role="listitem" key={review.id}>
                <span className="nd-id">#{review.pr_number}</span>
                <span className="nd-meta">
                  <span className="nd-title">repo {review.repo_id}</span>
                  <span className="nd-sub">
                    <span className="mono">{review.head_sha.slice(0, 9)}</span>
                    {review.is_eval ? " · is_eval" : ""}
                  </span>
                </span>
                <span className="nd-tally">
                  <StatusPill status={review.status} />
                  <span className="nd-cost">
                    {formatReviewCost(review.metrics)}
                  </span>
                </span>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function Analytics({
  data,
  replay,
  stale,
}: {
  data: components["schemas"]["DashboardMetricsResponse"];
  replay: ReplayMetricsResponse | undefined;
  stale: boolean;
}) {
  const current = data.deltas.current;
  const previous = data.deltas.previous;
  const severityCounts = data.severity_distribution;
  const buckets = data.buckets;
  const averageCost = current.reviews > 0 ? current.cost_usd / current.reviews : 0;
  // Missing completeness fields keep the original exact-cost display.
  const costComplete = current.cost_complete !== false;
  // Seeded demo windows end at their snapshot date, not today.
  const snapshotDate = data.anchored
    ? new Date(data.window_end).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null;
  const windowLabel = snapshotDate ? `${data.window} ending ${snapshotDate}` : `last ${data.window}`;

  // No verdicts means an unknown rate; rate deltas use percentage points.
  const replayCurrent = replay?.deltas.current;
  const currentReplayRate = replayCurrent ? replayRate(replayCurrent.equivalent, replayCurrent.total) : null;
  const previousReplayRate = replay
    ? replayRate(replay.deltas.previous.equivalent, replay.deltas.previous.total)
    : null;
  const replayDelta = replayDeltaInfo(currentReplayRate, previousReplayRate);

  return (
    <>
      {stale ? (
        <p className="queue-notice" role="alert">
          Couldn&rsquo;t refresh metrics — showing the last loaded window.
        </p>
      ) : null}
      {snapshotDate ? (
        <p className="queue-notice" role="note">
          Snapshot through {snapshotDate} — windows end at the seeded data, not today.
        </p>
      ) : null}

      <div className="stat-row">
        <MetricCard
          label="Reviews"
          value={current.reviews}
          cap={windowLabel}
          delta={deltaInfo(current.reviews, previous.reviews, "up-good")}
          spark={buckets.map((bucket) => bucket.reviews)}
          sparkVariant="accent"
        />
        <MetricCard
          label="Cost"
          value={`${costComplete ? "" : "\u2265"}$${current.cost_usd.toFixed(2)}`}
          cap={`${costComplete ? "" : "\u2265"}$${averageCost.toFixed(2)} avg / review${
            costComplete ? "" : " \u00b7 incomplete"
          }`}
          delta={
            // Comparing incomplete lower bounds cannot establish a cost change.
            costComplete && previous.cost_complete !== false
              ? deltaInfo(current.cost_usd, previous.cost_usd, "up-bad")
              : { cls: "flat", glyph: "\u2014", label: "n/a \u00b7 incomplete" }
          }
          spark={buckets.map((bucket) => bucket.cost_usd)}
          sparkVariant="neg"
          sparkIncomplete={buckets.map((bucket) => bucket.cost_complete === false)}
        />
        <MetricCard
          label="Findings"
          value={current.findings}
          cap={`${severityCounts.critical ?? 0} crit · ${severityCounts.high ?? 0} high · ${severityCounts.medium ?? 0} med`}
          delta={deltaInfo(current.findings, previous.findings, "neutral")}
          spark={buckets.map((bucket) => bucket.findings)}
          sparkVariant="muted"
        />
        <MetricCard
          label="Failed"
          value={current.failed}
          cap={windowLabel}
          delta={deltaInfo(current.failed, previous.failed, "up-bad")}
          spark={buckets.map((bucket) => bucket.failed)}
          sparkVariant="pos"
        />
        <MetricCard
          label="Replay-equiv"
          value={currentReplayRate === null ? "—" : `${currentReplayRate.toFixed(1)}%`}
          cap={
            replayCurrent
              ? replayCurrent.total > 0
                ? `${replayCurrent.equivalent}/${replayCurrent.total} verified`
                : "no verdicts yet"
              : "replay equivalence"
          }
          delta={replayDelta}
          // Empty buckets contribute zero to the rate sparkline.
          spark={
            replay
              ? replay.buckets.map((bucket) => (bucket.total > 0 ? (bucket.equivalent / bucket.total) * 100 : 0))
              : []
          }
          sparkVariant="accent"
        />
      </div>

      <div className="grid-2">
        <HeroChart buckets={buckets} granularity={data.granularity} />
        <div className="panel dist-panel">
          <div className="panel-h">
            <h2>Findings distribution</h2>
            <div className="sub">
              {current.findings} in {data.window}
              {snapshotDate ? ` (ending ${snapshotDate})` : ""}
            </div>
          </div>
          <div className="panel-b">
            <div className="dist-sub-h">by severity</div>
            <SeverityBar distribution={severityCounts} />
            <div className="dist-sub-h">by evidence tier</div>
            <TierRows distribution={data.evidence_tier_distribution} />
          </div>
        </div>
      </div>
    </>
  );
}
