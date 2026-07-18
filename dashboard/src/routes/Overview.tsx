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
import { deltaInfo, replayDeltaInfo, replayRate, type ReplayMetricsResponse } from "../lib/metrics";
import { useFilters } from "../state/filters";

type ReviewListItem = components["schemas"]["ReviewListItem"];

// The Overview is the operational landing screen + the Signal analytics surface
// (DECISIONS#039). The KPI cards, sparklines, hero chart, and distributions are
// REAL read-only aggregations over the audit stream (GET /api/metrics, windowed by
// the range selector) — honest zeros on a sparse window, never an invented series.
// Below the analytics is the "needs your decision" HITL rail (server-side
// status-filtered, fails closed). The Replay-equivalence KPI reads the sibling
// /api/metrics/replay endpoint (the persisted replay_verdict projection, DECISIONS#039)
// — equivalent/total over the same window; "—" when no reviews are verdicted yet.
export function Overview() {
  const includeEval = useFilters((s) => s.includeEval);
  const [window, setWindow] = useState<MetricsWindow>("7d");
  const opts = { refetchInterval: 2000 } as const;

  const metrics = $api.useQuery(
    "get",
    "/api/metrics",
    { params: { query: { window, include_eval: includeEval } } },
    opts,
  );
  // Sibling Replay-% query (same window). Independent of /api/metrics: it degrades the
  // ONE card to "—" if it's still loading or errors, without blocking the other four. The
  // global eval toggle is NOT passed: replay verdicts are projected for PRODUCTION reviews
  // only (the projector's scope, DECISIONS#039), so the card is always production-scoped.
  const replay = $api.useQuery(
    "get",
    "/api/metrics/replay",
    { params: { query: { window } } },
    opts,
  );

  // The HITL rail's two approval-queue queries (kept at limit 200 because they
  // ALSO feed the rail rows). The analytics above read from /api/metrics instead.
  const awaiting = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: includeEval, status: "awaiting_approval", limit: 200 } } },
    opts,
  );
  const expired = $api.useQuery(
    "get",
    "/api/reviews",
    {
      params: {
        query: { include_eval: includeEval, status: "awaiting_approval_expired", limit: 200 },
      },
    },
    opts,
  );

  // HITL visibility fails CLOSED (unchanged from the pre-analytics Overview): never
  // render a confident "0 awaiting" or an all-clear rail if either approval query
  // errored before it ever resolved. react-query keeps last-good data across a
  // failed background refetch, so a stale-but-real count still shows.
  const awaitingResolved = awaiting.data !== undefined && expired.data !== undefined;
  const awaitingError = Boolean(awaiting.error || expired.error);
  const railRows: ReviewListItem[] = [
    ...(awaiting.data?.reviews ?? []),
    ...(expired.data?.reviews ?? []),
  ].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const railLoaded = railRows.length;
  const railTotal = (awaiting.data?.total ?? 0) + (expired.data?.total ?? 0);

  const d = metrics.data;

  return (
    <section>
      <div className="ov-head">
        <RangeSeg value={window} onChange={setWindow} />
      </div>

      {metrics.error && !d ? (
        <div className="panel">
          <div className="panel-b">
            <p className="error">Couldn&rsquo;t load metrics — retrying.</p>
          </div>
        </div>
      ) : !d ? (
        <div className="panel">
          <div className="panel-b">
            <p style={{ color: "var(--muted)" }}>Loading metrics…</p>
          </div>
        </div>
      ) : (
        <Analytics data={d} replay={replay.data} stale={Boolean(metrics.error)} />
      )}

      <div className="panel">
        <div className="panel-h">
          <h2>Needs decision</h2>
          <div className="sub">{awaitingResolved ? `${railTotal} awaiting` : "…"}</div>
          <div className="right">
            <span className="pill">HITL gate · policy-set severity</span>
          </div>
        </div>
        {awaitingError && !awaitingResolved ? (
          <div className="panel-b">
            <p className="error">
              Couldn&rsquo;t load the approval queue — retrying. Reviews may be awaiting a decision.
            </p>
          </div>
        ) : !awaitingResolved ? (
          <div className="panel-b">
            <p style={{ color: "var(--muted)" }}>Loading…</p>
          </div>
        ) : railRows.length === 0 ? (
          <div className="panel-b">
            <p style={{ color: "var(--muted)" }}>Nothing is awaiting your decision.</p>
          </div>
        ) : (
          <>
            {railTotal > railLoaded ? (
              <div className="panel-b" style={{ paddingBottom: 0 }}>
                <p className="queue-notice">
                  Showing {railLoaded} of {railTotal} reviews awaiting a decision — the oldest may be
                  beyond this page.
                </p>
              </div>
            ) : null}
            <div role="list">
              {railRows.map((r) => (
                <Link to={`/reviews/${r.id}`} className="nd-item" role="listitem" key={r.id}>
                  <span className="nd-id">#{r.pr_number}</span>
                  <span className="nd-meta">
                    <span className="nd-title">repo {r.repo_id}</span>
                    <span className="nd-sub">
                      <span className="mono">{r.head_sha.slice(0, 9)}</span>
                      {r.is_eval ? " · is_eval" : ""}
                    </span>
                  </span>
                  <span className="nd-tally">
                    <StatusPill status={r.status} />
                    <span className="nd-cost">
                      {`${r.metrics.cost_complete === false ? "\u2265" : ""}$${r.metrics.total_cost_usd.toFixed(2)}`}
                    </span>
                  </span>
                </Link>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

// The analytics block — pure render off a resolved metrics payload. Split out so
// the loading/error/empty branching above stays readable and every value here is
// a non-null field of `data` (the parent guards `data` before mounting this).
function Analytics({
  data,
  replay,
  stale,
}: {
  data: components["schemas"]["DashboardMetricsResponse"];
  replay: ReplayMetricsResponse | undefined;
  stale: boolean;
}) {
  const cur = data.deltas.current;
  const prev = data.deltas.previous;
  const sev = data.severity_distribution;
  const buckets = data.buckets;
  const avgCost = cur.reviews > 0 ? cur.cost_usd / cur.reviews : 0;
  // Explicit-false semantics: pre-field payloads (no completeness key) render
  // exactly as before — only a stated false marks the lower bound.
  const costComplete = cur.cost_complete !== false;
  // Demo-snapshot anchoring (#039-honest label): when the backend anchored the
  // window to the seeded data's latest instant, every window caption says so —
  // "7d ending <date>" — instead of implying live recency with "last 7d".
  const snapshotDate = data.anchored
    ? new Date(data.window_end).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null;
  const windowCap = snapshotDate ? `${data.window} ending ${snapshotDate}` : `last ${data.window}`;

  // Replay-% (5th card): equivalent/total over the window. Degrades to a muted "—"
  // when the sibling query hasn't resolved (loading/error) OR resolved with no
  // verdicts — never a fabricated 0%. The delta is a rate-appropriate percentage-POINT
  // change (replayDeltaInfo), not deltaInfo's count-relative %.
  const rc = replay?.deltas.current;
  const curRate = rc ? replayRate(rc.equivalent, rc.total) : null;
  const prevRate = replay
    ? replayRate(replay.deltas.previous.equivalent, replay.deltas.previous.total)
    : null;
  const replayDelta = replayDeltaInfo(curRate, prevRate);

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
          value={cur.reviews}
          cap={windowCap}
          delta={deltaInfo(cur.reviews, prev.reviews, "up-good")}
          spark={buckets.map((b) => b.reviews)}
          sparkVariant="accent"
        />
        <MetricCard
          label="Cost"
          value={`${costComplete ? "" : "\u2265"}$${cur.cost_usd.toFixed(2)}`}
          cap={`${costComplete ? "" : "\u2265"}$${avgCost.toFixed(2)} avg / review${
            costComplete ? "" : " \u00b7 incomplete"
          }`}
          delta={
            // A delta between two incomplete lower bounds is not a valid lower
            // bound — suppress it whenever EITHER period is incomplete
            // (openai-native-host arc).
            costComplete && prev.cost_complete !== false
              ? deltaInfo(cur.cost_usd, prev.cost_usd, "up-bad")
              : { cls: "flat", glyph: "\u2014", label: "n/a \u00b7 incomplete" }
          }
          spark={buckets.map((b) => b.cost_usd)}
          sparkVariant="neg"
          sparkIncomplete={buckets.map((b) => b.cost_complete === false)}
        />
        <MetricCard
          label="Findings"
          value={cur.findings}
          cap={`${sev.critical ?? 0} crit · ${sev.high ?? 0} high · ${sev.medium ?? 0} med`}
          delta={deltaInfo(cur.findings, prev.findings, "neutral")}
          spark={buckets.map((b) => b.findings)}
          sparkVariant="muted"
        />
        <MetricCard
          label="Failed"
          value={cur.failed}
          cap={windowCap}
          delta={deltaInfo(cur.failed, prev.failed, "up-bad")}
          spark={buckets.map((b) => b.failed)}
          sparkVariant="pos"
        />
        <MetricCard
          label="Replay-equiv"
          value={curRate === null ? "—" : `${curRate.toFixed(1)}%`}
          cap={
            rc
              ? rc.total > 0
                ? `${rc.equivalent}/${rc.total} verified`
                : "no verdicts yet"
              : "replay equivalence"
          }
          delta={replayDelta}
          // Spark the per-bucket RATE (matches the headline), not raw counts like the
          // sibling cards — an empty bucket (no verdicts) has no rate, shown as 0.
          spark={replay ? replay.buckets.map((b) => (b.total > 0 ? (b.equivalent / b.total) * 100 : 0)) : []}
          sparkVariant="accent"
        />
      </div>

      <div className="grid-2">
        <HeroChart buckets={buckets} granularity={data.granularity} />
        <div className="panel dist-panel">
          <div className="panel-h">
            <h2>Findings distribution</h2>
            <div className="sub">
              {cur.findings} in {data.window}
              {snapshotDate ? ` (ending ${snapshotDate})` : ""}
            </div>
          </div>
          <div className="panel-b">
            <div className="dist-sub-h">by severity</div>
            <SeverityBar distribution={sev} />
            <div className="dist-sub-h">by evidence tier</div>
            <TierRows distribution={data.evidence_tier_distribution} />
          </div>
        </div>
      </div>
    </>
  );
}
