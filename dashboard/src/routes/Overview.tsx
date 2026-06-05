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
import { deltaInfo } from "../lib/metrics";
import { useFilters } from "../state/filters";

type ReviewListItem = components["schemas"]["ReviewListItem"];

// The Overview is the operational landing screen + the Signal analytics surface
// (DECISIONS#039). The KPI cards, sparklines, hero chart, and distributions are
// REAL read-only aggregations over the audit stream (GET /api/metrics, windowed by
// the range selector) — honest zeros on a sparse window, never an invented series.
// Below the analytics is the "needs your decision" HITL rail (server-side
// status-filtered, fails closed). Replay-% is intentionally absent — it has no
// cross-review aggregate source yet (sibling replay-verdict-projection feature).
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
        <Analytics data={d} stale={Boolean(metrics.error)} />
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
                    <span className="nd-cost">${r.metrics.total_cost_usd.toFixed(2)}</span>
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
  stale,
}: {
  data: components["schemas"]["DashboardMetricsResponse"];
  stale: boolean;
}) {
  const cur = data.deltas.current;
  const prev = data.deltas.previous;
  const sev = data.severity_distribution;
  const buckets = data.buckets;
  const avgCost = cur.reviews > 0 ? cur.cost_usd / cur.reviews : 0;

  return (
    <>
      {stale ? (
        <p className="queue-notice" role="alert">
          Couldn&rsquo;t refresh metrics — showing the last loaded window.
        </p>
      ) : null}

      <div className="stat-row">
        <MetricCard
          label="Reviews"
          value={cur.reviews}
          cap={`last ${data.window}`}
          delta={deltaInfo(cur.reviews, prev.reviews, "up-good")}
          spark={buckets.map((b) => b.reviews)}
          sparkVariant="accent"
        />
        <MetricCard
          label="Cost"
          value={`$${cur.cost_usd.toFixed(2)}`}
          cap={`$${avgCost.toFixed(2)} avg / review`}
          delta={deltaInfo(cur.cost_usd, prev.cost_usd, "up-bad")}
          spark={buckets.map((b) => b.cost_usd)}
          sparkVariant="neg"
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
          cap={`last ${data.window}`}
          delta={deltaInfo(cur.failed, prev.failed, "up-bad")}
          spark={buckets.map((b) => b.failed)}
          sparkVariant="pos"
        />
      </div>

      <div className="grid-2">
        <HeroChart buckets={buckets} granularity={data.granularity} />
        <div className="panel dist-panel">
          <div className="panel-h">
            <h2>Findings distribution</h2>
            <div className="sub">
              {cur.findings} in {data.window}
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
