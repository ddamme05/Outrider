import { Link } from "react-router";

import { $api } from "../api/client";
import type { components } from "../api/schema";
import { StatCard } from "../components/StatCard";
import { StatusPill } from "../components/StatusPill";
import { useFilters } from "../state/filters";

type ReviewListItem = components["schemas"]["ReviewListItem"];

// The Overview is the operational landing screen. Stat cards show CURRENT counts
// from server-side totals (each by-status count is a status-filtered query's
// `total`, never a count of a loaded page — the queue is paginated). NO deltas,
// sparklines, charts, or time-range: those need a metrics/time-series endpoint we
// don't have, and inventing them would fabricate analytics (spec non-goal). Below
// the cards is the "needs your decision" HITL rail (server-side status-filtered).
export function Overview() {
  const includeEval = useFilters((s) => s.includeEval);
  const opts = { refetchInterval: 2000 } as const;
  // Unfiltered: grand total + the loaded rows we sum cost over.
  const all = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: includeEval, limit: 200 } } },
    opts,
  );
  // Status-filtered counts use each query's `total` (server-side), not loaded rows.
  // awaiting/expired keep limit 200 because they ALSO feed the rail rows below.
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
  const completed = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: includeEval, status: "completed", limit: 1 } } },
    opts,
  );
  const failed = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: includeEval, status: "failed", limit: 1 } } },
    opts,
  );

  if (all.isLoading) {
    return <p>Loading…</p>;
  }
  if (all.error) {
    return <p className="error">Failed to load the overview.</p>;
  }

  const count = (query: { data?: { total: number } | undefined }): string =>
    query.data ? String(query.data.total) : "…";

  // HITL visibility fails CLOSED. The two approval-queue queries gate the rail
  // (and the "Awaiting decision" card) independently of `all`: if either errors
  // before it ever resolved, never render a confident count or an "all clear"
  // empty rail. The backend gate still holds the review at AWAITING_APPROVAL —
  // this guards the operator's window into it, so "couldn't check" never reads as
  // "nothing awaiting." Once resolved, react-query keeps last-good data across a
  // failed background refetch, so a stale-but-real count/list still shows.
  const awaitingResolved = awaiting.data !== undefined && expired.data !== undefined;
  const awaitingError = Boolean(awaiting.error || expired.error);
  const awaitingTotal = awaitingResolved
    ? String((awaiting.data?.total ?? 0) + (expired.data?.total ?? 0))
    : awaitingError
      ? "—"
      : "…";
  const loadedCost = (all.data?.reviews ?? []).reduce(
    (sum, r) => sum + r.metrics.total_cost_usd,
    0,
  );

  const railRows: ReviewListItem[] = [
    ...(awaiting.data?.reviews ?? []),
    ...(expired.data?.reviews ?? []),
  ].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const railLoaded = railRows.length;
  const railTotal = (awaiting.data?.total ?? 0) + (expired.data?.total ?? 0);

  return (
    <section>
      <div className="stat-row">
        <StatCard label="Reviews" value={count(all)} cap="total" />
        <StatCard label="Awaiting decision" value={awaitingTotal} cap="at the HITL gate" />
        <StatCard label="Completed" value={count(completed)} />
        <StatCard label="Failed" value={count(failed)} />
        <StatCard label="Cost" value={`$${loadedCost.toFixed(2)}`} cap="across loaded reviews" />
      </div>

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
