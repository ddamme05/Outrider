import { Link } from "react-router";

import { $api } from "../api/client";
import { StatusPill } from "../components/StatusPill";
import type { components } from "../api/schema";

type ReviewListItem = components["schemas"]["ReviewListItem"];

// The "Needs your decision" rail: reviews sitting at the HITL gate. We filter
// SERVER-SIDE per status, not client-side off an unfiltered page — the list is
// ordered created_at DESC and paginated before returning, so an unfiltered first
// page of 200 newest rows could bury an older awaiting_approval blocker. The
// status filter is single-valued, so we issue one query per awaiting status and
// merge. The list contract carries no per-review finding count, so we don't
// fabricate a gated count; the rail routes into detail where the decision UX lives.
export function Home() {
  const awaiting = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: false, status: "awaiting_approval", limit: 200 } } },
    { refetchInterval: 2000 },
  );
  const expired = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: false, status: "awaiting_approval_expired", limit: 200 } } },
    { refetchInterval: 2000 },
  );

  if (awaiting.isLoading || expired.isLoading) {
    return <p>Loading…</p>;
  }
  if (awaiting.error || expired.error) {
    return <p className="error">Failed to load the decision queue.</p>;
  }

  const rows: ReviewListItem[] = [...(awaiting.data?.reviews ?? []), ...(expired.data?.reviews ?? [])].sort(
    (a, b) => b.created_at.localeCompare(a.created_at),
  );
  // Honest truncation: either status filter could itself exceed the page.
  const loaded = (awaiting.data?.reviews.length ?? 0) + (expired.data?.reviews.length ?? 0);
  const total = (awaiting.data?.total ?? 0) + (expired.data?.total ?? 0);
  const truncated = total > loaded;

  return (
    <section>
      <h1>Needs your decision</h1>

      {truncated ? (
        <p className="queue-notice">
          Showing {loaded} of {total} reviews awaiting a decision — the oldest may
          be beyond this page.
        </p>
      ) : null}

      {rows.length === 0 ? (
        <p style={{ color: "var(--text-2)" }}>Nothing is awaiting your decision.</p>
      ) : (
        <div className="rlist">
          {rows.map((r) => (
            <div className="rrow" key={r.id}>
              <div className="r-status">
                <StatusPill status={r.status} />
              </div>
              <div className="r-main">
                <div className="r-title">
                  <Link to={`/reviews/${r.id}`}>repo {r.repo_id}</Link>
                  <span className="prnum">#{r.pr_number}</span>
                  {r.is_eval ? <span className="eval-tag mono">is_eval</span> : null}
                </div>
                <div className="r-sub">
                  {/* expires_at is detail-only; the list contract omits it. */}
                  <span className="mono">{r.head_sha.slice(0, 9)}</span>
                </div>
              </div>
              <div className="r-cost">${r.metrics.total_cost_usd.toFixed(2)}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
