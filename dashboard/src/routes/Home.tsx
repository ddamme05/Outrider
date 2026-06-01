import { Link } from "react-router";

import { $api } from "../api/client";
import { StatusPill } from "../components/StatusPill";

// The "Needs your decision" rail: reviews sitting at the HITL gate. The list
// contract carries no per-review finding counts, so we don't fabricate a gated
// count here — the rail routes into the detail view, where the decision UX and
// full finding context live. Status filter is single-valued, so we fetch
// unfiltered and keep both awaiting_approval and ..._expired.
export function Home() {
  const { data, error, isLoading } = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: false, limit: 200 } } },
    { refetchInterval: 2000 },
  );

  if (isLoading) {
    return <p>Loading…</p>;
  }
  if (error) {
    return <p className="error">Failed to load the decision queue.</p>;
  }

  const awaiting = (data?.reviews ?? []).filter((r) => r.status.startsWith("awaiting_approval"));

  return (
    <section>
      <h1>Needs your decision</h1>

      {awaiting.length === 0 ? (
        <p style={{ color: "var(--text-2)" }}>Nothing is awaiting your decision.</p>
      ) : (
        <div className="rlist">
          {awaiting.map((r) => (
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
