import { Link } from "react-router";

import { $api } from "../api/client";
import type { components } from "../api/schema";
import { StatusPill } from "../components/StatusPill";
import { REVIEW_STATUSES, type ReviewStatus, useFilters } from "../state/filters";

type ReviewListItem = components["schemas"]["ReviewListItem"];

// Display grouping (the mockup's status sections), in render order.
const GROUP_ORDER = ["Awaiting", "Running", "Completed", "Failed", "Skipped"] as const;
type Group = (typeof GROUP_ORDER)[number];

const STATUS_GROUP: Record<string, Group> = {
  awaiting_approval: "Awaiting",
  awaiting_approval_expired: "Awaiting",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  skipped: "Skipped",
};

export function Reviews() {
  const includeEval = useFilters((s) => s.includeEval);
  const status = useFilters((s) => s.status);
  const search = useFilters((s) => s.search);
  const setStatus = useFilters((s) => s.setStatus);

  // Request the backend max in one call (limit≤200). The mockup is a single
  // scrolling list with no pager, and there is no server-side text search — so
  // client search must run over the loaded set. Past 200 we don't silently
  // truncate; we surface an honest notice (below) and lean on the status filter
  // to narrow. True offset/limit pagination is a follow-up if a deployment
  // outgrows this.
  const { data, error, isLoading } = $api.useQuery(
    "get",
    "/api/reviews",
    { params: { query: { include_eval: includeEval, status: status ?? undefined, limit: 200 } } },
    { refetchInterval: 2000 },
  );

  if (isLoading) {
    return <p>Loading…</p>;
  }
  if (error) {
    return <p className="error">Failed to load reviews.</p>;
  }

  const loaded = data?.reviews ?? [];
  // The queue holds more reviews than this page loaded — search/grouping below
  // only see `loaded`, so we say so rather than implying the page is the whole.
  const truncated = (data?.total ?? 0) > loaded.length;

  const term = search.trim().toLowerCase();
  const rows = loaded.filter((review) => {
    if (term === "") {
      return true;
    }
    const haystack =
      `${review.repo_id} ${review.pr_number} ${review.status} ${review.head_sha} ${review.id}`.toLowerCase();
    return haystack.includes(term);
  });

  const groups = new Map<Group, ReviewListItem[]>();
  for (const review of rows) {
    const group = STATUS_GROUP[review.status] ?? "Completed";
    const bucket = groups.get(group) ?? [];
    bucket.push(review);
    groups.set(group, bucket);
  }

  return (
    <section>
      <div className="filterbar">
        <select
          className="filter-select"
          aria-label="Filter by status"
          value={status ?? ""}
          onChange={(event) =>
            setStatus(event.target.value === "" ? null : (event.target.value as ReviewStatus))
          }
        >
          <option value="">All statuses</option>
          {REVIEW_STATUSES.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
        <span className="spacer" />
      </div>

      {truncated ? (
        <p className="queue-notice">
          Showing {loaded.length} of {data?.total} reviews. Search and grouping
          cover only the loaded set — narrow with the status filter to reach the
          rest.
        </p>
      ) : null}

      {rows.length === 0 ? (
        <p style={{ color: "var(--muted)" }}>
          {term !== "" && truncated
            ? "No matches in the loaded reviews — narrow by status to search the rest."
            : "No reviews."}
        </p>
      ) : (
        GROUP_ORDER.filter((group) => groups.has(group)).map((group) => {
          const bucket = groups.get(group) ?? [];
          return (
            <div key={group}>
              <div className="group-label">
                {group}
                <span className="badge">{bucket.length}</span>
              </div>
              <div className="rlist">
                {bucket.map((review) => (
                  <div className="rrow" key={review.id}>
                    <div className="r-status">
                      <StatusPill status={review.status} />
                    </div>
                    <div className="r-main">
                      <div className="r-title">
                        <Link to={`/reviews/${review.id}`}>repo {review.repo_id}</Link>
                        <span className="prnum">#{review.pr_number}</span>
                        {review.is_eval ? <span className="eval-tag mono">is_eval</span> : null}
                      </div>
                      <div className="r-sub">
                        <span className="mono">{review.head_sha.slice(0, 9)}</span>
                        <span>{review.metrics.llm_calls_made} calls</span>
                      </div>
                    </div>
                    <div className="r-cost">${review.metrics.total_cost_usd.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })
      )}
    </section>
  );
}
