import { Link } from "react-router";

import { $api } from "../api/client";
import type { components } from "../api/schema";
import { StatusPill } from "../components/StatusPill";
import { type ReviewStatus, useFilters } from "../state/filters";

type ReviewListItem = components["schemas"]["ReviewListItem"];
type SeverityCounts = components["schemas"]["SeverityCounts"];

// The filter chips (mockup order). "expired" is the human label for the
// `awaiting_approval_expired` status; "All" is the null filter. `skipped`
// has no chip (matches the mockup) — those reviews surface only under "All".
const STATUS_CHIPS: { label: string; status: ReviewStatus }[] = [
  { label: "running", status: "running" },
  { label: "awaiting_approval", status: "awaiting_approval" },
  { label: "expired", status: "awaiting_approval_expired" },
  { label: "completed", status: "completed" },
  { label: "failed", status: "failed" },
];

// Severity tiers in display order, mapped to the mockup's tally classes.
const SEV_TIERS = [
  { key: "critical", cls: "c" },
  { key: "high", cls: "h" },
  { key: "medium", cls: "m" },
  { key: "low", cls: "l" },
  { key: "info", cls: "i" },
] as const;

function formatLatency(seconds: number | null): string {
  if (seconds == null) {
    return "—";
  }
  const total = Math.round(seconds);
  if (total < 60) {
    return `${total}s`;
  }
  const minutes = Math.floor(total / 60);
  const rem = total % 60;
  return `${minutes}m${rem.toString().padStart(2, "0")}s`;
}

function SeverityTally({ counts }: { counts: SeverityCounts | null }) {
  // `null` = the review hasn't reached synthesize, so there is no
  // report-equivalent set to count (rendered as an em-dash, like cost/latency
  // pending). A present-but-all-zero tally also renders as "—".
  if (counts == null) {
    return <span className="row-repo">—</span>;
  }
  const tiers = SEV_TIERS.filter(({ key }) => counts[key] > 0);
  if (tiers.length === 0) {
    return <span className="row-repo">—</span>;
  }
  return (
    <span className="tally">
      {tiers.map(({ key, cls }) => (
        <span key={key} className={`t ${cls}`} title={`${counts[key]} ${key}`}>
          {counts[key]}
        </span>
      ))}
    </span>
  );
}

export function Reviews() {
  const includeEval = useFilters((s) => s.includeEval);
  const status = useFilters((s) => s.status);
  const search = useFilters((s) => s.search);
  const setStatus = useFilters((s) => s.setStatus);

  // Request the backend max in one call (limit≤200). The page is a single
  // scrolling table with no pager, and there is no server-side text search —
  // so client search runs over the loaded set. Past 200 we surface an honest
  // notice (below) rather than silently truncating.
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
  const truncated = (data?.total ?? 0) > loaded.length;

  // Chip counts come from the response's status_counts (computed server-side
  // over the base filters, independent of the active status). "All" = the sum.
  const statusCounts = data?.status_counts;
  const allCount = statusCounts
    ? Object.values(statusCounts).reduce((sum, n) => sum + n, 0)
    : null;

  const term = search.trim().toLowerCase();
  const rows = loaded.filter((review) => {
    if (term === "") {
      return true;
    }
    const haystack =
      `${review.repo_full_name ?? review.repo_id} ${review.pr_title ?? ""} ${review.pr_number} ${review.status} ${review.head_sha} ${review.id}`.toLowerCase();
    return haystack.includes(term);
  });

  return (
    <section>
      <div className="panel-h">
        <h2>Reviews</h2>
        <div className="right filter-row" role="group" aria-label="Filter by status">
          <button
            type="button"
            className="fbtn"
            aria-pressed={status === null}
            onClick={() => setStatus(null)}
          >
            All{allCount === null ? null : <span className="num"> {allCount}</span>}
          </button>
          {STATUS_CHIPS.map((chip) => (
            <button
              type="button"
              key={chip.status}
              className="fbtn"
              aria-pressed={status === chip.status}
              onClick={() => setStatus(status === chip.status ? null : chip.status)}
            >
              {chip.label}
              {statusCounts ? <span className="num"> {statusCounts[chip.status]}</span> : null}
            </button>
          ))}
        </div>
      </div>

      {truncated ? (
        <p className="queue-notice">
          Showing {loaded.length} of {data?.total} reviews. Search covers only
          the loaded set — narrow with the status filter to reach the rest.
        </p>
      ) : null}

      {rows.length === 0 ? (
        <p style={{ color: "var(--muted)" }}>
          {term !== "" && truncated
            ? "No matches in the loaded reviews — narrow by status to search the rest."
            : "No reviews."}
        </p>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>PR</th>
                <th>Repo</th>
                <th>Title</th>
                <th>Status</th>
                <th>Severity</th>
                <th>Cost</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((review: ReviewListItem) => (
                <tr key={review.id}>
                  <td className="row-id">
                    <Link to={`/reviews/${review.id}`}>#{review.pr_number}</Link>
                  </td>
                  <td className="row-repo">{review.repo_full_name ?? `repo ${review.repo_id}`}</td>
                  <td>
                    {review.pr_title ?? <span className="row-repo">—</span>}
                    {review.is_eval ? <span className="eval-tag mono"> is_eval</span> : null}
                  </td>
                  <td>
                    <StatusPill status={review.status} />
                    {review.status === "completed" ? (
                      <Link
                        to={`/reviews/${review.id}/replay`}
                        className="abtn"
                        style={{ marginLeft: 8 }}
                        aria-label={`Open the replay reconstruction for review ${review.id}`}
                      >
                        ↻ replay
                      </Link>
                    ) : null}
                  </td>
                  <td>
                    <SeverityTally counts={review.severity_counts} />
                  </td>
                  <td className="row-cost">${review.metrics.total_cost_usd.toFixed(2)}</td>
                  <td className="row-lat">{formatLatency(review.metrics.wall_clock_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
