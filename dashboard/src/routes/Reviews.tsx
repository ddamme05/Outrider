import { Link, useNavigate } from "react-router";

import { $api } from "../api/client";
import type { components } from "../api/schema";
import { StatusPill } from "../components/StatusPill";
import { formatReviewCost } from "../lib/format";
import { SEVERITY_ORDER } from "../lib/metrics";
import { type ReviewStatus, useFilters } from "../state/filters";

type ReviewListItem = components["schemas"]["ReviewListItem"];
type SeverityCounts = components["schemas"]["SeverityCounts"];

// Skipped reviews appear under All; expired is the short approval-expiry label.
const STATUS_CHIPS: { label: string; status: ReviewStatus }[] = [
  { label: "running", status: "running" },
  { label: "awaiting_approval", status: "awaiting_approval" },
  { label: "expired", status: "awaiting_approval_expired" },
  { label: "completed", status: "completed" },
  { label: "failed", status: "failed" },
];

// Every severity must have a tally class.
const SEV_CLASS: Record<(typeof SEVERITY_ORDER)[number], string> = {
  critical: "c",
  high: "h",
  medium: "m",
  low: "l",
  info: "i",
};

function formatLatency(seconds: number | null): string {
  if (seconds == null) {
    return "—";
  }
  const total = Math.round(seconds);
  if (total < 60) {
    return `${total}s`;
  }
  const minutes = Math.floor(total / 60);
  const remainingSeconds = total % 60;
  return `${minutes}m${remainingSeconds.toString().padStart(2, "0")}s`;
}

function SeverityTally({ counts }: { counts: SeverityCounts | null }) {
  // Null means synthesis has not recorded a tally; an empty tally also shows —.
  if (counts == null) {
    return <span className="row-repo">—</span>;
  }
  const visibleSeverities = SEVERITY_ORDER.filter((severity) => counts[severity] > 0);
  if (visibleSeverities.length === 0) {
    return <span className="row-repo">—</span>;
  }
  return (
    <span className="tally">
      {visibleSeverities.map((severity) => (
        <span key={severity} className={`t ${SEV_CLASS[severity]}`} title={`${counts[severity]} ${severity}`}>
          {counts[severity]}
        </span>
      ))}
    </span>
  );
}

export function Reviews() {
  const includeEval = useFilters((filters) => filters.includeEval);
  const status = useFilters((filters) => filters.status);
  const search = useFilters((filters) => filters.search);
  const setStatus = useFilters((filters) => filters.setStatus);
  const navigate = useNavigate();

  // Text search covers the loaded page only; request the maximum and show truncation.
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

  const loadedReviews = data?.reviews ?? [];
  const truncated = (data?.total ?? 0) > loadedReviews.length;

  // Server counts include every status, regardless of the active status filter.
  const statusCounts = data?.status_counts;
  const allCount = statusCounts
    ? Object.values(statusCounts).reduce((sum, count) => sum + count, 0)
    : null;

  const searchTerm = search.trim().toLowerCase();
  const matchingReviews = loadedReviews.filter((review) => {
    if (searchTerm === "") {
      return true;
    }
    const searchText =
      `${review.repo_full_name ?? review.repo_id} ${review.pr_title ?? ""} ${review.pr_number} ${review.status} ${review.head_sha} ${review.id}`.toLowerCase();
    return searchText.includes(searchTerm);
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
          Showing {loadedReviews.length} of {data?.total} reviews. Search covers only
          the loaded set — narrow with the status filter to reach the rest.
        </p>
      ) : null}

      {matchingReviews.length === 0 ? (
        <p style={{ color: "var(--muted)" }}>
          {searchTerm !== "" && truncated
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
              {matchingReviews.map((review: ReviewListItem) => (
                <ReviewRow
                  key={review.id}
                  review={review}
                  onOpen={() => navigate(`/reviews/${review.id}`)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function ReviewRow({ review, onOpen }: { review: ReviewListItem; onOpen: () => void }) {
  return (
    // Links keep their own destinations and keyboard behavior inside the clickable row.
    <tr
      key={review.id}
      className="review-row"
      onClick={onOpen}
    >
      <td className="row-id">
        <Link to={`/reviews/${review.id}`} onClick={(event) => event.stopPropagation()}>
          #{review.pr_number}
        </Link>
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
            onClick={(event) => event.stopPropagation()}
            aria-label={`Open the replay reconstruction for review ${review.id}`}
          >
            ↻ replay
          </Link>
        ) : null}
      </td>
      <td>
        <SeverityTally counts={review.severity_counts} />
      </td>
      <td className="row-cost">
        {formatReviewCost(review.metrics)}
      </td>
      <td className="row-lat">{formatLatency(review.metrics.wall_clock_seconds)}</td>
    </tr>
  );
}
