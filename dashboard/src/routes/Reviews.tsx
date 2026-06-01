import { Link } from "react-router";

import { $api } from "../api/client";

export function Reviews() {
  // ~2s polling per the architecture doc; query key is [method, path, params].
  const { data, error, isLoading } = $api.useQuery(
    "get",
    "/api/reviews",
    {},
    { refetchInterval: 2000 },
  );

  if (isLoading) {
    return <p>Loading…</p>;
  }
  if (error) {
    return <p className="error">Failed to load reviews.</p>;
  }

  const reviews = data?.reviews ?? [];

  return (
    <section>
      <h1>Reviews</h1>
      {reviews.length === 0 ? (
        <p style={{ color: "var(--text-2)" }}>No reviews yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Repo / PR</th>
              <th>Calls</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {reviews.map((review) => (
              <tr key={review.id}>
                <td>{review.status}</td>
                <td>
                  <Link to={`/reviews/${review.id}`} className="mono">
                    {review.repo_id} #{review.pr_number}
                  </Link>
                </td>
                <td className="mono">{review.metrics.llm_calls_made}</td>
                <td className="mono">${review.metrics.total_cost_usd.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
