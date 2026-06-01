import { $api } from "../api/client";

// The FindingType → severity table for a review's policy_version (FUP-132).
// Fetched from GET /api/policy/{version} — the STORED versioned policy, so a
// historical review shows the table it was actually classified under. Severity is
// policy-set + versioned; the model never sets it.
export function PolicyTable({ version }: { version: string }) {
  const { data, error, isLoading } = $api.useQuery("get", "/api/policy/{version}", {
    params: { path: { version } },
  });

  if (isLoading) {
    return <p className="policy-status">Loading policy {version}…</p>;
  }
  if (error) {
    return <p className="policy-status error">Couldn't load policy {version}.</p>;
  }
  if (!data) {
    return null;
  }

  return (
    <div className="card policy-pop">
      <div className="pc-head">
        <span className="chip policy mono">policy {data.version}</span>
        <span className="muted">
          FindingType → severity. Deterministic, versioned — the model never sets it.
        </span>
      </div>
      <table className="policy">
        <thead>
          <tr>
            <th>FindingType</th>
            <th>Dimension</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>
          {data.entries.map((e) => (
            <tr key={e.finding_type}>
              <td className="ft mono">{e.finding_type}</td>
              <td className="muted">{e.dimension}</td>
              <td>
                <span className={`pill sev-pill sev-${e.severity}`}>
                  <span className="dot" aria-hidden="true" />
                  {e.severity}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
