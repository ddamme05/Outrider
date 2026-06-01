import { useEffect } from "react";

import { $api } from "../api/client";

// The FindingType → severity table for a review's policy_version (FUP-132),
// rendered as a modal dialog opened by the policy chip. Fetched from
// GET /api/policy/{version} — the STORED versioned policy, so a historical review
// shows the table it was actually classified under. Severity is policy-set +
// versioned; the model never sets it. Closes on backdrop click or Escape.
export function PolicyModal({ version, onClose }: { version: string; onClose: () => void }) {
  const { data, error, isLoading } = $api.useQuery("get", "/api/policy/{version}", {
    params: { path: { version } },
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={`Severity policy ${version}`}
      onClick={onClose}
    >
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-h">
          <div>
            <h2>Severity policy {version}</h2>
            <div className="sub">
              FindingType → severity. Deterministic, versioned — the model never sets it.
            </div>
          </div>
          <button type="button" className="modal-close" aria-label="Close policy" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-b">
          {isLoading ? (
            <p className="modal-note">Loading policy {version}…</p>
          ) : error ? (
            <p className="modal-note">Couldn&rsquo;t load policy {version}.</p>
          ) : !data ? (
            <p className="modal-note">No policy data for {version}.</p>
          ) : (
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
          )}
        </div>
      </div>
    </div>
  );
}
