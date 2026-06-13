import { $api } from "../api/client";
import { Modal } from "./Modal";

// The FindingType → severity table for a review's policy_version (FUP-132),
// rendered as a modal dialog opened by the policy chip. Fetched from
// GET /api/policy/{version} — the STORED versioned policy, so a historical review
// shows the table it was actually classified under. Severity is policy-set +
// versioned; the model never sets it. Shell (backdrop/Escape/close) is shared
// via Modal; this component owns only the policy-table body.
export function PolicyModal({ version, onClose }: { version: string; onClose: () => void }) {
  const { data, error, isLoading } = $api.useQuery("get", "/api/policy/{version}", {
    params: { path: { version } },
  });

  return (
    <Modal
      ariaLabel={`Severity policy ${version}`}
      title={`Severity policy ${version}`}
      sub="FindingType → severity. Deterministic, versioned — the model never sets it."
      closeLabel="Close policy"
      onClose={onClose}
    >
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
    </Modal>
  );
}
