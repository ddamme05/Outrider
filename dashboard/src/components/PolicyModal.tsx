import { $api } from "../api/client";
import { Modal } from "./Modal";

// Read the stored policy for the review's version, including historical reviews (FUP-132).
// Modal owns closing behavior; this component loads and displays the policy table.
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
            {data.entries.map((policyEntry) => (
              <tr key={policyEntry.finding_type}>
                <td className="ft mono">{policyEntry.finding_type}</td>
                <td className="muted">{policyEntry.dimension}</td>
                <td>
                  <span className={`pill sev-pill sev-${policyEntry.severity}`}>
                    <span className="dot" aria-hidden="true" />
                    {policyEntry.severity}
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
