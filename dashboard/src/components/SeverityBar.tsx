// Display policy-assigned severity counts as proportional segments and a legend.
// Zero-count severities remain in the legend; no severity is derived here.
import { SEVERITY_ORDER } from "../lib/metrics";

export function SeverityBar({ distribution }: { distribution: Record<string, number> }) {
  const severityRows = SEVERITY_ORDER.map((sev) => ({ sev, count: distribution[sev] ?? 0 }));
  const total = severityRows.reduce((count, severityRow) => count + severityRow.count, 0);

  return (
    <div className="dist-block">
      {total === 0 ? (
        <div className="dist-empty">No findings in this window.</div>
      ) : (
        <div className="seg-bar" role="img" aria-label="findings by severity">
          {severityRows
            .filter((severityRow) => severityRow.count > 0)
            .map((severityRow) => (
              <div
                key={severityRow.sev}
                className="seg"
                style={{ flex: severityRow.count, background: `var(--sev-${severityRow.sev})` }}
                title={`${severityRow.sev}: ${severityRow.count}`}
              >
                {severityRow.count}
              </div>
            ))}
        </div>
      )}
      <div className="seg-legend">
        {severityRows.map((severityRow) => (
          <span key={severityRow.sev} className="seg-leg-item">
            <span className="sw" style={{ background: `var(--sev-${severityRow.sev})` }} aria-hidden="true" />
            {severityRow.sev} <b>{severityRow.count}</b>
          </span>
        ))}
      </div>
    </div>
  );
}
