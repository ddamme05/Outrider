// Findings distribution by SEVERITY — a stacked horizontal segmented bar (mockup
// .seg-bar) + a swatch legend. Severity is policy-set and read straight off the
// aggregate (display only, never re-derived). Each segment's width is its share of
// the total; counts are printed inside. Zero-count severities are dropped from the
// bar (no zero-width segment) but still listed in the legend so the full 5-value
// enum stays visible and honest.
import { SEVERITY_ORDER } from "../lib/metrics";

export function SeverityBar({ distribution }: { distribution: Record<string, number> }) {
  const rows = SEVERITY_ORDER.map((sev) => ({ sev, count: distribution[sev] ?? 0 }));
  const total = rows.reduce((s, r) => s + r.count, 0);

  return (
    <div className="dist-block">
      {total === 0 ? (
        <div className="dist-empty">No findings in this window.</div>
      ) : (
        <div className="seg-bar" role="img" aria-label="findings by severity">
          {rows
            .filter((r) => r.count > 0)
            .map((r) => (
              <div
                key={r.sev}
                className="seg"
                style={{ flex: r.count, background: `var(--sev-${r.sev})` }}
                title={`${r.sev}: ${r.count}`}
              >
                {r.count}
              </div>
            ))}
        </div>
      )}
      <div className="seg-legend">
        {rows.map((r) => (
          <span key={r.sev} className="seg-leg-item">
            <span className="sw" style={{ background: `var(--sev-${r.sev})` }} aria-hidden="true" />
            {r.sev} <b>{r.count}</b>
          </span>
        ))}
      </div>
    </div>
  );
}
