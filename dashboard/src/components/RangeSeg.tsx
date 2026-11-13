// Choose the time range for Overview's charts. The page owns the selection
// and data queries; this control does not change filters on other pages.

export type MetricsWindow = "24h" | "7d" | "30d";

const WINDOWS: MetricsWindow[] = ["24h", "7d", "30d"];

export function RangeSeg({
  value,
  onChange,
}: {
  value: MetricsWindow;
  onChange: (next: MetricsWindow) => void;
}) {
  return (
    <div className="rangeseg" role="group" aria-label="metrics window">
      {WINDOWS.map((metricsWindow) => (
        <button
          key={metricsWindow}
          type="button"
          className="rseg"
          aria-pressed={metricsWindow === value}
          onClick={() => onChange(metricsWindow)}
        >
          {metricsWindow}
        </button>
      ))}
    </div>
  );
}
