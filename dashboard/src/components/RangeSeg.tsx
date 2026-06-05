// Window selector for the Overview analytics (mockup .rangeseg): 24h | 7d | 30d,
// a mutually-exclusive segmented control. The selected window is the /api/metrics
// query param, so changing it refetches the whole analytics surface. Lives on the
// Overview (not the shared topbar) because it scopes Overview's data only — a
// global topbar control would be inert on every other route.
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
      {WINDOWS.map((w) => (
        <button
          key={w}
          type="button"
          className="rseg"
          aria-pressed={w === value}
          onClick={() => onChange(w)}
        >
          {w}
        </button>
      ))}
    </div>
  );
}
