// The Overview hero chart (mockup left panel): cost-per-day or reviews-per-day as
// a line/area or bar series, with gridlines, axis labels, a stats legend, and
// keyboard-focusable point tooltips. Hand-rolled inline SVG (no charting dep) so
// it references the Signal theme tokens directly and renders honest zeros — an
// empty window draws a flat baseline, never an interpolated curve (DECISIONS#039).
import { useState } from "react";

import {
  formatBucketLabel,
  seriesStats,
  thinLabels,
  type MetricBucket,
} from "../lib/metrics";

type Series = "cost" | "reviews";
type ChartType = "line" | "bar";

const VB_W = 640;
const VB_H = 168;
const PAD = { l: 40, r: 14, t: 14, b: 22 };
const PLOT_W = VB_W - PAD.l - PAD.r;
const PLOT_H = VB_H - PAD.t - PAD.b;
const BASELINE = PAD.t + PLOT_H;

export function HeroChart({
  buckets,
  granularity,
}: {
  buckets: MetricBucket[];
  granularity: string;
}) {
  const [series, setSeries] = useState<Series>("cost");
  const [type, setType] = useState<ChartType>("line");
  const [hover, setHover] = useState<number | null>(null);

  const values = buckets.map((b) => (series === "cost" ? b.cost_usd : b.reviews));
  // Per-bucket completeness (openai-native-host arc): an incomplete cost bucket is a
  // LOWER BOUND — mark it in the tooltip/aria rather than rendering it as exact.
  const complete = buckets.map((b) => (series === "cost" ? b.cost_complete !== false : true));
  const labels = buckets.map((b) => formatBucketLabel(b.bucket, granularity));
  const stats = seriesStats(values);
  const n = values.length;

  const fmt = (v: number): string =>
    series === "cost" ? `$${v.toFixed(2)}` : String(Math.round(v));
  const fmtAxis = (v: number): string =>
    series === "cost" ? `$${v.toFixed(1)}` : String(Math.round(v));

  // Headroom so the peak doesn't touch the top gridline; floor at 1 to avoid /0 on
  // an all-zero (honest-empty) window — the series then flatlines at the baseline.
  const yMax = stats.peak > 0 ? stats.peak * 1.1 : 1;
  const x = (i: number): number => (n <= 1 ? PAD.l + PLOT_W / 2 : PAD.l + (i / (n - 1)) * PLOT_W);
  const y = (v: number): number => PAD.t + PLOT_H - (v / yMax) * PLOT_H;

  const linePts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const areaPath =
    n >= 2
      ? `M ${x(0).toFixed(1)},${BASELINE} L ${values
          .map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`)
          .join(" L ")} L ${x(n - 1).toFixed(1)},${BASELINE} Z`
      : "";
  const barW = n > 0 ? Math.min(28, (PLOT_W / n) * 0.6) : 0;

  const gridTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTickLabels = thinLabels(labels, 7);

  // null on an all-zero window (peakIndex -1) → the legend omits "on <day>" rather than
  // claiming a peak day that didn't happen (honest-zeros).
  const peakDay = stats.peakIndex >= 0 ? (labels[stats.peakIndex] ?? null) : null;

  return (
    <div className="panel hero-chart">
      <div className="chart-head">
        <div className="chart-titles">
          <h2 className="chart-title">{series === "cost" ? "Cost per day" : "Reviews per day"}</h2>
          <div className="chart-sub">honest aggregation · focus a point for the value</div>
        </div>
        <div className="chart-controls">
          <div className="chart-toggle" role="group" aria-label="series">
            <button type="button" aria-pressed={series === "cost"} onClick={() => setSeries("cost")}>
              Cost
            </button>
            <button
              type="button"
              aria-pressed={series === "reviews"}
              onClick={() => setSeries("reviews")}
            >
              Reviews
            </button>
          </div>
          <div className="chart-toggle" role="group" aria-label="chart type">
            <button type="button" aria-pressed={type === "line"} onClick={() => setType("line")}>
              Line
            </button>
            <button type="button" aria-pressed={type === "bar"} onClick={() => setType("bar")}>
              Bar
            </button>
          </div>
        </div>
      </div>

      <div className="chart-plot">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className={`chart-svg type-${type}`} role="img"
          aria-label={`${series} per ${granularity === "hour" ? "hour" : "day"}`}>
          {/* gridlines + y-axis labels */}
          {gridTicks.map((t) => {
            const gy = PAD.t + PLOT_H - t * PLOT_H;
            return (
              <g key={t}>
                <line x1={PAD.l} y1={gy} x2={VB_W - PAD.r} y2={gy} className="chart-grid" />
                <text x={PAD.l - 6} y={gy + 3} className="chart-ytick" textAnchor="end">
                  {fmtAxis(t * yMax)}
                </text>
              </g>
            );
          })}

          {/* data: area+line OR bars */}
          {type === "line" ? (
            <>
              {areaPath ? <path d={areaPath} className="chart-area" /> : null}
              {n >= 2 ? (
                <polyline points={linePts} className="chart-line" fill="none" />
              ) : n === 1 ? (
                <circle cx={x(0)} cy={y(values[0] ?? 0)} r={3} className="chart-dot" />
              ) : null}
            </>
          ) : (
            values.map((v, i) => (
              <rect
                key={i}
                x={x(i) - barW / 2}
                y={y(v)}
                width={barW}
                height={Math.max(0, BASELINE - y(v))}
                className="chart-bar"
              />
            ))
          )}

          {/* x-axis labels (thinned) */}
          {xTickLabels.map((l, i) =>
            l ? (
              <text key={i} x={x(i)} y={VB_H - 6} className="chart-xtick" textAnchor="middle">
                {l}
              </text>
            ) : null,
          )}

          {/* focusable hit areas + hover/focus markers */}
          {values.map((v, i) => (
            <g key={i}>
              {hover === i ? <circle cx={x(i)} cy={y(v)} r={3.5} className="chart-marker" /> : null}
              <rect
                x={x(i) - (n > 1 ? PLOT_W / n / 2 : PLOT_W / 2)}
                y={PAD.t}
                width={n > 1 ? PLOT_W / n : PLOT_W}
                height={PLOT_H}
                className="chart-hit"
                tabIndex={0}
                role="button"
                aria-label={`${labels[i]}: ${complete[i] ? "" : "\u2265"}${fmt(v)}`}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              />
            </g>
          ))}
        </svg>

        {hover !== null ? (
          <div
            className="chart-tip"
            style={{ left: `${(x(hover) / VB_W) * 100}%` }}
            role="status"
          >
            <span className="tip-val">
              {complete[hover] ? "" : "\u2265"}
              {fmt(values[hover] ?? 0)}
            </span>
            <span className="tip-day">{labels[hover] ?? ""}</span>
          </div>
        ) : null}
      </div>

      <div className="chart-legend">
        <span>
          <b>
            {complete.every(Boolean) ? "" : "\u2265"}
            {fmt(stats.total)}
          </b>{" "}
          total
        </span>
        <span>
          <b>{fmt(stats.avg)}</b> avg/{granularity === "hour" ? "hr" : "day"}
        </span>
        <span>
          peak <b>{fmt(stats.peak)}</b>
          {peakDay ? ` on ${peakDay}` : ""}
        </span>
      </div>
    </div>
  );
}
