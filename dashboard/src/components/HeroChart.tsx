// Overview's cost/review chart: select a series, scale its buckets, then render
// SVG marks, keyboard-accessible tooltips, and totals. Empty and incomplete
// series retain their actual values and limits (DECISIONS#039).
import { useState } from "react";

import {
  formatBucketLabel,
  seriesStats,
  thinLabels,
  type MetricBucket,
} from "../lib/metrics";

type Series = "cost" | "reviews";
type ChartType = "line" | "bar";

const VIEWBOX_WIDTH = 640;
const VIEWBOX_HEIGHT = 168;
const PAD = { l: 40, r: 14, t: 14, b: 22 };
const PLOT_W = VIEWBOX_WIDTH - PAD.l - PAD.r;
const PLOT_H = VIEWBOX_HEIGHT - PAD.t - PAD.b;
const BASELINE = PAD.t + PLOT_H;

export function HeroChart({
  buckets,
  granularity,
}: {
  buckets: MetricBucket[];
  granularity: string;
}) {
  const [series, setSeries] = useState<Series>("cost");
  const [chartType, setChartType] = useState<ChartType>("line");
  const [activePointIndex, setActivePointIndex] = useState<number | null>(null);

  const seriesValues = buckets.map((bucket) =>
    (series === "cost" ? bucket.cost_usd : bucket.reviews));
  // Incomplete costs are lower bounds, including in tooltips and accessible labels.
  const pointComplete = buckets.map((bucket) =>
    (series === "cost" ? bucket.cost_complete !== false : true));
  const pointLabels = buckets.map((bucket) =>
    formatBucketLabel(bucket.bucket, granularity));
  const statistics = seriesStats(seriesValues);
  const pointCount = seriesValues.length;

  const formatValue = (value: number): string =>
    series === "cost" ? `$${value.toFixed(2)}` : String(Math.round(value));
  const formatAxisValue = (value: number): string =>
    series === "cost" ? `$${value.toFixed(1)}` : String(Math.round(value));

  // Leave headroom above the peak and avoid dividing by zero for an all-zero series.
  const axisMax = statistics.peak > 0 ? statistics.peak * 1.1 : 1;
  const pointX = (index: number): number =>
    (pointCount <= 1 ? PAD.l + PLOT_W / 2 : PAD.l + (index / (pointCount - 1)) * PLOT_W);
  const pointY = (value: number): number => PAD.t + PLOT_H - (value / axisMax) * PLOT_H;

  const linePoints = seriesValues
    .map((value, index) => `${pointX(index).toFixed(1)},${pointY(value).toFixed(1)}`)
    .join(" ");
  const areaPath =
    pointCount >= 2
      ? `M ${pointX(0).toFixed(1)},${BASELINE} L ${seriesValues
          .map((value, index) => `${pointX(index).toFixed(1)},${pointY(value).toFixed(1)}`)
          .join(" L ")} L ${pointX(pointCount - 1).toFixed(1)},${BASELINE} Z`
      : "";
  const barWidth = pointCount > 0 ? Math.min(28, (PLOT_W / pointCount) * 0.6) : 0;

  const gridTicks = [0, 0.25, 0.5, 0.75, 1];
  const axisLabels = thinLabels(pointLabels, 7);

  // Zero-only series have no peak day. Incomplete costs also hide the day:
  // the largest known subtotal may not be the true peak.
  const peakDay =
    statistics.peakIndex >= 0 && pointComplete.every(Boolean)
      ? (pointLabels[statistics.peakIndex] ?? null)
      : null;

  return (
    <div className="panel hero-chart">
      <div className="chart-head">
        <div className="chart-titles">
          <h2 className="chart-title">{series === "cost" ? "Cost per day" : "Reviews per day"}</h2>
          <div className="chart-sub">honest aggregation · focus a point for the value</div>
        </div>
        <div className="chart-controls">
          <div className="chart-toggle" role="group" aria-label="series">
            <button
              type="button"
              aria-pressed={series === "cost"}
              onClick={() => setSeries("cost")}>
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
            <button
              type="button"
              aria-pressed={chartType === "line"}
              onClick={() => setChartType("line")}>
              Line
            </button>
            <button
              type="button"
              aria-pressed={chartType === "bar"}
              onClick={() => setChartType("bar")}>
              Bar
            </button>
          </div>
        </div>
      </div>

      <div className="chart-plot">
        <svg
          viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
          className={`chart-svg type-${chartType}`}
          role="img"
          aria-label={`${series} per ${granularity === "hour" ? "hour" : "day"}`}>
          {/* gridlines + y-axis labels */}
          {gridTicks.map((tickFraction) => {
            const gridY = PAD.t + PLOT_H - tickFraction * PLOT_H;
            return (
              <g key={tickFraction}>
                <line x1={PAD.l} y1={gridY} x2={VIEWBOX_WIDTH - PAD.r} y2={gridY} className="chart-grid" />
                <text x={PAD.l - 6} y={gridY + 3} className="chart-ytick" textAnchor="end">
                  {formatAxisValue(tickFraction * axisMax)}
                </text>
              </g>
            );
          })}

          {/* data: area+line OR bars */}
          {chartType === "line" ? (
            <>
              {areaPath ? <path d={areaPath} className="chart-area" /> : null}
              {pointCount >= 2 ? (
                <polyline points={linePoints} className="chart-line" fill="none" />
              ) : pointCount === 1 ? (
                <circle cx={pointX(0)} cy={pointY(seriesValues[0] ?? 0)} r={3} className="chart-dot" />
              ) : null}
            </>
          ) : (
            seriesValues.map((value, index) => (
              <rect
                key={index}
                x={pointX(index) - barWidth / 2}
                y={pointY(value)}
                width={barWidth}
                height={Math.max(0, BASELINE - pointY(value))}
                className="chart-bar"
                fillOpacity={pointComplete[index] ? undefined : 0.45}
              />
            ))
          )}

          {/* x-axis labels (thinned) */}
          {axisLabels.map((label, index) =>
            label ? (
              <text
                key={index}
                x={pointX(index)}
                y={VIEWBOX_HEIGHT - 6}
                className="chart-xtick"
                textAnchor="middle">
                {label}
              </text>
            ) : null,
          )}

          {/* Focusable points; dashed rings mark lower-bound costs. */}
          {seriesValues.map((value, index) => (
            <g key={index}>
              {!pointComplete[index] ? (
                <circle
                  cx={pointX(index)}
                  cy={pointY(value)}
                  r={2.5}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1"
                  strokeDasharray="2 1.5"
                  className="chart-incomplete"
                />
              ) : null}
              {activePointIndex === index ?
                <circle cx={pointX(index)} cy={pointY(value)} r={3.5} className="chart-marker" />
                : null}
              <rect
                x={pointX(index) - (pointCount > 1 ? PLOT_W / pointCount / 2 : PLOT_W / 2)}
                y={PAD.t}
                width={pointCount > 1 ? PLOT_W / pointCount : PLOT_W}
                height={PLOT_H}
                className="chart-hit"
                tabIndex={0}
                role="button"
                aria-label={`${pointLabels[index]}: ${pointComplete[index] ? "" : "\u2265"}${formatValue(value)}`}
                onMouseEnter={() => setActivePointIndex(index)}
                onMouseLeave={() => setActivePointIndex(null)}
                onFocus={() => setActivePointIndex(index)}
                onBlur={() => setActivePointIndex(null)}
              />
            </g>
          ))}
        </svg>

        {activePointIndex !== null ? (
          <div
            className="chart-tip"
            style={{ left: `${(pointX(activePointIndex) / VIEWBOX_WIDTH) * 100}%` }}
            role="status"
          >
            <span className="tip-val">
              {pointComplete[activePointIndex] ? "" : "\u2265"}
              {formatValue(seriesValues[activePointIndex] ?? 0)}
            </span>
            <span className="tip-day">{pointLabels[activePointIndex] ?? ""}</span>
          </div>
        ) : null}
      </div>

      <div className="chart-legend">
        <span>
          <b>
            {pointComplete.every(Boolean) ? "" : "\u2265"}
            {formatValue(statistics.total)}
          </b>{" "}
          total
        </span>
        <span>
          <b>
            {pointComplete.every(Boolean) ? "" : "\u2265"}
            {formatValue(statistics.avg)}
          </b>{" "}
          avg/{granularity === "hour" ? "hr" : "day"}
        </span>
        <span>
          peak{" "}
          <b>
            {pointComplete.every(Boolean) ? "" : "\u2265"}
            {formatValue(statistics.peak)}
          </b>
          {peakDay ? ` on ${peakDay}` : ""}
        </span>
      </div>
    </div>
  );
}
