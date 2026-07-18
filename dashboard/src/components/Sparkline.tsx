// Inline SVG sparkline (mockup .card-spark): one polyline, 62×22 viewBox, stroke
// follows `currentColor` set by the variant class (.sk-accent/.sk-pos/.sk-neg/
// .sk-muted → theme tokens). N-agnostic: scales whatever per-day series it's given
// (7 for 7d, 24 for 24h, 30 for 30d). Honest — an all-zero series draws a flat
// midline, never a faked wiggle.
export type SparkVariant = "accent" | "pos" | "neg" | "muted";

const W = 62;
const H = 22;
const PAD_X = 2;
const TOP = 4;
const BOTTOM = 18;

export function Sparkline({
  values,
  variant,
  label,
  incomplete,
}: {
  values: number[];
  variant: SparkVariant;
  label?: string;
  /** Per-point lower-bound flags (openai-native-host arc): a true entry marks the
   * point with a dashed ring — its value is a known subtotal, not an exact figure. */
  incomplete?: boolean[];
}) {
  if (values.length === 0) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  const innerW = W - PAD_X * 2;
  const denom = values.length > 1 ? values.length - 1 : 1;

  const xy = values.map((v, i) => {
    const x = PAD_X + (i / denom) * innerW;
    // Flat series (span 0) sits on the midline; otherwise scale into [TOP,BOTTOM].
    const y = span === 0 ? (TOP + BOTTOM) / 2 : BOTTOM - ((v - min) / span) * (BOTTOM - TOP);
    return [x, y] as const;
  });
  const points = xy.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

  return (
    <svg
      className={`card-spark sk-${variant}`}
      viewBox={`0 0 ${W} ${H}`}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {incomplete
        ? xy.map(([x, y], i) =>
            incomplete[i] ? (
              <circle
                key={i}
                cx={x}
                cy={y}
                r={1.8}
                fill="none"
                stroke="currentColor"
                strokeWidth="0.8"
                strokeDasharray="1.5 1"
              />
            ) : null,
          )
        : null}
    </svg>
  );
}
