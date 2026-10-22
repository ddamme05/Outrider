// Scale a numeric series into a small SVG trend. Flat series stay on the
// midline; incomplete points get dashed rings. Color comes from the variant.
export type SparkVariant = "accent" | "pos" | "neg" | "muted";

const VIEWBOX_WIDTH = 62;
const VIEWBOX_HEIGHT = 22;
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
  /** True marks a point as a lower-bound subtotal rather than an exact value. */
  incomplete?: boolean[];
}) {
  if (values.length === 0) return null;

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valueSpan = maxValue - minValue;
  const plotWidth = VIEWBOX_WIDTH - PAD_X * 2;
  const indexSpan = values.length > 1 ? values.length - 1 : 1;

  const pointCoordinates = values.map((value, index) => {
    const x = PAD_X + (index / indexSpan) * plotWidth;
    // Flat series (span 0) sits on the midline; otherwise scale into [TOP,BOTTOM].
    const y = valueSpan === 0 ? (TOP + BOTTOM) / 2 : BOTTOM - ((value - minValue) / valueSpan) * (BOTTOM - TOP);
    return [x, y] as const;
  });
  const points = pointCoordinates.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

  return (
    <svg
      className={`card-spark sk-${variant}`}
      viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
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
        ? pointCoordinates.map(([x, y], index) =>
            incomplete[index] ? (
              <circle
                key={index}
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
