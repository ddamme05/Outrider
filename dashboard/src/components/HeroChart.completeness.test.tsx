// Completeness rendering in the hero chart + sparkline (openai-native-host arc):
// an incomplete cost bucket is a LOWER BOUND — the value gets a ≥ bound, the
// bucket gets a dashed marker, and the peak DAY is suppressed (the largest
// known subtotal need not belong to the true peak bucket).
import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import type { components } from "../api/schema";
import { HeroChart } from "./HeroChart";
import { Sparkline } from "./Sparkline";

type Bucket = components["schemas"]["MetricBucket"];

function bucket(day: string, cost: number, complete: boolean): Bucket {
  return {
    bucket: `2026-07-${day}T00:00:00Z`,
    reviews: 1,
    cost_usd: cost,
    cost_complete: complete,
    findings: 0,
    failed: 0,
  } as Bucket;
}

test("complete series: exact legend figures and a named peak day", () => {
  const { container } = render(
    <HeroChart
      buckets={[bucket("01", 1.0, true), bucket("02", 3.0, true)]}
      granularity="day"
    />,
  );
  expect(screen.getByText(/peak/i).textContent).toContain("on ");
  expect(container.textContent).not.toContain("≥");
  expect(container.querySelectorAll(".chart-incomplete")).toHaveLength(0);
});

test("incomplete bucket: ≥ bounds on total/avg/peak, dashed marker, NO peak day", () => {
  const { container } = render(
    <HeroChart
      buckets={[bucket("01", 1.0, true), bucket("02", 3.0, false)]}
      granularity="day"
    />,
  );
  const legend = container.querySelector(".chart-legend");
  expect(legend?.textContent).toContain("≥");
  // The known-subtotal peak (day 02) is incomplete — naming a day would assert
  // an attribution the data cannot support.
  expect(screen.getByText(/peak/i).textContent).not.toContain("on ");
  expect(container.querySelectorAll(".chart-incomplete").length).toBeGreaterThan(0);
});

test("sparkline marks exactly the incomplete points", () => {
  const { container } = render(
    <Sparkline
      values={[1, 2, 3]}
      variant="neg"
      label="Cost trend"
      incomplete={[false, true, false]}
    />,
  );
  expect(container.querySelectorAll("circle")).toHaveLength(1);
});

test("sparkline without flags renders no markers (pre-field payloads unchanged)", () => {
  const { container } = render(<Sparkline values={[1, 2, 3]} variant="neg" label="Cost trend" />);
  expect(container.querySelectorAll("circle")).toHaveLength(0);
});
