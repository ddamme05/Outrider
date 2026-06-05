import { describe, expect, test } from "vitest";

import { deltaInfo, formatBucketLabel, seriesStats, thinLabels } from "./metrics";

describe("deltaInfo", () => {
  test("up-good: an increase is a green up-arrow", () => {
    expect(deltaInfo(24, 20, "up-good")).toEqual({ cls: "up", glyph: "▲", label: "20%" });
  });
  test("up-good: a decrease is a red down-arrow", () => {
    expect(deltaInfo(16, 20, "up-good")).toEqual({ cls: "down", glyph: "▼", label: "20%" });
  });
  test("up-bad: cost rising is a red up-arrow", () => {
    const d = deltaInfo(14.2, 11.68, "up-bad");
    expect(d.cls).toBe("up-bad");
    expect(d.glyph).toBe("▲");
  });
  test("up-bad: failures falling is a green down-arrow", () => {
    expect(deltaInfo(2, 4, "up-bad")).toEqual({ cls: "down-good", glyph: "▼", label: "50%" });
  });
  test("neutral polarity is always flat regardless of direction", () => {
    expect(deltaInfo(63, 1, "neutral").cls).toBe("flat");
  });
  test("equal current/previous is flat", () => {
    expect(deltaInfo(5, 5, "up-good").cls).toBe("flat");
  });
  test("empty prior window (previous 0) shows 'new', never an ∞% divide-by-zero", () => {
    expect(deltaInfo(5, 0, "up-good")).toEqual({ cls: "up", glyph: "▲", label: "new" });
  });
  test("small deltas keep one decimal; large deltas round", () => {
    expect(deltaInfo(21, 20, "up-good").label).toBe("5.0%");
    expect(deltaInfo(40, 20, "up-good").label).toBe("100%");
  });
});

describe("seriesStats", () => {
  test("total / avg / peak / peakIndex", () => {
    expect(seriesStats([2, 5, 3])).toEqual({ total: 10, avg: 10 / 3, peak: 5, peakIndex: 1 });
  });
  test("empty series", () => {
    expect(seriesStats([])).toEqual({ total: 0, avg: 0, peak: 0, peakIndex: -1 });
  });
  test("all-zero (honest-empty) series has peak 0, not -Infinity", () => {
    expect(seriesStats([0, 0, 0])).toEqual({ total: 0, avg: 0, peak: 0, peakIndex: 0 });
  });
});

describe("formatBucketLabel", () => {
  test("day granularity → 'Mon D' in UTC", () => {
    expect(formatBucketLabel("2026-05-29T00:00:00Z", "day")).toBe("May 29");
  });
  test("hour granularity → 'HH:00' in UTC", () => {
    expect(formatBucketLabel("2026-06-04T14:00:00Z", "hour")).toBe("14:00");
  });
});

describe("thinLabels", () => {
  test("returns all labels when under the max", () => {
    expect(thinLabels(["a", "b", "c"], 7)).toEqual(["a", "b", "c"]);
  });
  test("thins to evenly-spaced ticks, always keeping the last", () => {
    const out = thinLabels(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 5);
    expect(out[0]).toBe("0");
    expect(out[9]).toBe("9");
    expect(out.filter(Boolean).length).toBeLessThanOrEqual(6);
  });
});
