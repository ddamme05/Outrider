import { describe, expect, test } from "vitest";

import {
  deltaInfo,
  formatBucketLabel,
  replayDeltaInfo,
  replayRate,
  seriesStats,
  thinLabels,
} from "./metrics";

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
  test("neutral polarity stays grey (flat) but still shows direction + magnitude", () => {
    expect(deltaInfo(63, 60, "neutral")).toEqual({ cls: "flat", glyph: "▲", label: "5.0%" });
    expect(deltaInfo(60, 63, "neutral")).toEqual({ cls: "flat", glyph: "▼", label: "4.8%" });
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

describe("replayRate", () => {
  test("equivalent/total as a percentage", () => {
    expect(replayRate(23, 24)).toBeCloseTo(95.833, 2);
    expect(replayRate(9, 10)).toBe(90);
    expect(replayRate(5, 5)).toBe(100);
  });
  test("zero total → null (no DEFINED rate), never 0% — the honest-zeros guard", () => {
    // total=0 means no reviews verdicted yet; 0/0 has no rate. Returning 0 would wrongly
    // read as "every replay diverged"; null lets the card render "—".
    expect(replayRate(0, 0)).toBeNull();
  });
  test("zero equivalent over a non-zero total → an honest 0%", () => {
    expect(replayRate(0, 4)).toBe(0);
  });
});

describe("replayDeltaInfo", () => {
  test("percentage-POINT change with a directional glyph (rate, not relative %)", () => {
    // 95.8% vs 90.0% → +5.8pp up-good (NOT the 6.5% relative change deltaInfo would give).
    expect(replayDeltaInfo(95.833, 90)).toEqual({ cls: "up", glyph: "▲", label: "5.8pp" });
    expect(replayDeltaInfo(90, 95)).toEqual({ cls: "down", glyph: "▼", label: "5.0pp" });
  });
  test("a REAL prior 0% (every replay diverged) yields a defined pp delta, NOT 'new'", () => {
    // The bug deltaInfo(cur, prevRate ?? 0) had: it collapsed a real 0% baseline with "no data".
    expect(replayDeltaInfo(95, 0)).toEqual({ cls: "up", glyph: "▲", label: "95.0pp" });
  });
  test("no PRIOR verdicts (prevRate null) is 'new'; no CURRENT verdicts is flat", () => {
    expect(replayDeltaInfo(95, null)).toEqual({ cls: "flat", glyph: "—", label: "new" });
    expect(replayDeltaInfo(null, 90)).toEqual({ cls: "flat", glyph: "—", label: "vs prev" });
  });
  test("a sub-0.05pp change is flat (no noise)", () => {
    expect(replayDeltaInfo(90.02, 90).cls).toBe("flat");
  });
});

describe("seriesStats", () => {
  test("total / avg / peak / peakIndex", () => {
    expect(seriesStats([2, 5, 3])).toEqual({ total: 10, avg: 10 / 3, peak: 5, peakIndex: 1 });
  });
  test("empty series", () => {
    expect(seriesStats([])).toEqual({ total: 0, avg: 0, peak: 0, peakIndex: -1 });
  });
  test("all-zero (honest-empty) series has peak 0 and peakIndex -1 (no fabricated peak day)", () => {
    expect(seriesStats([0, 0, 0])).toEqual({ total: 0, avg: 0, peak: 0, peakIndex: -1 });
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
