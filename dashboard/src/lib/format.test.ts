import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { expiresLabel, unionDurationMs } from "./format";

describe("unionDurationMs", () => {
  const at = (ms: number) => new Date(ms).toISOString();
  const span = (startMs: number, endMs: number) => ({ start: at(startMs), end: at(endMs) });

  test("null when no valid intervals", () => {
    expect(unionDurationMs([])).toBeNull();
    expect(unionDurationMs([{ start: null, end: at(1000) }])).toBeNull();
    expect(unionDurationMs([{ start: at(2000), end: at(1000) }])).toBeNull(); // negative
    expect(unionDurationMs([{ start: "nope", end: "also-nope" }])).toBeNull();
  });

  test("single interval is its own span", () => {
    expect(unionDurationMs([span(0, 5000)])).toBe(5000);
  });

  test("overlapping concurrent phases merge — NOT summed (the parallel-analyze bug)", () => {
    // Four workers of 5s each, heavily overlapping under concurrency: wall-time is the union
    // (~6s here), not 4×5s=20s. Summing was the pre-fix double-count.
    const workers = [span(0, 5000), span(1000, 6000), span(500, 5500), span(200, 5200)];
    expect(unionDurationMs(workers)).toBe(6000);
    // Order-independence: shuffling the same intervals yields the same union.
    expect(unionDurationMs([...workers].reverse())).toBe(6000);
  });

  test("contiguous intervals (touching endpoints) merge into one", () => {
    expect(unionDurationMs([span(0, 3000), span(3000, 7000)])).toBe(7000);
  });

  test("multi-pass: trace gap between analyze passes is EXCLUDED (finding 1)", () => {
    // pass-0 workers span [0,5s]; trace runs [5s,20s]; pass-1 workers span [20s,23s].
    // A naive earliest→latest span would be 23s (swallowing the 15s trace gap); the union of
    // the two disjoint analyze clusters is 5s + 3s = 8s.
    const pass0 = [span(0, 5000), span(1000, 4000)];
    const pass1 = [span(20000, 23000), span(20500, 22000)];
    expect(unionDurationMs([...pass0, ...pass1])).toBe(8000);
  });

  test("invalid intervals are dropped, valid ones still counted", () => {
    expect(unionDurationMs([span(0, 5000), { start: null, end: at(9000) }])).toBe(5000);
  });
});

describe("expiresLabel", () => {
  const NOW = new Date("2026-01-01T00:00:00Z").getTime();
  const inMs = (ms: number) => new Date(NOW + ms).toISOString();

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("null for absent or unparseable input", () => {
    expect(expiresLabel(null)).toBeNull();
    expect(expiresLabel("not-a-date")).toBeNull();
  });

  test("'expired' for a past timestamp", () => {
    expect(expiresLabel(inMs(-60_000))).toBe("expired");
  });

  test("minutes tier under an hour", () => {
    expect(expiresLabel(inMs(30 * 60_000))).toBe("expires in 30m");
  });

  test("hours tier under two days", () => {
    expect(expiresLabel(inMs(3 * 60 * 60_000))).toBe("expires in 3h");
  });

  test("days tier under a year (was an awkward bare-hours label before)", () => {
    expect(expiresLabel(inMs(7 * 24 * 60 * 60_000))).toBe("expires in 7d");
  });

  test("null beyond a year — the demo's pinned-pending HITL never shows a countdown", () => {
    expect(expiresLabel(inMs(100 * 365 * 24 * 60 * 60_000))).toBeNull();
  });
});
