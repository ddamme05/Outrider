import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { expiresLabel } from "./format";

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
