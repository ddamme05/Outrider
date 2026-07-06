import { render } from "@testing-library/react";
import { expect, test } from "vitest";

import type { components } from "../api/schema";
import { PipelineStrip } from "./PipelineStrip";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
type Phase = NonNullable<TimelineData["phases"]>[number];

// A single analyze llm_call. The `model` is what the pipeline card must surface — the bug
// was collapsing a node's many calls to the last one's model.
function llmCall(id: string, model: string, cost: number): Record<string, unknown> {
  return {
    event_id: id,
    review_id: "r1",
    event_type: "llm_call",
    timestamp: "2026-06-01T00:00:00Z",
    sequence_number: 1,
    is_eval: false,
    model,
    node_id: "analyze",
    input_tokens: 100,
    output_tokens: 40,
    cached_tokens: 0,
    cost_usd: cost,
    pricing_version: "v1",
    latency_ms: 1200,
    prompt_hash: "a".repeat(64),
    cache_hit: false,
    context_summary: [],
    prompt_template_version: "analyze.v1",
    system_prompt_hash: "b".repeat(64),
    degraded_mode: false,
  };
}

function marker(m: "start" | "end", ts: string, key: string | null): Record<string, unknown> {
  return {
    event_id: `e-analyze-${m}-${key ?? "none"}`,
    review_id: "r1",
    event_type: "review_phase",
    timestamp: ts,
    sequence_number: 1,
    is_eval: false,
    node_id: "analyze",
    marker: m,
    phase_key: key,
  };
}

// One keyed analyze worker phase (plan/file:<path>#<pass>/aggregate under the fan-out).
function phase(
  phaseId: string,
  phaseKey: string,
  startTs: string,
  endTs: string,
  events: Record<string, unknown>[],
): Record<string, unknown> {
  return {
    phase_id: phaseId,
    node_id: "analyze",
    phase_key: phaseKey,
    start: marker("start", startTs, phaseKey),
    end: marker("end", endTs, phaseKey),
    events,
  };
}

function renderStrip(phases: Record<string, unknown>[]) {
  return render(
    <PipelineStrip status="completed" phases={phases as unknown as Phase[]} gatedCount={null} />,
  );
}

// The analyze node's card. Node names are unique, so match the pnode whose name span contains
// "analyze" (a done node also renders a ✓, hence `includes` not `===`).
function analyzeCard(container: HTMLElement): HTMLElement {
  const card = [...container.querySelectorAll<HTMLElement>(".pnode")].find((n) =>
    n.querySelector(".pn-name")?.textContent?.includes("analyze"),
  );
  if (!card) throw new Error("analyze card not found");
  return card;
}

const ts = (sec: number) => `2026-06-01T00:00:${String(sec).padStart(2, "0")}Z`;

// A DEEP-tier Sonnet worker and a STANDARD-tier Haiku worker — the exact mixed-tier review
// (DECISIONS.md#041) whose card used to show only "Haiku" (the last call).
const sonnetWorker = () =>
  phase("p-deep", "file:src/a.py#0", ts(0), ts(5), [llmCall("l1", "claude-sonnet-4-6", 0.06)]);
const haikuWorker = () =>
  phase("p-std", "file:src/b.py#0", ts(1), ts(3), [llmCall("l2", "claude-haiku-4-5-20251001", 0.01)]);

test("analyze surfaces EVERY distinct model, not just the last call's", () => {
  const { container } = renderStrip([sonnetWorker(), haikuWorker()]);
  expect(analyzeCard(container).querySelector(".pn-model")?.textContent).toBe("Haiku + Sonnet");
});

test("model label is stable under reversed fan-out completion order (finding 3)", () => {
  // Same two workers, opposite order — the label must not flip with nondeterministic ordering.
  const { container } = renderStrip([haikuWorker(), sonnetWorker()]);
  expect(analyzeCard(container).querySelector(".pn-model")?.textContent).toBe("Haiku + Sonnet");
});

test("analyze duration is the interval UNION of concurrent workers, not the sum", () => {
  // Two overlapping workers: [0,5s] and [1,6s]. Union = 6s; the pre-fix sum was 5+5 = 10s.
  const w1 = phase("p1", "file:a#0", ts(0), ts(5), [llmCall("l1", "claude-sonnet-4-6", 0.06)]);
  const w2 = phase("p2", "file:b#0", ts(1), ts(6), [llmCall("l2", "claude-haiku-4-5-20251001", 0.01)]);
  const stat = analyzeCard(renderStrip([w1, w2]).container).querySelector(".pn-stat")?.textContent ?? "";
  expect(stat).toContain("6.0s");
  expect(stat).not.toContain("10s");
});

test("analyze duration excludes the trace gap between passes (multi-pass union)", () => {
  // pass-0 [0,5s]; trace runs; pass-1 [20,23s]. Union = 5+3 = 8s, NOT the 23s earliest→latest.
  const p0 = phase("p0", "file:a#0", ts(0), ts(5), [llmCall("l1", "claude-sonnet-4-6", 0.06)]);
  const p1 = phase("p1", "file:c#1", ts(20), ts(23), [llmCall("l2", "claude-haiku-4-5-20251001", 0.01)]);
  const stat = analyzeCard(renderStrip([p0, p1]).container).querySelector(".pn-stat")?.textContent ?? "";
  expect(stat).toContain("8.0s");
  expect(stat).not.toContain("23s");
});
