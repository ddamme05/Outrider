import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import type { components } from "../api/schema";
import { ReplayTimeline } from "./ReplayTimeline";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];

function llmEvent(id: string, node: string, cost: number): Record<string, unknown> {
  return {
    event_id: id,
    review_id: "r1",
    event_type: "llm_call",
    timestamp: "2026-06-01T00:00:00Z",
    sequence_number: 1,
    is_eval: false,
    model: "claude-sonnet-4-5",
    node_id: node,
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

function phaseMarker(node: string, marker: "start" | "end", ts: string): Record<string, unknown> {
  return {
    event_id: `e-${node}-${marker}`,
    review_id: "r1",
    event_type: "review_phase",
    timestamp: ts,
    sequence_number: 1,
    is_eval: false,
    node_id: node,
    marker,
    phase_key: null,
  };
}

function data(overrides: Record<string, unknown> = {}): TimelineData {
  return {
    review_id: "r1",
    replay_equivalent: true,
    mode: "full",
    reason: null,
    status: "completed",
    events: [],
    phases: [],
    inter_phase_events: [],
    findings: [],
    llm_exchanges: [],
    ...overrides,
  } as unknown as TimelineData;
}

// FUP-125: phase grouping is trustworthy only on a replay-equivalent verdict. A
// non-equivalent verdict carries phases:null from the server; the component must
// degrade to the flat ordered feed + a banner, never render an untrusted grouping.
test("non-equivalent verdict degrades to the flat feed with a banner", () => {
  const ev = llmEvent("e1", "analyze", 0.27);
  render(
    <ReplayTimeline
      data={data({
        replay_equivalent: false,
        phases: null,
        reason: "finding_count mismatch: 5 vs 4",
        events: [ev],
      })}
    />,
  );
  expect(screen.getByText(/phase grouping is unavailable/)).toBeInTheDocument();
  expect(screen.getByText("finding_count mismatch: 5 vs 4")).toBeInTheDocument();
  // The flat AuditFeed renders the event; no scrubber, no phase cards.
  expect(screen.getByText("llm_call")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Play/ })).toBeNull();
  expect(document.querySelector(".tl-phase")).toBeNull();
  expect(screen.getByLabelText("replay verdict")).toHaveTextContent("not replay-equivalent");
});

test("equivalent verdict renders phase cards, the scrubber, and the verdict header", () => {
  const ev = llmEvent("e1", "analyze", 0.27);
  render(
    <ReplayTimeline
      data={data({
        events: [ev],
        phases: [
          {
            phase_id: "p-analyze",
            node_id: "analyze",
            start: phaseMarker("analyze", "start", "2026-06-01T00:00:00Z"),
            end: phaseMarker("analyze", "end", "2026-06-01T00:00:01Z"),
            events: [ev],
          },
        ],
      })}
    />,
  );
  expect(screen.getByLabelText("replay verdict")).toHaveTextContent("replay-equivalent");
  expect(screen.getByRole("button", { name: /Play/ })).toBeInTheDocument();
  expect(document.querySelector(".tl-phase .tl-node")).toHaveTextContent("analyze");
  // Duration derived from the start/end markers (1000ms → "1.0s").
  expect(screen.getByText("1.0s")).toBeInTheDocument();
  expect(screen.getByText("llm_call")).toBeInTheDocument();
  // Resting (full) view → no playback class on the row.
  expect(document.querySelector(".tl-phase .tl-evrow")?.className).not.toContain("future");
});

test("events outside any phase render in the between-phases bucket", () => {
  const inter = llmEvent("e-inter", "intake", 0);
  render(
    <ReplayTimeline
      data={data({
        events: [inter],
        phases: [],
        inter_phase_events: [inter],
      })}
    />,
  );
  expect(screen.getByText(/between phases/)).toBeInTheDocument();
  expect(document.querySelector(".tl-inter .tl-evrow")).toHaveTextContent("llm_call");
});

test("an in-flight phase (no end marker) is labeled, not given a fabricated duration", () => {
  const ev = llmEvent("e1", "analyze", 0.27);
  render(
    <ReplayTimeline
      data={data({
        status: "running",
        events: [ev],
        phases: [
          {
            phase_id: "p-analyze",
            node_id: "analyze",
            start: phaseMarker("analyze", "start", "2026-06-01T00:00:00Z"),
            end: null,
            events: [ev],
          },
        ],
      })}
    />,
  );
  // The in-flight pill marks the open phase (start present, end absent).
  expect(document.querySelector(".tl-phase-head .pill")).toHaveTextContent("in-flight");
});

test("the step-back control moves the play cursor off the resting full view", async () => {
  const user = userEvent.setup();
  const start = phaseMarker("analyze", "start", "2026-06-01T00:00:00Z");
  const e1 = llmEvent("e1", "analyze", 0.1);
  const e2 = llmEvent("e2", "analyze", 0.2);
  const end = phaseMarker("analyze", "end", "2026-06-01T00:00:01Z");
  render(
    <ReplayTimeline
      data={data({
        // Real wire shape: the flat stream CARRIES the phase start/end markers (the backend
        // only strips the projected verdict). The two visible rows are e1/e2; the markers are
        // represented by the phase card, never as rows — so playback must count 2, not 4.
        events: [start, e1, e2, end],
        phases: [
          {
            phase_id: "p-analyze",
            node_id: "analyze",
            start,
            end,
            events: [e1, e2],
          },
        ],
      })}
    />,
  );
  // Resting → the cursor sits at total/total; total is the 2 RENDERED rows, NOT 4 (markers excluded).
  expect(screen.getByText("2/2")).toBeInTheDocument();
  // Step back once → cursor 1/2; the un-played last event goes "future", the cursor "current".
  await user.click(screen.getByRole("button", { name: "◀" }));
  expect(screen.getByText("1/2")).toBeInTheDocument();
  const rows = document.querySelectorAll(".tl-phase .tl-evrow");
  expect(rows[0]?.className).toContain("current");
  expect(rows[1]?.className).toContain("future");
});

// ---- PR 2: expand-on-click content panels ----
function findingEvent(eventId: string, findingId: string): Record<string, unknown> {
  return {
    event_id: eventId,
    review_id: "r1",
    event_type: "finding",
    timestamp: "2026-06-01T00:00:00Z",
    sequence_number: 2,
    is_eval: false,
    finding_id: findingId,
    finding_type: "sql_injection",
    severity: "critical",
    file_path: "src/app.py",
    line_start: 10,
    line_end: 20,
    dimension: "security",
    finding_content_hash: "a".repeat(64),
    evidence_tier: "judged",
    query_match_id: null,
    trace_path: null,
    policy_version: "1.0.0",
    proposal_hash: "b".repeat(64),
  };
}

function findingContent(
  findingId: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    finding_id: findingId,
    content_redacted: false,
    title: "SQL injection",
    description: "Unparameterized query.",
    evidence: "cur.execute(f\"...{x}\")",
    suggested_fix: "Use parameters.",
    hitl_decision: null,
    redaction_sweep_at: null,
    ...overrides,
  };
}

function llmContent(eventId: string, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    event_id: eventId,
    content_redacted: false,
    prompt: "the prompt text",
    completion: "the completion text",
    redaction_sweep_at: null,
    ...overrides,
  };
}

function phaseWith(events: Record<string, unknown>[]): Record<string, unknown> {
  return {
    phase_id: "p-analyze",
    node_id: "analyze",
    start: phaseMarker("analyze", "start", "2026-06-01T00:00:00Z"),
    end: phaseMarker("analyze", "end", "2026-06-01T00:00:01Z"),
    events,
  };
}

test("clicking a finding row expands its content panel", async () => {
  const user = userEvent.setup();
  const fev = findingEvent("e-finding", "f-1");
  render(
    <ReplayTimeline
      data={data({ events: [fev], phases: [phaseWith([fev])], findings: [findingContent("f-1")] })}
    />,
  );
  // Collapsed by default.
  expect(screen.queryByText("Unparameterized query.")).toBeNull();
  await user.click(screen.getByRole("button", { name: /finding/ }));
  expect(screen.getByText("Unparameterized query.")).toBeInTheDocument();
  expect(screen.getByText("SQL injection")).toBeInTheDocument();
});

test("clicking an llm_call row expands the prompt + completion", async () => {
  const user = userEvent.setup();
  const lev = llmEvent("e-llm", "analyze", 0.1);
  render(
    <ReplayTimeline
      data={data({ events: [lev], phases: [phaseWith([lev])], llm_exchanges: [llmContent("e-llm")] })}
    />,
  );
  await user.click(screen.getByRole("button", { name: /llm_call/ }));
  expect(screen.getByText("the prompt text")).toBeInTheDocument();
  expect(screen.getByText("the completion text")).toBeInTheDocument();
});

test("a redacted finding row shows the retention stub, not content", async () => {
  const user = userEvent.setup();
  const fev = findingEvent("e-fr", "f-r");
  render(
    <ReplayTimeline
      data={data({
        events: [fev],
        phases: [phaseWith([fev])],
        findings: [
          findingContent("f-r", {
            content_redacted: true,
            title: null,
            description: null,
            evidence: null,
            suggested_fix: null,
            redaction_sweep_at: "2026-05-20T00:00:00Z",
          }),
        ],
      })}
    />,
  );
  await user.click(screen.getByRole("button", { name: /finding/ }));
  expect(
    screen.getByText(/Content redacted in the retention sweep on 2026-05-20/),
  ).toBeInTheDocument();
  // The redaction stub carries no content text.
  expect(screen.queryByText("Unparameterized query.")).toBeNull();
});

test("an overridden finding surfaces its HITL provenance (stream-canonical)", async () => {
  const user = userEvent.setup();
  const fev = findingEvent("e-fo", "f-o");
  render(
    <ReplayTimeline
      data={data({
        events: [fev],
        phases: [phaseWith([fev])],
        findings: [
          findingContent("f-o", {
            hitl_decision: {
              outcome: "severity_override",
              reviewer_id: "admin",
              reason: "downgraded: test-only path",
              original_severity: "critical",
              override_severity: "high",
            },
          }),
        ],
      })}
    />,
  );
  await user.click(screen.getByRole("button", { name: /finding/ }));
  const prov = document.querySelector(".tl-content .f-prov");
  expect(prov).not.toBeNull();
  expect(prov).toHaveTextContent("severity_override");
  expect(prov).toHaveTextContent("critical → high");
  expect(prov).toHaveTextContent("by admin");
});

test("an expanded panel survives a refetch (2s poll) without perturbing the scrubber", async () => {
  const user = userEvent.setup();
  const fev = findingEvent("e-finding", "f-1");
  const d = data({ events: [fev], phases: [phaseWith([fev])], findings: [findingContent("f-1")] });
  const { rerender } = render(<ReplayTimeline data={d} />);
  await user.click(screen.getByRole("button", { name: /finding/ }));
  expect(screen.getByText("Unparameterized query.")).toBeInTheDocument();
  // A 2s poll: same review_id, a fresh data object. Expand state is keyed by event_id and
  // reset only on review change, so the open panel must survive.
  rerender(<ReplayTimeline data={{ ...d, findings: [findingContent("f-1")] } as typeof d} />);
  expect(screen.getByText("Unparameterized query.")).toBeInTheDocument();
});
