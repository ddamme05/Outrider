import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import type { components } from "../api/schema";
import { ReplayFeed } from "./ReplayFeed";

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

function findingContent(
  findingId: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    finding_id: findingId,
    content_redacted: false,
    title: "SQL injection",
    description: "Unparameterized query.",
    evidence: "cur.execute(...)",
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

test("equivalent verdict renders phase cards + the per-operation rows", () => {
  const ev = llmEvent("e1", "analyze", 0.27);
  render(<ReplayFeed data={data({ events: [ev], phases: [phaseWith([ev])] })} />);
  expect(document.querySelector(".tl-phase .tl-node")).toHaveTextContent("analyze");
  expect(screen.getByText("llm_call")).toBeInTheDocument();
});

test("non-equivalent verdict degrades to the flat feed + banner (FUP-125)", () => {
  const ev = llmEvent("e1", "analyze", 0.27);
  render(
    <ReplayFeed
      data={data({ replay_equivalent: false, phases: null, reason: "drift", events: [ev] })}
    />,
  );
  expect(screen.getByText(/phase grouping is unavailable/)).toBeInTheDocument();
  expect(screen.getByText("llm_call")).toBeInTheDocument();
  expect(document.querySelector(".tl-phase")).toBeNull();
});

test("events outside any phase render in the between-phases bucket", () => {
  const inter = llmEvent("e-i", "intake", 0);
  render(<ReplayFeed data={data({ events: [inter], phases: [], inter_phase_events: [inter] })} />);
  expect(screen.getByText(/between phases/)).toBeInTheDocument();
  expect(document.querySelector(".tl-inter .tl-evrow")).toHaveTextContent("llm_call");
});

test("an in-flight phase (no end marker) is labeled", () => {
  const ev = llmEvent("e1", "analyze", 0.27);
  render(
    <ReplayFeed
      data={data({
        events: [ev],
        phases: [{ ...phaseWith([ev]), end: null }],
      })}
    />,
  );
  expect(document.querySelector(".tl-phase-head .pill")).toHaveTextContent("in-flight");
});

test("clicking a finding row expands its content + proof + HITL provenance", async () => {
  const user = userEvent.setup();
  const fev = findingEvent("e-finding", "f-1");
  render(
    <ReplayFeed
      data={data({
        events: [fev],
        phases: [phaseWith([fev])],
        findings: [
          findingContent("f-1", {
            hitl_decision: {
              outcome: "severity_override",
              reviewer_id: "admin",
              reason: "downgraded",
              original_severity: "critical",
              override_severity: "high",
            },
          }),
        ],
      })}
    />,
  );
  expect(screen.queryByText("Unparameterized query.")).toBeNull();
  await user.click(screen.getByRole("button", { name: /finding/ }));
  expect(screen.getByText("Unparameterized query.")).toBeInTheDocument();
  const prov = document.querySelector(".tl-content .f-prov");
  expect(prov).toHaveTextContent("critical → high");
});

test("clicking an llm_call row expands the prompt + completion", async () => {
  const user = userEvent.setup();
  const lev = llmEvent("e-llm", "analyze", 0.1);
  render(
    <ReplayFeed
      data={data({ events: [lev], phases: [phaseWith([lev])], llm_exchanges: [llmContent("e-llm")] })}
    />,
  );
  await user.click(screen.getByRole("button", { name: /llm_call/ }));
  expect(screen.getByText("the prompt text")).toBeInTheDocument();
  expect(screen.getByText("the completion text")).toBeInTheDocument();
});

test("a redacted finding shows the retention stub but keeps its proof line", async () => {
  const user = userEvent.setup();
  const fev = { ...findingEvent("e-p", "f-p"), evidence_tier: "observed", query_match_id: "sql.scm" };
  render(
    <ReplayFeed
      data={data({
        events: [fev],
        phases: [phaseWith([fev])],
        findings: [
          findingContent("f-p", {
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
  const proof = document.querySelector(".tl-c-proof");
  expect(proof).toHaveTextContent("observed");
  expect(proof).toHaveTextContent("sql.scm");
  expect(screen.getByText(/Content redacted in the retention sweep on 2026-05-20/)).toBeInTheDocument();
  expect(screen.queryByText("Unparameterized query.")).toBeNull();
});

test("an INFERRED finding's proof line renders the trace path", async () => {
  const user = userEvent.setup();
  const fev = {
    ...findingEvent("e-i", "f-i"),
    evidence_tier: "inferred",
    query_match_id: null,
    trace_path: ["auth.login", "db.query"],
  };
  render(
    <ReplayFeed
      data={data({ events: [fev], phases: [phaseWith([fev])], findings: [findingContent("f-i")] })}
    />,
  );
  await user.click(screen.getByRole("button", { name: /finding/ }));
  expect(document.querySelector(".tl-c-proof")).toHaveTextContent("auth.login → db.query");
});

test("the `shown` count hides not-yet-reached rows (progressive reveal)", () => {
  const e1 = llmEvent("e1", "analyze", 0.1);
  const e2 = llmEvent("e2", "analyze", 0.2);
  // shown=1 → e1 revealed (current), e2 not yet (future → hidden via .tl-evrow.future).
  render(<ReplayFeed data={data({ events: [e1, e2], phases: [phaseWith([e1, e2])] })} shown={1} />);
  const rows = document.querySelectorAll(".tl-phase .tl-evrow");
  expect(rows[0]?.className).toContain("current");
  expect(rows[1]?.className).toContain("future");
});

test("flat mode renders the append-only banner + .ae-phase dividers + .ae rows", () => {
  const ev = llmEvent("e1", "analyze", 0.27);
  render(<ReplayFeed data={data({ events: [ev], phases: [phaseWith([ev])] })} flat />);
  expect(screen.getByText(/Append-only by database policy/)).toBeInTheDocument();
  expect(document.querySelector(".ae-phase .pname")).toHaveTextContent("analyze");
  const aeRow = document.querySelector(".ae");
  expect(aeRow?.querySelector(".ae-type")).toHaveTextContent("llm_call");
  // Relative timestamp column is present and prefixed with "+".
  expect(aeRow?.querySelector(".ae-time")?.textContent).toMatch(/^\+/);
  // Flat mode does NOT render the collapsible card chrome.
  expect(document.querySelector(".tl-phase")).toBeNull();
});

test("flat mode keeps expand-on-click content panels", async () => {
  const user = userEvent.setup();
  const fev = findingEvent("e-finding", "f-1");
  render(
    <ReplayFeed
      data={data({ events: [fev], phases: [phaseWith([fev])], findings: [findingContent("f-1")] })}
      flat
    />,
  );
  expect(screen.queryByText("Unparameterized query.")).toBeNull();
  await user.click(screen.getByRole("button", { name: /finding/ }));
  expect(screen.getByText("Unparameterized query.")).toBeInTheDocument();
});
