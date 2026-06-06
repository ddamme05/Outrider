import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { ReplayReconstruct } from "./ReplayReconstruct";

const BASE = "http://localhost/api/reviews/r1";

function detail(overrides: Record<string, unknown> = {}) {
  return {
    id: "r1",
    installation_id: 1,
    repo_id: 100,
    pr_number: 7,
    head_sha: "abc1234def567",
    status: "completed",
    is_eval: false,
    created_at: "2026-05-31T00:00:00Z",
    updated_at: "2026-05-31T00:00:00Z",
    completed_at: "2026-05-31T00:01:00Z",
    expires_at: null,
    metrics: {
      llm_calls_made: 2,
      total_input_tokens: 40,
      total_output_tokens: 20,
      total_cost_usd: 0.41,
      files_examined: 5,
      files_traced_beyond_diff: 2,
      wall_clock_seconds: 72,
    },
    policy_version: "v3",
    ...overrides,
  };
}

function llmEv(id: string): Record<string, unknown> {
  return {
    event_id: id,
    review_id: "r1",
    event_type: "llm_call",
    timestamp: "2026-06-01T00:00:00Z",
    sequence_number: 1,
    is_eval: false,
    model: "claude-sonnet-4-5",
    node_id: "analyze",
    input_tokens: 100,
    output_tokens: 40,
    cached_tokens: 0,
    cost_usd: 0.27,
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

function phaseMk(marker: "start" | "end"): Record<string, unknown> {
  return {
    event_id: `e-analyze-${marker}`,
    review_id: "r1",
    event_type: "review_phase",
    timestamp: "2026-06-01T00:00:00Z",
    sequence_number: 1,
    is_eval: false,
    node_id: "analyze",
    marker,
    phase_key: null,
  };
}

function timeline(overrides: Record<string, unknown> = {}) {
  const e1 = llmEv("e1");
  const e2 = llmEv("e2");
  return {
    review_id: "r1",
    replay_equivalent: true,
    mode: "full",
    reason: null,
    status: "completed",
    events: [phaseMk("start"), e1, e2, phaseMk("end")],
    phases: [
      { phase_id: "p", node_id: "analyze", start: phaseMk("start"), end: phaseMk("end"), events: [e1, e2] },
    ],
    inter_phase_events: [],
    findings: [],
    llm_exchanges: [],
    ...overrides,
  };
}

// jsdom doesn't implement matchMedia (it's undefined); capture that so afterEach can restore it
// and the reduced-motion stub doesn't leak into later tests in the shared jsdom worker.
const originalMatchMedia = window.matchMedia;

function setReducedMotion(reduced: boolean) {
  window.matchMedia = ((query: string) => ({
    matches: reduced,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

function mount(opts: { reduced?: boolean; timeline?: unknown } = {}) {
  setReducedMotion(opts.reduced ?? true);
  server.use(
    http.get(BASE, () => HttpResponse.json(detail())),
    http.get(`${BASE}/replay-timeline`, () => HttpResponse.json(opts.timeline ?? timeline())),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/reviews/r1/replay"]}>
        <Routes>
          <Route path="/reviews/:reviewId/replay" element={<ReplayReconstruct />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  useTokenStore.setState({ token: "test-key" });
});
afterEach(() => {
  vi.useRealTimers();
  window.matchMedia = originalMatchMedia;
});

test("renders the reconstruction title + back link to the review", async () => {
  mount();
  expect(await screen.findByText(/repo 100/)).toBeInTheDocument();
  expect(screen.getByText(/Replay · reconstruct/)).toBeInTheDocument();
  const back = screen.getByRole("link", { name: /back to review/ });
  expect(back).toHaveAttribute("href", "/reviews/r1");
});

test("under reduced motion the reconstruction renders instantly with the verdict", async () => {
  mount({ reduced: true });
  // Instant render: counter at total/total, verdict shown, retention note.
  expect(await screen.findByText("2", { selector: ".rp-counter b" })).toBeInTheDocument();
  expect(screen.getByText(/replay-equivalent/)).toBeInTheDocument();
  expect(screen.getByText(/2 events · 0 findings · full/)).toBeInTheDocument();
  expect(screen.getByText(/metadata-only/)).toBeInTheDocument();
  // The phase-grouped feed renders both events.
  expect(screen.getAllByText("llm_call")).toHaveLength(2);
});

test("verdict counts finding EVENTS, not the suppressed content array, on a non-equivalent timeline", async () => {
  const findingEv = {
    event_id: "f-ev",
    review_id: "r1",
    event_type: "finding",
    timestamp: "2026-06-01T00:00:00Z",
    sequence_number: 1,
    is_eval: false,
    finding_id: "f1",
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
  // Non-equivalent verdict: the `findings` CONTENT array is suppressed to [] by the backend, but the
  // FindingEvent still rides `events`. The verdict must report the real count (1), not a fabricated 0.
  mount({
    reduced: true,
    timeline: timeline({
      replay_equivalent: false,
      phases: null,
      reason: "drift",
      events: [findingEv],
      findings: [],
      llm_exchanges: [],
    }),
  });
  expect(await screen.findByText(/not replay-equivalent/)).toBeInTheDocument();
  expect(screen.getByText(/1 events · 1 findings/)).toBeInTheDocument();
});

test("the speed multiplier offers 1×/2×/4×/8× and toggles the active one", async () => {
  const user = userEvent.setup();
  mount({ reduced: true });
  await screen.findByText(/Real-time reconstruction/);
  for (const label of ["1×", "2×", "4×", "8×"]) {
    expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
  }
  // 1× is the default; clicking 2× moves the active state.
  expect(screen.getByRole("button", { name: "1×" })).toHaveAttribute("aria-pressed", "true");
  await user.click(screen.getByRole("button", { name: "2×" }));
  expect(screen.getByRole("button", { name: "2×" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "1×" })).toHaveAttribute("aria-pressed", "false");
});

test("non-reduced-motion auto-plays — the event counter advances from 0", async () => {
  mount({ reduced: false });
  await screen.findByText(/Real-time reconstruction/);
  // Auto-play reveals events progressively (450ms / 1× step); the counter climbs off 0.
  await waitFor(() => expect(screen.getByText("1", { selector: ".rp-counter b" })).toBeInTheDocument(), {
    timeout: 1500,
  });
});
