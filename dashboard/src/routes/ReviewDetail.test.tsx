import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, expect, test } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { ReviewDetail } from "./ReviewDetail";

const BASE = "http://localhost/api/reviews/r1";

function detail(overrides: Record<string, unknown> = {}) {
  return {
    id: "r1",
    installation_id: 1,
    repo_id: 100,
    pr_number: 7,
    head_sha: "abc1234def567",
    status: "awaiting_approval",
    is_eval: false,
    created_at: "2026-05-31T00:00:00Z",
    updated_at: "2026-05-31T00:00:00Z",
    completed_at: null,
    expires_at: "2099-01-01T00:00:00Z",
    metrics: {
      llm_calls_made: 4,
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

function finding(overrides: Record<string, unknown> = {}) {
  return {
    finding_id: "f1",
    finding_type: "sql_injection",
    dimension: "security",
    severity: "high",
    evidence_tier: "observed",
    file_path: "src/auth/session.py",
    line_start: 43,
    line_end: 43,
    content_redacted: false,
    title: "SQL injection",
    description: "Unparameterized SQL string interpolation.",
    evidence: "cur.execute(f\"... {token}\")",
    suggested_fix: "Use parameterized queries.",
    query_match_id: "sql_string_format.scm",
    trace_path: null,
    publish_destination: "INLINE_COMMENT",
    eligibility: "withheld",
    eligibility_reason: "hitl_required_node_absent",
    redaction_sweep_at: null,
    ...overrides,
  };
}

function replay(overrides: Record<string, unknown> = {}) {
  return {
    review_id: "r1",
    replay_equivalent: true,
    mode: "full",
    event_count: 47,
    finding_count: 5,
    orphan_finding_count: 0,
    reason: null,
    ...overrides,
  };
}

function mount(
  responses: {
    detail?: unknown;
    findings?: unknown[];
    replay?: unknown;
    detailStatus?: number;
    events?: unknown[];
    policyEntries?: unknown[];
  } = {},
) {
  server.use(
    http.get("http://localhost/api/policy/:version", ({ params }) =>
      HttpResponse.json({ version: params.version, entries: responses.policyEntries ?? [] }),
    ),
    http.get(BASE, () =>
      responses.detailStatus
        ? HttpResponse.json({ detail: "not found" }, { status: responses.detailStatus })
        : HttpResponse.json(responses.detail ?? detail()),
    ),
    http.get(`${BASE}/findings`, () =>
      HttpResponse.json({ review_id: "r1", findings: responses.findings ?? [finding()] }),
    ),
    http.get(`${BASE}/replay`, () => HttpResponse.json(responses.replay ?? replay())),
    http.get(`${BASE}/events`, () => {
      const events = responses.events ?? [];
      return HttpResponse.json({ review_id: "r1", events, total: events.length });
    }),
  );

  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/reviews/r1"]}>
        <Routes>
          <Route path="/reviews/:reviewId" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  useTokenStore.setState({ token: "test-key" });
});

test("renders header, aggregate metrics, and the replay verdict", async () => {
  mount();
  expect(await screen.findByText(/repo 100/)).toBeInTheDocument();
  expect(screen.getByText("#7")).toBeInTheDocument();
  // Aggregate metric backed by the contract (72s → "1m12s").
  expect(screen.getByText("1m12s")).toBeInTheDocument();
  // Replay verdict — fully contract-backed (hero badge + detail panel both show it).
  expect((await screen.findAllByText("replay-equivalent")).length).toBeGreaterThan(0);
  expect(screen.getByText(/47 events · 5 findings · full/)).toBeInTheDocument();
});

test("renders a finding with severity pill, type, and destination", async () => {
  mount();
  expect(await screen.findByText(/sql_injection/)).toBeInTheDocument();
  const sevPill = screen.getByText("high");
  expect(sevPill.className).toContain("sev-high");
  expect(screen.getByText("INLINE_COMMENT")).toBeInTheDocument();
});

test("renders a redaction stub for content_redacted findings", async () => {
  mount({
    findings: [
      finding({
        finding_id: "fr",
        content_redacted: true,
        title: null,
        description: null,
        evidence: null,
        suggested_fix: null,
        query_match_id: null,
        redaction_sweep_at: "2026-05-20T00:00:00Z",
      }),
    ],
  });
  expect(await screen.findByText(/Content redacted in the findings retention sweep on 2026-05-20/)).toBeInTheDocument();
});

test("shows the failure reason when not replay-equivalent", async () => {
  mount({ replay: replay({ replay_equivalent: false, reason: "finding_count mismatch: 5 vs 4" }) });
  expect((await screen.findAllByText("not replay-equivalent")).length).toBeGreaterThan(0);
  expect(screen.getByText("finding_count mismatch: 5 vs 4")).toBeInTheDocument();
});

test("labels a mixed-mode replay distinctly (not as full reconstruction)", async () => {
  mount({ replay: replay({ mode: "mixed" }) });
  expect((await screen.findAllByText("replay-equivalent")).length).toBeGreaterThan(0);
  expect(screen.getByText(/Mixed reconstruction/)).toBeInTheDocument();
  expect(screen.queryByText(/Full reconstruction/)).not.toBeInTheDocument();
});

test("handles a null-mode non-equivalent verdict (reconstruct failed)", async () => {
  mount({
    replay: replay({
      replay_equivalent: false,
      mode: null,
      event_count: null,
      finding_count: null,
      orphan_finding_count: null,
      reason: "corrupt audit payload at event 12",
    }),
  });
  expect((await screen.findAllByText("not replay-equivalent")).length).toBeGreaterThan(0);
  expect(screen.getByText("corrupt audit payload at event 12")).toBeInTheDocument();
  expect(screen.getByText(/Reconstruction did not complete/)).toBeInTheDocument();
  expect(screen.queryByText(/Full reconstruction/)).not.toBeInTheDocument();
});

test("shows an explicit state when the replay endpoint fails (not silent omission)", async () => {
  server.use(
    http.get(BASE, () => HttpResponse.json(detail())),
    http.get(`${BASE}/findings`, () => HttpResponse.json({ review_id: "r1", findings: [finding()] })),
    http.get(`${BASE}/replay`, () => HttpResponse.json({ detail: "boom" }, { status: 500 })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/reviews/r1"]}>
        <Routes>
          <Route path="/reviews/:reviewId" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/Replay verdict unavailable/)).toBeInTheDocument();
});

test("renders an honest error when the review can't be loaded", async () => {
  mount({ detailStatus: 404 });
  expect(await screen.findByText(/Couldn't load this review/)).toBeInTheDocument();
});

// --- FUP-133: audit feed + per-node details ---

function llmEvent(node: string, cost: number): Record<string, unknown> {
  return {
    event_id: `e-${node}`,
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

test("audit-feed tab renders the event stream from the events endpoint", async () => {
  const user = userEvent.setup();
  mount({ events: [llmEvent("analyze", 0.27)] });
  await screen.findByText(/sql_injection/); // findings tab is default
  // Tab shows the count from the events endpoint.
  await user.click(screen.getByRole("tab", { name: /Audit feed/ }));
  // The feed renders the event by type + node.
  expect(await screen.findByText("llm_call")).toBeInTheDocument();
  expect(screen.getByText(/claude-sonnet-4-5 · \$0.27/)).toBeInTheDocument();
});

test("policy chip opens the versioned policy table from the endpoint", async () => {
  const user = userEvent.setup();
  mount({
    policyEntries: [{ finding_type: "hardcoded_secret", dimension: "security", severity: "high" }],
  });
  await screen.findByText(/repo 100/);
  // Chip is closed by default — no table yet.
  expect(screen.queryByText(/Deterministic, versioned/)).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /policy v3/ }));
  // The versioned table renders (header phrase is unique to PolicyTable).
  expect(await screen.findByText(/Deterministic, versioned/)).toBeInTheDocument();
  expect(screen.getByText("hardcoded_secret")).toBeInTheDocument();
});

test("pipeline node cards derive per-node model and cost from llm_call events", async () => {
  // Completed review so analyze/triage are "done" and show their derived stats.
  mount({
    detail: detail({ status: "completed" }),
    events: [llmEvent("analyze", 0.27), llmEvent("triage", 0.01)],
  });
  await screen.findByText(/sql_injection/);
  // Per-node costs derived from the events, shown on the pipeline node cards.
  expect(await screen.findByText("$0.27")).toBeInTheDocument();
  expect(screen.getByText("$0.01")).toBeInTheDocument();
  // model prettified from claude-sonnet-4-5 → Sonnet (analyze + triage nodes).
  expect(screen.getAllByText("Sonnet").length).toBe(2);
});
