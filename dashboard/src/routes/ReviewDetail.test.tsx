import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
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
      wall_clock_seconds: 42.5,
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
  responses: { detail?: unknown; findings?: unknown[]; replay?: unknown; detailStatus?: number } = {},
) {
  server.use(
    http.get(BASE, () =>
      responses.detailStatus
        ? HttpResponse.json({ detail: "not found" }, { status: responses.detailStatus })
        : HttpResponse.json(responses.detail ?? detail()),
    ),
    http.get(`${BASE}/findings`, () =>
      HttpResponse.json({ review_id: "r1", findings: responses.findings ?? [finding()] }),
    ),
    http.get(`${BASE}/replay`, () => HttpResponse.json(responses.replay ?? replay())),
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
  // Aggregate metric backed by the contract.
  expect(screen.getByText("42.5s")).toBeInTheDocument();
  // Replay verdict — fully contract-backed.
  expect(await screen.findByText("replay-equivalent")).toBeInTheDocument();
  expect(screen.getByText(/47 events · 5 findings · full/)).toBeInTheDocument();
});

test("renders a finding with severity pill, type, and destination", async () => {
  mount();
  expect(await screen.findByText("sql_injection")).toBeInTheDocument();
  const sevPill = screen.getByText("high");
  expect(sevPill.className).toContain("sev-high");
  expect(screen.getByText("→ INLINE_COMMENT")).toBeInTheDocument();
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
  expect(await screen.findByText("not replay-equivalent")).toBeInTheDocument();
  expect(screen.getByText("finding_count mismatch: 5 vs 4")).toBeInTheDocument();
});

test("labels a mixed-mode replay distinctly (not as full reconstruction)", async () => {
  mount({ replay: replay({ mode: "mixed" }) });
  expect(await screen.findByText("replay-equivalent")).toBeInTheDocument();
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
  expect(await screen.findByText("not replay-equivalent")).toBeInTheDocument();
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
