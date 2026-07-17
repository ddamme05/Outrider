import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, expect, test } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { ReviewDetail } from "./ReviewDetail";

// Demo-mode read-only affordances: a parked review must render its gated findings
// but lock the decision controls, because the demo box has no /decide route
// (unmounted AND unproxied). Covers the useDemoStatus → ReviewDetail wiring the
// code review flagged as untested (the whole affordance chain was invisible).

const BASE = "http://localhost/api/reviews/r1";

function detail() {
  return {
    findings_requiring_approval: ["f-high"],
    id: "r1",
    installation_id: 1,
    repo_id: 100,
    pr_number: 7,
    head_sha: "abc1234def",
    status: "awaiting_approval",
    is_eval: false,
    created_at: "2026-06-01T00:00:00Z",
    updated_at: "2026-06-01T00:00:00Z",
    completed_at: null,
    expires_at: "2099-01-01T00:00:00Z",
    metrics: {
      llm_calls_made: 1,
      total_input_tokens: 1,
      total_output_tokens: 1,
      total_cost_usd: 0.1,
      files_examined: 1,
      files_traced_beyond_diff: 0,
      wall_clock_seconds: 1,
    },
    policy_version: "v3",
  };
}

function finding() {
  return {
    finding_id: "f-high",
    finding_type: "sql_injection",
    dimension: "security",
    severity: "high",
    evidence_tier: "observed",
    file_path: "a.py",
    line_start: 1,
    line_end: 1,
    content_redacted: false,
    title: "t",
    description: "d",
    evidence: null,
    suggested_fix: null,
    query_match_id: "q.scm",
    trace_path: null,
    publish_destination: null,
    eligibility: null,
    eligibility_reason: null,
    redaction_sweep_at: null,
  };
}

// Non-boolean meta shapes the status resolver must treat as UNKNOWN (never
// production): a never-resolving hang, a hard 500 (real rejected request), and a
// malformed 200 whose demo_mode isn't a strict boolean.
type MetaMode = boolean | "hang" | "error" | "malformed";

function metaResponderFor(mode: MetaMode): () => Response | Promise<Response> {
  switch (mode) {
    case "hang":
      return () => new Promise<Response>(() => {});
    case "error":
      return () => new HttpResponse(null, { status: 500 });
    case "malformed":
      return () => HttpResponse.json({ demo_mode: null });
    default:
      return () => HttpResponse.json({ demo_mode: mode });
  }
}

function mount(demoMode: MetaMode) {
  server.use(
    http.get("http://localhost/api/meta", metaResponderFor(demoMode)),
    http.get(BASE, () => HttpResponse.json(detail())),
    http.get(`${BASE}/findings`, () =>
      HttpResponse.json({ review_id: "r1", findings: [finding()] }),
    ),
    http.get(`${BASE}/replay-timeline`, () =>
      HttpResponse.json({
        review_id: "r1",
        replay_equivalent: true,
        mode: "full",
        reason: null,
        status: "awaiting_approval",
        events: [],
        phases: [],
        inter_phase_events: [],
        findings: [],
        llm_exchanges: [],
      }),
    ),
    http.get(`${BASE}/events`, () => HttpResponse.json({ review_id: "r1", events: [], total: 0 })),
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

beforeEach(() => useTokenStore.setState({ token: "k" }));

test("demo mode locks the gate: read-only message, no Submit, disabled controls", async () => {
  mount(true);
  expect(
    await screen.findByText(/Read-only demo — decisions are disabled/),
  ).toBeInTheDocument();
  // The Submit button is not rendered at all in demo mode.
  expect(screen.queryByRole("button", { name: /Submit decision/ })).not.toBeInTheDocument();
  // The gated finding still renders (wasGated preserved), and its outcome buttons
  // are disabled rather than actionable.
  const approve = screen.getByRole("button", { name: "approve" });
  expect(approve).toBeDisabled();
});

test("production mode still offers a live Submit on the same parked review", async () => {
  mount(false);
  expect(await screen.findByRole("button", { name: /Submit decision/ })).toBeInTheDocument();
  expect(
    screen.queryByText(/Read-only demo — decisions are disabled/),
  ).not.toBeInTheDocument();
});

test("unresolved meta fails closed with an explicit checking message", async () => {
  mount("hang");
  // The review renders (its finding title proves ReviewDetail mounted) while the
  // meta query is still pending...
  expect(await screen.findByText("t")).toBeInTheDocument();
  // ...Submit must NOT appear and the outcome controls stay disabled, because
  // demo_mode has not resolved to production...
  expect(screen.queryByRole("button", { name: /Submit decision/ })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "approve" })).toBeDisabled();
  // ...the sticky bar SAYS why (not a silently broken-looking page)...
  expect(screen.getByText(/Checking deployment status/)).toBeInTheDocument();
  // ...and the demo banner (which fails to no-banner) does NOT claim demo.
  expect(
    screen.queryByText(/Read-only demo — decisions are disabled/),
  ).not.toBeInTheDocument();
});

test("a hard 500 from /api/meta keeps the gate closed with the checking message", async () => {
  // A real rejected request (not a hang): while retries are in flight, meta.data
  // stays undefined → status "loading" → fail closed, exactly like an outage.
  mount("error");
  expect(await screen.findByText("t")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Submit decision/ })).not.toBeInTheDocument();
  expect(screen.getByText(/Checking deployment status/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "approve" })).toBeDisabled();
});

test("a malformed 200 (demo_mode not a strict boolean) fails closed, not open to production", async () => {
  // {demo_mode: null} is a truthy response body whose flag is falsy — the old
  // `?? false` / truthy check would have called this production and re-enabled
  // the live Submit. Strict resolution treats it as unknown → fail closed.
  mount("malformed");
  expect(await screen.findByText("t")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Submit decision/ })).not.toBeInTheDocument();
  expect(screen.getByText(/Checking deployment status/)).toBeInTheDocument();
});
