import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { RESUME_WINDOW_MS } from "../lib/hitl";
import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { ReviewDetail } from "./ReviewDetail";

const BASE = "http://localhost/api/reviews/r1";
const DECIDE = "http://localhost/reviews/r1/decide";

function detail(status = "awaiting_approval", gatedIds: string[] | null = ["f-high"]) {
  return {
    findings_requiring_approval: gatedIds,
    id: "r1",
    installation_id: 1,
    repo_id: 100,
    pr_number: 7,
    head_sha: "abc1234def",
    status,
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

function finding(overrides: Record<string, unknown> = {}) {
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
    ...overrides,
  };
}

function mount(opts: {
  status?: string;
  findings?: unknown[];
  decideStatus?: number;
  capture?: (b: unknown) => void;
  resumeTo?: string; // status the detail GET returns AFTER a successful decide
  gatedIds?: string[] | null; // authoritative findings_requiring_approval
} = {}) {
  let decided = false;
  const gatedIds = opts.gatedIds === undefined ? ["f-high"] : opts.gatedIds;
  server.use(
    http.get(BASE, () =>
      HttpResponse.json(detail(decided && opts.resumeTo ? opts.resumeTo : opts.status, gatedIds)),
    ),
    http.get(`${BASE}/findings`, () =>
      HttpResponse.json({ review_id: "r1", findings: opts.findings ?? [finding()] }),
    ),
    // ReviewDetail reads the replay verdict + audit feed + pipeline off /replay-timeline (PR 1
    // retired the standalone /replay endpoint). Mocking the old /replay left the timeline query
    // erroring, so the verdict badge / feed / pipeline rendered in an error state across every
    // decide test — a minimal equivalent timeline restores that coverage.
    http.get(`${BASE}/replay-timeline`, () =>
      HttpResponse.json({
        review_id: "r1",
        replay_equivalent: true,
        mode: "full",
        reason: null,
        status: opts.status,
        events: [],
        phases: [],
        inter_phase_events: [],
        findings: [],
        llm_exchanges: [],
      }),
    ),
    http.get(`${BASE}/events`, () =>
      HttpResponse.json({ review_id: "r1", events: [], total: 0 }),
    ),
    http.post(DECIDE, async ({ request }) => {
      if (opts.capture) opts.capture(await request.json());
      if (opts.decideStatus === 422) {
        return HttpResponse.json({ detail: { missing: [], extras: ["f-high"] } }, { status: 422 });
      }
      decided = true;
      return HttpResponse.json({ review_id: "r1", status: "running" }, { status: 202 });
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

beforeEach(() => useTokenStore.setState({ token: "k" }));

test("submit is gated until every gated finding is decided, then posts the right payload", async () => {
  const user = userEvent.setup();
  let body: unknown;
  mount({ capture: (b) => (body = b) });

  const submit = await screen.findByRole("button", { name: /Submit decision/ });
  expect(submit).toBeDisabled();
  expect(screen.getByText(/0 \/ 1/)).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "approve" }));
  expect(screen.getByText(/1 \/ 1/)).toBeInTheDocument();
  expect(submit).toBeEnabled();

  await user.click(submit);
  await waitFor(() => expect(screen.getByText(/Decision submitted/)).toBeInTheDocument());
  // approve sends reason: "" and no override_severity / original_severity / reviewer_id.
  expect(body).toEqual({ decisions: [{ finding_id: "f-high", outcome: "approve", reason: "" }] });
});

test("reject requires a non-blank reason before submit enables", async () => {
  const user = userEvent.setup();
  mount();
  const submit = await screen.findByRole("button", { name: /Submit decision/ });
  await user.click(screen.getByRole("button", { name: "reject" }));
  expect(submit).toBeDisabled();
  await user.type(screen.getByLabelText(/Reason/), "not a real risk");
  expect(submit).toBeEnabled();
});

test("severity_override requires a severity different from the finding's", async () => {
  const user = userEvent.setup();
  let body: unknown;
  mount({ capture: (b) => (body = b) });
  const submit = await screen.findByRole("button", { name: /Submit decision/ });
  await user.click(screen.getByRole("button", { name: "severity_override" }));
  await user.type(screen.getByLabelText(/Reason/), "actually medium");
  // The finding is "high"; that option is disabled, so picking "medium" is the path.
  expect(submit).toBeDisabled();
  await user.selectOptions(screen.getByLabelText("Override severity"), "medium");
  expect(submit).toBeEnabled();
  await user.click(submit);
  await waitFor(() => expect(screen.getByText(/Decision submitted/)).toBeInTheDocument());
  expect(body).toEqual({
    decisions: [
      { finding_id: "f-high", outcome: "severity_override", reason: "actually medium", override_severity: "medium" },
    ],
  });
});

test("non-gated findings on an actionable review show no decision controls", async () => {
  mount({ findings: [finding({ finding_id: "f-low", severity: "low" })] });
  await screen.findByText(/sql_injection/);
  expect(screen.queryByRole("button", { name: "approve" })).not.toBeInTheDocument();
  // No gated findings → no submit bar at all.
  expect(screen.queryByRole("button", { name: /Submit decision/ })).not.toBeInTheDocument();
});

test("a gated finding on a non-actionable review is read-only", async () => {
  mount({ status: "completed" }); // gatedIds defaults to ["f-high"] — still in the snapshot
  await screen.findByText(/sql_injection/);
  expect(screen.queryByRole("button", { name: "approve" })).not.toBeInTheDocument();
  expect(screen.getByText(/gated the PR/)).toBeInTheDocument();
});

test("gating follows the authoritative set, not severity — a high finding not in the set is read-only", async () => {
  // Severity is "high" (the old inference would have gated it) but the server's
  // findings_requiring_approval does NOT include it → no controls, no submit bar.
  mount({ gatedIds: [] });
  await screen.findByText(/sql_injection/);
  expect(screen.queryByRole("button", { name: "approve" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Submit decision/ })).not.toBeInTheDocument();
});

test("a 422 set-mismatch surfaces an explicit refresh message", async () => {
  const user = userEvent.setup();
  mount({ decideStatus: 422 });
  const submit = await screen.findByRole("button", { name: /Submit decision/ });
  await user.click(screen.getByRole("button", { name: "approve" }));
  await user.click(submit);
  await waitFor(() =>
    expect(screen.getByText(/no longer matches the review's gated findings/)).toBeInTheDocument(),
  );
});

// --- FUP-135: submitted-state recovery ---

test("surfaces the status advancing off the gate after submit (resume took)", async () => {
  const user = userEvent.setup();
  mount({ resumeTo: "completed" });
  const submit = await screen.findByRole("button", { name: /Submit decision/ });
  await user.click(screen.getByRole("button", { name: "approve" }));
  await user.click(submit);
  // onSuccess invalidates → detail refetches → returns completed → "now completed".
  expect(await screen.findByText(/review is now completed/)).toBeInTheDocument();
});

test("offers a manual refresh while awaiting resume", async () => {
  const user = userEvent.setup();
  mount();
  const submit = await screen.findByRole("button", { name: /Submit decision/ });
  await user.click(screen.getByRole("button", { name: "approve" }));
  await user.click(submit);
  expect(await screen.findByRole("button", { name: "Refresh status" })).toBeInTheDocument();
});

test("a stuck resume re-enables submit but keeps controls read-only and resends the same payload", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  const bodies: unknown[] = [];
  mount({ capture: (b) => bodies.push(b) }); // status stays awaiting — resume never advances
  const submit = await screen.findByRole("button", { name: /Submit decision/ });
  await user.click(screen.getByRole("button", { name: "reject" }));
  await user.type(screen.getByLabelText(/Reason/), "real risk");
  await user.click(submit);
  await screen.findByText(/resuming the review/);

  await act(async () => {
    vi.advanceTimersByTime(RESUME_WINDOW_MS + 100);
  });

  expect(await screen.findByText(/Resume hasn't completed/)).toBeInTheDocument();
  const resubmit = screen.getByRole("button", { name: /Re-submit/ });
  expect(resubmit).toBeEnabled();
  // Controls stay read-only — a divergent payload would wedge the audit row.
  expect(screen.getByRole("button", { name: "approve" })).toBeDisabled();

  await user.click(resubmit);
  await waitFor(() => expect(bodies.length).toBe(2));
  expect(bodies[1]).toEqual(bodies[0]); // byte-identical re-submit
});

afterEach(() => vi.useRealTimers());
