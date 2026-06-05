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

// The replay-timeline DTO (ROADMAP feature 6): metadata-only, replay-verified, phase-grouped.
// `phases` is non-null only on an equivalent verdict (FUP-125); a non-equivalent verdict
// carries phases:null and the flat `events` stream is rendered as the AuditFeed fallback.
function timelineData(overrides: Record<string, unknown> = {}) {
  return {
    review_id: "r1",
    replay_equivalent: true,
    mode: "full",
    reason: null,
    status: "completed",
    events: [],
    phases: [],
    inter_phase_events: [],
    ...overrides,
  };
}

function mount(
  responses: {
    detail?: unknown;
    findings?: unknown[];
    timeline?: unknown;
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
    http.get(`${BASE}/replay-timeline`, () =>
      HttpResponse.json(responses.timeline ?? timelineData()),
    ),
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
  // Replay verdict — the hero badge reads off the timeline DTO's replay_equivalent.
  const verdict = await screen.findByLabelText("replay verdict");
  expect(verdict).toHaveTextContent("replay-equivalent");
  expect(verdict).not.toHaveTextContent("not replay-equivalent");
});

test("metrics strip surfaces files_examined + files_traced_beyond_diff", async () => {
  mount(); // detail() defaults: 5 examined, 2 traced beyond the diff
  const filesCard = (await screen.findByText("Files")).closest(".ms-card");
  expect(filesCard).toHaveTextContent("5");
  expect(filesCard).toHaveTextContent("examined");
  expect(filesCard).toHaveTextContent("2 traced beyond diff");
});

test("files render '—', never a misleading 0, before synthesize completes", async () => {
  mount({
    detail: detail({
      metrics: {
        llm_calls_made: 0,
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_cost_usd: 0,
        files_examined: null,
        files_traced_beyond_diff: null,
        wall_clock_seconds: null,
      },
    }),
  });
  const filesCard = (await screen.findByText("Files")).closest(".ms-card");
  expect(filesCard).toHaveTextContent("—");
  expect(filesCard).not.toHaveTextContent("0 examined");
});

test("renders a finding with severity pill, type, and destination", async () => {
  mount();
  expect(await screen.findByText(/sql_injection/)).toBeInTheDocument();
  const sevPill = screen.getByText("high");
  expect(sevPill.className).toContain("sev-high");
  expect(screen.getByText("INLINE_COMMENT")).toBeInTheDocument();
});

test("a finding with a HITL decision shows its override provenance", async () => {
  mount({
    findings: [
      finding({
        finding_id: "f2",
        hitl_decision: {
          outcome: "severity_override",
          reviewer_id: "admin",
          reason: "downgraded: test-only path",
          original_severity: "high",
          override_severity: "medium",
        },
      }),
    ],
  });
  await screen.findByText(/sql_injection/);
  const prov = document.querySelector(".f-prov");
  expect(prov).not.toBeNull();
  expect(prov).toHaveTextContent("severity_override");
  expect(prov).toHaveTextContent("high → medium");
  expect(prov).toHaveTextContent("by admin");
  expect(prov).toHaveTextContent("downgraded: test-only path");
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
  const user = userEvent.setup();
  mount({
    timeline: timelineData({
      replay_equivalent: false,
      phases: null,
      reason: "finding_count mismatch: 5 vs 4",
    }),
  });
  // Hero badge reflects the negative verdict at a glance.
  expect(await screen.findByText("not replay-equivalent")).toBeInTheDocument();
  // The reason surfaces in the timeline tab's banner (phase grouping suppressed, FUP-125).
  await user.click(screen.getByRole("tab", { name: /Timeline/ }));
  expect(await screen.findByText("finding_count mismatch: 5 vs 4")).toBeInTheDocument();
});

test("surfaces the reconstruction mode verbatim in the timeline header", async () => {
  const user = userEvent.setup();
  mount({ timeline: timelineData({ mode: "mixed" }) });
  // Equivalent verdict in the hero…
  expect(await screen.findByText("replay-equivalent")).toBeInTheDocument();
  // …and the timeline header shows the raw mode — "mixed" is never relabeled.
  await user.click(screen.getByRole("tab", { name: /Timeline/ }));
  expect(await screen.findByText(/· mixed/)).toBeInTheDocument();
});

test("handles a null-mode non-equivalent verdict (reconstruct failed)", async () => {
  const user = userEvent.setup();
  mount({
    timeline: timelineData({
      replay_equivalent: false,
      mode: null,
      status: null,
      phases: null,
      reason: "corrupt audit payload at event 12",
    }),
  });
  expect(await screen.findByText("not replay-equivalent")).toBeInTheDocument();
  await user.click(screen.getByRole("tab", { name: /Timeline/ }));
  expect(await screen.findByText("corrupt audit payload at event 12")).toBeInTheDocument();
  expect(screen.getByText(/phase grouping is unavailable/)).toBeInTheDocument();
});

test("shows an explicit state when the replay-timeline endpoint fails (not silent omission)", async () => {
  server.use(
    http.get("http://localhost/api/policy/:version", ({ params }) =>
      HttpResponse.json({ version: params.version, entries: [] }),
    ),
    http.get(BASE, () => HttpResponse.json(detail())),
    http.get(`${BASE}/findings`, () => HttpResponse.json({ review_id: "r1", findings: [finding()] })),
    http.get(`${BASE}/replay-timeline`, () => HttpResponse.json({ detail: "boom" }, { status: 500 })),
    http.get(`${BASE}/events`, () => HttpResponse.json({ review_id: "r1", events: [], total: 0 })),
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
  // The hero badge fails loud: an explicit "unavailable", never an absent badge.
  expect(await screen.findByText(/replay verdict unavailable/)).toBeInTheDocument();
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

test("timeline tab renders the reconstructed event stream grouped by phase", async () => {
  const user = userEvent.setup();
  const ev = llmEvent("analyze", 0.27);
  mount({
    timeline: timelineData({
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
    }),
  });
  await screen.findByText(/sql_injection/); // findings tab is default
  await user.click(screen.getByRole("tab", { name: /Timeline/ }));
  // The phase card surfaces the node header + the llm_call's metadata summary from the DTO.
  expect(await screen.findByText("llm_call")).toBeInTheDocument();
  const phaseNode = document.querySelector(".tl-phase .tl-node");
  expect(phaseNode).toHaveTextContent("analyze");
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

test("pipeline node cards derive per-node model and cost from the verified phases", async () => {
  // Completed review so analyze/triage are "done" and show their derived stats. The
  // per-node stats now come from the server's replay-verified phases (the FUP-125
  // closure), not from a client re-grouping of the raw /events stream.
  const triageEv = llmEvent("triage", 0.01);
  const analyzeEv = llmEvent("analyze", 0.27);
  mount({
    detail: detail({ status: "completed" }),
    timeline: timelineData({
      events: [triageEv, analyzeEv],
      phases: [
        {
          phase_id: "p-triage",
          node_id: "triage",
          start: phaseMarker("triage", "start", "2026-06-01T00:00:00Z"),
          end: phaseMarker("triage", "end", "2026-06-01T00:00:01Z"),
          events: [triageEv],
        },
        {
          phase_id: "p-analyze",
          node_id: "analyze",
          start: phaseMarker("analyze", "start", "2026-06-01T00:00:01Z"),
          end: phaseMarker("analyze", "end", "2026-06-01T00:00:03Z"),
          events: [analyzeEv],
        },
      ],
    }),
  });
  await screen.findByText(/sql_injection/);
  // Per-node costs derived from the phases, shown on the pipeline node cards.
  expect(await screen.findByText(/\$0\.27/)).toBeInTheDocument();
  expect(screen.getByText(/\$0\.01/)).toBeInTheDocument();
  // model prettified from claude-sonnet-4-5 → Sonnet (analyze + triage nodes).
  expect(screen.getAllByText("Sonnet").length).toBe(2);
});

test("completed review fails CLOSED when the verified phases are unavailable — no fabricated audit facts", async () => {
  // A completed review whose verdict is non-equivalent → the server suppresses phases
  // (the FUP-125 gate), AND whose raw /events count stream 500s. Node states are
  // status-backed (completed → done) but per-node stats + the event count must NOT be
  // asserted from thin air: never "0 audit events", never hitl "passed" / publish
  // "posted" without the verified reconstruction. Regression for the fail-open bug.
  server.use(
    http.get("http://localhost/api/policy/:version", ({ params }) =>
      HttpResponse.json({ version: params.version, entries: [] }),
    ),
    http.get(BASE, () => HttpResponse.json(detail({ status: "completed" }))),
    http.get(`${BASE}/findings`, () => HttpResponse.json({ review_id: "r1", findings: [finding()] })),
    http.get(`${BASE}/replay-timeline`, () =>
      HttpResponse.json(
        timelineData({ replay_equivalent: false, phases: null, reason: "drift", events: [] }),
      ),
    ),
    http.get(`${BASE}/events`, () => HttpResponse.json({ detail: "boom" }, { status: 500 })),
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
  // Wait for the phases-unavailable fallback note (proves phasesLoaded=false rendered).
  await screen.findByText(/load from the replay-verified timeline/);
  expect(screen.queryByText(/0 audit events/)).toBeNull();
  expect(screen.queryByText("passed")).toBeNull();
  expect(screen.queryByText("posted")).toBeNull();
});

test("completed review shows the authoritative gated count, not 0, in the findings header", async () => {
  // 2 findings gated the PR (decided at review time). Non-actionable now, so the
  // live `gated` set is empty — but the header must still show the server snapshot
  // count, not "0 gated" (the proof/HITL-side analogue of the /events fail-open).
  mount({
    detail: detail({ status: "completed", findings_requiring_approval: ["f1", "f2"] }),
    findings: [
      finding({ finding_id: "f1" }),
      finding({ finding_id: "f2", finding_type: "xss" }),
    ],
  });
  await screen.findByText(/sql_injection/);
  expect(screen.getByText(/2 findings · 2 gated/)).toBeInTheDocument();
});

test("findings header fails CLOSED when /findings is unavailable — no fabricated 0 findings", async () => {
  server.use(
    http.get("http://localhost/api/policy/:version", ({ params }) =>
      HttpResponse.json({ version: params.version, entries: [] }),
    ),
    http.get(BASE, () => HttpResponse.json(detail())),
    http.get(`${BASE}/findings`, () => HttpResponse.json({ detail: "boom" }, { status: 500 })),
    http.get(`${BASE}/replay-timeline`, () => HttpResponse.json(timelineData())),
    http.get(`${BASE}/events`, () => HttpResponse.json({ review_id: "r1", events: [], total: 0 })),
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
  // The findings tab surfaces the load error explicitly…
  expect(await screen.findByText("Failed to load findings.")).toBeInTheDocument();
  // …and the findings count fails closed to "—", never a fabricated "0 findings"
  // (asserted on the metrics card, which is unique — the pipeline's "paused · N
  // findings" is a separate gate count from the detail snapshot).
  expect(screen.getByText(/policy v3 · — findings/)).toBeInTheDocument();
});

test("awaiting review with no gate snapshot shows 'paused' without a fabricated 0 count", async () => {
  // findings_requiring_approval is null (no HITL-request snapshot) — distinct from
  // [] (snapshot, nothing gated). The pipeline must not claim "paused · 0 findings"
  // or "0 critical/high"; it shows just "paused" and a count-free gate note.
  mount(); // default detail() is awaiting_approval with no findings_requiring_approval
  await screen.findByText(/repo 100/);
  expect(screen.queryByText(/paused · 0 findings/)).toBeNull();
  expect(screen.queryByText(/0 critical\/high/)).toBeNull();
  expect(screen.getByText("paused")).toBeInTheDocument();
});
