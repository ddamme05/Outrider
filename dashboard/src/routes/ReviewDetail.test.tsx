import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
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
      wall_clock_seconds: 3,
      pre_approval_seconds: 72,
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
    // Use the lowercase wire value; the display label is checked below.
    publish_destination: "inline_comment",
    eligibility: "withheld",
    eligibility_reason: "hitl_required_node_absent",
    redaction_sweep_at: null,
    ...overrides,
  };
}

// Replay-verified phases feed the pipeline and replay view (FUP-125).
// Divergent verdicts carry phases:null and use the flat AuditFeed fallback.
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
    findings: [],
    llm_exchanges: [],
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
    search?: string;
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
      <MemoryRouter initialEntries={[`/reviews/r1${responses.search ?? ""}`]}>
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
  const reviewTime = screen.getByText("Review time");
  expect(reviewTime).toHaveAccessibleDescription(/intake.*synthesis.*approval.*publication/i);
  expect(reviewTime.closest(".ms-card")).toHaveTextContent("1m12s");
  // Replay verdict — the hero pill reads off the timeline DTO's replay_equivalent
  // and is a clickable button (opens the explainer).
  const verdict = await screen.findByRole("button", { name: /what replay-equivalent means/i });
  expect(verdict).toHaveTextContent("replay-equivalent");
  expect(verdict).not.toHaveTextContent("not replay-equivalent");
});

test.each([
  [undefined, "—"], [null, "—"], [0, "0ms"],
])("displays review time %s as %s", async (duration, expected) => {
  mount({ detail: detail({ metrics: { ...detail().metrics, pre_approval_seconds: duration } }) });
  const card = (await screen.findByText("Review time")).closest(".ms-card");
  expect(card?.querySelector(".ms-val")?.textContent).toBe(expected);
});

test("clicking the replay verdict opens the explainer modal", async () => {
  const user = userEvent.setup();
  mount();
  await user.click(await screen.findByRole("button", { name: /what replay-equivalent means/i }));
  expect(await screen.findByText("Replay equivalence")).toBeInTheDocument();
  expect(
    screen.getByText("Every review is reconstructable from its append-only audit log."),
  ).toBeInTheDocument();
});

test("metrics strip surfaces files_examined + files_traced_beyond_diff", async () => {
  mount(); // detail() defaults: 5 examined, 2 traced beyond the diff
  const filesCard = (await screen.findByText("Files")).closest(".ms-card");
  expect(filesCard).toHaveTextContent("5");
  expect(filesCard).toHaveTextContent("examined");
  expect(filesCard).toHaveTextContent("2 traced beyond diff");
});

test.each([
  [18480, 512, `${(18480).toLocaleString()} in`],
  [0, 0, "0 in"],
  [null, 40, "Input unavailable"],
  [undefined, 40, "Input unavailable"],
] as const)("shows the recorded prompt total %s in review metrics", async (total, ordinary, label) => {
  const review = detail();
  mount({
    detail: {
      ...review,
      metrics: { ...review.metrics, total_input_tokens: ordinary, total_prompt_tokens: total },
    },
  });
  const card = (await screen.findByText("Tokens")).closest(".ms-card");
  expect(card?.querySelector(".ms-val")?.textContent).toBe(label);
  expect(card).toHaveTextContent("20 out");
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

test("renders a finding with humanized severity pill, type, dimension, and destination", async () => {
  mount();
  // Render-wait on a unique humanized label (single finding → one destination tag).
  await screen.findByText("Inline comment");
  // Severity pill: humanized text "High", CSS class still keyed off the lowercase wire value.
  const sevPill = document.querySelector(".sev-pill");
  expect(sevPill).toHaveTextContent("High");
  expect(sevPill?.className).toContain("sev-high");
  // The title repeats the type label; scope this check to the type/dimension chip.
  const tag = document.querySelector(".ft-tag");
  expect(tag).toHaveTextContent("SQL injection");
  expect(tag).toHaveTextContent("Security");
  // The display label and CSS class use different forms of the wire value.
  const dest = screen.getByText("Inline comment");
  expect(dest.className).toContain("dest-inline");
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
  await screen.findAllByText(/SQL injection/); // render-wait; humanized label may repeat (title + type chip)
  const prov = document.querySelector(".f-prov");
  expect(prov).not.toBeNull();
  expect(prov).toHaveTextContent("Severity overridden");
  expect(prov).toHaveTextContent("High → Medium");
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
  mount({
    timeline: timelineData({
      replay_equivalent: false,
      phases: null,
      reason: "finding_count mismatch: 5 vs 4",
    }),
  });
  // Hero badge reflects the negative verdict at a glance.
  expect(await screen.findByText("not replay-equivalent")).toBeInTheDocument();
  // The audit banner explains why phase grouping is unavailable (FUP-125).
  expect(await screen.findByText("finding_count mismatch: 5 vs 4")).toBeInTheDocument();
});

test("handles a null-mode non-equivalent verdict (reconstruct failed)", async () => {
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

test("the always-shown audit feed renders the reconstructed stream as a flat phase-divided feed", async () => {
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
  // The mockup's flat .ae feed renders inline below the findings (no tab): an append-only banner,
  // an .ae-phase divider naming the node, and .ae event rows with a type chip + rich body + time.
  expect(await screen.findByText(/Append-only by database policy/)).toBeInTheDocument();
  expect(document.querySelector(".ae-phase .pname")).toHaveTextContent("analyze");
  expect(screen.getByText("llm_call")).toBeInTheDocument();
  // The flat body bolds the model (prettified via prettyModel, like every model label); scope to
  // the .ae feed since the pipeline strip shows the same "Sonnet". The relative-timestamp is present.
  expect(document.querySelector(".ae")?.textContent).toContain("Sonnet");
  expect(document.querySelector(".ae .ae-time")).toBeInTheDocument();
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
  // Completed node statistics come from server-verified phases (FUP-125),
  // rather than regrouping the raw /events response in the client.
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
  await screen.findAllByText(/SQL injection/); // render-wait; humanized label may repeat (title + type chip)
  // Scope to pipeline cards because the audit feed repeats these costs.
  const pnStats = Array.from(document.querySelectorAll(".pn-stat")).map((e) => e.textContent ?? "");
  expect(pnStats.some((s) => /\$0\.27/.test(s))).toBe(true);
  expect(pnStats.some((s) => /\$0\.01/.test(s))).toBe(true);
  // Count model labels on the two active cards, excluding audit-feed labels.
  const pnModels = Array.from(document.querySelectorAll(".pn-model")).map((e) => e.textContent ?? "");
  expect(pnModels.filter((m) => m === "Sonnet").length).toBe(2);
});

test("completed review fails CLOSED when the verified phases are unavailable — no fabricated audit facts", async () => {
  // Missing verified phases (FUP-125) and a failed event read must leave audit
  // facts unknown, even when the review's status is completed.
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
  await screen.findAllByText(/SQL injection/); // render-wait; humanized label may repeat (title + type chip)
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
  // The findings panel surfaces the load error explicitly.
  expect(await screen.findByText("Failed to load findings.")).toBeInTheDocument();
  // The metrics count stays unknown; the pipeline uses a separate approval snapshot.
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

test("saved warnings link to each file's original review, including empty reused results", async () => {
  const user = userEvent.setup();
  const source = "302e176f-e13e-4e5b-a931-b19d0fe836eb";
  const other = "f3073fb9-60bc-4bb6-9dbe-eaa94c43d4bb";
  const cache = "a".repeat(64);
  const otherCache = "b".repeat(64);
  const served = { event_type: "cache_serve", source_review_id: source, cache_key: cache, served_finding_count: 2 };
  mount({ events: [
    { ...served, event_id: "e1", file_path: "src/bin.js" },
    { ...served, event_id: "e2", file_path: "test/bin-test.js", served_finding_count: 0, source_review_id: other, cache_key: otherCache },
    { event_type: "cache_lookup", outcome: "would_hit", file_path: "README.md" },
  ] });
  server.use(http.get(`${BASE}/candidates`, () => HttpResponse.json({
    review_id: "r1", total: 0, limit: 20, offset: 0, candidates: [],
  })));
  await user.click(await screen.findByRole("button", { name: "Show saved warnings" }));
  expect(await screen.findByRole("link", { name: "Original review for src/bin.js" }))
    .toHaveAttribute("href", `/reviews/${source}?cache=${cache}`);
  expect(screen.getByRole("link", { name: "Original review for test/bin-test.js" }))
    .toHaveAttribute("href", `/reviews/${other}?cache=${otherCache}`);
  expect(screen.queryByRole("link", { name: /Original review for README/ })).toBeNull();
  expect(await screen.findByText("No new model warnings were saved in this review.")).toBeVisible();
});

test.each(["retained", "expired", "unverified", "missing phases"])("the original exchange honors %s content", async (mode) => {
  const user = userEvent.setup();
  const cache = "c".repeat(64);
  const phase = "file:tests/bin.js#0";
  const lookup = { event_type: "cache_lookup", node_id: "analyze", event_id: "lookup", phase_key: phase,
    outcome: "miss", cache_key: cache, file_path: "tests/bin.js" };
  const call = { event_type: "llm_call", node_id: "analyze", event_id: "wanted", phase_key: phase,
    model: "gpt", input_tokens: 10, output_tokens: 2, cost_usd: mode === "expired" ? null : 0,
    cached_tokens: 7, ...(mode === "expired" ? {} : { cache_write_tokens: 3 }),
    context_summary: [{ file_path: "lib/bin.js", line_start: 3, line_end: 9,
      scope_unit_name: "bucket", inclusion_reason: "captured_reference" }] };
  const sibling = { ...call, event_id: "sibling", phase_key: "file:tests/other.js#0",
    context_summary: [{ ...call.context_summary[0], file_path: "lib/other.js" }] };
  mount({ search: `?cache=${cache}`, timeline: timelineData({
    replay_equivalent: mode !== "unverified", phases: mode === "unverified" || mode === "missing phases" ? null : [],
    events: [lookup, sibling, call], llm_exchanges: [
      { event_id: "sibling", prompt: "Sibling prompt", completion: "Sibling answer", content_redacted: false },
      { event_id: "wanted", prompt: "Target prompt", completion: "Target answer", content_redacted: mode === "expired" },
    ],
  }) });
  if (mode === "unverified" || mode === "missing phases") {
    expect(await screen.findByText(/original file analysis could not be identified/i)).toBeVisible();
    expect(screen.queryByText(/Recorded source ranges/)).toBeNull();
    expect(screen.queryByText("Target answer")).toBeNull();
    return;
  }
  const heading = await screen.findByRole("heading", { name: "Original analysis of tests/bin.js" });
  const section = within(heading.closest("section")!);
  await user.click(section.getByText("Model exchange for tests/bin.js"));
  expect(section.queryByText("Sibling answer")).toBeNull();
  const tokens = mode === "expired" ? /input unavailable \/ 2 out/ : /20 in \/ 2 out/;
  expect(section.getByText(tokens)).toHaveTextContent(mode === "expired" ? "unpriced" : "$0.00");
  const ranges = section.getByText("Recorded source ranges (1)");
  expect(ranges.closest("details")).not.toHaveAttribute("open");
  await user.click(ranges);
  expect(section.getByText("lib/bin.js:3–9")).toBeVisible();
  expect(section.getByRole("listitem")).toHaveTextContent("Captured reference · bucket");
  expect(section.getByRole("listitem")).not.toHaveTextContent(/HEAD|BASE/);
  expect(section.queryByText(/lib\/other\.js/)).toBeNull();
  if (mode === "expired") {
    expect(section.getByText(/Content redacted/)).toBeVisible();
    expect(section.queryByText("Target answer")).toBeNull();
  } else {
    expect(section.getByText("Target prompt")).toBeVisible();
    expect(section.getByText("Target answer")).toBeVisible();
  }
});

test.each([false, true])("saved warnings load on demand and honor expiry=%s", async (expired) => {
  const user = userEvent.setup();
  let reads = 0;
  server.use(http.get(`${BASE}/candidates`, () => {
    reads += 1;
    return HttpResponse.json({
      review_id: "r1", total: 1, limit: 20, offset: 0,
      candidates: [{
        candidate: { candidate_id: "c1", file_path: "account.py", severity: null, source_side: "base" },
        draft_disposition: "needs_classification",
        original: expired ? null : {
          title: "<script>original warning</script>",
          description: "The account identifier is inconsistent.",
          finding_type: "identity_mixup", evidence: "user.id", line_start: 8, line_end: 8,
        },
        raw_preview: null, preview_truncated: false, content_redacted: expired,
        assessment: { status: "uncertain", outcome: "completed", suggested_finding_type: null },
        assessment_reason: expired ? null : "The caller context is missing.",
        assessment_content_redacted: expired,
      }],
    });
  }));
  mount();
  const expand = await screen.findByRole("button", { name: "Show saved warnings" });
  expect(reads).toBe(0);
  await user.click(expand);
  expect(await screen.findByText(/Needs classification/)).toBeInTheDocument();
  expect(screen.getByText(/Removed BASE \(absent from HEAD\)/)).toBeInTheDocument();
  expect(reads).toBe(1);
  if (expired) {
    expect(screen.getByText("Original content is no longer available.")).toBeInTheDocument();
    expect(screen.queryByText("<script>original warning</script>")).toBeNull();
  } else {
    expect(screen.getByText("<script>original warning</script>")).toBeInTheDocument();
    expect(screen.getByText("The caller context is missing.")).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
  }
});

test("saved checks distinguish current uncertainty from PR relevance", async () => {
  const user = userEvent.setup();
  server.use(http.get(`${BASE}/candidates`, () => HttpResponse.json({
    review_id: "r1", total: 1, limit: 20, offset: 0,
    candidates: [{
      candidate: { candidate_id: "c1", file_path: "account.py", severity: null },
      draft_disposition: "needs_classification",
      original: {
        title: "Existing account concern", description: "The account constraint is unknown.",
        finding_type: "identity_mixup", evidence: "user.email", line_start: 8, line_end: 8,
      },
      raw_preview: null, preview_truncated: false, content_redacted: false,
      assessment: {
        status: "refutes", outcome: "completed", suggested_finding_type: null,
        head_assessment: "uncertain", change_assessment: "unrelated_preexisting",
      },
      assessment_reason: "The change does not affect this existing concern.",
      assessment_content_redacted: false,
    }],
  })));
  mount();
  await user.click(await screen.findByRole("button", { name: "Show saved warnings" }));
  expect(await screen.findByText("Existing account concern")).toBeInTheDocument();
  expect(screen.getByText("Follow-up check: refutes · completed")).toBeInTheDocument();
  expect(screen.getByText(
    "Local bug: uncertain · PR relation: existing concern unrelated to this change.",
  )).toBeInTheDocument();
  expect(screen.queryByText(/Claimed impact:/)).toBeNull();
});

test.each([
  ["uncertain", "uncertain", "unproved"],
  ["not_claimed", "supports", "no broader impact claimed"],
])("saved checks show claimed impact separately: %s", async (impact, status, label) => {
  const user = userEvent.setup();
  server.use(http.get(`${BASE}/candidates`, () => HttpResponse.json({
    review_id: "r1", total: 1, limit: 20, offset: 0,
    candidates: [{
      candidate: { candidate_id: "c1", file_path: "app/request.py", severity: "low" },
      draft_disposition: "included_in_draft",
      original: {
        title: "Large integer response", description: "An oversized integer raises ValueError.",
        finding_type: "missing_error_handling", evidence: "json.loads(body)",
        line_start: 8, line_end: 8,
      },
      raw_preview: null, preview_truncated: false, content_redacted: false,
      assessment: {
        status, outcome: "completed", suggested_finding_type: null,
        head_assessment: "supports", impact_assessment: impact,
        change_assessment: "introduced_or_worsened",
      },
      assessment_reason: "The local exception is supported; the caller response is not shown.",
      assessment_content_redacted: false,
    }],
  })));
  mount();
  await user.click(await screen.findByRole("button", { name: "Show saved warnings" }));
  expect(await screen.findByText("Large integer response")).toBeInTheDocument();
  expect(screen.getByText(
    `Local bug: supported · Claimed impact: ${label} · PR relation: introduced or worsened by this change.`,
  )).toBeInTheDocument();
  expect(screen.getByText("Proposed repair: not assessed.")).toBeInTheDocument();
});

test("saved checks separate a supported bug from a refuted repair", async () => {
  const user = userEvent.setup();
  const originalClaim = "A cookie can shadow hasOwnProperty. Use Object.create(null) for the map.";
  server.use(http.get(`${BASE}/candidates`, () => HttpResponse.json({
    review_id: "r1", total: 1, limit: 20, offset: 0,
    candidates: [{
      candidate: { candidate_id: "c1", file_path: "cookies.js", severity: "medium" },
      draft_disposition: "included_in_draft",
      original: {
        title: "Cookie shadows an object method", description: originalClaim,
        finding_type: "missing_error_handling", evidence: "cookies.hasOwnProperty(name)",
        line_start: 8, line_end: 8,
      },
      raw_preview: null, preview_truncated: false, content_redacted: false,
      assessment: {
        status: "uncertain", outcome: "completed", suggested_finding_type: null,
        head_assessment: "supports", impact_assessment: "not_claimed",
        change_assessment: "introduced_or_worsened", repair_assessment: "refutes",
      },
      assessment_reason: "A null-prototype map still cannot call hasOwnProperty directly.",
      assessment_content_redacted: false,
    }],
  })));
  mount();
  await user.click(await screen.findByRole("button", { name: "Show saved warnings" }));
  expect(await screen.findByText(originalClaim)).toBeInTheDocument();
  expect(screen.getByText("Follow-up check: uncertain · completed")).toBeInTheDocument();
  expect(screen.getByText(
    "Local bug: supported · Claimed impact: no broader impact claimed · PR relation: introduced or worsened by this change.",
  )).toBeInTheDocument();
  expect(screen.getByText("Proposed repair: refuted.")).toBeInTheDocument();
});

test("a low priority finding held for verification can be decided without a severity override", async () => {
  const user = userEvent.setup();
  mount({
    detail: detail({ findings_requiring_approval: ["f1"] }),
    findings: [finding({ severity: "low", eligibility_reason: "hitl_required_node_absent" })],
  });
  const approve = await screen.findByRole("button", { name: "approve" });
  expect(screen.queryByRole("button", { name: "severity_override" })).toBeNull();
  await user.click(approve);
  expect(approve).toHaveAttribute("aria-pressed", "true");
});
