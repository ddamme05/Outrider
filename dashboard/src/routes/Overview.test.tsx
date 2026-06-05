import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, test } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { Overview } from "./Overview";

function review(overrides: Record<string, unknown>) {
  return {
    id: "x",
    installation_id: 1,
    repo_id: 1,
    pr_number: 1,
    head_sha: "deadbeef00",
    status: "awaiting_approval",
    is_eval: false,
    created_at: "2026-06-01T00:00:00Z",
    updated_at: "2026-06-01T00:00:00Z",
    completed_at: null,
    metrics: {
      llm_calls_made: 1,
      total_input_tokens: 1,
      total_output_tokens: 1,
      total_cost_usd: 0.1,
      files_examined: null,
      files_traced_beyond_diff: null,
      wall_clock_seconds: null,
    },
    ...overrides,
  };
}

// A DashboardMetricsResponse fixture (DECISIONS#039). Defaults mirror the mockup's
// example window; the current totals deliberately differ from the previous window
// so delta polarity (up-good / up-bad) is exercised.
function metrics(overrides: Record<string, unknown> = {}) {
  return {
    window: "7d",
    granularity: "day",
    generated_at: "2026-06-04T12:00:00Z",
    buckets: [
      { bucket: "2026-05-29T00:00:00Z", reviews: 2, cost_usd: 1.9, findings: 9, failed: 0 },
      { bucket: "2026-05-30T00:00:00Z", reviews: 4, cost_usd: 2.4, findings: 12, failed: 1 },
      { bucket: "2026-05-31T00:00:00Z", reviews: 3, cost_usd: 1.2, findings: 8, failed: 0 },
      { bucket: "2026-06-01T00:00:00Z", reviews: 5, cost_usd: 2.8, findings: 11, failed: 0 },
      { bucket: "2026-06-02T00:00:00Z", reviews: 4, cost_usd: 1.7, findings: 10, failed: 0 },
      { bucket: "2026-06-03T00:00:00Z", reviews: 2, cost_usd: 2.1, findings: 7, failed: 1 },
      { bucket: "2026-06-04T00:00:00Z", reviews: 4, cost_usd: 2.1, findings: 6, failed: 0 },
    ],
    severity_distribution: { critical: 4, high: 12, medium: 23, low: 18, info: 6 },
    evidence_tier_distribution: { observed: 29, inferred: 14, judged: 20 },
    deltas: {
      current: { reviews: 24, cost_usd: 14.2, findings: 63, failed: 2 },
      previous: { reviews: 20, cost_usd: 11.68, findings: 63, failed: 4 },
    },
    ...overrides,
  };
}

function mount(opts: { metrics?: unknown; awaiting?: unknown[]; awaitingTotal?: number } = {}) {
  server.use(
    http.get("http://localhost/api/metrics", () => HttpResponse.json(opts.metrics ?? metrics())),
    http.get("http://localhost/api/reviews", ({ request }) => {
      const status = new URL(request.url).searchParams.get("status") ?? "";
      const reviews =
        status === "awaiting_approval" ? (opts.awaiting ?? [review({ id: "a", repo_id: 100 })]) : [];
      const total =
        status === "awaiting_approval" ? (opts.awaitingTotal ?? reviews.length) : 0;
      return HttpResponse.json({ reviews, total, limit: 200, offset: 0 });
    }),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Overview />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => useTokenStore.setState({ token: "k" }));

test("KPI cards render the windowed metric totals + period-over-period deltas", async () => {
  mount();
  // $14.20 also appears in the chart legend (the cost buckets sum to the window
  // total — correct), so anchor on the unique avg-per-review cap.
  expect(await screen.findByText("$0.59 avg / review")).toBeInTheDocument(); // 14.20 / 24
  expect(screen.getByText("24")).toBeInTheDocument(); // Reviews total (windowed)
  // Reviews 24 vs prior 20 → up-good ▲ 20%. Failed 2 vs prior 4 → fewer-is-good ▼ 50%.
  expect(screen.getByText("20%")).toBeInTheDocument();
  expect(screen.getByText("50%")).toBeInTheDocument();
});

test("renders the analytics chrome — delta, sparkline, hero chart, range selector", async () => {
  const { container } = mount();
  await screen.findByText("$0.59 avg / review");
  // The #039 reversal: Signal's time-series chrome is now PRESENT (was asserted-absent).
  expect(container.querySelector(".delta")).not.toBeNull();
  expect(container.querySelector(".card-spark")).not.toBeNull();
  expect(container.querySelector(".chart-svg")).not.toBeNull();
  expect(container.querySelector(".rangeseg")).not.toBeNull();
});

test("findings distributions render severity + tier counts and the proof footnote", async () => {
  mount();
  await screen.findByText("$0.59 avg / review");
  // the proof-boundary footnote renders verbatim (unique strings)
  expect(screen.getByText("query_match_id")).toBeInTheDocument();
  expect(screen.getByText("trace_path")).toBeInTheDocument();
  expect(screen.getByText("model interpretation")).toBeInTheDocument();
  // tier tracks (scoped — "observed" also appears in the footnote)
  const tierRows = document.querySelector(".tier-rows");
  expect(tierRows).toHaveTextContent("observed");
  expect(tierRows).toHaveTextContent("29"); // observed representative count
  // the full 5-value severity enum is listed in the legend
  const segLegend = document.querySelector(".seg-legend");
  expect(segLegend).toHaveTextContent("critical");
  expect(segLegend).toHaveTextContent("info");
});

test("honest zeros on an empty window — no fabricated series", async () => {
  mount({
    metrics: metrics({
      buckets: [{ bucket: "2026-06-04T00:00:00Z", reviews: 0, cost_usd: 0, findings: 0, failed: 0 }],
      severity_distribution: { critical: 0, high: 0, medium: 0, low: 0, info: 0 },
      evidence_tier_distribution: { observed: 0, inferred: 0, judged: 0 },
      deltas: {
        current: { reviews: 0, cost_usd: 0, findings: 0, failed: 0 },
        previous: { reviews: 0, cost_usd: 0, findings: 0, failed: 0 },
      },
    }),
  });
  // Empty distribution renders an explicit honest message, never a faked bar.
  expect(await screen.findByText("No findings in this window.")).toBeInTheDocument();
  expect(screen.getByText("$0.00 avg / review")).toBeInTheDocument();
});

test("metrics fail CLOSED — explicit error, never fabricated zeros", async () => {
  server.use(
    http.get("http://localhost/api/metrics", () => new HttpResponse(null, { status: 500 })),
    http.get("http://localhost/api/reviews", () =>
      HttpResponse.json({ reviews: [], total: 0, limit: 200, offset: 0 }),
    ),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { container } = render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Overview />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/couldn.t load metrics/i)).toBeInTheDocument();
  // No KPI card is rendered with a fabricated value on error.
  expect(container.querySelector(".metric-card")).toBeNull();
});

test("the needs-your-decision rail lists awaiting reviews", async () => {
  mount({ awaiting: [review({ id: "a", repo_id: 100, pr_number: 7 })] });
  expect(await screen.findByText("Needs decision")).toBeInTheDocument();
  // rail rows render once the approval queries resolve (the header is immediate)
  expect(await screen.findByText("repo 100")).toBeInTheDocument();
});

test("rail empty state when nothing awaits", async () => {
  mount({ awaiting: [], awaitingTotal: 0 });
  expect(await screen.findByText("Nothing is awaiting your decision.")).toBeInTheDocument();
});

test("rail fails CLOSED when an approval-queue query errors — no false 'all clear'", async () => {
  // Metrics succeed (analytics render), but the awaiting_approval query 500s before
  // it ever resolved. The rail must NOT collapse to the empty state — that would
  // tell the operator nothing awaits when we simply couldn't check, hiding a review
  // potentially blocked at the high/critical gate.
  server.use(
    http.get("http://localhost/api/metrics", () => HttpResponse.json(metrics())),
    http.get("http://localhost/api/reviews", ({ request }) => {
      const status = new URL(request.url).searchParams.get("status") ?? "";
      if (status === "awaiting_approval") {
        return new HttpResponse(null, { status: 500 });
      }
      return HttpResponse.json({ reviews: [], total: 0, limit: 200, offset: 0 });
    }),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Overview />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  // Analytics render off the (successful) metrics query…
  expect(await screen.findByText("$0.59 avg / review")).toBeInTheDocument();
  // …but the rail fails closed: explicit error, never the all-clear empty state.
  expect(await screen.findByText(/couldn.t load the approval queue/i)).toBeInTheDocument();
  expect(screen.queryByText("Nothing is awaiting your decision.")).toBeNull();
});
