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

// Per-status server totals keyed off the `status` query param (the endpoint
// behavior). `total` is deliberately DIFFERENT from the loaded row count to prove
// the cards read server totals, not loaded-page length.
function mount(opts: { totals?: Record<string, number>; awaiting?: unknown[] } = {}) {
  const totals: Record<string, number> = {
    "": 240,
    awaiting_approval: 3,
    awaiting_approval_expired: 1,
    completed: 210,
    failed: 6,
    ...opts.totals,
  };
  server.use(
    http.get("http://localhost/api/reviews", ({ request }) => {
      const status = new URL(request.url).searchParams.get("status") ?? "";
      const reviews =
        status === "awaiting_approval" ? (opts.awaiting ?? [review({ id: "a", repo_id: 100 })]) : [];
      return HttpResponse.json({ reviews, total: totals[status] ?? 0, limit: 200, offset: 0 });
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

test("stat cards show server-side totals, not loaded-page counts", async () => {
  // Only 1 awaiting review is loaded, but the server total for awaiting is 3 (+1
  // expired = 4) — the card must show 4, not the 1 loaded row.
  mount();
  expect(await screen.findByText("240")).toBeInTheDocument(); // Reviews (unfiltered total)
  expect(screen.getByText("4")).toBeInTheDocument(); // Awaiting = 3 + 1, not loaded count
  expect(screen.getByText("210")).toBeInTheDocument(); // Completed
  expect(screen.getByText("6")).toBeInTheDocument(); // Failed
});

test("renders NO fabricated analytics — no delta, sparkline, chart, or range selector", async () => {
  const { container } = mount();
  await screen.findByText("240");
  // The honest-Overview guarantee: none of Signal's time-series chrome is present.
  expect(container.querySelector(".delta")).toBeNull();
  expect(container.querySelector(".card-spark")).toBeNull();
  expect(container.querySelector("svg")).toBeNull(); // no hero chart / sparkline svg
  expect(container.querySelector(".rangeseg")).toBeNull();
});

test("the needs-your-decision rail lists awaiting reviews", async () => {
  mount({ awaiting: [review({ id: "a", repo_id: 100, pr_number: 7 })] });
  expect(await screen.findByText("Needs your decision")).toBeInTheDocument();
  expect(screen.getByText("repo 100")).toBeInTheDocument();
});

test("rail empty state when nothing awaits", async () => {
  mount({ totals: { awaiting_approval: 0, awaiting_approval_expired: 0 }, awaiting: [] });
  expect(await screen.findByText("Nothing is awaiting your decision.")).toBeInTheDocument();
});
