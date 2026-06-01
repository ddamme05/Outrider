import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, test } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { Home } from "./Home";

function review(overrides: Record<string, unknown>) {
  return {
    id: "x",
    installation_id: 1,
    repo_id: 1,
    pr_number: 1,
    head_sha: "deadbeef00",
    status: "completed",
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

function renderHome(reviews: Array<ReturnType<typeof review>>) {
  // Home now filters server-side (one query per awaiting status), so the handler
  // must honor the `status` query param the way the backend does.
  server.use(
    http.get("http://localhost/api/reviews", ({ request }) => {
      const status = new URL(request.url).searchParams.get("status");
      const matched = status ? reviews.filter((r) => r.status === status) : reviews;
      return HttpResponse.json({ reviews: matched, total: matched.length, limit: 200, offset: 0 });
    }),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Home />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => useTokenStore.setState({ token: "k" }));

test("lists only reviews awaiting a decision", async () => {
  renderHome([
    review({ id: "a", repo_id: 100, pr_number: 7, status: "awaiting_approval" }),
    review({ id: "b", repo_id: 200, pr_number: 9, status: "awaiting_approval_expired" }),
    review({ id: "c", repo_id: 300, pr_number: 1, status: "completed" }),
  ]);
  expect(await screen.findByText("repo 100")).toBeInTheDocument();
  expect(screen.getByText("repo 200")).toBeInTheDocument();
  expect(screen.queryByText("repo 300")).not.toBeInTheDocument();
});

test("shows an empty state when nothing is awaiting", async () => {
  renderHome([review({ id: "c", status: "completed" })]);
  expect(await screen.findByText("Nothing is awaiting your decision.")).toBeInTheDocument();
});

test("surfaces an honest notice when the awaiting set exceeds the page", async () => {
  // total (from the status-filtered query) exceeds the loaded rows.
  server.use(
    http.get("http://localhost/api/reviews", ({ request }) => {
      const status = new URL(request.url).searchParams.get("status");
      if (status === "awaiting_approval") {
        return HttpResponse.json({
          reviews: [review({ id: "a", repo_id: 100, status: "awaiting_approval" })],
          total: 250,
          limit: 200,
          offset: 0,
        });
      }
      return HttpResponse.json({ reviews: [], total: 0, limit: 200, offset: 0 });
    }),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Home />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/Showing 1 of 250 reviews awaiting a decision/)).toBeInTheDocument();
});
