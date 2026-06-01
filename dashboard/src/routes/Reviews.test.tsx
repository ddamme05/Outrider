import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, test } from "vitest";

import { useTokenStore } from "../auth/token";
import { useFilters } from "../state/filters";
import { server } from "../test/server";
import { Reviews } from "./Reviews";

function renderWithProviders(ui: ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

function review(overrides: Record<string, unknown>) {
  return {
    id: "00000000-0000-0000-0000-000000000000",
    installation_id: 1,
    repo_id: 100,
    pr_number: 7,
    head_sha: "abc1234def",
    status: "running",
    is_eval: false,
    created_at: "2026-05-31T00:00:00Z",
    updated_at: "2026-05-31T00:00:00Z",
    completed_at: null,
    metrics: {
      llm_calls_made: 2,
      total_input_tokens: 10,
      total_output_tokens: 5,
      total_cost_usd: 0.05,
      files_examined: null,
      files_traced_beyond_diff: null,
      wall_clock_seconds: null,
    },
    ...overrides,
  };
}

beforeEach(() => {
  useTokenStore.setState({ token: "test-key" });
  useFilters.setState({ includeEval: false, status: null, search: "" });
});

test("renders reviews grouped by status, with pills + cost", async () => {
  // Absolute URL matching the test base (VITE_API_BASE_URL=http://localhost) —
  // a relative handler would resolve against jsdom's location origin/port and
  // miss the port-80 fetch.
  server.use(
    http.get("http://localhost/api/reviews", () =>
      HttpResponse.json({
        reviews: [
          review({ id: "r1", repo_id: 100, pr_number: 7, status: "running" }),
          review({
            id: "r2",
            repo_id: 200,
            pr_number: 9,
            status: "completed",
            metrics: {
              llm_calls_made: 4,
              total_input_tokens: 40,
              total_output_tokens: 20,
              total_cost_usd: 0.2,
              files_examined: 5,
              files_traced_beyond_diff: 2,
              wall_clock_seconds: 42.5,
            },
          }),
        ],
        total: 2,
        limit: 50,
        offset: 0,
      }),
    ),
  );

  renderWithProviders(<Reviews />);

  // Status group headers (capitalized) appear once the query resolves.
  expect(await screen.findByText("Running")).toBeInTheDocument();
  expect(screen.getByText("Completed")).toBeInTheDocument();
  // Rows render real fields: repo, cost, and the status pill (raw status text).
  expect(screen.getByText("repo 100")).toBeInTheDocument();
  expect(screen.getByText("repo 200")).toBeInTheDocument();
  expect(screen.getByText("$0.05")).toBeInTheDocument();
  expect(screen.getByText("$0.20")).toBeInTheDocument();
  // Status text also appears in the filter <option>s, so target the pill span.
  const isPill = (el: HTMLElement) => el.classList.contains("pill");
  expect(screen.getAllByText("running").some(isPill)).toBe(true);
  expect(screen.getAllByText("completed").some(isPill)).toBe(true);
});
