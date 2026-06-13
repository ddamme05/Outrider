import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactElement } from "react";
import { MemoryRouter, useLocation } from "react-router";
import { beforeEach, expect, test } from "vitest";

import { useTokenStore } from "../auth/token";
import { useFilters } from "../state/filters";
import { server } from "../test/server";
import { Reviews } from "./Reviews";

// Probe that surfaces the current router location for navigation assertions.
function LocationProbe() {
  const loc = useLocation();
  return <div data-testid="loc">{loc.pathname}</div>;
}

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
    repo_full_name: "acme/api",
    pr_number: 7,
    pr_title: "Add session token storage",
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
    severity_counts: null,
    ...overrides,
  };
}

function statusCounts(overrides: Record<string, number> = {}) {
  return {
    running: 0,
    awaiting_approval: 0,
    awaiting_approval_expired: 0,
    completed: 0,
    failed: 0,
    skipped: 0,
    ...overrides,
  };
}

const isPill = (el: HTMLElement) => el.classList.contains("pill");

beforeEach(() => {
  useTokenStore.setState({ token: "test-key" });
  useFilters.setState({ includeEval: false, status: null, search: "" });
});

test("renders the table with repo, title, status, severity tally, cost, latency", async () => {
  // Absolute URL matching the test base (VITE_API_BASE_URL=http://localhost).
  server.use(
    http.get("http://localhost/api/reviews", () =>
      HttpResponse.json({
        reviews: [
          review({ id: "r1", repo_full_name: "acme/api", pr_number: 7, status: "running" }),
          review({
            id: "r2",
            repo_full_name: "acme/web",
            pr_number: 9,
            pr_title: "Add session token storage",
            status: "completed",
            metrics: {
              llm_calls_made: 4,
              total_input_tokens: 40,
              total_output_tokens: 20,
              total_cost_usd: 0.2,
              files_examined: 5,
              files_traced_beyond_diff: 2,
              wall_clock_seconds: 72,
            },
            severity_counts: { critical: 1, high: 2, medium: 1, low: 1, info: 0 },
          }),
        ],
        total: 2,
        limit: 50,
        offset: 0,
        status_counts: statusCounts({ running: 1, completed: 1 }),
      }),
    ),
  );

  renderWithProviders(<Reviews />);

  // Real repo names + titles + PR numbers — not "repo <id>".
  expect(await screen.findByText("acme/api")).toBeInTheDocument();
  expect(screen.getByText("acme/web")).toBeInTheDocument();
  expect(screen.getByText("#9")).toBeInTheDocument();
  expect(screen.getAllByText("Add session token storage").length).toBeGreaterThan(0);
  // Cost + latency render real, formatted values.
  expect(screen.getByText("$0.05")).toBeInTheDocument();
  expect(screen.getByText("$0.20")).toBeInTheDocument();
  expect(screen.getByText("1m12s")).toBeInTheDocument();
  // Status pills (not the same text in the filter chips).
  expect(screen.getAllByText("running").some(isPill)).toBe(true);
  expect(screen.getAllByText("completed").some(isPill)).toBe(true);
  // Severity tally for the completed review (non-zero tiers: 1,2,1,1; info=0 omitted).
  const tally = document.querySelector(".tally");
  expect(tally?.textContent).toBe("1211");
  // "All N" chip shows the summed status_counts.
  expect(screen.getByText(/^All/).textContent).toContain("2");
  // The ↻ replay control on the completed row routes to the surviving replay surface.
  const replay = screen.getByRole("link", { name: /replay reconstruction for review r2/i });
  expect(replay.getAttribute("href")).toBe("/reviews/r2/replay");
});

test("repo name and title fall back gracefully when null", async () => {
  server.use(
    http.get("http://localhost/api/reviews", () =>
      HttpResponse.json({
        reviews: [
          review({ id: "r1", repo_id: 555, repo_full_name: null, pr_title: null, status: "running" }),
        ],
        total: 1,
        limit: 50,
        offset: 0,
        status_counts: statusCounts({ running: 1 }),
      }),
    ),
  );

  renderWithProviders(<Reviews />);

  // No membership row → fall back to `repo <id>`; no title → em-dash.
  expect(await screen.findByText("repo 555")).toBeInTheDocument();
});

test("severity tally renders an em-dash before the review reaches synthesize", async () => {
  server.use(
    http.get("http://localhost/api/reviews", () =>
      HttpResponse.json({
        reviews: [review({ id: "r1", status: "running", severity_counts: null })],
        total: 1,
        limit: 50,
        offset: 0,
        status_counts: statusCounts({ running: 1 }),
      }),
    ),
  );

  renderWithProviders(<Reviews />);
  await screen.findByText("acme/api");
  // No `.tally` rendered when severity_counts is null.
  expect(document.querySelector(".tally")).toBeNull();
});

test("tells the truth when the queue is larger than the loaded page", async () => {
  server.use(
    http.get("http://localhost/api/reviews", () =>
      HttpResponse.json({
        reviews: [review({ id: "r1", status: "running" })],
        total: 80,
        limit: 200,
        offset: 0,
        status_counts: statusCounts({ running: 1 }),
      }),
    ),
  );

  renderWithProviders(<Reviews />);

  expect(await screen.findByText(/Showing 1 of 80 reviews/)).toBeInTheDocument();
});

test("clicking anywhere on a row navigates to the review detail", async () => {
  server.use(
    http.get("http://localhost/api/reviews", () =>
      HttpResponse.json({
        reviews: [review({ id: "r1", repo_full_name: "acme/api", pr_number: 7 })],
        total: 1,
        limit: 50,
        offset: 0,
        status_counts: statusCounts({ running: 1 }),
      }),
    ),
  );

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/reviews"]}>
        <LocationProbe />
        <Reviews />
      </MemoryRouter>
    </QueryClientProvider>,
  );

  // Click a NON-link cell (the repo name) — the row-level handler navigates.
  fireEvent.click(await screen.findByText("acme/api"));
  expect(screen.getByTestId("loc").textContent).toBe("/reviews/r1");
});
