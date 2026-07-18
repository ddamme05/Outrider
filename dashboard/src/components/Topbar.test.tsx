import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, test } from "vitest";

import { useTokenStore } from "../auth/token";
import { useNav } from "../state/nav";
import { server } from "../test/server";
import { Topbar } from "./Topbar";

// A 30d ReplayMetricsResponse for the global health pill. 47/50 = 94% equivalent.
function replay(overrides: Record<string, unknown> = {}) {
  return {
    window: "30d",
    granularity: "day",
    buckets: [],
    generated_at: "2026-06-04T12:00:00Z",
    window_end: "2026-06-04T12:00:00Z",
    anchored: false,
    deltas: {
      current: { equivalent: 47, total: 50 },
      previous: { equivalent: 40, total: 45 },
    },
    ...overrides,
  };
}

function mount(replayResponder?: Parameters<typeof http.get>[1]) {
  server.use(
    http.get(
      "http://localhost/api/metrics/replay",
      replayResponder ?? (() => HttpResponse.json(replay())),
    ),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Topbar />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  useTokenStore.setState({ token: "k" });
  useNav.setState({ open: false });
});

test("renders the global replay-equivalence pill from /api/metrics/replay (30d)", async () => {
  mount();
  // 47/50 = 94% → rounded to whole percent.
  expect(await screen.findByText("94% replay")).toBeInTheDocument();
});

test("the pill fails CLOSED — no fabricated value when /api/metrics/replay errors", async () => {
  mount(() => new HttpResponse(null, { status: 500 }));
  // The search field always renders (a stable anchor); the pill must NOT appear.
  expect(await screen.findByPlaceholderText("filter reviews…")).toBeInTheDocument();
  expect(screen.queryByText(/replay/)).toBeNull();
});

test("no pill when no reviews are verdicted (total 0) — honest, never 0%", async () => {
  mount(() =>
    HttpResponse.json(
      replay({ deltas: { current: { equivalent: 0, total: 0 }, previous: { equivalent: 0, total: 0 } } }),
    ),
  );
  expect(await screen.findByPlaceholderText("filter reviews…")).toBeInTheDocument();
  expect(screen.queryByText(/replay/)).toBeNull();
});

test("brands the header 'read-only demo' on a confirmed demo deployment", async () => {
  server.use(http.get("http://localhost/api/meta", () => HttpResponse.json({ demo_mode: true })));
  mount();
  expect(await screen.findByText(/read-only demo/i)).toBeInTheDocument();
});

test("no demo tag on a production deployment (fails to no-tag)", async () => {
  // The default /api/meta handler is production; the tag must never flash there.
  mount();
  expect(await screen.findByPlaceholderText("filter reviews…")).toBeInTheDocument();
  expect(screen.queryByText(/read-only demo/i)).toBeNull();
});

test("the hamburger toggles the mobile nav drawer open via the store", async () => {
  mount();
  const toggle = screen.getByLabelText("Open navigation");
  expect(toggle).toHaveAttribute("aria-expanded", "false");

  await userEvent.click(toggle);

  expect(toggle).toHaveAttribute("aria-expanded", "true");
  expect(useNav.getState().open).toBe(true);
});
