import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { expect, test } from "vitest";

import { server } from "../test/server";
import { DemoBanner } from "./DemoBanner";

const BANNER = /Read-only demo — seeded snapshot data/;

function mount(meta: () => Response | Promise<Response>) {
  server.use(http.get("http://localhost/api/meta", meta));
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <DemoBanner />
    </QueryClientProvider>,
  );
}

test("renders on a confirmed demo deployment", async () => {
  mount(() => HttpResponse.json({ demo_mode: true }));
  expect(await screen.findByText(BANNER)).toBeInTheDocument();
});

test("stays hidden on production (fails to no-banner)", async () => {
  mount(() => HttpResponse.json({ demo_mode: false }));
  // Give the query a tick to resolve, then assert absence.
  await Promise.resolve();
  expect(screen.queryByText(BANNER)).not.toBeInTheDocument();
});

test("stays hidden while discovery is unresolved (no false demo flash)", () => {
  mount(() => new Promise<Response>(() => {}));
  expect(screen.queryByText(BANNER)).not.toBeInTheDocument();
});

test("stays hidden on a malformed 200 (demo_mode not a strict boolean)", async () => {
  // {demo_mode: null} must not be coerced either way; the banner stays hidden.
  mount(() => HttpResponse.json({ demo_mode: null }));
  await Promise.resolve();
  expect(screen.queryByText(BANNER)).not.toBeInTheDocument();
});
