import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { ConnectSlack } from "./ConnectSlack";

const INSTALL = "http://localhost/slack/install";
const SLACK_URL = "https://slack.com/oauth/v2/authorize?state=signed-state";

const realLocation = window.location;
function stubAssign(): ReturnType<typeof vi.fn> {
  const assign = vi.fn();
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: { ...realLocation, assign },
  });
  return assign;
}

beforeEach(() => useTokenStore.setState({ token: "admin-key" }));
afterEach(() => {
  vi.restoreAllMocks();
  useTokenStore.setState({ token: null });
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: realLocation,
  });
});

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <ConnectSlack />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

async function fillAndConnect(id = "12345678", channel = "C0123456789"): Promise<void> {
  await userEvent.type(screen.getByLabelText(/github installation id/i), id);
  await userEvent.type(screen.getByLabelText(/slack channel id/i), channel);
  await userEvent.click(screen.getByRole("button", { name: /^connect slack$/i }));
}

test("valid inputs → fetches the authorize URL and navigates the browser to Slack", async () => {
  const assign = stubAssign();
  let seenQuery = "";
  server.use(
    http.get(INSTALL, ({ request }) => {
      seenQuery = new URL(request.url).search;
      return HttpResponse.json({ authorize_url: SLACK_URL });
    }),
  );
  mount();
  await fillAndConnect();
  expect(assign).toHaveBeenCalledWith(SLACK_URL);
  expect(seenQuery).toContain("installation_id=12345678");
  expect(seenQuery).toContain("channel_id=C0123456789");
});

test("a non-numeric installation id is rejected client-side with no request", async () => {
  const spy = vi.fn();
  server.use(http.get(INSTALL, () => (spy(), HttpResponse.json({ authorize_url: SLACK_URL }))));
  mount();
  await fillAndConnect("not-a-number", "C0123456789");
  expect(await screen.findByRole("alert")).toHaveTextContent(/numeric github installation id/i);
  expect(spy).not.toHaveBeenCalled();
});

test("a malformed channel id is rejected client-side with no request", async () => {
  const spy = vi.fn();
  server.use(http.get(INSTALL, () => (spy(), HttpResponse.json({ authorize_url: SLACK_URL }))));
  mount();
  await fillAndConnect("12345678", "nope");
  expect(await screen.findByRole("alert")).toHaveTextContent(/starts with C/i);
  expect(spy).not.toHaveBeenCalled();
});

test("a backend-valid 6-char channel (C+5) is NOT rejected client-side", async () => {
  // The client regex must mirror the backend's [CG][A-Z0-9]{5,}, not narrow it. `C12345` is the
  // shortest id the backend accepts; the client must let it through to the request.
  const assign = stubAssign();
  server.use(http.get(INSTALL, () => HttpResponse.json({ authorize_url: SLACK_URL })));
  mount();
  await fillAndConnect("12345678", "C12345");
  expect(assign).toHaveBeenCalledWith(SLACK_URL);
});

test("a Slack-not-configured 503 → explicit set-OUTRIDER_SLACK message, and no navigation", async () => {
  const assign = stubAssign();
  server.use(
    http.get(INSTALL, () =>
      HttpResponse.json({ detail: "Slack OAuth is not configured" }, { status: 503 }),
    ),
  );
  mount();
  await fillAndConnect();
  expect(await screen.findByRole("alert")).toHaveTextContent(/Slack isn.t configured/i);
  expect(assign).not.toHaveBeenCalled();
});

test("a bodyless 503 (backend down) → generic unavailable, not a Slack-config claim", async () => {
  const assign = stubAssign();
  server.use(http.get(INSTALL, () => new HttpResponse(null, { status: 503 })));
  mount();
  await fillAndConnect();
  expect(await screen.findByRole("alert")).toHaveTextContent(/temporarily unavailable/i);
  expect(screen.queryByText(/Slack isn.t configured/i)).not.toBeInTheDocument();
  expect(assign).not.toHaveBeenCalled();
});

test("an HTML response (wrong topology / demo box) shows the topology note, not Start success", async () => {
  const assign = stubAssign();
  server.use(
    http.get(INSTALL, () =>
      new HttpResponse("<!doctype html><html></html>", {
        headers: { "content-type": "text/html" },
      }),
    ),
  );
  mount();
  await fillAndConnect();
  expect(await screen.findByText(/isn.t reaching the Outrider API/i)).toBeInTheDocument();
  expect(assign).not.toHaveBeenCalled();
  // Fail closed: once we know the peer isn't the API, the form is gone (connecting again
  // would fail identically), mirroring the GitHub setup page dropping Start on topology.
  expect(screen.queryByRole("button", { name: /^connect slack$/i })).not.toBeInTheDocument();
  expect(screen.queryByLabelText(/github installation id/i)).not.toBeInTheDocument();
});

test("SECURITY: a non-slack.com authorize URL is refused and never navigated to", async () => {
  const assign = stubAssign();
  server.use(
    http.get(INSTALL, () =>
      HttpResponse.json({ authorize_url: "https://evil.example.com/oauth?state=s" }),
    ),
  );
  mount();
  await fillAndConnect();
  expect(await screen.findByRole("alert")).toHaveTextContent(/non-slack\.com origin/i);
  expect(assign).not.toHaveBeenCalled();
});
