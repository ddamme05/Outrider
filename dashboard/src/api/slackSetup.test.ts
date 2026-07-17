import { http, HttpResponse } from "msw";
import { afterEach, expect, test, vi } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import {
  SlackNotConfiguredError,
  SlackSetupError,
  SlackSetupProtocolError,
  SlackSetupUnreachableError,
  navigateToSlack,
  startSlackInstall,
} from "./slackSetup";

// API-boundary tests: the client maps the `/slack/install` HTTP contract onto typed errors, and
// `navigateToSlack` refuses any origin but slack.com. The component test proves the operator sees
// the right copy; this proves the client rejects the wrong payload / destination first.

const INSTALL = "http://localhost/slack/install";

// jsdom's window.location.assign is non-configurable (vi.spyOn can't wrap it), so swap the whole
// location object with a stub carrying a mock assign. Restored after each test.
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

afterEach(() => {
  vi.restoreAllMocks();
  useTokenStore.setState({ token: null });
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: realLocation,
  });
});

test("a well-formed authorize URL is returned and carries the admin bearer + Accept json", async () => {
  useTokenStore.setState({ token: "admin-key" });
  let seenAuth: string | null = null;
  let seenAccept: string | null = null;
  server.use(
    http.get(INSTALL, ({ request }) => {
      seenAuth = request.headers.get("Authorization");
      seenAccept = request.headers.get("Accept");
      return HttpResponse.json({ authorize_url: "https://slack.com/oauth/v2/authorize?state=s" });
    }),
  );
  await expect(startSlackInstall(42, "C0ABCDE")).resolves.toEqual({
    authorize_url: "https://slack.com/oauth/v2/authorize?state=s",
  });
  expect(seenAuth).toBe("Bearer admin-key");
  expect(seenAccept).toContain("application/json");
});

test("503 whose detail names Slack maps to SlackNotConfiguredError", async () => {
  server.use(
    http.get(INSTALL, () =>
      HttpResponse.json({ detail: "Slack OAuth is not configured" }, { status: 503 }),
    ),
  );
  await expect(startSlackInstall(42, "C0ABCDE")).rejects.toBeInstanceOf(SlackNotConfiguredError);
});

test("the #070 credential-gate 503 is NOT misreported as a Slack-config problem", async () => {
  // GET /slack/install sits behind require_credentials_configured, which 503s with
  // "setup incomplete" BEFORE Slack config is evaluated — a different fix than "set OUTRIDER_SLACK_*".
  server.use(
    http.get(INSTALL, () => HttpResponse.json({ detail: "setup incomplete" }, { status: 503 })),
  );
  const err = await startSlackInstall(42, "C0ABCDE").catch((e: unknown) => e);
  expect(err).toBeInstanceOf(SlackSetupError);
  expect(err).not.toBeInstanceOf(SlackNotConfiguredError);
  expect((err as Error).message).toMatch(/GitHub App setup first/i);
});

test("a bodyless / infra 503 gets a GENERIC unavailable message, not Slack-config or GitHub-setup", async () => {
  // A gateway/proxy 503 (backend down) has no JSON detail → the fallback must not be mislabeled as
  // either a Slack-config problem or a GitHub-setup problem. It should say "temporarily unavailable".
  server.use(http.get(INSTALL, () => new HttpResponse(null, { status: 503 })));
  const err = await startSlackInstall(42, "C0ABCDE").catch((e: unknown) => e);
  expect(err).toBeInstanceOf(SlackSetupError);
  expect(err).not.toBeInstanceOf(SlackNotConfiguredError);
  expect((err as Error).message).toMatch(/temporarily unavailable/i);
  expect((err as Error).message).not.toMatch(/GitHub App setup|OUTRIDER_SLACK/i);
});

test("401 clears the stored admin token so TokenGate re-prompts", async () => {
  useTokenStore.setState({ token: "stale-key" });
  server.use(http.get(INSTALL, () => new HttpResponse(null, { status: 401 })));
  await expect(startSlackInstall(42, "C0ABCDE")).rejects.toBeInstanceOf(SlackSetupError);
  expect(useTokenStore.getState().token).toBeNull();
});

test("400 surfaces the backend detail as a SlackSetupError", async () => {
  server.use(
    http.get(INSTALL, () => HttpResponse.json({ detail: "invalid channel id" }, { status: 400 })),
  );
  await expect(startSlackInstall(42, "nope")).rejects.toThrow(/invalid channel id/);
});

test("an HTML page (wrong topology) rejects as a protocol error", async () => {
  server.use(
    http.get(INSTALL, () =>
      new HttpResponse("<!doctype html><html></html>", {
        headers: { "content-type": "text/html" },
      }),
    ),
  );
  await expect(startSlackInstall(42, "C0ABCDE")).rejects.toBeInstanceOf(SlackSetupProtocolError);
});

test("valid JSON of the wrong shape rejects as a protocol error", async () => {
  server.use(http.get(INSTALL, () => HttpResponse.json({ unexpected: "payload" })));
  await expect(startSlackInstall(42, "C0ABCDE")).rejects.toBeInstanceOf(SlackSetupProtocolError);
});

test("a transport rejection maps to SlackSetupUnreachableError", async () => {
  server.use(http.get(INSTALL, () => HttpResponse.error()));
  await expect(startSlackInstall(42, "C0ABCDE")).rejects.toBeInstanceOf(SlackSetupUnreachableError);
});

test("navigateToSlack sends the browser to a slack.com URL", () => {
  const assign = stubAssign();
  navigateToSlack("https://slack.com/oauth/v2/authorize?state=s");
  expect(assign).toHaveBeenCalledWith("https://slack.com/oauth/v2/authorize?state=s");
});

test("SECURITY: navigateToSlack refuses a non-slack.com origin", () => {
  const assign = stubAssign();
  expect(() => navigateToSlack("https://evil.example.com/oauth?state=s")).toThrow(SlackSetupError);
  expect(assign).not.toHaveBeenCalled();
});
