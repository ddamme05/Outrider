// Slack OAuth start (DECISIONS.md#051 / #052). Request JSON from /slack/install
// instead of following its browser redirect. Raw fetch preserves the typed
// recovery errors used by ConnectSlack; api/client.ts supplies the normal
// schema-based query client. Attach the admin token, clear it on 401, and
// validate the response before handing navigation to the Slack origin guard.

import { clearToken, getToken } from "../auth/token";

// Same-origin by default (co-deployed / FastAPI-serves-SPA); matches `api/client.ts`.
const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

/** The ONLY origin `navigateToSlack` will send the browser to. */
const SLACK_ORIGIN = "https://slack.com";

/** Typed errors so the page can distinguish the flow's failures from generic ones. */
export class SlackSetupError extends Error {}

/** Invalid JSON or an unexpected response shape, often from a dev/static
 * server. Slack OAuth is absent in demo mode and not proxied by Vite. */
export class SlackSetupProtocolError extends SlackSetupError {}

/** Fetch rejected without a usable response (offline, DNS, connection failure). */
export class SlackSetupUnreachableError extends SlackSetupError {}

/** Slack OAuth is not configured on this instance (OUTRIDER_SLACK_CLIENT_ID unset → 503). */
export class SlackNotConfiguredError extends SlackSetupError {}

/** Response of `GET /slack/install` (Accept: application/json) — the Slack authorize URL. */
export interface SlackInstallURL {
  authorize_url: string;
}

function isSlackInstallURL(value: unknown): value is SlackInstallURL {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Record<string, unknown>).authorize_url === "string"
  );
}

function authHeaders(): Headers {
  const headers = new Headers({ Accept: "application/json" });
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

async function _detail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
    }
  } catch {
    /* non-JSON body — fall through to a status-code message */
  }
  // Keep a bodyless failure neutral so it cannot imply a configuration problem.
  return `Install request failed (${response.status}).`;
}

/** Fetch an authorize URL for this installation/channel. A 401 clears the key;
 * 503 details distinguish missing Slack config, incomplete GitHub setup (#070),
 * and an unavailable backend. Other HTTP, protocol, and transport failures
 * keep their separate typed errors. */
export async function startSlackInstall(
  installationId: number,
  channelId: string,
): Promise<SlackInstallURL> {
  const query = new URLSearchParams({
    installation_id: String(installationId),
    channel_id: channelId,
  });
  let response: Response;
  try {
    response = await fetch(`${baseUrl}/slack/install?${query.toString()}`, {
      headers: authHeaders(),
      // The response carries a short-lived signed state; never let a cache serve one at all.
      cache: "no-store",
    });
  } catch {
    throw new SlackSetupUnreachableError(
      "couldn't reach the Outrider API (no response). Is the backend running?",
    );
  }
  if (response.status === 401) {
    // Clear the key and still propagate the rejected request.
    clearToken();
    throw new SlackSetupError("Admin key was rejected. Enter it again.");
  }
  if (response.status === 503) {
    // Known backend details select recovery guidance. Unknown/bodyless 503s
    // stay unavailable; the #070 credential gate runs before Slack settings.
    const detail = await _detail(response);
    if (/not configured/i.test(detail)) {
      throw new SlackNotConfiguredError(
        "Slack isn’t configured on this instance. Set the OUTRIDER_SLACK_* variables and restart.",
      );
    }
    if (/setup incomplete/i.test(detail)) {
      throw new SlackSetupError(
        "This instance isn’t finished setting up. Complete GitHub App setup first, then connect Slack.",
      );
    }
    throw new SlackSetupError(
      "Outrider is temporarily unavailable (the backend may be starting up or down). Try again in a moment.",
    );
  }
  if (!response.ok) {
    throw new SlackSetupError(await _detail(response));
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    const isHtml = (response.headers.get("content-type") ?? "").includes("text/html");
    throw new SlackSetupProtocolError(
      isHtml
        ? "GET /slack/install returned an HTML page instead of JSON. The request reached a " +
          "static/dev server rather than the Outrider API — the Slack flow needs FastAPI serving " +
          "the built dashboard (it is not available on the Vite dev server or the demo box)."
        : "GET /slack/install returned a body that is not valid JSON, so the peer is not the " +
          "Outrider API.",
    );
  }
  if (!isSlackInstallURL(body)) {
    throw new SlackSetupProtocolError("GET /slack/install returned JSON of an unexpected shape.");
  }
  return body;
}

/** Hand the browser to the exact https://slack.com origin. Reject malformed
 * or foreign URLs before navigation so ConnectSlack can show the error. */
export function navigateToSlack(authorizeUrl: string): void {
  let authorizeURL: URL;
  try {
    authorizeURL = new URL(authorizeUrl);
  } catch {
    throw new SlackSetupError("Slack returned an unparseable authorize URL.");
  }
  if (authorizeURL.origin !== SLACK_ORIGIN) {
    throw new SlackSetupError(`refusing to navigate to a non-slack.com origin (${authorizeURL.origin}).`);
  }
  window.location.assign(authorizeUrl);
}
