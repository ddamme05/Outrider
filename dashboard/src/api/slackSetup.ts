// Client for the Slack "Connect Slack" install flow (DECISIONS.md#051 / #052).
//
// `/slack/install` IS in the generated OpenAPI schema (it mounts in production, the canonical
// `demo_mode=False` surface), but this is still a thin, locally-typed raw-fetch client rather than
// the openapi-fetch `$api`: the endpoint is content-negotiated (Accept: application/json → a 200
// JSON body, else a 302) and needs custom error mapping the generated client doesn't express well —
// the dual-503 branch (credential gate vs Slack-not-configured), 401 → clear the stale token, and
// HTML-body topology detection. Same-origin, admin-token-authed, mirroring api/setup.ts.
//
// The flow: GET /slack/install with `Accept: application/json` returns the Slack authorize URL
// (a fetch cannot follow a cross-origin 302, so we read it and navigate the browser ourselves),
// then `navigateToSlack` sends the browser there — but only after asserting the URL is on
// slack.com, the one origin we will ever hand off to (mirrors setup.ts's GITHUB_ORIGIN guard).

import { clearToken, getToken } from "../auth/token";

// Same-origin by default (co-deployed / FastAPI-serves-SPA); matches `api/client.ts`.
const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

/** The ONLY origin `navigateToSlack` will send the browser to. */
const SLACK_ORIGIN = "https://slack.com";

/** Typed errors so the page can distinguish the flow's failures from generic ones. */
export class SlackSetupError extends Error {}

/** The peer answered, but not as the Outrider API (HTML shell / wrong shape). In practice a
 *  TOPOLOGY fault: the Vite dev server or a proxy with no `/slack` rule answered, or this is the
 *  demo box (Slack OAuth is production-only). Same class as setup.ts's SetupProtocolError. */
export class SlackSetupProtocolError extends SlackSetupError {}

/** The request never reached a server (connection refused, DNS, offline). */
export class SlackSetupUnreachableError extends SlackSetupError {}

/** Slack OAuth is not configured on this instance (OUTRIDER_SLACK_CLIENT_ID unset → 503). */
export class SlackNotConfiguredError extends SlackSetupError {}

/** Response of `GET /slack/install` (Accept: application/json) — the Slack authorize URL. */
export interface SlackInstallURL {
  authorize_url: string;
}

function isSlackInstallURL(v: unknown): v is SlackInstallURL {
  return (
    typeof v === "object" &&
    v !== null &&
    typeof (v as Record<string, unknown>).authorize_url === "string"
  );
}

function authHeaders(): Headers {
  const headers = new Headers({ Accept: "application/json" });
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

async function _detail(resp: Response): Promise<string> {
  try {
    const body: unknown = await resp.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
    }
  } catch {
    /* non-JSON body — fall through to a status-code message */
  }
  // Neutral fallback: must NOT contain "Slack", or a bodyless 503 would match the 503
  // classifier's Slack branch and mislabel an infra outage as a Slack-config problem.
  return `Install request failed (${resp.status}).`;
}

/**
 * `GET /slack/install` (admin) — begin connecting Slack for `installationId` + `channelId`.
 * Returns the Slack authorize URL to navigate to. Maps the flow's HTTP contract onto typed errors:
 *   - 401 → clears the stale token + SlackSetupError (TokenGate re-prompts)
 *   - 503 whose detail names Slack → SlackNotConfiguredError; otherwise (the #070 credential gate)
 *     → SlackSetupError pointing at GitHub App setup
 *   - other non-2xx → SlackSetupError with the backend's detail
 *   - non-JSON / wrong-shape body → SlackSetupProtocolError (wrong topology / demo box)
 *   - transport rejection → SlackSetupUnreachableError
 */
export async function startSlackInstall(
  installationId: number,
  channelId: string,
): Promise<SlackInstallURL> {
  const query = new URLSearchParams({
    installation_id: String(installationId),
    channel_id: channelId,
  });
  let resp: Response;
  try {
    resp = await fetch(`${baseUrl}/slack/install?${query.toString()}`, {
      headers: authHeaders(),
      // The response carries a short-lived signed state; never let a cache serve one at all.
      cache: "no-store",
    });
  } catch {
    throw new SlackSetupUnreachableError(
      "couldn't reach the Outrider API (no response). Is the backend running?",
    );
  }
  if (resp.status === 401) {
    // Stale/rejected admin key — drop it so TokenGate re-prompts (mirrors api/setup.ts + the
    // openapi-fetch authMiddleware). The error still propagates as the query's failure.
    clearToken();
    throw new SlackSetupError("Admin key was rejected. Enter it again.");
  }
  if (resp.status === 503) {
    // THREE distinct 503s share this route, and each wants a different fix — so classify on the
    // SPECIFIC backend detail, never a loose "slack" match (the fallback detail contains "Slack"):
    //   - Slack-not-configured ("Slack OAuth is not configured") → set OUTRIDER_SLACK_*;
    //   - the #070 credential gate ("setup incomplete", returned BEFORE Slack config is evaluated)
    //     → finish GitHub App setup;
    //   - anything else, incl. a bodyless infra/gateway 503 → the backend is unavailable, retry.
    // Handing any of these the wrong recovery instruction is worse than a generic one.
    const detail = await _detail(resp);
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
  if (!resp.ok) {
    throw new SlackSetupError(await _detail(resp));
  }
  let body: unknown;
  try {
    body = await resp.json();
  } catch {
    const isHtml = (resp.headers.get("content-type") ?? "").includes("text/html");
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

/**
 * Navigate the browser to Slack's authorize page. REFUSES any URL not on `https://slack.com`
 * (defence-in-depth: the URL is backend-built, but this is the one hand-off origin we allow, the
 * same posture as setup.ts's `submitManifestToGitHub`). Returns nothing on success (the browser
 * leaves the page); throws `SlackSetupError` on a non-Slack origin so the caller can surface it.
 */
export function navigateToSlack(authorizeUrl: string): void {
  let parsed: URL;
  try {
    parsed = new URL(authorizeUrl);
  } catch {
    throw new SlackSetupError("Slack returned an unparseable authorize URL.");
  }
  if (parsed.origin !== SLACK_ORIGIN) {
    throw new SlackSetupError(`refusing to navigate to a non-slack.com origin (${parsed.origin}).`);
  }
  window.location.assign(authorizeUrl);
}
