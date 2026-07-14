// Client for the App-Manifest onboarding surface (DECISIONS.md#070).
//
// The `/setup*` endpoints mount ONLY in `database` credential mode, so they are deliberately NOT in
// the generated OpenAPI schema (the canonical schema is the env-mode surface, and shifting it to the
// database-mode superset would ripple through the freshness guard). This is therefore a thin,
// locally-typed raw-fetch client for that one surface — same-origin, admin-token-authed, mirroring
// the openapi-fetch `authMiddleware` (attach Bearer; drop a stale key on 401).

import { clearToken, getToken } from "../auth/token";

// Same-origin by default (co-deployed / dev proxy); matches `api/client.ts`.
const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

/** `GITHUB_ORIGIN` is the ONLY origin the manifest form may POST to (see `submitManifestToGitHub`). */
const GITHUB_ORIGIN = "https://github.com";

/** Response of `POST /setup` — the GitHub target + the pre-filled App manifest to submit to it. */
export interface SetupStartResponse {
  /** GitHub's "create a new App" page for the chosen owner, carrying the backend-signed `state`. */
  target_url: string;
  /** The App manifest, POSTed to `target_url` as the `manifest` form field. */
  manifest: Record<string, unknown>;
}

/**
 * The setup state machine's CLOSED vocabulary — mirrors `SETUP_STATUSES` in
 * `src/outrider/db/models/setup_state.py`, which a DB CHECK constraint enforces. `router.py`
 * annotates the field `str`, but the wire value is one of exactly these five: the UI's affordances
 * (Start / Retry / Reset / installed) are defined only over them. A status outside this set means
 * the client is talking to something it does not understand, so it fails loudly rather than render
 * a blank, actionless page — and a future backend state must be taught here before the client can
 * drive it.
 */
export const SETUP_STATUSES = [
  "UNCONFIGURED",
  "AWAITING_CALLBACK",
  "CONVERTING",
  "CONFIGURED",
  "ORPHANED",
] as const;

export type SetupStatusValue = (typeof SETUP_STATUSES)[number];

/** Response of `GET /setup/status` — the state-machine status + configured/install-known flags. */
export interface SetupStatus {
  status: SetupStatusValue;
  /** Credentials obtained (`status === "CONFIGURED"`). */
  configured: boolean;
  /** Outrider has seen the App installed (an active installation) — distinct from `configured`. */
  install_known: boolean;
}

/** Typed error for the onboarding flow so callers can distinguish it from generic failures. */
export class SetupError extends Error {}

/**
 * The peer answered, but not as the Outrider API: unparseable body or the wrong shape. In practice
 * this is a TOPOLOGY error, not a backend fault — the SPA dev server (or a proxy with no `/setup`
 * rule) answered with its own HTML shell. `/setup*` mounts only in `database` credential mode and is
 * deliberately NOT proxied by the Vite dev server, so the full onboarding flow is supported ONLY
 * when FastAPI serves the built SPA (`OUTRIDER_SERVE_SPA=1`). See `dashboard/vite.config.ts`.
 */
export class SetupProtocolError extends SetupError {}

/** The request never reached a server at all (connection refused, DNS, offline). */
export class SetupUnreachableError extends SetupError {}

/** Runtime shape guard — `as` casts are erased at compile time and cannot validate a wire payload. */
function isSetupStatus(v: unknown): v is SetupStatus {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.status === "string" &&
    (SETUP_STATUSES as readonly string[]).includes(o.status) &&
    typeof o.configured === "boolean" &&
    typeof o.install_known === "boolean"
  );
}

function isSetupStartResponse(v: unknown): v is SetupStartResponse {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.target_url === "string" &&
    typeof o.manifest === "object" &&
    o.manifest !== null &&
    !Array.isArray(o.manifest)
  );
}

/**
 * Parse a success body as JSON, or raise `SetupProtocolError`. NEVER surfaces peer-supplied content
 * — not the body, and not headers either: both are peer-controlled and unbounded, and neither is
 * signal for the operator. The `content-type` is read ONLY to pick between two fixed messages.
 */
async function _json(resp: Response, what: string): Promise<unknown> {
  try {
    return await resp.json();
  } catch {
    const isHtml = (resp.headers.get("content-type") ?? "").includes("text/html");
    throw new SetupProtocolError(
      isHtml
        ? `${what} returned an HTML page instead of JSON. The request reached a static/dev server ` +
          `rather than the Outrider API — onboarding requires FastAPI serving the built SPA.`
        : `${what} returned a body that is not valid JSON, so the peer is not the Outrider API.`,
    );
  }
}

/** Run `fetch`, converting a transport-level rejection into a typed `SetupUnreachableError`. */
async function _fetch(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch {
    throw new SetupUnreachableError(
      "couldn't reach the Outrider API (no response). Is the backend running?",
    );
  }
}

/** Build request headers with the admin bearer token attached (same-origin, per authMiddleware). */
function authHeaders(extra?: Record<string, string>): Headers {
  const headers = new Headers(extra);
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
    /* non-JSON body — fall through to the status-code message */
  }
  return `setup request failed (${resp.status})`;
}

/**
 * `GET /setup/status` — PUBLIC (no auth). Returns `null` on 404: the `/setup` router is absent, i.e.
 * this instance uses `env` credentials and App-Manifest onboarding does not apply.
 */
export async function fetchSetupStatus(): Promise<SetupStatus | null> {
  const resp = await _fetch(`${baseUrl}/setup/status`);
  // 404 is the ACCEPTED production contract: the router is absent => `env` credential mode. Kept
  // ahead of every other branch, and never conflated with the protocol errors below.
  if (resp.status === 404) return null;
  if (!resp.ok) throw new SetupError(await _detail(resp));
  const body = await _json(resp, "GET /setup/status");
  if (!isSetupStatus(body)) {
    throw new SetupProtocolError("GET /setup/status returned JSON of an unexpected shape.");
  }
  return body;
}

/**
 * `POST /setup` (admin) — begin onboarding for `org`. Returns the GitHub target URL + the manifest
 * to submit there. The admin token rides the `Authorization` header (same-origin) and NEVER leaves
 * in the GitHub-bound form.
 */
export async function startSetup(org: string): Promise<SetupStartResponse> {
  const resp = await _fetch(`${baseUrl}/setup`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ org }),
  });
  if (resp.status === 401) clearToken(); // mirror authMiddleware: stale key → the token-gate re-prompts
  if (!resp.ok) throw new SetupError(await _detail(resp));
  const body = await _json(resp, "POST /setup");
  if (!isSetupStartResponse(body)) {
    throw new SetupProtocolError("POST /setup returned JSON of an unexpected shape.");
  }
  return body;
}

/**
 * `POST /setup/reset` (admin) — recover an `ORPHANED` instance to `UNCONFIGURED` so onboarding can be
 * retried. Returns the new status. 409 if the instance is not in a resettable (ORPHANED) state.
 */
export async function resetSetup(): Promise<SetupStatus> {
  const resp = await _fetch(`${baseUrl}/setup/reset`, { method: "POST", headers: authHeaders() });
  if (resp.status === 401) clearToken();
  if (!resp.ok) throw new SetupError(await _detail(resp));
  const body = await _json(resp, "POST /setup/reset");
  if (!isSetupStatus(body)) {
    throw new SetupProtocolError("POST /setup/reset returned JSON of an unexpected shape.");
  }
  return body;
}

/**
 * Auto-submit the manifest to GitHub via a POST form. GitHub's App-Manifest flow REQUIRES the
 * manifest as a form field (it is too large + structured for a URL), so a plain link cannot do this.
 *
 * SECURITY — a form whose `action` is attacker-controlled can exfiltrate whatever it carries, so:
 *   1. REFUSE to submit unless `targetUrl` is exactly the `https://github.com` origin. A compromised
 *      or misconfigured backend therefore cannot make the browser POST the manifest + signed `state`
 *      to any other host.
 *   2. Field values are set via the DOM `.value` property — NEVER `innerHTML` — so nothing in the
 *      manifest is ever parsed as HTML (no injection from manifest content).
 * Call ONLY from an explicit user action (never on load), so it can't be driven as a silent CSRF.
 */
export function submitManifestToGitHub(
  targetUrl: string,
  manifest: Record<string, unknown>,
): void {
  let url: URL;
  try {
    url = new URL(targetUrl);
  } catch {
    throw new SetupError("setup returned a malformed target URL; refusing to submit.");
  }
  if (url.origin !== GITHUB_ORIGIN) {
    throw new SetupError(
      `refusing to submit the App manifest to a non-GitHub origin (${url.origin}).`,
    );
  }
  const form = document.createElement("form");
  form.method = "POST";
  form.action = targetUrl;
  const field = document.createElement("input");
  field.type = "hidden";
  field.name = "manifest";
  field.value = JSON.stringify(manifest); // `.value` assignment — never parsed as HTML
  form.appendChild(field);
  document.body.appendChild(form);
  form.submit();
  form.remove();
}
