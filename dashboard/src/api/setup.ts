// GitHub App manifest onboarding (DECISIONS.md#070). These endpoints mount
// in database credential mode and are absent from the canonical env-mode
// OpenAPI schema. This raw-fetch client validates their responses locally
// and applies the same bearer-token/401 policy as api/client.ts.

import { clearToken, getToken } from "../auth/token";

// Same-origin by default; matches api/client.ts.
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

/** Known setup states mirror the DB CHECK vocabulary in src/outrider/db/models/setup_state.py.
 * Reject unknown states before the page chooses Start, Retry, or Reset controls. */
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

/** Invalid JSON or an unexpected response shape. Often a dev/static server
 * answered instead of the API: /setup is not proxied by Vite. The supported
 * onboarding flow uses FastAPI serving the built SPA (OUTRIDER_SERVE_SPA=1). */
export class SetupProtocolError extends SetupError {}

/** Fetch rejected without a usable response (offline, DNS, connection failure). */
export class SetupUnreachableError extends SetupError {}

/** Validate the wire payload; TypeScript casts cannot check it at runtime. */
function isSetupStatus(value: unknown): value is SetupStatus {
  if (typeof value !== "object" || value === null) return false;
  const payload = value as Record<string, unknown>;
  return (
    typeof payload.status === "string" &&
    (SETUP_STATUSES as readonly string[]).includes(payload.status) &&
    typeof payload.configured === "boolean" &&
    typeof payload.install_known === "boolean"
  );
}

function isSetupStartResponse(value: unknown): value is SetupStartResponse {
  if (typeof value !== "object" || value === null) return false;
  const payload = value as Record<string, unknown>;
  return (
    typeof payload.target_url === "string" &&
    typeof payload.manifest === "object" &&
    payload.manifest !== null &&
    !Array.isArray(payload.manifest)
  );
}

/** Parse JSON without exposing an invalid body or peer-controlled headers.
 * Content type only selects between fixed protocol-error messages. */
async function _json(response: Response, requestName: string): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    const isHtml = (response.headers.get("content-type") ?? "").includes("text/html");
    throw new SetupProtocolError(
      isHtml
        ? `${requestName} returned an HTML page instead of JSON. The request reached a static/dev server ` +
          `rather than the Outrider API — onboarding requires FastAPI serving the built SPA.`
        : `${requestName} returned a body that is not valid JSON, so the peer is not the Outrider API.`,
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

async function _detail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
    }
  } catch {
    /* non-JSON body — fall through to the status-code message */
  }
  return `setup request failed (${response.status})`;
}

/** Public status read. Treat 404 as an absent setup router (env credentials). */
export async function fetchSetupStatus(): Promise<SetupStatus | null> {
  const response = await _fetch(`${baseUrl}/setup/status`);
  // Handle the accepted 404 contract before HTTP or payload validation.
  if (response.status === 404) return null;
  if (!response.ok) throw new SetupError(await _detail(response));
  const body = await _json(response, "GET /setup/status");
  if (!isSetupStatus(body)) {
    throw new SetupProtocolError("GET /setup/status returned JSON of an unexpected shape.");
  }
  return body;
}

/** Begin admin-authenticated setup. The returned form sends only the manifest
 * to GitHub; the admin key stays in the API request header. */
export async function startSetup(org: string): Promise<SetupStartResponse> {
  const response = await _fetch(`${baseUrl}/setup`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ org }),
  });
  if (response.status === 401) clearToken(); // Clear a rejected key so TokenGate reopens.
  if (!response.ok) throw new SetupError(await _detail(response));
  const body = await _json(response, "POST /setup");
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
  const response = await _fetch(`${baseUrl}/setup/reset`, { method: "POST", headers: authHeaders() });
  if (response.status === 401) clearToken();
  if (!response.ok) throw new SetupError(await _detail(response));
  const body = await _json(response, "POST /setup/reset");
  if (!isSetupStatus(body)) {
    throw new SetupProtocolError("POST /setup/reset returned JSON of an unexpected shape.");
  }
  return body;
}

/** Submit from an explicit user action: GitHub requires a POST form carrying
 * the manifest. Require the exact https://github.com origin before creating
 * the form: its action also carries signed state. Assign DOM values so manifest
 * text cannot become HTML. */
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
  field.value = JSON.stringify(manifest); // DOM value assignment keeps manifest text out of HTML parsing.
  form.appendChild(field);
  document.body.appendChild(form);
  form.submit();
  form.remove();
}
