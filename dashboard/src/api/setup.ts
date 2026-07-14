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

/** Response of `GET /setup/status` — the state-machine status + configured/install-known flags. */
export interface SetupStatus {
  status: string;
  /** Credentials obtained (`status === "CONFIGURED"`). */
  configured: boolean;
  /** Outrider has seen the App installed (an active installation) — distinct from `configured`. */
  install_known: boolean;
}

/** Typed error for the onboarding flow so callers can distinguish it from generic failures. */
export class SetupError extends Error {}

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
  const resp = await fetch(`${baseUrl}/setup/status`);
  if (resp.status === 404) return null;
  if (!resp.ok) throw new SetupError(await _detail(resp));
  return (await resp.json()) as SetupStatus;
}

/**
 * `POST /setup` (admin) — begin onboarding for `org`. Returns the GitHub target URL + the manifest
 * to submit there. The admin token rides the `Authorization` header (same-origin) and NEVER leaves
 * in the GitHub-bound form.
 */
export async function startSetup(org: string): Promise<SetupStartResponse> {
  const resp = await fetch(`${baseUrl}/setup`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ org }),
  });
  if (resp.status === 401) clearToken(); // mirror authMiddleware: stale key → the token-gate re-prompts
  if (!resp.ok) throw new SetupError(await _detail(resp));
  return (await resp.json()) as SetupStartResponse;
}

/**
 * `POST /setup/reset` (admin) — recover an `ORPHANED` instance to `UNCONFIGURED` so onboarding can be
 * retried. Returns the new status. 409 if the instance is not in a resettable (ORPHANED) state.
 */
export async function resetSetup(): Promise<SetupStatus> {
  const resp = await fetch(`${baseUrl}/setup/reset`, { method: "POST", headers: authHeaders() });
  if (resp.status === 401) clearToken();
  if (!resp.ok) throw new SetupError(await _detail(resp));
  return (await resp.json()) as SetupStatus;
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
