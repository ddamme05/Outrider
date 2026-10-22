import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import {
  SetupError,
  SetupProtocolError,
  fetchSetupStatus,
  resetSetup,
  startSetup,
  submitManifestToGitHub,
  type SetupStatus,
} from "../api/setup";

// F5 (DECISIONS.md#070): read setup status, start or retry onboarding, then submit the
// manifest to GitHub through api/setup.ts's origin-checked form. TokenGate requires a token;
// the backend authorizes setup writes.
// Retry asks the backend to repair abandoned attempts (#070). ORPHANED requires GitHub-side
// deletion before Reset (spec F4); CONFIGURED distinguishes credentials from installation.

// A protocol failure hides Start; an ordinary request error leaves it available for retry.
type Phase = "loading" | "unavailable" | "error" | "topology";
type StatusState = SetupStatus | Phase;

function statusTone(status: string): string {
  if (status === "CONFIGURED") return "pos";
  if (status === "ORPHANED") return "neg";
  if (status === "AWAITING_CALLBACK" || status === "CONVERTING") return "warn";
  return "info";
}

/** Add a period only when the message lacks terminal punctuation. */
function sentence(text: string): string {
  return /[.?!]$/.test(text) ? text : `${text}.`;
}

// GitHub's installation webhook can arrive after the browser returns. Poll briefly
// for the display hint, with a fixed limit if the webhook never arrives.
const INSTALL_POLL_ATTEMPTS = 10;
const INSTALL_POLL_INTERVAL_MS = 1000;

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Use the untrusted redirect hint only to start status polling, then remove it from the URL.
 * The id is never stored or sent. install_known reads the backend's active installation rows.
 */
function consumeInstallHint(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("setup_action") !== "install" && !params.has("installation_id")) {
    return false;
  }
  params.delete("setup_action");
  params.delete("installation_id");
  const remainingQuery = params.toString();
  window.history.replaceState(
    null,
    "",
    window.location.pathname + (remainingQuery ? `?${remainingQuery}` : ""),
  );
  return true;
}

export function SetupGitHubApp() {
  const [setupState, setSetupState] = useState<StatusState>("loading");
  const [organizationLogin, setOrganizationLogin] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [confirmDeleted, setConfirmDeleted] = useState(false);
  const [confirmingInstall, setConfirmingInstall] = useState(false);

  const refreshSetupStatus = useCallback(async (): Promise<SetupStatus | null> => {
    try {
      const setupStatus = await fetchSetupStatus();
      setSetupState(setupStatus ?? "unavailable");
      setStatusError(null);
      return setupStatus;
    } catch (refreshError) {
      // Surface typed SetupError messages; other exceptions use the page's generic fallback.
      setSetupState(refreshError instanceof SetupProtocolError ? "topology" : "error");
      setStatusError(refreshError instanceof SetupError ? refreshError.message : null);
      return null;
    }
  }, []);

  // Scrub the URL in an effect to keep render pure. The ref retains the hint across
  // StrictMode's effect cleanup/setup, when the URL has already been cleaned.
  const installHint = useRef<boolean | null>(null);

  useEffect(() => {
    installHint.current ??= consumeInstallHint();
    if (!installHint.current) {
      void refreshSetupStatus();
      return;
    }
    let cancelled = false;
    setConfirmingInstall(true);
    void (async () => {
      for (let attempt = 0; attempt < INSTALL_POLL_ATTEMPTS; attempt += 1) {
        const setupStatus = await refreshSetupStatus();
        if (cancelled) return;
        if (setupStatus?.install_known) break;
        if (attempt < INSTALL_POLL_ATTEMPTS - 1) {
          await sleep(INSTALL_POLL_INTERVAL_MS);
          if (cancelled) return;
        }
      }
      setConfirmingInstall(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshSetupStatus]);

  async function onStart(event: FormEvent): Promise<void> {
    event.preventDefault();
    const ownerLogin = organizationLogin.trim();
    if (!ownerLogin) {
      setError("Enter the GitHub organization that will own the App.");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const { target_url, manifest } = await startSetup(ownerLogin);
      // Keep the form busy during navigation; failures below unlock it.
      submitManifestToGitHub(target_url, manifest);
    } catch (startError) {
      setError(
        startError instanceof SetupError ? startError.message : "Setup failed — check the server logs.",
      );
      // begin_setup can commit stale CONVERTING → ORPHANED before rejecting Start.
      // Refresh so the page offers the cleanup/reset flow for that committed state.
      await refreshSetupStatus();
      setBusy(false);
    }
  }

  async function onReset(): Promise<void> {
    setError(null);
    setBusy(true);
    try {
      await resetSetup();
      setConfirmDeleted(false);
      await refreshSetupStatus();
    } catch (resetError) {
      setError(resetError instanceof SetupError ? resetError.message : "Reset failed.");
    } finally {
      setBusy(false);
    }
  }

  const status = typeof setupState === "object" ? setupState.status : null;
  const configured = typeof setupState === "object" && setupState.configured;
  const installed = typeof setupState === "object" && setupState.install_known;
  const inFlight = status === "AWAITING_CALLBACK" || status === "CONVERTING";
  // Offer Start for fresh, retryable, or request-error states. Orphans need Reset;
  // protocol failures need a working API, and configured instances need no new App.
  const showStart = status === "UNCONFIGURED" || inFlight || setupState === "error";
  const startLabel = inFlight ? "Retry setup" : "Set up GitHub App";

  return (
    <div className="setup-page">
      <div className="card setup-card">
        <header className="setup-head">
          <h1 className="setup-title">Set up the GitHub App</h1>
          <p className="setup-lead">
            Create this deployment&rsquo;s GitHub App from a pre-filled manifest — one click here,
            one confirmation on GitHub. GitHub hands the credentials back automatically; no secrets
            to copy and no restart. The App must be owned by the organization whose repositories
            Outrider reviews (personal-account onboarding is not supported in V1).
          </p>
        </header>

        {status && (
          <div className="setup-status">
            <span className="setup-status__label">Current status</span>
            <span className={`setup-badge setup-badge--${statusTone(status)}`}>{status}</span>
          </div>
        )}

        {setupState === "loading" && <p className="setup-muted">Checking setup status&hellip;</p>}

        {setupState === "unavailable" && (
          <p className="setup-note">
            This instance uses environment credentials, so App-Manifest onboarding isn&rsquo;t
            available here.
          </p>
        )}

        {setupState === "topology" && (
          <p className="setup-note setup-note--error">
            This page isn&rsquo;t reaching the Outrider API &mdash; something else answered. The
            onboarding flow is supported only when FastAPI serves the built dashboard (
            <code>OUTRIDER_SERVE_SPA=1</code>); the Vite dev server deliberately does not proxy{" "}
            <code>/setup</code>, so <code>npm run dev</code> cannot run it. Starting here would fail
            the same way.
          </p>
        )}

        {/* Render the status error as its own sentence before the retry guidance. */}
        {setupState === "error" && (
          <p className="setup-note setup-note--error">
            {sentence(
              statusError
                ? `Couldn’t read setup status: ${statusError}`
                : "Couldn’t read setup status",
            )}{" "}
            You can still try below.
          </p>
        )}

        {/* A returning browser may still be waiting for the installation webhook. */}
        {confirmingInstall && !installed && (
          <p className="setup-note">
            Confirming the installation&hellip; GitHub delivers the install event separately, so this
            can take a few seconds.
          </p>
        )}

        {/* Credentials and installation are separate steps (spec §Land). */}
        {configured &&
          !confirmingInstall &&
          (installed ? (
            <p className="setup-note setup-note--ok">
              This instance is fully set up — the App is configured and installed. &#10003;
            </p>
          ) : (
            <p className="setup-note">
              Credentials are configured, but the App isn&rsquo;t installed on any repositories yet.
              Open your organization&rsquo;s GitHub App settings and install it to start receiving
              reviews.
            </p>
          ))}

        {/* Retry invokes backend repair; a live attempt can still return 409. */}
        {inFlight && (
          <p className="setup-note">
            An onboarding attempt is already in progress. If you didn&rsquo;t finish it on GitHub,
            retry below to start over — an abandoned attempt is cleared automatically. If it&rsquo;s
            genuinely mid-flight, you&rsquo;ll be told it&rsquo;s already running.
          </p>
        )}

        {/* Require explicit GitHub-side deletion confirmation before Reset (spec F4). */}
        {status === "ORPHANED" && (
          <div className="setup-block">
            <p className="setup-note setup-note--error">
              The last onboarding attempt failed. GitHub had already created the App before the
              failure, so it still exists and holds credentials. Delete it first: open your
              organization&rsquo;s GitHub App settings, remove the partial App, then confirm and
              reset below.
            </p>
            <label className="setup-check">
              <input
                type="checkbox"
                className="setup-check__box"
                checked={confirmDeleted}
                onChange={(event) => setConfirmDeleted(event.target.checked)}
                disabled={busy}
              />
              <span className="setup-check__text">I have deleted the orphaned App on GitHub.</span>
            </label>
            <button
              className="btn setup-btn setup-btn--danger"
              type="button"
              onClick={() => void onReset()}
              disabled={busy || !confirmDeleted}
            >
              {busy ? "Resetting…" : "Reset and start over"}
            </button>
          </div>
        )}

        {showStart && (
          <form className="setup-form" onSubmit={onStart}>
            <div className="setup-field">
              <label className="setup-field__label" htmlFor="setup-org">
                Organization that will own the App
              </label>
              <input
                id="setup-org"
                className="setup-field__input"
                value={organizationLogin}
                onChange={(event) => setOrganizationLogin(event.target.value)}
                placeholder="acme-inc"
                autoComplete="off"
                spellCheck={false}
                disabled={busy}
              />
              <span className="setup-field__hint">
                The GitHub organization login, exactly as it appears in your org URL.
              </span>
            </div>
            <button className="btn primary setup-btn" type="submit" disabled={busy}>
              {busy ? "Opening GitHub…" : startLabel}
            </button>
          </form>
        )}

        {error && (
          <p className="setup-note setup-note--error" role="alert">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
