import { useCallback, useEffect, useState, type FormEvent } from "react";

import {
  SetupError,
  SetupProtocolError,
  fetchSetupStatus,
  resetSetup,
  startSetup,
  submitManifestToGitHub,
  type SetupStatus,
} from "../api/setup";

// F5 (DECISIONS.md#070): the click-through "Set up GitHub App" action. It reads the public
// GET /setup/status and, when the instance is not yet configured, POSTs /setup (admin-authed) and
// auto-submits the returned App manifest to GitHub. The dangerous step (a cross-origin form POST
// carrying the manifest + signed state) is hardened in `api/setup.ts::submitManifestToGitHub`: it
// refuses any target that is not exactly the https://github.com origin and sets field values via the
// DOM `.value` (never innerHTML). The whole page sits behind the dashboard's admin TokenGate.
//
// The UI is state-machine-aware (#070 recovery states):
//   - UNCONFIGURED → Start.
//   - AWAITING_CALLBACK / CONVERTING → RETRY, which re-POSTs /setup — that is the actual repair
//     path (begin_setup resets an expired AWAITING_CALLBACK and orphans a stale CONVERTING; a plain
//     status refresh would NOT, so it would be a dead end). A genuinely in-flight attempt returns a
//     409 with a clear message.
//   - ORPHANED → Reset, GATED on the operator confirming they deleted the partial App on GitHub
//     first (spec F4): GitHub creates the App before redirecting, so a failed attempt leaves a real
//     App holding root credentials; resetting without deleting it accumulates orphaned Apps.
//   - CONFIGURED → distinguish credentials-obtained from App-installed via the install-known flag.

// `topology` is distinct from `error` on purpose: it means the reply did not come from the Outrider
// API at all (an HTML shell / wrong shape), which is a deployment-topology fault, not a backend one.
// Collapsing it into `error` produced the misleading "you can still try below" — Start fails
// identically, so the page must say WHY rather than offer an action that cannot work.
type Phase = "loading" | "unavailable" | "error" | "topology";
type StatusState = SetupStatus | Phase;

/**
 * Terminate a sentence exactly once. Neither assumption holds on its own: a backend `detail` is
 * usually unpunctuated ("setup incomplete") while our own messages sometimes end in their own
 * punctuation ("...Is the backend running?"), so blindly appending "." yields "running?." and
 * blindly omitting it yields "incomplete You can still try below."
 */
function sentence(text: string): string {
  return /[.?!]$/.test(text) ? text : `${text}.`;
}

export function SetupGitHubApp() {
  const [state, setState] = useState<StatusState>("loading");
  const [org, setOrg] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [confirmDeleted, setConfirmDeleted] = useState(false);

  const refresh = useCallback(async (): Promise<void> => {
    try {
      const s = await fetchSetupStatus();
      setState(s ?? "unavailable");
      setStatusError(null);
    } catch (err) {
      // Bind the error and route on TYPE. A protocol error is never a backend fault, so it must not
      // render as one; only `SetupError` messages are surfaced (they are ours), never a raw body.
      setState(err instanceof SetupProtocolError ? "topology" : "error");
      setStatusError(err instanceof SetupError ? err.message : null);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function onStart(e: FormEvent): Promise<void> {
    e.preventDefault();
    const trimmed = org.trim();
    if (!trimmed) {
      setError("Enter the GitHub organization that will own the App.");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const { target_url, manifest } = await startSetup(trimmed);
      // On success this navigates the browser to GitHub (form POST) and the component unmounts;
      // `busy` is only reset on the error path below.
      submitManifestToGitHub(target_url, manifest);
    } catch (err) {
      setError(err instanceof SetupError ? err.message : "Setup failed — check the server logs.");
      // A rejected Start may have TRANSITIONED the machine: begin_setup commits a stale
      // CONVERTING → ORPHANED in its own transaction BEFORE the 409 (see state_machine.begin_setup).
      // Re-sync so the UI shows the real state (ORPHANED → the reset/cleanup flow) instead of the
      // pre-click one with its now-wrong "Retry will proceed" affordance.
      await refresh();
      setBusy(false);
    }
  }

  async function onReset(): Promise<void> {
    setError(null);
    setBusy(true);
    try {
      await resetSetup();
      setConfirmDeleted(false);
      await refresh();
    } catch (err) {
      setError(err instanceof SetupError ? err.message : "Reset failed.");
    } finally {
      setBusy(false);
    }
  }

  const status = typeof state === "object" ? state.status : null;
  const configured = typeof state === "object" && state.configured;
  const installed = typeof state === "object" && state.install_known;
  const inFlight = status === "AWAITING_CALLBACK" || status === "CONVERTING";
  // The Start/Retry form: a fresh instance, an in-flight one (retry re-POSTs /setup = the repair
  // path), or a transient status-read error. NOT for ORPHANED (Start 409s until reset), CONFIGURED,
  // or `topology` — there POST /setup would hit the same non-API peer, so offering Start would be a
  // convincing but nonfunctional control.
  //
  // `error` DOES keep Start, and the distinction from `topology` is deliberate: an unreachable or
  // erroring backend can recover between the status fetch and the click (it was starting up, the
  // network blipped), so retrying is a real action. A peer that answered as not-the-API cannot
  // recover by being clicked at — that needs a config change. Recoverable ⇒ offer; misconfigured ⇒ explain.
  const showStart = status === "UNCONFIGURED" || inFlight || state === "error";
  const startLabel = inFlight ? "Retry setup" : "Set up GitHub App";

  return (
    <div className="content">
      <div className="card" style={{ maxWidth: 620 }}>
        <h1 className="rd-title">Set up the GitHub App</h1>
        <p>
          Create this deployment&rsquo;s GitHub App from a pre-filled manifest — one click here, one
          confirmation on GitHub. GitHub hands the credentials back automatically; no secrets to copy
          and no restart. The App must be owned by the organization whose repositories Outrider
          reviews (personal-account onboarding is not supported in V1).
        </p>

        {state === "loading" && <p>Checking setup status&hellip;</p>}
        {state === "unavailable" && (
          <p>
            This instance uses environment credentials, so App-Manifest onboarding isn&rsquo;t
            available here.
          </p>
        )}
        {state === "topology" && (
          <p className="error">
            This page isn&rsquo;t reaching the Outrider API &mdash; something else answered. The
            onboarding flow is supported only when FastAPI serves the built dashboard (
            <code>OUTRIDER_SERVE_SPA=1</code>); the Vite dev server deliberately does not proxy{" "}
            <code>/setup</code>, so <code>npm run dev</code> cannot run it. Starting here would fail
            the same way.
          </p>
        )}
        {/* Start STAYS available here, unlike `topology` — see `showStart`. The message may end in
            its own punctuation, so it is rendered as its own sentence rather than spliced into one. */}
        {state === "error" && (
          <p className="error">
            {sentence(
              statusError
                ? `Couldn’t read setup status: ${statusError}`
                : "Couldn’t read setup status",
            )}{" "}
            You can still try below.
          </p>
        )}
        {status && (
          <p>
            Current status: <span className="badge">{status}</span>
          </p>
        )}

        {/* CONFIGURED: credentials obtained — but the operator may not have finished GitHub's
            separate install step, so distinguish configured from installed (spec §Land). */}
        {configured &&
          (installed ? (
            <p>This instance is fully set up — the App is configured and installed. &#10003;</p>
          ) : (
            <p>
              Credentials are configured, but the App isn&rsquo;t installed on any repositories yet.
              Open your organization&rsquo;s GitHub App settings and install it to start receiving
              reviews.
            </p>
          ))}

        {/* In-flight: an attempt was started but not completed. Retry re-POSTs /setup, which is the
            repair path (an expired AWAITING_CALLBACK is reset and a stale CONVERTING is orphaned);
            a genuinely in-progress attempt returns a clear 409 below. */}
        {inFlight && (
          <p>
            An onboarding attempt is already in progress. If you didn&rsquo;t finish it on GitHub,
            retry below to start over — an abandoned attempt is cleared automatically. If it&rsquo;s
            genuinely mid-flight, you&rsquo;ll be told it&rsquo;s already running.
          </p>
        )}

        {/* ORPHANED: a failed attempt. GitHub already created the App, so it must be deleted before
            resetting (spec F4) — otherwise repeated resets accumulate orphaned root-credential Apps.
            The reset is gated on an explicit deletion confirmation. */}
        {status === "ORPHANED" && (
          <div>
            <p className="error">
              The last onboarding attempt failed. GitHub had already created the App before the
              failure, so it still exists and holds credentials. Delete it first: open your
              organization&rsquo;s GitHub App settings, remove the partial App, then confirm and
              reset below.
            </p>
            <label style={{ display: "block", margin: "0.5rem 0" }}>
              <input
                type="checkbox"
                checked={confirmDeleted}
                onChange={(e) => setConfirmDeleted(e.target.checked)}
                disabled={busy}
              />{" "}
              I have deleted the orphaned App on GitHub.
            </label>
            <button
              className="btn"
              type="button"
              onClick={() => void onReset()}
              disabled={busy || !confirmDeleted}
            >
              {busy ? "Resetting…" : "Reset and start over"}
            </button>
          </div>
        )}

        {showStart && (
          <form onSubmit={onStart}>
            <label>
              Organization that will own the App
              <input
                className="field"
                value={org}
                onChange={(e) => setOrg(e.target.value)}
                placeholder="acme-inc"
                autoComplete="off"
                spellCheck={false}
                disabled={busy}
              />
            </label>
            <button className="btn" type="submit" disabled={busy}>
              {busy ? "Opening GitHub…" : startLabel}
            </button>
          </form>
        )}

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
