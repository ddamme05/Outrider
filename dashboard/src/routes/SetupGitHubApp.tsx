import { useCallback, useEffect, useState, type FormEvent } from "react";

import {
  SetupError,
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
// The UI is state-machine-aware (#070 recovery states): only UNCONFIGURED shows the Start form;
// ORPHANED offers Reset; in-flight states offer Refresh (a fresh Start would 409); and CONFIGURED
// distinguishes credentials-obtained from App-installed via the install-known flag.

type Phase = "loading" | "unavailable" | "error";
type StatusState = SetupStatus | Phase;

export function SetupGitHubApp() {
  const [state, setState] = useState<StatusState>("loading");
  const [org, setOrg] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    try {
      const s = await fetchSetupStatus();
      setState(s ?? "unavailable");
    } catch {
      setState("error");
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
      setBusy(false);
    }
  }

  async function onReset(): Promise<void> {
    setError(null);
    setBusy(true);
    try {
      await resetSetup();
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
  // The Start form: fresh instance (UNCONFIGURED) or a transient status-read error (let them retry).
  const showStart = status === "UNCONFIGURED" || state === "error";

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
        {state === "error" && (
          <p className="error">Couldn&rsquo;t read setup status; you can still try below.</p>
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

        {/* ORPHANED: a failed attempt. A fresh Start would 409 until the state is reset. */}
        {status === "ORPHANED" && (
          <>
            <p>The last onboarding attempt failed. Reset to start over.</p>
            <button className="btn" type="button" onClick={() => void onReset()} disabled={busy}>
              {busy ? "Resetting…" : "Reset and start over"}
            </button>
          </>
        )}

        {/* In-flight: a fresh Start would 409; offer a refresh once the attempt completes/expires. */}
        {(status === "AWAITING_CALLBACK" || status === "CONVERTING") && (
          <>
            <p>
              An onboarding attempt is in progress. Finish creating the App on GitHub, or wait for it
              to time out, then refresh.
            </p>
            <button className="btn" type="button" onClick={() => void refresh()} disabled={busy}>
              Refresh status
            </button>
          </>
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
              {busy ? "Opening GitHub…" : "Set up GitHub App"}
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
