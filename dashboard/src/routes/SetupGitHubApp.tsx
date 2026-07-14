import { useEffect, useState, type FormEvent } from "react";

import {
  SetupError,
  fetchSetupStatus,
  startSetup,
  submitManifestToGitHub,
  type SetupStatus,
} from "../api/setup";

// F5 (DECISIONS.md#070): the click-through "Set up GitHub App" action. It reads the public
// GET /setup/status, and — when the instance is not yet configured — POSTs /setup (admin-authed)
// and auto-submits the returned App manifest to GitHub. The dangerous step (a cross-origin form POST
// carrying the manifest + signed state) is hardened in `api/setup.ts::submitManifestToGitHub`:
// it refuses any target that is not exactly the https://github.com origin and sets field values via
// the DOM `.value` (never innerHTML). The whole page sits behind the dashboard's admin TokenGate.

type StatusState = SetupStatus | "loading" | "unavailable" | "error";

export function SetupGitHubApp() {
  const [statusState, setStatusState] = useState<StatusState>("loading");
  const [org, setOrg] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchSetupStatus()
      .then((s) => {
        if (active) setStatusState(s ?? "unavailable");
      })
      .catch(() => {
        if (active) setStatusState("error");
      });
    return () => {
      active = false;
    };
  }, []);

  const statusObj = typeof statusState === "object" ? statusState : null;
  const configured = statusObj?.configured === true;
  // Show the onboarding form unless we're still loading, onboarding is unavailable (env mode), or
  // the instance is already configured. A status-read error still offers the form (POST /setup
  // surfaces its own error) so a transient blip doesn't block onboarding.
  const showForm = !configured && statusState !== "loading" && statusState !== "unavailable";

  async function onSubmit(e: FormEvent): Promise<void> {
    e.preventDefault();
    const trimmed = org.trim();
    if (!trimmed) {
      setError("Enter the GitHub organization or account that will own the App.");
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

  return (
    <div className="content">
      <div className="card" style={{ maxWidth: 620 }}>
        <h1 className="rd-title">Set up the GitHub App</h1>
        <p>
          Create this deployment&rsquo;s GitHub App from a pre-filled manifest — one click here, one
          confirmation on GitHub. GitHub hands the credentials back automatically; no secrets to copy
          and no restart.
        </p>

        {statusState === "loading" && <p>Checking setup status&hellip;</p>}
        {statusState === "unavailable" && (
          <p>
            This instance uses environment credentials, so App-Manifest onboarding isn&rsquo;t
            available here.
          </p>
        )}
        {statusState === "error" && (
          <p className="error">Couldn&rsquo;t read setup status; you can still try below.</p>
        )}
        {statusObj && (
          <p>
            Current status: <span className="badge">{statusObj.status}</span>
          </p>
        )}
        {configured && <p>This instance is already configured. &#10003;</p>}

        {showForm && (
          <form onSubmit={onSubmit}>
            <label>
              Owner (organization or account that will own the App)
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
