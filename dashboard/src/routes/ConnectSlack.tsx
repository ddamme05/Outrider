import { useState, type FormEvent } from "react";

import {
  SlackNotConfiguredError,
  SlackSetupError,
  SlackSetupProtocolError,
  SlackSetupUnreachableError,
  navigateToSlack,
  startSlackInstall,
} from "../api/slackSetup";

// Connect-Slack onboarding (DECISIONS.md#051/#052): request an admin-authorized OAuth URL,
// then navigate to Slack. Slack returns to the backend's server-rendered callback.
// A non-API reply hides the form; the Vite dev server does not proxy /slack.

// Match the backend's [CG][A-Z0-9]{5,} channel check; the server validates again.
function channelLooksValid(channel: string): boolean {
  return /^[CG][A-Z0-9]{5,}$/.test(channel.trim());
}

export function ConnectSlack() {
  const [installationId, setInstallationId] = useState("");
  const [channelId, setChannelId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [protocolErrorMessage, setProtocolErrorMessage] = useState<string | null>(null);

  async function onConnect(event: FormEvent): Promise<void> {
    event.preventDefault();
    setError(null);
    setProtocolErrorMessage(null);
    const parsedInstallationId = Number(installationId.trim());
    if (!Number.isInteger(parsedInstallationId) || parsedInstallationId <= 0) {
      setError("Enter the numeric GitHub installation id (the number at the end of the App’s " +
        "installation settings URL).");
      return;
    }
    if (!channelLooksValid(channelId)) {
      setError("Enter a Slack channel id — it starts with C (or G for a private group), e.g. " +
        "C0123456789.");
      return;
    }
    setBusy(true);
    try {
      const { authorize_url } = await startSlackInstall(parsedInstallationId, channelId.trim());
      // Keep the form busy during navigation; failures below unlock it.
      navigateToSlack(authorize_url);
    } catch (connectionError) {
      if (connectionError instanceof SlackSetupProtocolError) {
        setProtocolErrorMessage(connectionError.message);
      } else if (
        connectionError instanceof SlackNotConfiguredError ||
        connectionError instanceof SlackSetupUnreachableError ||
        connectionError instanceof SlackSetupError
      ) {
        setError(connectionError.message);
      } else {
        setError("Couldn’t start the Slack connection — check the server logs.");
      }
      setBusy(false);
    }
  }

  return (
    <div className="setup-page">
      <div className="card setup-card">
        <header className="setup-head">
          <h1 className="setup-title">Connect Slack</h1>
          <p className="setup-lead">
            Send review notifications to a Slack channel. Pick the GitHub installation and the
            channel, then approve the app on Slack — Outrider stores the bot token encrypted and
            posts an approval card when a review needs a decision, plus a short note when one is
            published. Slack is notify-only: you can&rsquo;t trigger or approve reviews from it.
          </p>
        </header>

        {/* Hide the form after a protocol rejection; another attempt needs a working API. */}
        {protocolErrorMessage ? (
          <p className="setup-note setup-note--error">
            This page isn&rsquo;t reaching the Outrider API &mdash; something else answered. The
            Slack flow is supported only when FastAPI serves the built dashboard; the Vite dev
            server and the demo box don&rsquo;t run it. Connecting here would fail the same way.
          </p>
        ) : (
          <form className="setup-form" onSubmit={onConnect}>
            <div className="setup-field">
              <label className="setup-field__label" htmlFor="slack-install-id">
                GitHub installation id
              </label>
              <input
                id="slack-install-id"
                className="setup-field__input"
                value={installationId}
                onChange={(event) => setInstallationId(event.target.value)}
                placeholder="12345678"
                inputMode="numeric"
                autoComplete="off"
                spellCheck={false}
                disabled={busy}
              />
              <span className="setup-field__hint">
                The number at the end of your organization&rsquo;s GitHub App installation settings
                URL.
              </span>
            </div>

            <div className="setup-field">
              <label className="setup-field__label" htmlFor="slack-channel-id">
                Slack channel id
              </label>
              <input
                id="slack-channel-id"
                className="setup-field__input"
                value={channelId}
                onChange={(event) => setChannelId(event.target.value)}
                placeholder="C0123456789"
                autoComplete="off"
                spellCheck={false}
                disabled={busy}
              />
              <span className="setup-field__hint">
                Open the channel in Slack &rarr; its id sits at the bottom of the channel details
                pane. Invite the bot to the channel after connecting so posts land.
              </span>
            </div>

            <button className="btn primary setup-btn" type="submit" disabled={busy}>
              {busy ? "Opening Slack…" : "Connect Slack"}
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
