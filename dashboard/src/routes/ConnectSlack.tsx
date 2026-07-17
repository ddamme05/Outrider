import { useState, type FormEvent } from "react";

import {
  SlackNotConfiguredError,
  SlackSetupError,
  SlackSetupProtocolError,
  SlackSetupUnreachableError,
  navigateToSlack,
  startSlackInstall,
} from "../api/slackSetup";

// Connect-Slack onboarding (DECISIONS.md#051/#052): the dashboard front-end for the
// `GET /slack/install` OAuth start. The operator picks a GitHub installation + a Slack channel;
// we fetch the Slack authorize URL (admin-authed) and navigate the browser to Slack, which then
// redirects to the backend's `/slack/oauth/callback` (server-rendered — the SPA does not handle
// the return). Reuses the `.setup-*` chrome from the GitHub App setup page.
//
// Like `/setup`, the Slack routes mount only under FastAPI-serves-the-built-SPA (production), so on
// the Vite dev server or the demo box this page surfaces a topology error rather than a control
// that cannot work.

// A GitHub installation id is a positive integer; a Slack channel id is `C…`/`G…` (public/private).
// This friendly pre-check must not narrow the authoritative contract: it mirrors the backend's
// `_CHANNEL_ID_RE` (`[CG][A-Z0-9]{5,}` — six chars minimum) exactly, so no backend-valid id is
// rejected client-side. The server re-validates regardless.
function channelLooksValid(channel: string): boolean {
  return /^[CG][A-Z0-9]{5,}$/.test(channel.trim());
}

export function ConnectSlack() {
  const [installationId, setInstallationId] = useState("");
  const [channelId, setChannelId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topology, setTopology] = useState<string | null>(null);

  async function onConnect(e: FormEvent): Promise<void> {
    e.preventDefault();
    setError(null);
    setTopology(null);
    const idNum = Number(installationId.trim());
    if (!Number.isInteger(idNum) || idNum <= 0) {
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
      const { authorize_url } = await startSlackInstall(idNum, channelId.trim());
      // Navigates the browser to Slack (origin-guarded) and this component unmounts; `busy` is
      // only reset on the error paths below.
      navigateToSlack(authorize_url);
    } catch (err) {
      if (err instanceof SlackSetupProtocolError) {
        setTopology(err.message);
      } else if (
        err instanceof SlackNotConfiguredError ||
        err instanceof SlackSetupUnreachableError ||
        err instanceof SlackSetupError
      ) {
        setError(err.message);
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

        {/* Once we KNOW the peer isn't the Outrider API, fail closed: hide the form (connecting
            again would fail identically) and explain — mirrors the GitHub setup page, which drops
            Start on a topology fault. */}
        {topology ? (
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
                onChange={(e) => setInstallationId(e.target.value)}
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
                onChange={(e) => setChannelId(e.target.value)}
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
