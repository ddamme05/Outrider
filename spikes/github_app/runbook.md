# Live runbook — GitHub App + smee.io

**Purpose.** Walk through Q2 (installation-token minting) and Q6 (smee.io
tunnel setup) against a real GitHub App. The offline `run_all.py` proves
the primitives work; this runbook proves the end-to-end round trip works
on your account and captures the two findings that can only be observed
live: your installation_id, and any diff between the recorded
octokit-sample payloads in `fixtures/` and what GitHub actually delivers.

Budget: ~15 minutes from cold start. Credentials live in your shell env —
nothing created here gets committed.

---

## Prerequisites

- A GitHub account you own.
- A public or private test repository on that account — doesn't need to
  contain real code; an empty repo with one README suffices.
- Node.js installed locally (for `npx smee-client`). Any recent LTS works.
- The spike's Python env. From repo root:
  `.venv/bin/python --version` → `Python 3.13.13`.

---

## Step 1 — Create a smee.io channel

1. Open [smee.io](https://smee.io) in a browser.
2. Click **Start a new channel**.
3. Copy the channel URL. It looks like `https://smee.io/abc123xyz456`.
4. Leave the browser tab open. smee.io shows deliveries there too —
   useful as a backup view while debugging.

**Spike finding to capture:** confirm the channel URL persists across
browser reloads (it does, within a reasonable window). If you close the
tab and lose the URL, create a new channel and update every following
step accordingly.

---

## Step 2 — Register the GitHub App

Navigate to
[Settings → Developer settings → GitHub Apps → New GitHub App](https://github.com/settings/apps/new).

Fill in:

| Field | Value |
|---|---|
| GitHub App name | `outrider-spike-<your-handle>` (must be globally unique on GitHub) |
| Homepage URL | any valid URL, e.g. your GitHub profile |
| Webhook URL | the smee.io channel URL from step 1 |
| Webhook secret | generate a random string, e.g. `openssl rand -hex 32`. Save this. |
| **Repository permissions** | `Contents: Read-only`, `Pull requests: Read and write`. **Nothing else.** (per `docs/deployment.md` + the `github-token-scope-minimum-viable` invariant) |
| Subscribe to events | Check: `Pull request` and `Installation`. |
| Where can this GitHub App be installed? | `Only on this account` |

Click **Create GitHub App**. On the resulting page:

1. **App ID** — top of the page, numeric. Save this.
2. Scroll down, click **Generate a private key**. A `.pem` file downloads.
   Save it somewhere outside the Outrider repo. **Do not commit it.**

---

## Step 3 — Install the App on your test repo

From the App's settings page, click **Install App** (left sidebar). Select
your account. Choose **Only select repositories** → pick your test repo.
Click **Install**.

**Capture:** The redirect URL after install contains
`?installation_id=<NUMBER>` as a query parameter. That number is your
installation_id. Save it.

Also — at this point GitHub sends an `installation.created` webhook.
You'll see it in the smee.io browser tab. The receiver isn't running yet,
so it's just logged in smee's UI — that's fine; the webhook is visible
for later inspection.

---

## Step 4 — Run the smee client

In a new terminal:

```bash
npx smee-client --url <channel-url> --target http://localhost:8000/webhooks/github --port 3000
```

Leave this running. It'll print `Forwarding https://smee.io/... to http://localhost:8000/webhooks/github`
and then sit idle. When webhooks arrive they'll print here too.

**Failure modes to note in this section:**

- If smee-client loses its connection while the receiver is down, the
  webhooks delivered during the outage are **lost** — smee doesn't buffer.
  Acceptable for spike use, would not be acceptable for production.
- Channel URL stays stable across smee-client restarts as long as the
  browser tab is still holding it open.

---

## Step 5 — Run the receiver

In another terminal, from the repo root:

```bash
export OUTRIDER_SPIKE_WEBHOOK_SECRET='<the-secret-from-step-2>'
export OUTRIDER_SPIKE_CAPTURE_DIR="$(mktemp -d -t outrider-spike-payloads-XXXXXX)"
echo "Captured payloads will land in: $OUTRIDER_SPIKE_CAPTURE_DIR"
cd spikes/github_app
../../.venv/bin/python -m uvicorn receiver:app --host 127.0.0.1 --port 8000 --log-level info
```

You should see uvicorn startup logs. Leave it running.

The `OUTRIDER_SPIKE_CAPTURE_DIR` env var makes the receiver write each
accepted webhook's raw body to `<dir>/<correlation_id>.json`. Step 7 uses
those files for the real-vs-fixture diff.

Smoke test: in a third terminal:
```bash
curl http://127.0.0.1:8000/healthz
```
Expect `{"status":"ok"}`.

---

## Step 6 — Trigger a real webhook

In your test repo:

1. Create a branch, push a commit, open a pull request.
2. Watch the three surfaces in order:
   - **smee.io browser tab** — the raw webhook payload appears within a
     second or two.
   - **smee-client terminal** — `POST http://localhost:8000/webhooks/github`.
   - **receiver terminal** — a FastAPI access log line with `202`.

If everything lines up, the round trip works.

---

## Step 7 — Capture the two live findings into NOTES.md

After the webhook lands, edit `NOTES.md` with:

### Installation ID

Append to the Q2 section:

> **Live installation_id on test repo:** `<number from step 3>`.
> Minted on `YYYY-MM-DD`.

### Payload diff

Compare the real delivery against our recorded fixture. The receiver wrote
each accepted body to `$OUTRIDER_SPIKE_CAPTURE_DIR` keyed by
correlation_id (see step 5). Pick the most recent capture of the
`pull_request.opened` delivery and diff it:

```bash
# From the repo root. The capture_dir path is in step 5's terminal.
LATEST=$(ls -t "$OUTRIDER_SPIKE_CAPTURE_DIR"/*.json | head -1)
echo "Latest captured payload: $LATEST"

diff <(.venv/bin/python -m json.tool "$LATEST") \
     <(.venv/bin/python -m json.tool spikes/github_app/fixtures/sample_pull_request_opened.json) \
     | head -200
```

If multiple deliveries landed (e.g., `installation.created` + `pull_request.opened`),
the receiver's INFO log lines name each correlation_id alongside the event
so you can pick the one you want: `grep "webhook_received event='pull_request'" receiver.log`.

Most differences will be trivial (timestamps, IDs, URL hosts). Append any
**structural** differences to NOTES.md's Q4 section:

> **Real-delivery payload diff (YYYY-MM-DD):** [none / list fields that
> appeared in the real delivery but not in the octokit sample / or vice
> versa].

If a structural diff shows up (e.g., a new field the octokit sample doesn't
carry, or a different type), that's real drift that `api/webhooks/schemas.py`
will need to account for.

---

## Step 8 — Tear down

The App, key, and smee.io channel can all persist indefinitely if you want
to repeat the runbook. If you're cleaning up:

1. From the App settings page: **Delete GitHub App** at the bottom.
2. Close the smee.io tab and/or navigate to a new channel (channels expire
   from idleness; no explicit delete needed).
3. Delete the downloaded `.pem` file. Do not let it linger.

Do **not** move the downloaded `.pem` into `spikes/github_app/fixtures/` —
even in a gitignored state, that's the wrong home for a real key.
`fixtures/.gitignore` blocks `*.pem` as belt-and-suspenders against exactly
this mistake. The Q1 JWT demo generates its own RSA key in memory per run
(see `demos/demo_q1_jwt_app_auth.py`) — there's no on-disk test key to
confuse with a real one. Real App keys belong in your secrets manager.

---

## What this runbook does NOT cover

- Actually posting a review comment back on the PR — that's outbound
  publishing, a different concern from inbound webhook ingestion.
- Idempotency across duplicate deliveries — GitHub will retry failed
  webhooks, and the receiver does not dedupe. Acceptable for spike use.
- Installation-token rotation behavior — `githubkit` handles this
  transparently via `AppInstallationAuthStrategy`; observing it requires
  a longer session than the spike intends.
- Multi-account / multi-repo installations — spike is one account, one
  repo. Multi-install behavior is Month 1+ work.
