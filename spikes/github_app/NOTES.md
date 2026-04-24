# GitHub App + smee.io spike — notes

**Scope.** Per `DECISIONS.md#006-two-month-0-spikes-not-five` and `#007-smee-for-month-0-webhook-tunnel`:
register a GitHub App, sign a JWT, mint an installation token, receive a
webhook payload via smee.io, verify the signature, log the payload shape.
Throwaway code — no production paths depend on this directory.

**Versions (pinned in `requirements.txt`).**
- Python 3.13.13 (from `../../.python-version`)
- `fastapi==0.135.3`
- `httpx==0.28.1`
- `pyjwt[crypto]==2.12.1`
- `githubkit==0.15.3`

**Method.** Docs-first pass against `aegis-docs::githubkit/pr-review-bot.md`,
`.../usage/getting-started/authentication.md`, `.../usage/webhooks.md`, and
`.../usage/rest-api.md`. Offline-verifiable demos for the cryptographic
primitives, payload shapes, and the FastAPI receiver contract. Two
questions (Q2 installation-token minting, Q6 smee.io tunnel setup) require
live GitHub access and are documented in `runbook.md` rather than demo'd —
see "Live runbook split" below.

**Status.** `.venv/bin/python run_all.py` → **5/5 demos pass**.

---

## Live runbook split

The spike is deliberately bifurcated. Cryptography + parsing + route
behavior are verified mechanically and pass on any machine. The live
round-trip requires a registered App and credentials that aren't in the
repo. `runbook.md` is the manual procedure that produces the two live
artifacts: your installation_id (a number) and a diff between the recorded
octokit fixtures and what a real webhook actually delivers.

Without the split, every rerun of the spike would require App registration
and the offline primitives would be unverifiable historically.

---

## Q1 — GitHub App JWT authentication

**Q1a (primitive).** `pyjwt[crypto]==2.12.1` signs and verifies an RS256 JWT
round-trip against a 2048-bit RSA key. Canonical claims per [GitHub App JWT docs](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-json-web-token-jwt-for-a-github-app):
- `iat`: issued-at, unix seconds. Small backdate (10s) is safe and GitHub
  documents a 60s tolerance for clock skew.
- `exp`: issued-at + up to **10 minutes** (hard cap on GitHub's side).
- `iss`: the numeric App ID **or** the client ID string.

Tamper detection works (changing the last 4 chars of a signed JWT raises
`jwt.InvalidSignatureError` on decode).

**Q1b (canonical).** `githubkit.AppAuthStrategy(app_id, private_key)` +
`GitHub(strategy)` constructs without error. `github.with_auth(github.auth.as_installation(id))`
returns a usable installation-scoped client, purely by construction — no
live call made in the demo.

**Consequence for `src/outrider/github/app_auth.py`.** Use
`AppAuthStrategy` as the production shape; `pyjwt` stays available as the
primitive if we ever need to debug SDK-level behavior. Don't hand-roll
JWT claims in production code unless `githubkit` stops being viable.

**Demo.** `demos/demo_q1_jwt_app_auth.py`.

---

## Q2 — installation access token minting

**Deferred to live runbook.** `githubkit` handles installation-token minting
automatically when you call any API through a client authenticated with
`AppInstallationAuthStrategy` or `GitHub.with_auth(github.auth.as_installation(id))`.
There's no documented method like `async_create_installation_access_token`
that an offline demo can verify against real credentials.

**Docs-confirmed call shape** (from `aegis-docs::githubkit/pr-review-bot.md`):

```python
app_github = GitHub(AppAuthStrategy(app_id, private_key))
installation_github = app_github.with_auth(
    app_github.auth.as_installation(installation_id)
)
# Any rest call on installation_github auto-mints and caches the token.
```

**Escape hatch** (per `aegis-docs::githubkit/pr-review-bot.md`):

```python
resp = await github.arequest(
    "POST",
    f"https://api.github.com/app/installations/{installation_id}/access_tokens",
    headers={"Accept": "application/vnd.github+json"},
)
```

Q7's demo confirms `github.arequest` exists and is callable. The
`runbook.md` step 5 produces the actual installation_id.

---

## Q3 — webhook signature verification

**Finding.** The invariant `webhook-signature-constant-time-compare` names
`hmac.compare_digest` as the primitive. Two verification paths agree:

1. Primitive: `hmac.new(secret, body, sha256).hexdigest()` → prefix with
   `"sha256="` → `hmac.compare_digest(expected, incoming)`.
2. Canonical: `githubkit.webhooks.verify(secret, body, signature)`.

Both agree on:
- Positive: a signature computed over the same body under the same secret.
- Negative: a same-length-but-wrong signature. (`==` would short-circuit
  on the first byte; `compare_digest` examines all of them — that's the
  invariant's whole point.)
- Negative: a malformed signature string (not hex).
- Negative: a signature computed under a different secret.

**Consequence for `src/outrider/api/webhooks/signature.py`.** Use
`githubkit.webhooks.verify`. It wraps the exact primitive the invariant
requires, and the test that proves this is `demo_q3_webhook_signature.py`.
If a future `githubkit` release changes `verify`'s implementation,
Q3 fails first with a clear error.

**Demo.** `demos/demo_q3_webhook_signature.py`.

---

## Q4 — webhook payload shapes

**Finding.** `githubkit.webhooks.parse(name, body)` validates against the
pinned `2026-03-10` schema. All three events we care about parse cleanly
from real octokit sample payloads:

- **`pull_request.opened`** → `WebhookPullRequestOpened`.
  `PRContext`-relevant fields all present and typed: `pull_request.number`,
  `pull_request.head.sha`, `pull_request.base.sha`, `pull_request.diff_url`,
  `pull_request.patch_url`, `repository.id`, `repository.full_name`,
  `installation.id`.
- **`pull_request.synchronize`** → `WebhookPullRequestSynchronize`.
  Carries `before` and `after` SHAs at the event top level. **Important:**
  `event.pull_request.head.sha == event.after` always — `pr.head.sha` is
  authoritative for PRContext; don't also read `event.after` as a second
  source of truth.
- **`installation.created`** → `WebhookInstallationCreated`. Delivers
  `installation.id`, `installation.app_slug`, `installation.account`, and
  a non-empty `repositories` list.

**Fixture provenance.** Payloads come from
[`octokit/webhooks` `main/payload-examples/api.github.com/`](https://github.com/octokit/webhooks).
One patch applied: added `installation.app_slug` to the `installation.created`
sample because the octokit version is stale (missing a field githubkit's
`2026-03-10` schema requires). Real deliveries will have this field.

**Demo.** `demos/demo_q4_payload_shapes.py`.

---

## Q5 — FastAPI receiver route

**Finding.** `receiver.py` demonstrates the whole contract from spec §6.3:
signature verification → event parsing → ACK 202 with a shape-summary
dict. Drive it via `httpx.AsyncClient` + `httpx.ASGITransport` against the
app directly (no uvicorn; no network sockets).

Four cases all behave correctly:
- Good signature → 202 with populated summary.
- Wrong signature (equal-length) → 401.
- No `X-Hub-Signature-256` header → 401.
- Unknown event name → 400 (parse error → client error, not server error).

**Cold-start parser import finding.** First call to
`githubkit.webhooks.parse(...)` within a process takes ~1 second
because it lazy-loads the tagged-union Pydantic models for all 80+ webhook
event types. Every subsequent call is ~11ms.

In a long-running FastAPI worker this is paid once at first webhook and
amortized forever; not a production concern by itself. But it means
**the first webhook after a cold deploy will look slow** unless we warm
the parser at startup. Actionable for `src/outrider/api/webhooks/router.py`:
add a FastAPI `startup` hook that calls `parse("ping", <minimal body>)`
once before the worker starts accepting traffic.

The Q5 demo warms the parser explicitly before timing the contract case
and asserts the warm path is under 100ms (observed ~12ms).

**Demo.** `demos/demo_q5_receiver_route.py`. **Route implementation.** `receiver.py`.

---

## Q6 — smee.io tunnel setup

**Live runbook only.** Per `DECISIONS.md#007`, smee.io is the Month 0
choice. Setup is `npx smee-client --url <channel> --target http://localhost:8000/webhooks/github`.
Findings from the live session (channel URL behavior, reconnection,
dropped-event semantics) go into `runbook.md` and into this section as
addenda after the runbook is walked.

---

## Q7 — githubkit surface audit

**Finding.** Every import path the V1 `src/outrider/github/` wrapper will use
resolves on pinned `0.15.3`:

| API | Call site |
|---|---|
| `githubkit.GitHub` | client constructor |
| `githubkit.AppAuthStrategy(app_id, private_key)` | App-level auth |
| `githubkit.AppInstallationAuthStrategy(app_id, private_key, installation_id)` | Installation-scoped client, direct |
| `GitHub.with_auth(github.auth.as_installation(id))` | Installation-scoped client, derived |
| `githubkit.webhooks.verify(secret, body, signature)` | Q3 path |
| `githubkit.webhooks.sign(secret, body, method='sha256')` | Q3 test-only, not production |
| `githubkit.webhooks.parse(name, body)` | Q4/Q5 path |
| `github.arequest(method, url, ...)` / `github.request(...)` | Raw escape hatch (`aegis-docs::githubkit/pr-review-bot.md`) |

Signatures for `verify` / `sign` / `parse` are asserted explicitly — if
any shift in a future upgrade, Q7 fails first with a clear error instead
of the receiver failing cryptically.

**Demo.** `demos/demo_q7_githubkit_surface.py`.

---

## Gotchas discovered during the spike

### The octokit sample for `installation.created` is stale

Missing `installation.app_slug`. Added it locally to make the fixture parse
against githubkit's `2026-03-10` schema. Real deliveries will include the
field; the runbook step 6 should diff the real payload against our patched
fixture to confirm.

### Cold-start parser import is ~1s

Covered at Q5 above. Startup-hook mitigation belongs in
`api/webhooks/router.py` when it's written — not a spike problem to solve.

### `pull_request.head.sha` vs `event.after` on `synchronize`

Both exist. They're always equal on a correct delivery. Pick one and stick
with it in `api/webhooks/schemas.py` → `PRContext` — the recommended
default below uses `pull_request.head.sha` because it's the same field
regardless of event action.

### Vendor-SDK import is in-spike by design

The spike formally violates `vendor-sdks-only-in-wrappers` (demos import
`githubkit` directly). That's the whole point of the spike — we're
auditing the SDK surface. Production code will consolidate inside
`src/outrider/github/`; ruff's `spikes/**/*.py` exceptions allow the
spike-level deviation.

---

## Spike findings — recommended defaults for V1 code

These are informed starting points, not locked-in decisions. The real
build can deviate with a `DECISIONS.md` entry if experience shows a better
choice.

| Question | Recommended default |
|---|---|
| App JWT shape | `pyjwt.encode({"iat": now-10, "exp": now+9*60, "iss": app_id}, private_key, "RS256")` — or just let `AppAuthStrategy` sign internally. |
| Installation-scoped client | `github.with_auth(github.auth.as_installation(installation_id))`. Tokens auto-minted and cached. |
| Webhook signature verification | `githubkit.webhooks.verify(secret, raw_body, header)`. Wraps `hmac.compare_digest`. |
| Webhook parsing | `githubkit.webhooks.parse(event_name, raw_body)`. One call per request. |
| Raw request fallback | `await github.arequest(method, url, ...)`. Never fabricate a `async_create_<thing>` method name unchecked against local docs. |
| FastAPI ACK shape | Return early on signature/event failures (401/400); 202 on success with a correlation_id echoing `X-GitHub-Delivery`. |
| Parser warmup | Add a startup hook that calls `parse("pull_request", <small body>)` once — pay the 1s lazy-load at boot, not on the first real webhook. |
| Raw body for signature | Always `await request.body()` before JSON decoding. `githubkit.webhooks.verify` must see the exact bytes GitHub signed. |
| Secret loading | Fail fast if missing (spike does this). `pydantic-settings` in V1 production. |

---

## What the spike did NOT cover — deferred to the real build

- **Idempotency (spec §6.5).** `UNIQUE(repo_id, pr_number, head_sha)` + fast-path `SELECT` + `IntegrityError` handling — production `db/` and `api/webhooks/idempotency.py`.
- **Dispatcher seam.** The spike ACKs 202 but does not dispatch. `ReviewDispatcher` → `BackgroundTasksDispatcher` (V1) → `CeleryDispatcher` (V2). Month 2.
- **SDK wrapper shape.** `src/outrider/github/` consolidates vendor-SDK imports per the invariant. The spike demonstrates operations through the SDK directly; production code consolidates.
- **Non-spike webhook events.** `ping`, `installation_repositories`, `repository`, etc. Ignored here — add Pydantic models in `api/webhooks/schemas.py` as needed.
- **Rate limiting + retry (spec §6.9).** Not observed in a spike; `githubkit` exposes rate-limit headers but wiring the reaction is production.
- **PR size limits (spec §6.10).** Diff-size preflight belongs in `api/webhooks/router.py`, not here.
- **Production secrets handling.** Throwaway RSA key; gitignored `.pem` under `fixtures/`. Production reads from `pydantic-settings`.
- **Installation event side effects.** Spec §6.2 says we persist the installation_id on `installation.created`. The spike parses and logs it; persistence is `db/` work.
- **Parse-without-name fallback.** `parse_without_name` exists but docs recommend against it (slower, may return wrong type). Production should always read `X-GitHub-Event`.
- **Webhook delivery retries.** GitHub retries failed deliveries; idempotency + the dispatcher handle this. Spike doesn't simulate.

---

## Reproducing

```bash
cd spikes/github_app
/home/spinbot/projects/outrider/.venv/bin/python run_all.py
```

Exits 0 iff every claim above reproduces on the pinned versions.

For the live round-trip — registered App + smee.io + real webhook delivery
— follow `runbook.md` with your own credentials.
