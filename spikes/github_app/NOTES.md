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

**Status.** `.venv/bin/python run_all.py` → **5/5 demos pass**. Live
runbook walk completed 2026-04-26 — installation_id captured, Q2 token
mint verified end-to-end against real GitHub, real-payload diff
recorded. Spike is complete. See "Live artifacts captured" below for
the runbook-walk evidence.

---

## Post-closeout review pass (2026-04-23)

Codex audit of the initial commit caught four genuine findings. Calibrated
response rather than reopening the spike:

Applied in-place:

- **Q1 no longer depends on a gitignored fixture.** The RSA key is now
  generated in-memory per run via `cryptography.hazmat.rsa.generate_private_key`.
  `fixtures/test_private_key.pem` is deleted. Offline reproduction now
  works on a fresh checkout without setup — the spike's advertised
  `python run_all.py` contract holds.
- **Receiver emits structured logs.** `receiver.py` uses
  `logging.getLogger("outrider.spike.github_app.receiver")` at INFO. The
  runbook's "receiver's structured logs" reference now has a real target.
  Log line is `key=value`-shaped so `grep webhook_received` during the
  runbook walk captures every shape summary.
- **Q5 coverage expanded.** Two additional route cases:
  (1) `installation.created` drives the installation branch of the
  receiver (previously untested — Q4 parsed the payload but Q5 never
  drove the route); (2) missing `X-GitHub-Event` header returns 400
  (receiver had the check but no test exercised it).
- **Q7 asserts the pinned `githubkit` version.** `importlib.metadata.version("githubkit") == "0.15.3"`.
  Prior version check was `hasattr(githubkit, "versions")` — a generic
  "does the SDK exist" check, not the pin assertion the audit expected.

Deferred, reasoning noted:

- **Live installation_id and payload diff remain pending a runbook walk.**
  `DECISIONS.md#006`'s contract names "the installation ID" as a spike
  deliverable. Producing it requires registering a GitHub App on a test
  repo, which needs credentials this session didn't have. `runbook.md`
  step 7 captures the ID and the real-vs-fixture payload diff into
  NOTES.md when walked. The offline surface is complete; the live surface
  is procedurally ready but not yet walked. This is an honest split, not
  a reframing of the contract — the spike is complete-modulo-runbook-walk.
- **`installation.deleted` route coverage.** Spec §6.2 requires both
  `created` and `deleted` action handling in production. The spike tests
  `created`; `deleted` has the same payload shape minus `repositories`,
  so the receiver's `installation` branch handles both without changes.
  Testing `deleted` in the spike would validate the same code path
  against a fixture that differs only in one field — Month 1's
  `api/webhooks/router.py` tests cover `deleted` against a real payload.

Why not reopen the spike for the live walk: spike's role is to retire
unknowns and hand findings to Month 1. The runbook is the handoff
artifact for the live portion; walking it is a separate session's work
because it depends on credentials.

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

### Live artifacts captured (runbook walk 2026-04-26)

`DECISIONS.md#006` names "the installation ID" as a spike deliverable.
The runbook walk on 2026-04-26 produced:

- **Live installation_id:** `127368814` on `ddamme05/outrider-spike-test`.
  First seen via `installation.created` webhook in the smee.io UI when
  the App was installed (pre-receiver, so the body is in smee's UI but
  not in the receiver's capture dir — acceptable, NOTES.md captures the
  ID itself).
- **Q2 token-mint round-trip verified:** `verify_installation_token.py`
  output:
  ```
  q2_verified installation_id=127368814 app_authenticated_as='outrider-spike-ddamme' app_id=3514536
  ```
  Confirms `AppAuthStrategy` + `with_auth(as_installation(id))` mints a
  valid installation token against real GitHub. Without `TEST_REPO` set
  the script fell back to `apps.async_get_authenticated`; with
  `TEST_REPO=ddamme05/outrider-spike-test` set it would also exercise
  `repos.async_get` (same token, different endpoint — both prove the
  installation-scoped client authenticates).
- **End-to-end webhook flow:** smee.io → smee-client → uvicorn
  receiver → 202 Accepted, two `pull_request` deliveries observed
  (`opened` and `review_requested`). Receiver returned 202 on every
  delivery; capture-dir wrote raw bodies as expected.
- **Real-vs-fixture payload diff (structural findings):** Real
  deliveries on 2026-04-26 carry `"user_view_type": "public"` on user
  objects (sender, pull_request.user, etc.); the octokit fixture does
  not. New field added by GitHub since the octokit sample was last
  refreshed. Otherwise the diff is value-level (timestamps, IDs, URLs,
  login names, body content) — no structural drift that would break
  `githubkit.webhooks.parse` or `PRContext` construction. The
  `user_view_type` field is informational and does not affect Q4's
  finding that all `PRContext`-relevant fields are present and typed.

Spike contract from `DECISIONS.md#006` is now complete-complete:
JWT signing (Q1), installation token mint (Q2 + verify script), webhook
signature verification (Q3), payload shapes (Q4 + real-payload diff),
FastAPI route (Q5), smee tunnel (Q6), githubkit surface (Q7).

---

## Q1 — GitHub App JWT authentication

**Q1a (primitive).** `pyjwt[crypto]==2.12.1` signs and verifies an RS256 JWT
round-trip against a 2048-bit RSA key. Canonical claims per [GitHub App JWT docs](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-json-web-token-jwt-for-a-github-app):
- `iat`: issued-at, unix seconds. Backdate **60 seconds** per current
  GitHub docs (the demo originally used 10s; corrected after the
  2026-04-26 audit). 60s gives the recommended clock-skew tolerance
  margin between the issuing machine and GitHub's servers.
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

**Verified live 2026-04-26.** `githubkit` handles installation-token
minting automatically when you call any API through a client
authenticated with `AppInstallationAuthStrategy` or
`GitHub.with_auth(github.auth.as_installation(id))`. There's no
documented method like `async_create_installation_access_token` —
`githubkit` mints the token transparently on the first API call from
the installation-scoped client.

**Live evidence (`verify_installation_token.py` output, 2026-04-26):**

```
q2_verified installation_id=127368814 app_authenticated_as='outrider-spike-ddamme' app_id=3514536
```

Run from repo root with the App ID, private-key path, and installation_id
in env vars (see `runbook.md` Step 7). The script constructs an
`AppAuthStrategy(app_id, private_key)` client, switches context via
`with_auth(github.auth.as_installation(installation_id))`, and makes
one real GitHub API call (`apps.async_get_authenticated` without
`TEST_REPO`, or `repos.async_get` with). The successful response
confirms the token round-trip works against production GitHub.

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

Q7's demo confirms `github.arequest` exists and is callable for cases
where `githubkit` doesn't expose a generated method. For Q2's specific
flow, the auto-mint via `with_auth(as_installation(...))` is the
canonical path and is now live-verified.

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

**Real-payload diff finding (2026-04-26 runbook walk).** A live
`pull_request` delivery against `ddamme05/outrider-spike-test` was
captured and diffed against `sample_pull_request_opened.json`. The
structural finding: real payloads carry **`"user_view_type": "public"`**
on every user object (sender, pull_request.user, assignees[*],
pull_request.assignees[*], etc.); the octokit fixture does not. This
is a new field GitHub added since the octokit sample was last refreshed.
It is informational and does not affect any field `PRContext`
construction reads, but `api/webhooks/schemas.py` may want to capture
or ignore it explicitly when written. All other diff lines were
value-level (timestamps, IDs, URLs, login names, body text) — no other
structural drift, no missing fields, no type changes on the fields
Q4 enumerates above.

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

### Octokit fixture permissions are NOT Outrider's permission spec

`fixtures/sample_installation_created.json` carries the broad permission set
from octokit's published sample (write access to many resources). That is
*not* Outrider's minimum-viable permission set, which is `contents: read` +
`pull_requests: write` only per `docs/deployment.md` and `runbook.md`
step 2. Q4 validates the *shape* of the `installation.created` payload, not
the permission contents. If a future test needs to assert the
minimum-permission story, it must inspect the App's actual installed
permissions via the GitHub API at install time, not the fixture's
`installation.permissions` field. Treat the fixture's permissions as
arbitrary — they're whatever the octokit sample author wrote, and they
will not match real Outrider installs.

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
| App JWT shape | `pyjwt.encode({"iat": now-60, "exp": now+9*60, "iss": app_id}, private_key, "RS256")` — 60s `iat` backdate per current GitHub docs (corrected from 10s after audit 2026-04-26); or just let `AppAuthStrategy` sign internally. |
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
- **`installation.deleted` route coverage.** Spec §6.2 requires both `created` and `deleted`. The spike's route case covers `created` only; `deleted` has the same payload shape minus `repositories`, so the receiver's `installation` branch handles both without code changes. Proper `deleted` coverage lives in `api/webhooks/router.py` tests in Month 1.
- **Parse-without-name fallback.** `parse_without_name` exists but docs recommend against it (slower, may return wrong type). Production should always read `X-GitHub-Event`.
- **Webhook delivery retries.** GitHub retries failed deliveries; idempotency + the dispatcher handle this. Spike doesn't simulate.

---

## Reproducing

```bash
cd spikes/github_app
/home/spinbot/projects/outrider/.venv/bin/python run_all.py
```

Exits 0 iff every claim above reproduces on the pinned versions. The
demos are self-contained — Q1 generates its own RSA key at runtime, so
a fresh checkout passes without any setup beyond `uv sync`.

For the live round-trip — registered App + smee.io + real webhook delivery
— follow `runbook.md` with your own credentials. The installation_id and
real-payload diff captured there land back into this file as addenda to
Q2 and Q4.
