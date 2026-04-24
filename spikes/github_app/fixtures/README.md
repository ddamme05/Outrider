# Fixtures

Fixtures used by the GitHub App + smee.io spike. Kept local to the spike — not
shared with `tests/fixtures/` — so nothing outside the spike starts depending
on them.

| File | Purpose | Source |
|---|---|---|
| `sample_pull_request_opened.json` | Real `pull_request.opened` webhook payload. Feeds Q4 (payload shape) and Q5 (receiver route). | [octokit/webhooks `main/payload-examples/api.github.com/pull_request/opened.payload.json`](https://raw.githubusercontent.com/octokit/webhooks/main/payload-examples/api.github.com/pull_request/opened.payload.json) |
| `sample_pull_request_synchronize.json` | Real `pull_request.synchronize` payload (force-push / new commit). | Same repo, `synchronize.payload.json`. |
| `sample_installation_created.json` | Real `installation.created` payload, patched with `app_slug: "outrider-spike-test"` — the upstream octokit sample is stale and githubkit 0.15.3's 2026-03-10 schema requires that field. | Same repo, `installation/created.payload.json`, patched. |
| `test_private_key.pem` | Throwaway 2048-bit RSA key for JWT demos only. **Not a real GitHub App private key.** Generated at spike-scaffold time via `cryptography.hazmat`. Gitignored locally via `fixtures/.gitignore`. |

## Why real payloads instead of constructed ones

`githubkit.webhooks.parse` validates against the 2026-03-10 webhook schema,
which has 127+ required fields on `pull_request` alone. Hand-constructing
a valid minimal fixture would cost more than just downloading a real one,
and would diverge from GitHub's actual wire format in subtle ways. Using the
octokit samples keeps the fixture grounded in reality, and the one spike
finding we already surfaced (the missing `app_slug` on `installation`) is
itself useful: it tells us the octokit samples can drift from GitHub's
current schema.
