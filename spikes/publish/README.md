# Publish-node smoke harness

Direct-invoke smoke harness for the V1 publish node
(`spikes/publish/smoke_publish.py`).

## Why this exists

The 1963-test unit suite uses a stub publisher; it proves the publish
node body wires together correctly under mock GitHub + mock Postgres,
but it cannot tell you whether githubkit actually accepts our request
shape, whether the body marker round-trips through the GitHub review
list API, or whether the intra-Outrider idempotency check fires on a
real re-invoke. That gap is the difference between "publish is shipped"
and "publish actually works against GitHub." This harness closes it.

## Two modes

### Mock mode (default, no external deps)

```bash
uv run python -m spikes.publish.smoke_publish
```

Stub publisher + in-memory recording sinks. Runs the eight-step
pre-flight, routing, eligibility gate, attempt emission, and
idempotency check end-to-end. No GitHub call, no Postgres write.
Exits 0 if all assertions pass; non-zero otherwise.

Use this to:
- Verify the publish node wiring after a refactor.
- Validate test-fixture changes.
- Sanity-check during local development without GitHub creds.

### Live mode (`--apply`; real GitHub + Postgres)

```bash
# Set env vars (see "Required env vars" below) then:
uv run python -m spikes.publish.smoke_publish --apply --pr 2
```

Real `GitHubKitPublisher` posts to the allowlisted smoke-test repo.
Real `AuditPersister` writes audit rows to the postgres-test container.
The harness queries `audit_events` after the first invoke and asserts
one row each of `publish_routing`, `publish_eligibility`,
`publish_attempt`, and `publish` landed for the harness's `review_id`
(pillar 3 — closes FUP-070). Second invocation with the same review_id
MUST hit the FUP-064 intra-Outrider idempotency path
(`PublishResult.idempotently_skipped`, no second POST).

Use this to:
- Validate before opening a PR for the publish-node arc.
- Confirm a githubkit / GitHub REST API behavior change after upgrades.
- Reproduce a publish-side bug reported from real usage.

### Hard gates on live mode

- `--repo` is allowlisted to `ddamme05/outrider-smoke-test` only. The
  allowlist is hardcoded in `smoke_publish.py:_REPO_ALLOWLIST`.
  Refuses anything else.
- `TEST_DATABASE_URL` must point at port 5433 with `outrider_test` in
  the database name. Mirrors `tests/integration/conftest.py`'s guard;
  refuses to run against the dev or production DB.
- `is_eval=True` is hardcoded on every constructed `ReviewState` so
  harness writes don't pollute production dashboard queries (per
  `docs/testing.md` "Eval isolation").
- Synthetic findings come from `_make_finding` which derives severity
  via `SEVERITY_POLICY[finding_type]` and refuses CRITICAL/HIGH (the
  V1 eligibility gate would withhold them → green run with zero
  comments posted → false-success).

## Required env vars (live mode)

| Variable | Why |
|---|---|
| `OUTRIDER_GITHUB_APP_ID` | Production GitHub App env var; consumed by `GitHubAppSettings` in `src/outrider/github/config.py`. |
| `OUTRIDER_GITHUB_APP_PRIVATE_KEY` | Full PEM content (BEGIN..END), not a path. Matches `lifespan.py`. |
| `OUTRIDER_GITHUB_WEBHOOK_SECRET` | Required by `GitHubAppSettings` even though smoke harness doesn't receive webhooks. |
| `OUTRIDER_SMOKE_INSTALLATION_ID` | Harness-specific; the App's installation id for the smoke-test repo. |
| `TEST_DATABASE_URL` | psycopg async URL pointing at the postgres-test container (port 5433, db name contains `outrider_test`). |
| `OUTRIDER_TRUNCATION_HMAC_SECRET` | Defaulted by the harness if absent; set explicitly for a non-default secret. |

## Prerequisite: postgres-test container up (live mode)

```bash
docker compose up -d postgres-test
docker compose exec postgres-test pg_isready -U outrider -d outrider_test
```

See `docs/testing.md` for the two-container model rationale (why the
test container, not the dev container, is the right target).

## Cleanup

Live mode writes a per-run entry to `spikes/publish/cleanup_manifest.jsonl`
listing every `(timestamp, review_id, github_review_id, owner, repo, pr_number)`
the harness posted. GitHub does not expose bulk-dismissal of submitted
review comments; operators manually dismiss via the GitHub UI or via a
separate cleanup script consuming this manifest.

Postgres cleanup: drop the `outrider_test` DB between runs. The
audit-events table is append-only by trigger, so `DELETE` is not an
option — that's per `audit-events-append-only` invariant.

## What this harness does NOT exercise

- The `IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD` branch (Step 6 of
  `agent/nodes/publish.py` — the external-record `find_existing_review_on_head_sha`
  query path). Hitting this requires a
  crash-after-success scenario where the GitHub review exists but the
  PublishEvent audit row does not. Future harness extension.
- The `WITHHELD` paths (CRITICAL/HIGH eligibility, fabricated overrides).
  Covered by unit tests; the harness deliberately rejects those severities
  at fixture-build time to surface the green-run-zero-comments
  false-success bug class.
- Full-graph end-to-end (`graph.ainvoke(seed)`): requires LLM credentials
  + a real PR the analyze node can produce model-eligible findings on.
  Tracked separately as a follow-up to the publish-node arc.
- Concurrent publish calls (V2 hardening per FUP-068). Run this harness
  serially; concurrent invocations against the same `(repo, pr)` are
  out of scope for V1.

## Trust-boundary notes

- The harness lives under `spikes/`, outside the `check-trust-boundaries`
  skill's globbed scope. All invariants still apply — the skill simply
  doesn't catch violations here at write-time, so the harness's design
  documents them explicitly:
  - `vendor-sdks-only-in-wrappers`: harness consumes `GitHubKitPublisher`
    + `make_installation_client_factory` from `outrider.github`; never
    `import githubkit` directly.
  - `audit-events-append-only`: harness uses fresh `review_id` per run
    (uuid4) + per-run DB drop; never DELETE.
  - `publish-routes-through-coordinates`: harness invokes the publish
    node body unchanged; routing decisions stay in `coordinates/` per
    the production path.
  - `state-is-pure-data`: harness round-trips the constructed
    `ReviewState` through `model_dump_json()` at boot to assert no
    embedded clients/sessions/callbacks.
