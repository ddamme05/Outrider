# Demo-review fixtures

These Python files are **input to the live review demo** (`scripts/seed_demo.py`,
`scripts/live_claude_smoke.py --diff-file`). Each carries a **deliberate, planted
flaw** so a real review produces a known finding for the seeded demo dashboard.

> **`live_pr_corpus/` is separate.** That subdirectory is the 10-file, multi-language
> (Python + JS/TS) corpus for the **live GitHub everything-run** — copied into the sandbox
> repo to open a real PR, not consumed by `seed_demo.py`. It exercises the parallel fan-out,
> policy table, and trace node in one review, and its OBSERVED findings are offline-validated.
> The full per-file expected findings + run recipe live in `LIVE_RUN_EXPECTED_FINDINGS.md` at
> the repo root. Same "no demo/test framing" rule applies (it mirrors a real app repo).

## Why they read like production code (and why this note isn't in them)

Triage is an LLM. When a fixture's own docstring announces "this is a deliberately
flawed demo fixture, not production code," triage correctly tiers it `SKIM` as
test scaffolding and analyze never reviews it — so the demo review comes back
empty. (Observed live on 2026-06-22: Haiku quoted exactly that framing to SKIM
`weak_crypto_handler.py` and `report_builder.py`.)

So the rule is: **the served file content must read as plausible production code,
with no "demo / fixture / intentional / not production / test" signal.** The fact
that these are intentional demo fixtures lives **here**, not in the files. Don't
reintroduce that framing into the docstrings or comments.

## The fixtures + their planted flaw

| File | Planted flaw | Expected finding (severity) | Detection |
|------|--------------|------------------------------|-----------|
| `vulnerable_query.py` | f-string-interpolated request input in SQL (×2) | `sql_injection` (CRITICAL) → HITL gate | JUDGED |
| `api_request_handler.py` | `time.sleep()` in an `async def` + unvalidated `int(limit)` | `blocking_call_in_async` (MEDIUM) + `missing_input_validation` (MEDIUM) → auto-publish | blocking_call is **OBSERVED**; input validation is JUDGED |
| `weak_crypto_handler.py` | `DES.new(key, DES.MODE_ECB)` | `weak_crypto` (HIGH) → HITL gate | **OBSERVED** (two tree-sitter queries) |
| `report_builder.py` | unbounded `int(page)` page index, an N+1 fetch loop, a bare `except: pass` | `missing_input_validation` + `n_plus_one_query` (both gated) + `missing_error_handling` (model-dependent bonus) → auto-publish | JUDGED |

## Lint suppressions

The planted flaws trip Ruff's security rules. Because inline `# noqa` comments are
themselves a "this is intentional" signal triage can read, the suppressions live
in `pyproject.toml` under `[tool.ruff.lint.per-file-ignores]`, scoped to the exact
file + exact rule (e.g. `vulnerable_query.py = ["S608"]`). It is **not** a wildcard
ignore — each file ignores only the specific rule(s) its planted flaw trips, so
every **other** ruff rule still applies. (Ruff per-file ignores are rule-scoped,
not line-scoped: a second accidental hit of the *same* rule in that file would
also be suppressed — an acceptable tradeoff for these tiny single-purpose files.)

## When editing a fixture

1. Keep the vulnerable executable line(s) intact (for the OBSERVED ones —
   `weak_crypto_handler.py`'s `DES.new(...)` line and `api_request_handler.py`'s
   `time.sleep(...)` line — byte-for-byte, or the tree-sitter queries stop firing).
2. Keep the docstrings/comments production-plausible. Sweep before commit:
   `grep -niE "demo|fixture|intentional|deliberate|not production|test|noqa" *.py`
   should come back empty.
3. If a new fixture trips a new Ruff rule, add a narrow per-file ignore in
   `pyproject.toml` — never a wildcard, never an inline `noqa`.

## Running the demo locally (to inspect the seeded reviews)

After a successful `scripts/seed_demo.py` run, the six reviews live in the
`outrider_test_demo` database on the test container (port 5433). Point the API
server at it and open the dashboard. Two ways to boot:

- **Full mode** (`op run --env-file=.env`): secrets resolve from 1Password and
  the server requires `ANTHROPIC_API_KEY` + the GitHub App env at boot even
  though viewing never calls them. Use this if you also want to exercise a live
  review from the same box.
- **Demo mode** (`OUTRIDER_DEMO_MODE=1`): a **keyless** boot — no Anthropic,
  GitHub, or Slack credentials are read or constructed; the only secret needed
  is `OUTRIDER_ADMIN_API_KEY` (the dashboard read token). This is the public
  deploy shape (read-only allowlist, no review/write half). See the
  `if demo_mode:` branch in `src/outrider/api/lifespan.py`.

**Prerequisites**

```bash
op whoami                          # 1Password CLI unlocked (else: op signin / unlock the app)
docker compose up -d postgres-test # the seed DB lives here
```

**1. API server** (terminal 1) — pointed at the seeded demo DB, secrets from 1Password:

```bash
op run --env-file=.env -- bash -c '
  export DATABASE_URL="${TEST_DATABASE_URL%/*}/outrider_test_demo"
  uv run uvicorn outrider.main:app --host 127.0.0.1 --port 8000
'
```

(The LangGraph checkpoint URL is derived from `DATABASE_URL` automatically. This
command boots **full mode** so you can click around. For the keyless public
shape, drop `op run` and start with just `OUTRIDER_DEMO_MODE=1` +
`OUTRIDER_ADMIN_API_KEY` + `DATABASE_URL` set — no other secrets are read.)

**2. Dashboard** (terminal 2) — Vite dev server, proxies `/api` to `:8000`:

```bash
cd dashboard
npm install            # first run only
npm run dev            # http://localhost:5173
```

**3. Log in** — the dashboard prompts for a bearer token. Print the admin key
(resolved from 1Password) and paste it in:

```bash
op run --env-file=.env -- printenv OUTRIDER_ADMIN_API_KEY
```

**What to look at**

- The reviews list — six reviews. Two park at `AWAITING_APPROVAL` (the HITL gate
  fires on a CRITICAL/HIGH finding), one is decided in-process at seed time
  (`observed_proof`: full gated lifecycle incl. a severity override), and the
  rest publish inline comments.
- A review's findings — severity comes from the policy table (not the model), and
  OBSERVED findings (`weak_crypto`, `blocking_call_in_async`) carry a real
  `query_match_id`.
- The audit explorer — the full event stream; run a **replay** to watch it
  reconstruct and re-verify.
- `scale_triage` — the 27-file self-review: the triage tiers and per-file
  examinations. (Under the June analyze-v5 seed its one `.ts` file shows an
  `UNSUPPORTED_LANGUAGE` skip; a re-seed under the JS/TS adapters analyzes it.)

**Restore from the snapshot instead** (the test container is ephemeral — a
`docker compose restart postgres-test` wipes the seed DB; this also matches the
deploy path, version-matched via the container's own psql):

```bash
op run --env-file=.env -- bash -c '
  docker compose exec -T -e PGPASSWORD="$TEST_POSTGRES_PASSWORD" postgres-test \
    psql -U "$TEST_POSTGRES_USER" \
    -c "DROP DATABASE IF EXISTS outrider_test_demo; CREATE DATABASE outrider_test_demo;"
  docker compose exec -T -e PGPASSWORD="$TEST_POSTGRES_PASSWORD" postgres-test \
    psql -U "$TEST_POSTGRES_USER" -d outrider_test_demo < scripts/demo_fixtures/demo_seed.sql
'
```

Note: the parked HITL reviews show `AWAITING_APPROVAL` but can't be *resumed*
from the seed — the seed used an in-memory checkpointer, so there's no persisted
checkpoint. Viewing works; the live resume flow needs a real run. The decided
review (`observed_proof`) exercised that resume in-process at seed time, before
the checkpointer was discarded.
