# Live Run — Expected Findings

The diff sheet for the "everything gets hit" live run. Open a PR in the sandbox repo with the
corpus below, run `scripts/live_github_demo.py` against it (once per host), then diff the actual
findings against this file. Anything that doesn't match is a real signal — a silent parse skip,
a provider drift, a coordinate bug, or a genuine model miss.

The PR is designed to exercise, in one review: **Python + JS/TS**, the **parallel analyze
fan-out**, the **deterministic severity policy table**, and the **trace node** (cross-file
resolution). It runs identically on Anthropic (Sonnet/Haiku tiered) and Fireworks GLM-5.2.

**Corpus location:** the 10 files live in `scripts/demo_fixtures/live_pr_corpus/`, mirroring the
sandbox repo layout. To build the PR: copy `src/app/db/queries.py` onto the sandbox repo's **base**
branch first (the trace target — must exist at head but be absent from the diff), then open a PR
that adds the other **9** files. The OBSERVED findings below were offline-validated against the
real query producer, so they will fire when the sandbox review runs.

---

## Two finding tiers, two confidence levels

- **OBSERVED** — a tree-sitter query in `queries/{python,javascript}/*.scm` fired. Deterministic:
  if the code shape matches the `.scm`, the finding is **guaranteed** and its severity comes
  from `SEVERITY_POLICY` (never the model). These are the hard assertions. Carries a
  `query_match_id`.
- **JUDGED** — the model proposed it; no query. Best-effort: **likely** to appear but
  model-dependent (and it may vary run-to-run and host-to-host). If it appears, its severity is
  still policy-assigned by `FindingType`, not model-set.
- **INFERRED** — produced in the trace pass-2 over a fetched out-of-diff file; carries a
  `trace_path`.

**Severity is deterministic** (`policy/severity.py`, `DECISIONS.md#001`): the model picks a
`FindingType` from a constrained enum; the table assigns severity. So the severities below are
fixed regardless of host — only *whether* a JUDGED finding appears can vary.

HITL fires on any **CRITICAL or HIGH** finding. This PR has several OBSERVED criticals, so the
gate is guaranteed to trip and the review will pause at `AWAITING_APPROVAL`.

---

## Prerequisites (one-time)

1. **Commit the trace target to the sandbox repo's BASE branch first.** `src/app/db/queries.py`
   must exist at the PR's head SHA but be **absent from the PR diff** — that's what gives the
   trace node something to resolve and fetch. Commit it to `main` in the sandbox repo, *then*
   open the PR that adds the other 9 files.
2. The demo now honors `OUTRIDER_ANALYZE_MAX_CONCURRENCY` (landed in `fix(scripts): live-run
   readiness…`) — set it to `1` for the sequential A/B baseline, `4` (default) for parallel.
3. Keep `OUTRIDER_LLM_REASONING` **off** for both hosts (the runner now fails clean at setup if
   it's on for a GLM host).
4. **Cache is cold-at-serve by construction.** The demo runs `cache_mode=SHADOW` (default): the
   Outrider analyze-cache only records would-hit/miss telemetry and **never serves** — the model
   runs on every pass, so the second run of the same head_sha is *not* served from cache. Do NOT
   set `OUTRIDER_CACHE_MODE=serve` for the demo. Belt-and-suspenders (only if you ever flip to
   serve, or want a truly pristine arm): clear the review's cache rows between runs —
   `DELETE FROM analyze_file_cache WHERE source_review_id = '<prior-review-id>';` — or push a new
   commit so the head_sha (part of the cache key) changes.

---

## File manifest (10 files)

| # | Path | Lang | Tier | In diff? | Proves |
|---|------|------|------|----------|--------|
| 1 | `src/app/db/user_repository.py` | py | DEEP | yes | Python OBSERVED SQLi + JUDGED spread; HITL |
| 2 | `src/app/tasks/report_runner.py` | py | DEEP | yes | densest Python OBSERVED; multi-query→one-finding dedup |
| 3 | `src/app/api/reports.py` | py | DEEP | yes | **trace source** (imports an out-of-diff sink) |
| 4 | `src/app/db/queries.py` | py | — | **base only** | **trace target** (fetched in pass-2) |
| 5 | `src/app/utils/pagination.py` | py | STANDARD | yes | Haiku-tier file, JUDGED-only (INFO band) |
| 6 | `src/app/models/invoice.py` | py | STANDARD | yes | second Haiku-tier file (fan-out width) |
| 7 | `src/ops/deploy_helpers.js` | js | DEEP | yes | JS OBSERVED (command injection + weak crypto) |
| 8 | `src/data/user_repository.ts` | ts | DEEP | yes | TS grammar fires the JS OBSERVED catalog |
| 9 | `src/components/ExpressionPreview.tsx` | tsx | DEEP | yes | TSX grammar + eval/hash OBSERVED |
| 10 | `README.md` | md | SKIM/SKIP | yes | triage excludes non-code (no worker Sent) |

Tier is **triage-decided** at runtime by security density — the split above is the intent; the
6 security-dense files realistically route DEEP, the 2 plain ones STANDARD. Verify actual tiers
in the `TriageResult` after the run rather than assuming.

---

## Expected findings, per file

Legend: **[O]** OBSERVED (guaranteed), **[J]** JUDGED (likely, model-dependent),
**[I]** INFERRED (trace pass-2). The OBSERVED rows below were **offline-validated** against the
real producer (`ast_facts.registry.parse_source` → `analyze_observed.run_observed_matches`) —
each listed query provably fires on the corpus file.

### 1. `src/app/db/user_repository.py` — DEEP
| Finding type | Severity | Tier | Note |
|---|---|---|---|
| `sql_injection` | CRITICAL | **[O]** | `python.sql_injection_string_concat` — fires on **all 5 DB methods** (f-string SQL into `cursor.execute`): `find_by_id`, `search_by_email`, `create_user`, `verify_password`, `load_team_roster` |
| `weak_password_hash` | CRITICAL | [J] | md5 password hashing — no Python hash query, so JUDGED |
| `hardcoded_secret` | HIGH | [J] | module-level `DB_PASSWORD` literal |
| `n_plus_one_query` | MEDIUM | [J] | per-row `find_by_id()` in a loop |
| `missing_error_handling` | LOW | [J] | raw cursor use, no try/except |

### 2. `src/app/tasks/report_runner.py` — DEEP
| Finding type | Severity | Tier | Note |
|---|---|---|---|
| `command_injection` | CRITICAL | **[O]** | `python.command_injection_subprocess_shell` (`subprocess(..., shell=True)`) |
| `unsafe_deserialization` | HIGH | **[O]** | `python.unsafe_deserialization_yaml` (`yaml.load` w/o SafeLoader) |
| `tls_verify_disabled` | HIGH | **[O]** | `python.tls_verify_disabled` (`requests(..., verify=False)`) |
| `weak_crypto` | HIGH | **[O]** | `weak_crypto_broken_cipher` + `weak_crypto_ecb_mode` both fire on the `DES.new` line → **dedup to ONE** finding (content-hash dedup) |

### 3. `src/app/api/reports.py` — DEEP (trace source)
| Finding type | Severity | Tier | Note |
|---|---|---|---|
| `sql_injection` | CRITICAL | [J] | JUDGED (not OBSERVED): the f-string is passed to the imported `run_raw_query`, whose `.execute` sink lives out-of-diff → emits a **TraceCandidate** for `app.db.queries` |

### 4. `src/app/db/queries.py` — trace target (base only, fetched in pass-2)
| Finding type | Severity | Tier | Note |
|---|---|---|---|
| `sql_injection` | CRITICAL | [I]/[J] | second analyze round over the trace-fetched file; confirms the raw `cursor.execute(sql)` sink. `analysis_rounds == 2`, `files_traced_beyond_diff >= 1`, published **DASHBOARD_ONLY** (not in the diff) |

### 5. `src/app/utils/pagination.py` — STANDARD (Haiku)
| Finding type | Severity | Tier |
|---|---|---|
| `missing_input_validation` | MEDIUM | [J] |
| `missing_error_handling` | LOW | [J] |
| `deprecated_api` | INFO | [J] — `datetime.utcnow()` |

### 6. `src/app/models/invoice.py` — STANDARD (Haiku)
| Finding type | Severity | Tier |
|---|---|---|
| `missing_input_validation` | MEDIUM | [J] |
| `missing_test` | LOW | [J] |
| `unused_import` | INFO | [J] — unreferenced `import json` |

### 7. `src/ops/deploy_helpers.js` — DEEP (JavaScript)
| Finding type | Severity | Tier | Note |
|---|---|---|---|
| `command_injection` | CRITICAL | **[O]** | `javascript.command_injection_child_process` (`cp.execSync(var)`) |
| `weak_crypto` | HIGH | **[O]** | `javascript.weak_crypto_hash` (`createHash('md5')`) |
| `weak_crypto` | HIGH | **[O]** | `javascript.weak_crypto_broken_cipher` (`des-ede3-cbc`) |
| `weak_crypto` | HIGH | **[O]** | `javascript.weak_crypto_ecb_mode` (`aes-128-ecb`) |

### 8. `src/data/user_repository.ts` — DEEP (TypeScript, uses the JS catalog)
| Finding type | Severity | Tier | Note |
|---|---|---|---|
| `sql_injection` | CRITICAL | **[O]** | `javascript.sql_injection_string_concat` (`pool.query(concat)`) |
| `tls_verify_disabled` | HIGH | **[O]** | `javascript.tls_verify_disabled` (`rejectUnauthorized: false`) |
| `tls_verify_disabled` | HIGH | **[O]** | `javascript.tls_env_verify_disabled` (`NODE_TLS_REJECT_UNAUTHORIZED = '0'`) |

### 9. `src/components/ExpressionPreview.tsx` — DEEP (TSX)
| Finding type | Severity | Tier | Note |
|---|---|---|---|
| `command_injection` | CRITICAL | **[O]** | `javascript.command_injection_eval` (`eval(memberExpr)`) |
| `weak_crypto` | HIGH | **[O]** | `javascript.weak_crypto_hash` bare-callee (`createHash` via named import) |

### 10. `README.md`
No `analyze_file` worker Sent (triage classifies non-code as SKIM/SKIP). Verify **no**
`FileExaminationEvent` for it — it keeps the kept-count honest for the concurrency A/B.

---

## Aggregate expected outcome (diff the run against this)

- **Files kept/examined:** 9 code files (README not analyzed). ~6 DEEP + ~2 STANDARD + 1 trace-fetched.
- **OBSERVED matches (guaranteed — offline-validated):** **20** query matches across 5 files:
  `user_repository.py` 5 SQLi · `report_runner.py` 5 (cmd-inj, unsafe-deser, tls, broken-cipher,
  ecb — the last two on one line → **1 weak_crypto finding**) · `deploy_helpers.js` 4 (cmd-inj +
  3 weak-crypto) · `user_repository.ts` 4 (SQLi, tls-verify ×2, tls-env) · `ExpressionPreview.tsx`
  2 (eval, md5). Each carries a `query_match_id`; a missing one = a silent parse/skip or catalog
  regression. (Match count > finding count where content-hash dedup collapses same-line/same-type
  hits, e.g. the broken-cipher+ecb pair.)
- **JUDGED findings (likely):** ~10, spanning CRITICAL→INFO. Count/wording varies by host.
- **Severity spread:** CRITICAL, HIGH, MEDIUM, LOW, INFO all represented.
- **HITL:** fires (multiple CRITICAL/HIGH) → `status=awaiting_approval`, checkpointed.
- **Trace:** `analysis_rounds == 2`, `files_traced_beyond_diff >= 1`, one DASHBOARD_ONLY finding.
- **Parallel:** at `n=4`, multiple `file:<path>#0` analyze phases overlap in the timeline.

---

## Coverage matrix

| Requested surface | Proven by | Post-run signal |
|---|---|---|
| **Python** | files 1–6 | Python OBSERVED (SQLi/cmd/deser/tls/crypto) + JUDGED spread |
| **JS/TS** | files 7–9 (.js, .ts, .tsx) | `javascript.*` OBSERVED fire under all three grammars |
| **Parallelization** | 8 kept files at `n=4` | overlapping `file:*#0` phases; `n=1` A/B: same findings, longer wall-clock |
| **Policy table** | severity spread across files | every band CRITICAL→INFO present; severity from `SEVERITY_POLICY`, not model |
| **Trace node** | files 3 → 4 | `analysis_rounds==2`, resolved TraceDecision, DASHBOARD_ONLY publish |

---

## Run recipe

Secrets are `op://` refs, so wrap in `op run`. The test/demo DB is plaintext (port 5433).

**Run 1 — Anthropic, parallel (n=4, default):**
```
op run --env-file=.env -- env \
  OUTRIDER_LLM_HOST=anthropic OUTRIDER_ANALYZE_MAX_CONCURRENCY=4 \
  uv run python scripts/live_github_demo.py \
    --owner <owner> --repo <repo> --pr <N> --installation-id <ID>
```
Needs `ANTHROPIC_API_KEY`. Interrupts at HITL — resume via the dashboard to reach publish.

**Run 1b — Anthropic sequential baseline (A/B), same PR/head_sha:** change **only**
`OUTRIDER_ANALYZE_MAX_CONCURRENCY=1`. This is **functional** validation, not a latency
benchmark: treat it as proving parallelization changes *scheduling, not outcome*. Wall-clock is
**not** a clean signal here — the provider's own prompt cache warms on Run 1, so Run 1b's
per-call latency/cost is confounded by provider-side cache hits (that's the LLM vendor's cache,
separate from Outrider's SHADOW analyze-cache). What to assert instead:
- Findings/severities/HITL/trace/`policy_version` **identical** to Run 1.
- **`LLMCallEvent` count identical** across the two arms — this is the real proof the sequential
  run isn't being served from Outrider's cache (SHADOW never serves, so every file still calls
  the model in both arms). A lower count in Run 1b = a cache-serve leak to investigate.
- Inspect `CacheLookupEvent` rows (would_hit/miss) to confirm SHADOW behavior, not serves.

**Run 2 — Fireworks GLM-5.2 (single model, all nodes), same PR/head_sha:**
```
op run --env-file=.env -- env \
  OUTRIDER_LLM_HOST=fireworks OUTRIDER_ANALYZE_MAX_CONCURRENCY=4 \
  uv run python scripts/live_github_demo.py \
    --owner <owner> --repo <repo> --pr <N> --installation-id <ID>
```
Changes vs Run 1: host `anthropic → fireworks`, key `ANTHROPIC_API_KEY → FIREWORKS_API_KEY`.
No code change (the demo is host-general). Expect `analyze_model == standard_analyze_model ==
accounts/fireworks/models/glm-5p2`, `profile_id=fireworks`, `pricing_version=v6`. (`baseten` is
the alternate GLM host: `OUTRIDER_LLM_HOST=baseten` + `BASETEN_API_KEY`.)

**Inspect (granular dump, after each run):**
```
uv run python scripts/inspect_review.py --review-id <id>
    # full per-phase timeline (phase_key-grouped) + curated LLM (profile_id/finish_reason) + findings + replay verdict
uv run python scripts/inspect_review.py --review-id <id> --phase-key file:src/ops/deploy_helpers.js
    # isolate ONE fan-out worker
uv run python scripts/inspect_review.py --review-id <id> --compact
    # scan the interleaved parallel stream (phase_key/model/profile_id per line)
```

**Cross-run invariants:** findings-by-severity, HITL-fired, trace rounds, and `policy_version`
must match across all three arms. Only `analyze_model`/`standard_analyze_model`, `profile_id`,
wall-clock, and phase overlap should differ.

---

## Caveats

- **OBSERVED requires the exact `.scm` code shape.** Every sink must sit inside a function body
  (a module-top-level sink is skipped `NO_CHANGED_SCOPE_UNITS` and never reaches the OBSERVED
  producer — only `tls_env_verify_disabled` is module-scope eligible). If an expected OBSERVED
  finding is missing, check the code shape and the fan-out worker's skip line first.
- **JS/TS structural OBSERVED is empty by design** — only the 8 `queries/javascript/*.scm`
  security queries exist (no structural queries, no `queries/typescript/` dir). A model claiming
  `evidence_tier=OBSERVED` on a JS/TS scope/import citation is default-denied at the proof
  boundary.
- **Trace is Python module-form here** (the safest path). JS/TS trace resolves only leading-dot
  relative specifiers and has no from-import correction; a single Python trace edge is deliberate.
- **JUDGED findings are model-dependent** — their presence/wording can differ between Anthropic
  and GLM. Only the OBSERVED set and the deterministic severities are invariant.
