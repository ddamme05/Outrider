# Demo-review fixtures

These Python files are **input to the live review demo** (`scripts/seed_demo.py`,
`scripts/live_claude_smoke.py --diff-file`). Each carries a **deliberate, planted
flaw** so a real review produces a known finding for the seeded demo dashboard.

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
| `report_builder.py` | unvalidated `int(page)` → SQL OFFSET, an N+1 query, a bare `except: pass` | `missing_input_validation` + `n_plus_one_query` + `missing_error_handling` → auto-publish | JUDGED |

## Lint suppressions

The planted flaws trip Ruff's security rules. Because inline `# noqa` comments are
themselves a "this is intentional" signal triage can read, the suppressions live
in `pyproject.toml` under `[tool.ruff.lint.per-file-ignores]`, scoped to the exact
file + exact rule (e.g. `vulnerable_query.py = ["S608"]`). It is **not** a wildcard
ignore — each file lists only the rule its flaw trips, so a real accidental issue
in a fixture still gets caught.

## When editing a fixture

1. Keep the vulnerable executable line(s) intact (for the OBSERVED ones —
   `weak_crypto_handler.py`'s `DES.new(...)` line and `api_request_handler.py`'s
   `time.sleep(...)` line — byte-for-byte, or the tree-sitter queries stop firing).
2. Keep the docstrings/comments production-plausible. Sweep before commit:
   `grep -niE "demo|fixture|intentional|deliberate|not production|test|noqa" *.py`
   should come back empty.
3. If a new fixture trips a new Ruff rule, add a narrow per-file ignore in
   `pyproject.toml` — never a wildcard, never an inline `noqa`.
