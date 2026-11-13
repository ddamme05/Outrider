# Mock GitHub fixtures — `EvalFixture` JSON

Each `*.json` here is one **`EvalFixture`** (the Pydantic model in
`src/outrider/agent/eval_driver.py` — that model is canonical and
`extra="forbid"`, so this README documents it but the model validates it).
`run_review("tests/eval/fixtures/mock_github/<name>.json")` loads one of these,
drives the real 7-node graph against it (real Postgres, scripted LLM + fake
GitHub at the two network boundaries), and returns an `EvalRunResult`.

## Supplemental model-comparison holdouts

`test_model_comparison.py` keeps a separate pool of two SQL-injection forms
(`string.Template.substitute`, `str.join`) and two safe forms (psycopg SQL
composition with `sql.Literal`, SQLAlchemy `select().where()` expressions).
The corresponding `*_unseen_*.json` fixtures enter the existing optional
STANDARD-model comparison and Sonnet comparison reports as `holdout-recall`
and `holdout-safe`. Together they add four scenarios, or eight model calls,
to a future explicitly enabled comparison. No paid result is claimed yet.

These are distinct forms, not merely additional examples of demonstrated
placeholder styles. Their novelty is manually reviewed against the clean
pass-zero prompt: `SYSTEM_PROMPT_STABLE_PREFIX`, `FILE_CONTEXT_TEMPLATE` and
`USER_TEMPLATE`. The explicit JSON list of those three strings is hashed with
SHA-256 into a separate review stamp. This is not the frozen grading identity.
Changing any of these strings stops comparison before calls until each form
is reviewed again. If a form is now demonstrated, replace it with another
unseen form, verify real admission and model visibility, then update the
review stamp. Each track must retain at least two distinct reviewed forms.

Imports needed to recognize the APIs appear inside changed functions because
pass-zero context does not include module-level imports. Offline checks prove
recall findings are admitted and disappear with an empty model, safe code is
actually reviewed, and a scripted SQL false positive on either safe call
remains visible rather than being hidden by the parameterized-call veto.
Scripted results establish instrumentation, not model quality.

The original recall/safe pools still define the frozen 32-fixture suite-v2.
Its identity, grading gates, prior fixtures and saved paid evidence remain
unchanged. The exemplar, GLM and OpenAI collection protocols do not acquire
these supplemental rows automatically.

## Schema

```jsonc
{
  // PR identity (seeds PRContext; intake enriches changed_files from `files`).
  "installation_id": 12345,
  "owner": "acme", "repo": "widget", "pr_number": 7,
  "base_sha": "a…40", "head_sha": "b…40",
  "pr_title": "…", "pr_body": null,           // pr_body optional
  "author": "someone",
  "total_additions": 3, "total_deletions": 0,

  // Changed files. intake fetches these via the fake GitHub client (the seed
  // PRContext.changed_files starts EMPTY — intake's real two-phase fetch runs).
  "files": [
    {
      "path": "app/views.py",
      "status": "modified",                    // added | removed | modified | renamed
      "additions": 3, "deletions": 0,
      "patch": "@@ …",                          // see "Patch wire shape" below
      "previous_path": null,                    // base-side path for renamed files only
      "content_base": "…",                      // by status: removed/modified/renamed
      "content_head": "…"                       // by status: added/modified/renamed
      // NOTE: no `language` — intake derives it; supplying it fails validation.
      // If a run 404s during intake, the file is missing the content_base/
      // content_head its status requires (intake fetches it → absent → 404).
    }
  ],

  // OPTIONAL. Repository content OUTSIDE the PR diff, served by the fake GitHub
  // client's async_get_content at head_sha (the ref the trace node probes).
  // Keyed by repo-relative path -> file content. Used by trace scenarios: a
  // changed handler in `files` imports a model that lives ONLY here (beyond the
  // diff), so trace's two-phase probe fetches + resolves it. Goes through the
  // SAME content path as `files`, so the base64 wire-shape is exercised
  // identically. An absent path mimics GitHub's 404 (so trace learns which of a
  // dotted import's candidate paths exists). Omit for non-trace fixtures.
  "repository_contents_head": { "app/models.py": "class QueryBuilder: ..." },

  // Scripted LLM responses, keyed by node_id -> ordered list of raw response
  // strings (index 0 = that node's first call). The string is the EXACT text
  // the real node parser consumes; shapes are proven by
  // tests/integration/test_e2e_smoke.py (_triage_response / _analyze_response).
  "llm_responses": {
    "triage": ["{\"file_tiers\": {\"app/views.py\": \"deep\"}, \"overall_risk\": \"high\", \"relevant_dimensions\": [\"security\"], \"reasoning\": \"…\"}"],
    "analyze": ["{\"findings\": [{\"finding_type\": \"sql_injection\", \"evidence_tier\": \"judged\", \"query_match_id\": null, \"trace_path\": null, \"title\": \"…\", \"description\": \"…\", \"evidence\": \"…\", \"line_start\": 4, \"line_end\": 4, \"trace_candidates\": []}]}"],
    "synthesize": ["Free-form summary prose."]
  }
}
```

## Proof boundary for scripted model responses

A scripted analyze response plays the model's role. Use `evidence_tier="judged"`
with `query_match_id=null` for model interpretations. The current parser rejects
model-produced `observed` claims, even if the named query fired. Structural
`OBSERVED` findings come from Outrider's independent producer and require its
vulnerability-specific proof; a generic function/class/import match does not
establish a bug. Post-trace `inferred` findings need the supported trace path.

`pygoat_sql_injection.json` remains part of the frozen 32-fixture suite and
historical captures. Do not rewrite its scripted response or prior hashes.
`pygoat_sql_injection_judged.json` preserves the same source, diff and warning
claim, changing only the scripted tier/query fields for active workflow and
host-gating tests. Model-refresh tests retain the legacy case for their separate
model-contribution and invalid-proof checks; the derived fixture does not join
the frozen benchmark pool.

## Patch wire shape

`patch` is **hunks-only** — `@@ … @@` hunk(s) with NO `--- a/…` / `+++ b/…`
headers and no `diff --git` line, mirroring what GitHub's `/pulls/{number}/files`
returns. `coordinates/diff_parser.py::_wrap_github_hunks_with_headers`
synthesizes the headers when the first non-blank line begins with `@@`. (The
smoke test's header-bearing patch is a tolerated quirk — the wrapper passes
already-headered input through — but hunks-only is the real wire shape; match
it.) Content (`content_base`/`content_head`) is plain UTF-8 text — do NOT
pre-base64-encode it; the driver's fake GitHub client base64-wraps it to match
the contents-API shape.

## HITL gating

A CRITICAL/HIGH finding trips the HITL gate: the graph `interrupt()`s and
`run_review` STOPS there (single-pass; no auto-resume). On a gated run
`EvalRunResult.findings` is still populated (synthesize ran first) but
`.published_comments == ()` and `.hitl_gated is True`. Use a sub-HIGH (MEDIUM/LOW)
finding when a scenario needs `run_review` itself to reach publish.

To drive a gated finding THROUGH the gate, use `run_review_with_resume(path,
db_url=...)` (the resume driver): it interrupts on a real `AsyncPostgresSaver`,
restarts on a fresh saver + graph (same DB, same `thread_id`), resumes with an
explicit approve-all `Command(resume=...)`, and publishes — returning a
`ResumedRunResult`. See `hitl_resume_critical.json` (a CRITICAL SQL-injection PR)
for a worked example.

## Trace scenarios (beyond-diff import resolution)

See `handler_to_model_trace.json` for a worked example. To drive the trace node
to a resolved `TraceDecision`:

- Give a round-1 finding a non-empty `trace_candidates` entry —
  `{"import_string_raw": "app.models", "reason": "…"}` (the raw shape; the parser
  canonicalizes it). A dotted import maps to TWO candidate paths
  (`app/models.py` AND `app/models/__init__.py`); serve **only one** in
  `repository_contents_head` so resolution is unambiguous (`resolved`, not
  `ambiguous`).
- A **single** candidate skips the Haiku ranking call, so no `trace` LLM
  response is needed.
- Trace fetching a beyond-diff file loops back to **analyze round 2**, so script
  a **second** `analyze` response (e.g. `{"findings": []}`) — the first call
  reviews the PR file, the second reviews the fetched model.
