# Mock GitHub fixtures — `EvalFixture` JSON

Each `*.json` here is one **`EvalFixture`** (the Pydantic model in
`src/outrider/agent/eval_driver.py` — that model is canonical and
`extra="forbid"`, so this README documents it but the model validates it).
`run_review("tests/eval/fixtures/mock_github/<name>.json")` loads one of these,
drives the real 7-node graph against it (real Postgres, scripted LLM + fake
GitHub at the two network boundaries), and returns an `EvalRunResult`.

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
    }
  ],

  // Scripted LLM responses, keyed by node_id -> ordered list of raw response
  // strings (index 0 = that node's first call). The string is the EXACT text
  // the real node parser consumes; shapes are proven by
  // tests/integration/test_e2e_smoke.py (_triage_response / _analyze_response).
  "llm_responses": {
    "triage": ["{\"file_tiers\": {\"app/views.py\": \"deep\"}, \"overall_risk\": \"high\", \"relevant_dimensions\": [\"security\"], \"reasoning\": \"…\"}"],
    "analyze": ["{\"findings\": [{\"finding_type\": \"sql_injection\", \"evidence_tier\": \"observed\", \"query_match_id\": \"…\", \"trace_path\": null, \"title\": \"…\", \"description\": \"…\", \"evidence\": \"…\", \"line_start\": 4, \"line_end\": 4, \"trace_candidates\": []}]}"],
    "synthesize": ["Free-form summary prose."]
  }
}
```

## Proof boundary: `OBSERVED` findings need a real `query_match_id`

An analyze finding with `evidence_tier="observed"` MUST carry a `query_match_id`
that is in the set of structural queries that actually fired on the file —
otherwise the parser rejects it (`query_match_id_not_in_registry`). The set is
computed by `agent/nodes/analyze._build_query_match_id_set(content_bytes)` from
the `.scm` queries in `src/outrider/queries/python/` (function/class/import
definitions). To author an OBSERVED fixture: write `content_head`, then discover
a real id empirically rather than fabricating a format:

```bash
uv run python -c "from outrider.agent.nodes.analyze import _build_query_match_id_set as q; print(q(open('PATH').read().encode()))"
```

`JUDGED` findings carry `query_match_id=null`; `INFERRED` carry a `trace_path`.

## Patch wire shape

`patch` is **hunks-only** — `@@ … @@` hunk(s) with NO `--- a/…` / `+++ b/…`
headers and no `diff --git` line, mirroring what GitHub's `/pulls/{number}/files`
actually returns. `coordinates/diff_parser.py::_wrap_github_hunks_with_headers`
synthesizes the headers when the first non-blank line begins with `@@`. (The
smoke test's header-bearing patch is a tolerated quirk — the wrapper passes
already-headered input through — but hunks-only is the real wire shape; match
it.) Content (`content_base`/`content_head`) is plain UTF-8 text — do NOT
pre-base64-encode it; the driver's fake GitHub client base64-wraps it to match
the contents-API shape.

## HITL gating

A CRITICAL/HIGH finding trips the HITL gate: the graph `interrupt()`s and
`run_review` STOPS there (single-pass; no auto-resume — that's the deferred
`run_review_with_resume`). On a gated run `EvalRunResult.findings` is still
populated (synthesize ran first) but `.published_comments == ()` and
`.hitl_gated is True`. Use a sub-HIGH (MEDIUM/LOW) finding when a scenario needs
publish to run.
