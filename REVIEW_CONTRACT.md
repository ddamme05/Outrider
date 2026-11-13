# Outrider review contract

Outrider selects code from a GitHub pull request, collects model proposals and
structural findings, and prepares accepted findings for human approval and GitHub
publication. This reference describes the current implementation, including
optional follow-up assessments of saved model warnings.

Read the [workflow](#review-flow) for responsibilities, the
[issue catalog](#issue-catalog) for accepted categories, and the
[JSON contracts](#json-contracts) for fields. The generated
[machine-readable catalog](docs/review-contract.json) contains the complete component
schemas, policy mappings, and checker decision table. It is a reference containing
separate shapes, not a new API endpoint or one combined runtime response.
The HTTP API has its own [OpenAPI document](dashboard/openapi.json).

This document records implemented behavior. Maintainers keep the design
specification and trust-boundary map in local `docs/spec.md` and
`docs/trust-boundaries.md`. Accepted design changes are recorded in
[DECISIONS.md](DECISIONS.md).

Detection quality and parity with a direct Claude review still require
measurement. Passing validation does not prove a bug.

## Review flow

| Stage | Model or human judgment | What code does |
|---|---|---|
| Intake | None. | Authorizes repository access, captures pinned BASE/HEAD source and diffs, and applies intake limits. |
| Triage | Model selects file review tiers, relevant dimensions, and overall PR risk. | Validates paths and tiers; snapshots the severity policy. A file marked `skim` receives no analyzer review. |
| Analyze | Model proposes individual issues, descriptions, locations, and supporting quotes. | Also produces structural findings from registered queries on current code. Whole-file removals receive bounded BASE/change review without structural proof. Checks category, source range, evidence and proof references; assigns severity, dimension, identities and confidence. |
| Trace | Model ranks supplied trace candidates. | Resolves imports, records ambiguous or unresolved paths, and bounds further analysis. Path resolution does not establish bug truth. |
| Optional candidate check | Model assesses the local bug, broader impact, PR relationship and repair advice separately. | Checks citations and derives an advisory status. Preserves the original warning. |
| Synthesize | Model writes a prose summary when findings remain. | Assembles accepted findings, applies deduplication and limits, and computes metrics. Empty reports use a fixed summary of retained count and upstream risk, without a summary model call. |
| Human approval | Reviewer approves, rejects, suppresses, or explicitly overrides severity. | Enforces the approval partition and records the decision and reviewer. |
| Publish | None. | Applies approval rules, chooses comment placement, sanitizes output, and posts an eligible GitHub review. Approved BASE removal findings go into a labeled review-body section. V1 posts `COMMENT`, not a GitHub approval or request for changes. |

The analyzer is instructed to check whether previously supported public calls now
fail, including valid non-default argument combinations. A concrete trigger can
follow from the supplied signatures and before/after code without a repository
call site. Unknown
contracts and inputs already rejected before the change do not establish a bug.

There are three different objects to follow through this flow:

- An **original warning**, called a native candidate in code, is a model proposal.
  Retained references distinguish admitted, rejected and malformed proposals.
- An **accepted finding** passed Outrider's admission checks and has a policy
  category, severity and evidence tier. It can originate from a model proposal
  or a deterministic structural producer, and may still be mistaken.
- A **follow-up assessment** is an additional model opinion about an original
  warning. It neither deletes that warning nor independently proves it correct.

Saved warnings and follow-up assessments remain separate from the final report's
`findings`. Retained source and model prose can expire; metadata and hashes are
not substitutes for the expired text.

Accepted findings with the same file, source side, line span and category share a
finding group. Outrider selects one representative; retained originals remain
available in the candidate list. This grouping does not prove they describe the
same defect.

## Issue catalog

The active severity policy is `1.3.0`. These 23 values are the categories accepted
by the finding schema. They are not 23 independently guaranteed detectors.
Severity comes from policy, not from a model's choice of wording.

| `finding_type` | Meaning | Policy `severity` | `dimension` |
|---|---|---|---|
| `sql_injection` | Untrusted data changes SQL structure. | `critical` | `security` |
| `xss` | Untrusted content can execute as browser script. | `high` | `security` |
| `hardcoded_secret` | A secret is embedded in source. | `high` | `security` |
| `auth_bypass` | An authentication or authorization requirement can be bypassed. | `critical` | `security` |
| `path_traversal` | Input can escape the intended filesystem path. | `high` | `security` |
| `missing_input_validation` | Missing input checks create a security issue. | `medium` | `security` |
| `command_injection` | Untrusted data changes command execution. | `critical` | `security` |
| `unsafe_deserialization` | Deserializing untrusted data can execute unsafe behavior. | `high` | `security` |
| `tls_verify_disabled` | TLS peer verification is disabled. | `high` | `security` |
| `weak_crypto` | Cryptography is unsuitable for its security purpose. | `high` | `security` |
| `weak_password_hash` | Password storage uses an unsuitable hash. | `critical` | `security` |
| `insecure_randomness` | Security-sensitive values use unsuitable randomness. | `high` | `security` |
| `ssrf` | Input can direct server requests to unintended destinations. | `high` | `security` |
| `ssrf_metadata` | Code permits server requests to cloud metadata, link-local endpoints or an internal control plane. | `critical` | `security` |
| `open_redirect` | Input can redirect a user to an unintended destination. | `medium` | `security` |
| `open_redirect_authed` | An open redirect affects an authenticated flow. | `high` | `security` |
| `n_plus_one_query` | Repeated queries grow with the number of records. | `medium` | `performance` |
| `blocking_call_in_async` | A blocking operation stalls an asynchronous execution path. | `medium` | `performance` |
| `unused_import` | An import is unused. | `info` | `code_quality` |
| `missing_error_handling` | An actionable failure path lacks required handling. | `low` | `code_quality` |
| `missing_test` | A supported coverage gap warrants a test. | `low` | `test_coverage` |
| `deprecated_api` | Code uses a deprecated API. | `info` | `best_practices` |
| `correctness_bug` | Supported wrong behavior with no more specific matching category. | `medium` | `code_quality` |

An unrestricted user-chosen URL can qualify as `ssrf_metadata` without a literal
metadata address in source. Visible effective destination restrictions can rule
out that subtype. A permitted destination does not prove successful network
access, a returned response or credential exposure; unseen validation remains
unknown.

Deterministic structural producers are active alongside model analysis. Current
Python query families cover `command_injection`, `unsafe_deserialization`,
`sql_injection`, `tls_verify_disabled`, `blocking_call_in_async` and `weak_crypto`.
JavaScript/TypeScript families cover `command_injection`, `sql_injection`,
`tls_verify_disabled` and `weak_crypto`. These queries identify supported code
patterns, subject to producer checks. All current registry entries are
`signal_only`; a match alone does not qualify the scope to skip model analysis.
The five shell/code-execution queries are marked `requires_context`: execution syntax
does not establish hostile input or an authority boundary. Their matches remain
available to code checks, but do not automatically emit `command_injection`.
The existing model analysis can still propose a supported `judged` injection,
which retains critical policy severity. Source requests are unchanged; no
additional model call or durable standalone signal event is added. Missing
context does not establish that execution is safe. See [decision 071](DECISIONS.md#071-execution-only-query-matches-require-context-before-vulnerability-emission).
See the [observed query catalog](src/outrider/queries/observed_catalog.py) and
[structural finding producer](src/outrider/agent/nodes/analyze_observed.py).

Use a more specific matching category first. `correctness_bug` covers actionable
wrong behavior, such as a demonstrated bad result or failure path; it does not
mean style preferences, speculative risks or missing context. Its severity is
always the policy's `medium`, not an invitation for the model to choose priority.
Other free-form labels still remain unclassified saved candidates rather than
silently becoming classified findings.

Policy `1.3.0` adds this category without changing earlier mappings. Upgrading the
application requires its migration to seed the new policy row; historical review
rows retain their recorded versions, including `1.2.0`.

Not seeing a test is not evidence that no test exists. Test coverage is unknown
when the supplied context does not establish it. Neither a parsed import nor a
possible caller proves runtime behavior.

Sources: [severity policy](src/outrider/policy/severity.py),
[dimension mapping](src/outrider/policy/dimensions.py).

## JSON contracts

JSON examples below illustrate field shapes, not results from a paid review.
Source locations are one-based, inclusive ranges in the recorded `source_side`.
JSON arrays represent Python
tuple fields. Required fields, bounds, defaults and nullable values are in the
generated catalog; custom validators and pipeline checks add rules beyond JSON
Schema. Model output and the assembled report have different contracts.

### Triage: what gets reviewed

The prompt numbers the changed files, such as `file_1: app.py`. The model returns:

```json
{
  "file_tiers": {"file_1": "standard"},
  "overall_risk": "medium",
  "relevant_dimensions": ["performance"],
  "reasoning": "The changed async handler contains a blocking operation."
}
```

| Field | Allowed values and ownership |
|---|---|
| `file_tiers` | Each supplied file ID exactly once, mapped to `deep`, `standard` or `skim`. Code restores the actual paths before constructing `TriageResult`. Code owns `skip`; the model cannot choose it. |
| `overall_risk` | Model opinion: `low`, `medium`, `high`, `critical`. It is PR-level risk, not a finding's policy severity. |
| `relevant_dimensions` | Model selection from the five dimensions in the issue catalog. |
| `reasoning` | Model explanation, bounded to 500 characters in the validated result. |
| `policy_version` | Code supplies the active policy when assembling `TriageResult`; it is omitted from the requested model example above. |

The generation schema requires the exact numbered IDs and excludes `skip`.
It depends on the file count, so filenames stay in message content rather than
the provider's compiled schema cache. The saved `TriageResult` still uses actual
paths. Extra or missing IDs fail validation; there is no filename guessing or
fallback to the old response format. Refusals, truncation and providers without
strict schema enforcement can still return invalid output.

The domain schema includes `skip`, but live model admission forbids it. The final
report propagates this triage risk; synthesis does not independently recompute it.
Source: [triage prompt](src/outrider/prompts/triage.py),
[triage admission](src/outrider/agent/nodes/triage.py).

### Analysis: an original warning

For a supplied source region containing `time.sleep(1)` on line 2 inside an async
function, an analysis response can look like this:

```json
{
  "findings": [{
    "finding_type": "blocking_call_in_async",
    "evidence_tier": "judged",
    "query_match_id": null,
    "trace_path": null,
    "title": "Blocking sleep stalls the event loop",
    "description": "time.sleep blocks this async handler's event-loop thread for one second.",
    "evidence": "time.sleep(1)",
    "line_start": 2,
    "line_end": 2,
    "trace_candidates": []
  }]
}
```

The required proposal fields are `finding_type`, `evidence_tier`, `title`,
`description`, `evidence`, `line_start` and `line_end`. Optional fields are
`query_match_id`, `trace_path` and `trace_candidates`. Each trace candidate has
`import_string_raw` and `reason`; it requests investigation rather than proving
a relationship. The per-file call supplies `file_path`, so it is not in this
model response. The model does not supply severity, confidence or finding IDs.

The provider schema accepts category and tier strings. Code then validates
canonical values and each proposal separately. An unknown category, invalid quote
or malformed sibling must not turn a rejected claim into an accepted finding.
Recorded rejection reasons explain the admission outcome; retained proposals do
not guarantee that every raw response fragment could be parsed.

Source: [analysis schemas](src/outrider/schemas/llm/analyze.py),
[analysis parser](src/outrider/agent/nodes/analyze_parser.py),
[evidence checks](src/outrider/agent/nodes/finding_evidence.py).

### Accepted finding: policy and provenance added by code

The complete `ReviewFinding` serialization is in the catalog. Its fields are:

| Fields | Meaning and owner |
|---|---|
| `finding_id`, `review_id`, `installation_id` | Code-assigned identities and installation scope. |
| `finding_type` | Canonical category from an admitted model proposal or deterministic query metadata. |
| `policy_version`, `dimension`, `severity` | Captured policy and its category mappings. |
| `evidence_tier`, `query_match_id`, `trace_path` | Validated basis of the claim; proof references are required for stronger tiers. |
| `file_path`, `line_start`, `line_end` | Validated repository path and source range. |
| `source_side` | Code-assigned `head` or `base`. Ordinary findings use HEAD; whole-file removal findings use BASE. Default HEAD is omitted from legacy-compatible serialization. |
| `title`, `description`, `evidence` | Model claim or structural producer description, with a source quote; bounded to 120, 1,000 and 2,000 characters respectively. |
| `suggested_fix` | Optional later patch suggestion; not part of analysis discovery. |
| `confidence` | Computed evidence-tier score in serialized output. |
| `original_severity`, `override_reason`, `overrider_id` | Optional human severity-override provenance. |
| `publish_destination` | Code-selected `inline_comment`, `review_body`, `dashboard_only`, or `null` before placement. |
| `content_hash`, `proposal_hash` | Code-computed identities for admitted content and its originating model proposal or structural producer record. |

| `evidence_tier` | Basis required | Fixed `confidence` |
|---|---|---|
| `judged` | Model interpretation with admitted source evidence. | `0.5` |
| `inferred` | An admissible recorded trace path. | `0.75` |
| `observed` | Admitted structural query evidence and match identity. | `0.9` |

These scores are not calibrated probabilities. Structural evidence does not prove
exploitability or deployment behavior. A model writing `observed` does not grant
that tier; live admission must establish the required producer and evidence.
Source: [finding schema](src/outrider/schemas/review_finding.py),
[proof boundary](src/outrider/policy/findings.py).

Whole-file removal findings are `judged` only, with quotes from bounded primary
BASE source. BASE means the file was removed, not that its old behavior still
exists. It does not allow structural query references, a trace path or a suggested
replacement. BASE and HEAD at the same path and lines have different content and
proposal identities; existing HEAD identity recipes remain unchanged.
Original candidate metadata, retained context, audit events, persistence and API
views preserve the source side, including after content expiry.
HTTP finding and candidate views render the side explicitly. Stored legacy
records that omit it still mean HEAD and retain their original identities.
Apply the included source-side storage migration before running the upgraded app.

### Follow-up check: bug, impact, PR relationship and repair advice

The optional `native-candidate-check-v5` separates four model judgments:

| Field | Values | Question |
|---|---|---|
| `head_assessment` | `supports`, `refutes`, `uncertain` | Does current HEAD support the local bug's trigger and consequence? |
| `impact_assessment` | `supports`, `refutes`, `uncertain`, `not_claimed` | Does captured evidence establish the warning's broader impact or required caller behavior? `not_claimed` means the warning describes only local behavior. |
| `change_assessment` | `introduced_or_worsened`, `fix_attempt`, `unrelated_preexisting`, `uncertain` | How does the PR change relate to that warning? |
| `repair_assessment` | `supports`, `refutes`, `uncertain`, `not_proposed` | Does repair advice in the original warning address the bug and respect the shown contract? `not_proposed` means the warning gives no repair advice. |

A PR can attempt a fix and still leave the bug present. For example, replacing
one unsafe branch while leaving another reachable branch unsafe can justify
`head_assessment: "supports"` together with `change_assessment: "fix_attempt"`.
“They attempted a fix” alone cannot dismiss the current-code claim.
Repair advice is a separate question: the model can identify a real bug but
suggest an incorrect fix. This assessment covers advice in the original warning,
not the PR's fix attempt or a later generated patch. Missing evidence means
`uncertain`, rather than assuming the advice works.

For a warning in a Python `.pyi` type declaration, the checker can also include
the complete matching top-level function from its exact captured `.py` sibling.
It uses only sources admitted by the existing index and remaining context space
after the original evidence is packed. Missing, ambiguous or oversized functions
stay unavailable. This does not fetch code, resolve runtime bindings or follow
the implementation's helpers. It does not enable the optional checker by default.

The full model response also requires `candidate_id`, `reason`,
`suggested_finding_type` (canonical value or `null`) and `citations`.
Each citation has `snippet_id`, `line_start`, `line_end` and an exact `quote`.
Candidate and snippet IDs are 64-character hashes supplied in the request.
The model returns neither combined `status` nor `suggested_severity`.

For an API required to reject nonpositive values with `ValueError`, suppose a PR
adds a negative-value guard to `divide(value)` but leaves `return 10 / value` on
HEAD line 4. Suppose the warning correctly identifies the zero divisor but
suggests returning zero for that input. That advice violates the required
`ValueError` behavior, so the bug and repair receive different judgments.
The repeated-letter IDs below stand for
request-supplied identities; they are not references to a real saved exchange.

```json
{
  "candidate_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "head_assessment": "supports",
  "impact_assessment": "not_claimed",
  "change_assessment": "fix_attempt",
  "repair_assessment": "refutes",
  "reason": "The negative-value guard still allows zero to reach division. Returning zero would hide the error instead of raising the required ValueError.",
  "suggested_finding_type": "correctness_bug",
  "citations": [{
    "snippet_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "line_start": 4,
    "line_end": 4,
    "quote": "    return 10 / value"
  }]
}
```

The category suggestion is advisory. Code derives its suggested `medium` severity
from policy `1.3.0`; it does not replace the original finding's classification.

Code derives the advisory `status` in this order:

1. A refuted HEAD claim **or** `unrelated_preexisting` yields `refutes`.
2. A supported local bug with `introduced_or_worsened` or `fix_attempt` yields
   `supports` when broader impact is supported or `not_claimed`, and repair
   advice is supported or `not_proposed`.
3. Other combinations yield `uncertain`.

An unknown or contradicted impact does not refute a supported local bug. It makes
the combined status uncertain, so the existing verification hold requires human
review of the original warning. For example, code can prove an exception escapes
without proving an HTTP 500 response in an unseen caller. The local bug can stay
supported while impact is uncertain; neither judgment deletes or rewrites it.
The same applies to refuted or uncertain repair advice. In the example above,
`head_assessment` remains `supports`, while the combined status is `uncertain`.
The original warning needs human review; it is not silently rewritten or removed.

The first rule can mean the warning is unrelated to this PR, without saying the
current code is safe. Consumers must retain both axes to explain that distinction.
Definite HEAD and impact assessments require HEAD citations; a preexisting/
unrelated judgment requires BASE and HEAD citations. Exact quote validation binds
the answer to supplied evidence; it does not prove the model's reasoning.

The checker can also infer an impact the warning never claimed and hold a valid
local finding unnecessarily. Separate fields make that mistake inspectable;
schema validation alone does not establish judgment quality.

V5 responses require all four judgments. Saved v1/v2 flat responses, v3
HEAD/change responses and v4 HEAD/change/impact responses retain their original
parsing, combination behavior and serialized shape. An absent historical field
means that question was not assessed; it is not equivalent to `not_claimed` or
`not_proposed`. A current response cannot omit a required judgment or be loaded
as an older response contract.

The resulting `CandidateAssessmentRef` stores references, derived status,
assessment axes, outcome and policy suggestions. Raw reasons stay in retained
model exchanges. Attempt outcomes are `completed`, `interrupted`,
`content_unavailable`, `invalid_response`, `provider_failed` or `budget_exhausted`.
An invalid or unfinished answer is not a successful verification.

Checks preserve originals and can put accepted findings on a human verification
hold. They do not automatically rewrite findings, upgrade evidence, assign human
approval, or remove a refuted warning. Legacy v1/v2 answers retain their old
single-status format; they are not silently converted into v3 judgments.
Saved-report readers use the receipt-bound historical policy and mappings rather
than today's defaults. They can inspect older answers; starting fresh checks
from a report with an older policy is refused before provider construction.
The feature is off by default and its model accuracy remains unqualified.
Its current-code checker does not assess BASE-only candidates; removal findings
require human review independently of optional model checks. The check-only
recovery tools still require clean HEAD analyzer exchanges and refuse mixed
BASE/degraded reports before starting generation.
Source: [checker](src/outrider/agent/candidate_checks.py),
[assessment metadata and combination](src/outrider/schemas/native_candidate.py).

### Trace, summary, patches and final report

| Shape | Fields and behavior |
|---|---|
| Trace model response | `ranked_candidate_ids`: ranks only the supplied IDs. |
| `TraceDecision` | Code supplies `source_finding_id`, `target_file`, `reason`, `resolution_status`, `proposed_import_strings`, `resolved_candidate_paths`, `trace_path`. Status is `resolved`, `unresolved` or `ambiguous`. |
| Summary | Nonempty reports use a plain-text model response bounded to 2,000 characters. Empty reports use the fixed `empty-findings-v1` summary without a model call. The completion event records the source and hashes the producer bytes, raw response for models or exact template text. Template summaries have no model identity. Counts in model prose are not authoritative metrics. |
| Patch model response | `items`, each with `finding_id`, nullable `original_line`, nullable `replacement_line`, and nullable `reason`. Applying a replacement requires an exact `original_line` echo of the target. Separate calls propose eligible single-line replacements; code validates placement and output. |
| `ReviewReport` | `summary`, `overall_risk`, `findings`, `metrics`. No `overall_assessment` field. Candidate assessments are stored separately. |
| `ReviewMetrics` | `files_examined`, `files_traced_beyond_diff`, `llm_calls_made`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `wall_clock_seconds`. Code computes these; historical nullable model metrics mean unknown, not zero. |

Sources: [trace parsing](src/outrider/agent/nodes/trace_parser.py),
[trace record](src/outrider/schemas/trace_decision.py),
[summary](src/outrider/prompts/synthesize.py),
[patch parsing](src/outrider/agent/nodes/patch_generation.py),
[report](src/outrider/schemas/review_report.py).

### Human decision API

`POST /reviews/{review_id}/decide` requires dashboard admin authentication and
accepts a body like:

```json
{
  "decisions": [{
    "finding_id": "00000000-0000-4000-8000-000000000001",
    "outcome": "reject",
    "reason": "The quoted branch is unreachable under the documented caller contract."
  }],
  "annotation": null
}
```

Outcomes are `approve`, `reject`, `suppress`, `severity_override`. All except
`approve` require a nonblank reason. Only `severity_override` accepts
`override_severity`; it must change the saved severity. The server supplies
`original_severity`, `reviewer_id` and `decided_at` in the internal `HITLDecision`.
The public payload does not accept those provenance fields.

The saved `HITLRequest` partitions `findings_requiring_approval` and
`auto_post_findings`, optionally records `verification_hold_finding_ids`, and
sets `created_at` and `expires_at`. High/critical findings and verification-held
accepted findings require approval. Verification-only holds permit approve,
reject or suppress, without a severity override. Unknown-category saved warnings
cannot be approved into the report through this panel.

All BASE removal findings require approval, including medium/low/info findings.
Code validates a complete removal patch against captured BASE and confirms the
file has no HEAD. Approved findings receive `review_body` placement with routing
reason `removed_source`, clearly labeled as removed BASE. They never create LEFT
inline comments or suggestions on deleted lines. Missing or mismatched provenance
remains dashboard-only; approval cannot bypass those source checks.

Read-only inspection surfaces include
`GET /api/reviews/{review_id}/candidates` and
`GET /api/reviews/{review_id}/agent-view`.
See the [HTTP schema](dashboard/openapi.json),
[decision endpoint](src/outrider/api/dashboard/hitl.py), and
[approval domain models](src/outrider/schemas/hitl.py).

## Coverage and defaults

Context selection happens in code before model transport. A prompt carries the
selected reference material and review instructions; it does not itself parse
the repository or discover tests.

Within their bounds, supplemental references follow surviving scopes affected by additions or
deletions. This includes relevant declarations, possible callers, and changed
test functions. Removing an entire function does not make its unchanged neighbor
a changed scope. Reference snippets do not expand allowed finding locations.

| Area | Current behavior and bounds |
|---|---|
| Parsed code | Python, JavaScript, TypeScript and TSX. Changed functions/classes are selected from current source. Deletion-only hunks can select retained HEAD scopes; changes outside functions get bounded current-source regions. |
| Declaration files | `.pyi`, `.d.ts`, `.d.mts` and `.d.cts` receive normal changed-scope or bounded changed-region review. The suffix alone is not evidence of generated content. Existing generated-marker, vendor, size, minification, triage and budget exclusions still apply; review does not execute declarations or guarantee agreement with an omitted implementation. |
| Nested scope rendering | Selected scope IDs, names, order, line ranges and finding bounds remain intact. A complete enclosing selected body is rendered once; contained scopes retain their headers and identify the selected body that includes them. Partial byte overlaps and different files retain separate bodies. Exact repeated clipped hunks render once in first-occurrence order; nonidentical clips keep their original ranges and separators. |
| Whole-file removals | Supported nonempty textual files receive a dedicated BASE/change review after the complete removal patch matches captured BASE and HEAD is absent. Primary BASE source is bounded to 100 lines / 16,384 bytes; removal diff to 100 line objects / 8,192 characters. Omitted source stays explicit. |
| Remaining deletion limits | Empty/metadata-only removals have no nonempty source quote and skip `NO_REVIEWABLE_CONTEXT`. Missing, partial or mismatched removal evidence skips `PATCH_BASE_MISALIGNED`. Existing language, intake, parser-exclusion, triage and budget limits still apply. |
| Text configuration | `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.env.example`, `.env.sample`, `.env.template` receive bounded text review without AST proof. |
| Base supplement | Enabled by default; up to 6,000 characters of imports, assignments, possible callers and related tests from captured changed files. It does not search unchanged tests. |
| Related Python tests | Up to two changed test files matched by filename, each with up to two complete changed functions sharing 1,200 source characters. Decorators count in the range and source limit. Oversized definitions, or decorator wrappers whose opening line cannot be recovered, are omitted whole. Tests are reference data, not executed or verified coverage. |
| Local parent constructors | From captured Python files up to 128 KiB, selected constructors calling `super().__init__` can receive up to two complete constructors from unique earlier same-file bases, at most 4,096 UTF-8 bytes each. Imported, ambiguous, modified and multiple-base relationships remain unresolved. |
| Complete parent class | One eligible parent can also be supplied as a complete class, including its documentation and methods. Its source must fit 16 KiB; the complete rendered supplement must fit 32 KiB. The existing base supplement keeps its 6,000-character limit. |
| Parent declarations | When parent source is supplied, remaining headroom can add imports, assignments and complete small undecorated classes used by that source and primary scopes. Complete declarations share a 1,600-character source limit inside the same 32-KiB rendered supplement. Visible competing bindings are omitted; names do not establish runtime resolution. |
| Parent reference allocation | References use only spare pool and per-file headroom after existing allocations. If an addition does not fit, prior context and funded primary scopes remain. These are syntactic source references, not runtime binding proof or additional finding locations; no extra discovery call is required. |
| Constructor callers | After existing references, selected own initializers in unambiguous top-level Python classes can receive complete same-file top-level direct-name callers. Existing and added callers share two excerpts / 1,000 source characters; the supplement retains its 6,000-character / 32-KiB rendered bounds. Visible decorators, metaclasses, rebinding, custom `__new__` and caller shadows withhold the new reference. Spare headroom funds additions without changing primary allocations or finding locations. This name heuristic does not prove runtime binding or reachability. |
| Declaration/implementation companions | Off by default. `OUTRIDER_ANALYZE_COMPANION_CONTEXT_ENABLED=true` can add matching top-level functions from an exact already-captured `.py`/`.pyi` sibling during clean first-pass analysis. Up to three complete functions fit 4 KiB each, 8 KiB of rendered growth and a 32-KiB total supplement. Ambiguous or visibly rebound names are omitted. Existing references and primary allocations take priority; missing source or insufficient headroom leaves the request unchanged. References affect cache identity but do not expand finding locations or prove runtime binding. No additional fetch or model call is introduced. |
| Surviving removal references | Reuses one bounded captured-HEAD index per analysis pass. Up to two small complete Python modules with an explicit import of the removed module can be supplied within the existing supplement cap. Names do not prove import roots, required behavior or reachability. Unchanged callers are not fetched; missing consumers remain unknown. |
| Module/called-function experiments | Off by default. Additional bounded references remain opt-in; name/import matching does not prove receiver resolution. |
| Called-function reference allocation | Base file requests are funded first. Helper text uses only spare pool and per-file headroom, within the existing supplement limits. If it cannot fit, the base request and its source manifest remain unchanged. Later optional parent references may receive less remaining headroom. The option remains evaluation-only; no additional source fetch is introduced. |
| Import reference experiment | Off by default (`import_reference_context_enabled`). For changed Python functions, explicit relative imports can retrieve exact named definitions or bindings at the PR's pinned HEAD. Aliases are followed conservatively; ambiguous bindings, absolute/dynamic/star imports and unknown import forms remain unresolved. References do not expand finding locations. |
| Import reference bounds | At most 16 repository read attempts, 4 new retained files, 128 KiB per file, 400 KiB retained source, 2 import hops, and 4 complete excerpts per imported module request (4,096 characters each). The rendered reference portion fits 12 KiB and the complete supplement fits 32 KiB. Module/package alternatives must be checked; unavailable alternatives cannot establish a unique match. Source paths, exact lines, HEAD and source hashes accompany excerpts; omissions stay explicit. |
| Import reference allocation | Base file allocations are funded first. Added reference text uses only spare pool and per-file headroom; otherwise the base context remains with an omission notice. This mode requires the base supplement and installation-scoped GitHub access, and cannot be combined with the module/called-function experiments. It is a bounded context experiment, not a quality or full-repository coverage guarantee. |
| Candidate checks | Off by default; at most 2 per review, 8 snippets, 2,000 bytes per snippet, 12,000 rendered context bytes. Uses captured source, including validated BASE reconstruction when available; no extra repository fetch. |
| Candidate callable defaults | Python checks can follow a called argument's explicit default to an unambiguous same-file function or `Class.method` definition. These complete bodies take priority over loose attribute-name matches within the shared four-dependency-reference limit and existing packet caps; primary and available BASE excerpts stay first. References are labeled as possible implementations because callers can override defaults. Imported, computed, decorated, nested and visibly rebound bindings remain unresolved; this is not runtime binding proof or an additional fetch or model call. |
| Intake limits | Defaults: 30 files and 1,000 total additions/deletions. |
| Analysis limits | Defaults: 200,000 review-budget tokens, concurrency 4, per-file share 25%, absolute per-file cap 60,000 tokens. These are allocation bounds, not dollar billing. |
| Analysis allocation | The planner and worker share source selection and initial prompt rendering. Each file reserves the estimated system/user tokens plus the response allowance for that request, including overlapping excerpts and supplied references. Whole-file bytes outside the request do not consume its allocation. The worker rechecks before filtering or cache lookup; unused allocations are not redistributed. Byte-based estimates are not exact provider token counts. |
| Finding limits | 50 raw proposals per call; 200 admitted findings soft limit per round/report, 256 hard limit. High/critical findings and already-known verification holds survive soft caps; exceeding hard approval capacity fails explicitly. This cannot retroactively save a warning capped before its first check. |
| Trace references | Python absolute modules and explicit package-relative modules (such as `.helpers` or `..shared`) are supported; JS/TS keeps relative path specifiers. Python relative candidates are anchored to the finding file without guessing runtime import roots. Module/package alternatives remain ambiguous when both exist. Invalid or repository-escaping forms are rejected. |
| Trace limits | Up to 5 selected imports per finding and 2 analysis rounds. Existing suffix-symbol checks, verified reads and per-pass fetch limits apply to Python relative candidates too. |
| Patches and approval | Patch suggestions enabled by default, maximum 5. Approval timeout defaults to 30 minutes with `expire_only`; timeout cannot approve publication. |

Generated, binary, oversized, unsupported, skimmed, malformed-source and
budget-excluded material may receive no full analysis. Skips and omitted context
must be considered when interpreting a quiet review or comparing cost and quality.
These bounds do not imply full repository or test coverage.

Current Anthropic defaults use Sonnet 5.5 for deep analysis and Haiku 5.5 for
standard analysis, triage, summaries, trace ranking and suggested patches.
Evaluations that also use Sonnet 5.5 for standard analysis override these defaults.
Prompt versions are
`triage-v3`, `analyze-v19`, `trace-v2`, `synthesize-v5`, `patch-v1`, and optional
`native-candidate-check-v5`. Settings can change; source and regenerated schemas
remain the implementation reference.

Sources: [graph defaults](src/outrider/agent/graph.py),
[analysis selection](src/outrider/agent/nodes/analyze.py),
[context collection](src/outrider/agent/nodes/analyze_context.py),
[model defaults](src/outrider/llm/config.py),
[intake limits](src/outrider/agent/nodes/intake_config.py).

## Keeping this reference current

After changing schemas, categories, policy mappings or checker combination rules,
regenerate and inspect the catalog:

```bash
.venv/bin/python scripts/gen_review_contract.py
.venv/bin/python scripts/gen_review_contract.py --check
```

Generation imports definitions only: it does not run a review, contact a model,
connect to a database or spend money. Keep the prose aligned with changed defaults
and admission rules; the JSON export cannot encode every custom validator or
cross-stage invariant. Regenerate the separate HTTP schema with
[gen_openapi.py](scripts/gen_openapi.py) when HTTP contracts change.

The existing automatic grading contracts are HEAD-only and explicitly refuse
BASE findings or primary BASE context. They must not score removed evidence as
current code. Manual review packets preserve source side; live deletion quality
still needs separate ground truth and semantic adjudication.
