# Eval tests

The suite checks finding quality, graph behavior and the evidence used to admit
models. Most tests use local fixtures or scripted responses. Database scenarios
use the isolated test Postgres instance on port 5433.

From the repository root:

```bash
uv run pytest tests/eval -q --is-eval
```

See [database setup](../../docs/testing.md#two-container-model-why-integration-tests-use-postgres-test-not-postgres)
before running database scenarios. The `--is-eval` flag is required; fixture
factories also mark their records so eval data stays out of normal review views.

## Where to start

For what to improve and how to measure it, start with
[Improving review quality](../../docs/review-quality.md).
Trace a recorded review through the real pipeline before choosing an evaluation.
Derive checks from the observed failure, and distinguish fresh model answers
from saved-answer replay and authored controls.

| Question | Read |
| --- | --- |
| How does a PR fixture run through the graph? | [runner.py](runner.py), [eval driver](../../src/outrider/agent/eval_driver.py) |
| How are findings scored? | [grading.py](grading.py), [scorecard.py](scorecard.py) |
| How do I compare replacement models with a spending limit? | [model_refresh.py](model_refresh.py), [comparison reports](model_refresh_report.py) |
| Does a simpler system prompt help with the same context and model? | [Prompt comparison](#compare-system-prompts), [collector](prompt_control.py) |
| Does added context help the workflow, and how does it compare with direct review? | [Workflow comparison](#compare-the-workflow-with-direct-review), [collector](review_comparison.py) |
| Why did the workflow miss something direct review found? | [Saved-request diagnostics](#diagnose-a-saved-review-miss), [collector](review_diagnostics.py) |
| Did a candidate lose useful baseline findings? | [Comparison floor](../../docs/review-quality.md#preserve-useful-baseline-findings), [evaluator](review_floor.py) |
| What happened to rejected or omitted model claims? | [Offline admission and selection audit](#audit-recorded-admission-and-selection) |
| How do I add examples with independently checked answers? | [Grounded examples](#grounded-examples) |
| How do I capture and compare real public PRs? | [Real PR guide](corpus/real_pr_refresh/README.md), [capture command](capture_pr.py) |
| What must saved probe evidence prove? | [probe_capture.py](probe_capture.py) |
| What rejects incomplete or inconsistent evidence? | [capture tests](test_probe_capture.py), [scorecard tests](test_openai_scorecard.py) |
| Do patch and trace responses satisfy their actual rules? | [patch admission](test_patch_admission.py), [trace admission](test_trace_admission.py) |
| Do the live-demo files still produce the documented matches? | [live corpus check](test_live_pr_corpus.py), [expected findings](../../LIVE_RUN_EXPECTED_FINDINGS.md) |

The model-comparison runs also include four supplemental SQL examples: two
injection forms and two safe forms absent from the reviewed prompt examples.
Their [replenishment procedure](fixtures/mock_github/README.md#supplemental-model-comparison-holdouts)
requires another review when the prompt changes. They extend future comparison
reports; the frozen 32-fixture suite and its saved evidence stay unchanged.

Capture validation follows four steps: verify the procedure and host, check the
declared model roles, hash-check every required fixture, then bind billing
evidence to those responses. Token classes must be nonnegative integers and
fit inside the prompt count. Synthetic captures test these rules locally; they
do not demonstrate real model behavior or establish real billing evidence.

## Compare replacement models

Start with a free plan. This needs no credentials and makes no model calls:

```bash
uv run python -m tests.eval.model_refresh \
  --phase compat --target anthropic-haiku-5-5 \
  --max-cost-usd 1 --max-calls 3
```

The dollar limit is a ceiling, not a planned charge. The command shows the
maximum headroom needed before a request and labels any estimate separately.
Actual usage is recorded after each completed call. If the next request cannot
fit, the command stops before sending it and keeps the partial report.

After reviewing the plan, run the same three compatibility cases with credentials:

```bash
op run --env-file=.env -- uv run python -m tests.eval.model_refresh \
  --phase compat --target anthropic-haiku-5-5 \
  --max-cost-usd 1 --max-calls 3 --paid --i-authorize-spend
```

Each paid run creates a new directory under `reports/model-refresh/`, with its plan,
individual exchanges, results and a readable summary. Existing runs and frozen
baselines are preserved. Costs use observed token usage and versioned rates;
they are not provider invoices. Unpriced calls cannot establish a cost winner.

For quality comparisons, select both a baseline and candidate with repeated
`--target` arguments, use `--phase analyze` and `--reps 3`. The 36 examples include
the four supplemental SQL cases. `--phase nodes` covers triage, summaries, trace
ranking and suggested fixes; `--phase all` runs both sets. Review the free plan
and set an explicit budget for each batch. Every paid phase starts with the three
compatibility cases, included in its planned call count. That small check does
not establish review quality or authorize a default change.

Reports separate model findings from structural-query findings. Summary prose
needs a separate review. Production defaults change only after the relevant
comparison is complete and reviewed; OpenAI also retains its production-admission
and billing requirements.

New comparisons use `refresh-grounded-visible-context-v2`. They report an
expected issue match, its exact source quote and its reported location separately.
A body-level finding is not missed just because an older answer pointed at the
function declaration. Wrong functions, wrong types, invented or hidden quotes,
and ambiguous repeated quotes cannot earn a clean match.

An expected answer also needs enough context. A function shown without tests
does not establish that tests are absent; a helper without a local exception
handler does not establish that it owns recovery. Such cases stay visible as
unscored coverage gaps. Findings on examples labeled safe are counted, with
correctness left for separate review. A grounded match is not a semantic verdict
or permission to change a default. Supported expectations keep a fixed total;
an ambiguous answer stays in that total as pending review. The [scorer](review_quality.py) and
[context reviews](fixtures/review_quality_context.json) record the rules and
which expectations their inputs support.

### Compare system prompts

Use `--compare-prompts` with one target, `--phase analyze`, and either
`--suite grounded` or `--suite real-pr`. Both arms use the same user context,
model settings, output schema, admission and scorer. The concise arm keeps the
essential output rules and allowed categories while removing worked examples
and the detailed review guidance.

This free plan needs no keys and makes no model calls:

```bash
uv run python -m tests.eval.model_refresh \
  --compare-prompts --suite grounded --phase analyze --reps 3 \
  --target anthropic-sonnet-5-5 --max-calls 66 --dry-run
```

The grounded v3 set has ten examples across five exposed bug/fix pairs.
Three repetitions require 30 analysis calls plus three compatibility calls per
arm: 66 total. The real-PR set uses three files from one bug family and requires
24 calls total with the same repetitions. Structural coverage remains separate
from model-only credit, including the new async pair.

Paid execution still requires both spending flags and an explicit dollar limit.
The two arms receive equal independent call/dollar caps; unused allowance is not
transferred. Each arm records its actual prompt identity and raw exchanges. The
combined output saves generation source, request-pair identities and separate
compatibility/quality cost and timing. A mismatch in context or settings stops
the second arm before its model call.

This is a system-prompt experiment inside the analyzer, not an unrestricted
direct-model or full-pipeline benchmark. Arms run in a fixed order with separate
system cache entries; these development cases cannot establish a general speed
or quality winner. Independently review extra warnings and validate promising
changes on new repositories and bug families before changing defaults.

## Compare the workflow with direct review

Use one model across three arms: Outrider with the context supplement disabled,
Outrider with it enabled, and a straightforward one-shot review. The workflow
arms run intake, triage, analysis, trace and synthesis through an automated draft
or approval pause. Patch generation is disabled; approval is not resumed and
GitHub publication is captured locally. Source access is limited to the same
frozen PR captures for all arms.

Start with this **free** command from the repository root. It needs no keys or
database and constructs no provider:

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --target anthropic-sonnet-5-5 --reps 1 --dry-run
```

It plans four public PR heads across two bug/fix families: upload rollover
blocking an event loop, and OAuth verification refreshing the wrong user's
session. The [case guide](corpus/review_comparison/README.md) records original
dates, capture dates, licenses and executable checks. These are exposed
development examples. A fixed counterpart establishes only its named repair.

The plan reports **upper bounds**, not scheduled calls or expected spend. With
the default 32-call cap per workflow review, four cases and one repetition allow
up to 260 calls; actual routing may use fewer. Three repetitions allow up to 780.
Use repeated `--case` options to select cases in execution order. Without them,
cases use corpus order. Dollar caps split equally across the three arms, with no
transfer of unused allowance. Direct review receives one
call per case/repetition; remaining call capacity splits between workflow arms.

Default analysis concurrency is four, matching production. Conservative budget
reservations must cover simultaneous calls: the printed headroom is not an
expected bill. `--analyze-concurrency 1` permits a smaller-budget serial quality
pilot, but its timing cannot establish production latency. Review the plan and
agree a total dollar/call ceiling before collecting paid answers.

For Sonnet 5.5, the current reservation upper bounds are $10.32768 per workflow
arm at four workers and $2.58192 at one worker. A $3 total cap splits into $1 per
arm and cannot fund those reservations. These are not expected charges or
sufficient spending caps: earlier usage also consumes each arm's allowance.

Paid collection also requires `OUTRIDER_IS_EVAL=1`, a configured isolated
`TEST_DATABASE_URL` ([setup](../../docs/testing.md#two-container-model-why-integration-tests-use-postgres-test-not-postgres)),
provider credentials, `--max-cost-usd`, `--max-calls`, `--paid` and
`--i-authorize-spend`. A workflow setup failure before provider construction stops
the experiment and preserves a partial report. Later model or budget failures
remain visible. Do not interpret an incomplete comparison as a winner.

Each run creates a new directory under `reports/review-comparison/` containing
the plan, generation source, actual requests/responses, usage, timing, workflow
diagnostics and an [adjudication packet](../../docs/review-quality.md#judge-claims-before-comparing-arms).
The direct arm allows free-text bug categories and locations outside changed
scopes. Its formatting checks do not apply Outrider's finding-admission policy.
Raw workflow proposals and final admitted findings both remain reviewable.

Exact saved direct requests and response assessment can be replayed for free:

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --replay-direct reports/review-comparison/EXPERIMENT_DIRECTORY
```

Replay verifies immutable receipts and runs archived rendering without network
access. Only `adjudication/review.json` is editable. Whole-workflow regeneration
is not implemented: generated finding IDs enter later prompts. The actual stage
exchanges are retained rather than relabeled as exact graph replay.

### Compare standalone and workflow costs

New workflow results include `cost_comparison`: API usage costs and call counts
for standalone review and each complete Outrider workflow, both per PR/repetition
and in total. These are recorded provider usage amounts, including cache billing,
not the spending cap or reservation. They are usage-based estimates under the
recorded pricing contract, not reconciled provider invoices. Server hosting costs
are not estimated.

Read an existing run without making calls or changing its original files:

```bash
.venv/bin/python -m tests.eval.review_costs \
  --report reports/review-comparison/20261101T162618Z-e1e80e4ab04d \
  --output .refactor/costs-oauth
```

Use a new output directory. The derived report records its source hashes and
reporter/runtime identity. It checks receipt-covered calls against saved exchanges,
the case/repetition schedule and arm budgets. Unpriced upper bounds remain separate
from measured spend. Incomplete or unpriced pairs get no numeric savings comparison.
Skipped files and missing analysis events remain visible: a cheaper review is not
an equal-quality review. Timings retain the original concurrency/cache limitations.

### Compare instructions through the workflow

The [completed Starlette comparison](../../reports/2026-11-01-workflow-instructions-results.md)
found and retained the original bug in three of three shorter-instruction runs,
versus zero with the current instructions. Misleading titles, uncertain extra
warnings and the separate concurrency check still prevent a broad quality claim.
Keep these exposed cases for regression checks; use another bug family before
promoting the candidate.

The [saved-request results](../../reports/2026-11-01-instruction-diagnostics-results.md)
support testing the shorter instructions beyond a single analyzer call. Use
`--compare-instructions` for three workflow conditions: current instructions,
the existing concise candidate, and the generic direct control. By default both
workflow conditions disable the supplemental context, keeping the selected-source setup
that performed best in the diagnostic.

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --compare-instructions --target anthropic-sonnet-5-5 \
  --case starlette-933 --case starlette-2962 --reps 3 \
  --analyze-concurrency 1 --calls-per-review 12 --dry-run
```

Only clean first-pass analysis gets the concise prefix. Its native schema,
user/source context, request settings and admission stay fixed. Degraded reviews,
post-trace analysis, triage, trace, synthesis and direct review retain their
original instructions. Actual model requests record the effective prefix/version.

This is an eval control at the provider boundary. Original planner estimates and
file selection remain in place, and each isolated review has no analysis-result
cache store. It does not establish production budget/cache migration behavior.
The three arms retain independent equal spending caps; the call count is an
upper bound because workflow routing varies. This serial two-case plan allows
up to 150 calls, with six direct calls included. It requires the same isolated
test database and explicit paid prerequisites as other workflow comparisons.

Add `--supplemental-context` to include the normal bounded context supplement in
both instruction conditions. The original default remains disabled so earlier
controls retain their meaning. The option requires `--compare-instructions` and
cannot alter a saved direct replay. A free plan for the context-enabled control is:

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --compare-instructions --supplemental-context \
  --target anthropic-sonnet-5-5 \
  --case flask-security-1216 --case flask-security-1183 --reps 1 \
  --analyze-concurrency 1 --calls-per-review 32 --dry-run
```

This still forces one chosen model into every role and uses the isolated driver
without a result-cache store. It is a controlled context/instruction experiment,
not qualification of all production model, cache and concurrency settings.
The filename-based test selector also supplies possible excerpts, not proof that
the changed behavior is tested. Inspect the actual source ranges before scoring.
It recognizes joined test names such as `oauth_glue.py` with `test_oauthglue.py`,
while preserving the context size and two-test-file limits. Newly eligible files
can compete for those limits; neither the matching filename nor a supplied snippet
establishes that the relevant assertion was included. Unchanged tests outside the
capture are still unavailable.

For a first check on the different OAuth bug family, use one repetition and run
the smaller fixed PR before the larger buggy PR:

```bash
OUTRIDER_IS_EVAL=1 op run --env-file=.env -- .venv/bin/python -m tests.eval.review_comparison \
  --compare-instructions --target anthropic-sonnet-5-5 \
  --case flask-security-1216 --case flask-security-1183 --reps 1 \
  --analyze-concurrency 1 --calls-per-review 32 \
  --max-cost-usd 9 --max-calls 130 \
  --paid --i-authorize-spend
```

Add `--dry-run` to print this plan without model calls or a database. The dollar
cap is a safety limit, not an expected bill. Putting the larger direct request
last helps it fit the existing reservation guard; it does not guarantee that
every workflow call will fit. Never raise the cap automatically after a stop.

Free checks confirm the named OAuth functions and tests fit their individual
analysis limits. When funded, the actual rendered source includes the relevant
callback ranges. Triage or earlier allocations can still omit either file in
the larger PR. Keep that as a workflow coverage miss; do not force the known
answer's file into review or credit an instruction comparison that never showed
it the relevant code. Tests receive separate analysis calls here, rather than
automatically becoming context for the source reviewer. Inspect actual requests,
skips and final findings before judging the result. Both cases remain historical,
exposed development examples.

The [completed OAuth pilot](../../reports/2026-11-01-oauth-workflow-results.md)
cost $1.24 and reviewed the named callbacks in both native methods. The shorter
candidate found the original bug, but direct review retained additional useful
issues. The larger PR's tests were skipped by the internal token budget. The
separate unsupported-file allocation fix was checked locally after that run;
these paid responses do not measure its effect on model quality.

Judge final retained findings as well as raw discovery. The shorter candidate
recovers the original bug, but the direct control also found a separate conditional
concurrency issue that all native diagnostic conditions missed. No complete
useful-discovery floor or production promotion has been established.

### Compare bounded called-function context

This experiment keeps the cheaper scope-based workflow. Both native arms use the
same concise clean-pass-0 instructions and the existing context supplement. Only
the candidate adds small helper bodies already captured in changed PR files.
Standalone review remains the evaluation reference, not a mandatory production
call. Plan it for free:

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --compare-called-functions --target anthropic-sonnet-5-5 \
  --case flask-security-1216 --case flask-security-1183 --reps 1 \
  --analyze-concurrency 1 --dry-run
```

The helper index uses registered language parsers, once per pass, with at most
32 captured files, 128 KiB per file and 400 KiB in total. At most two complete
helpers of 2,000 characters each fit the unused space in the existing 6,000-character
supplement. Ambiguous names and test implementations are excluded. A unique name
within that bounded index does not establish receiver or import resolution.

References carry the separate `captured_reference` manifest reason. They do not
expand finding locations, grant INFERRED evidence or perform repository fetches.
Their rendered cost is counted before file allocation, so other files can lose
budget; inspect those changes alongside quality, cost and timing. The conservative
production-prompt planning estimates still apply to both concise arms.

This option is evaluation-only, off by default, and limited to the initial pass.
Non-eval use fails before context selection or a provider call. Production prompts,
model defaults, approval rules and the normal context behavior remain unchanged.
The free plan authorizes no spending. Fresh generation requires both paid flags
and an explicit dollar/call ceiling, after the free checks are reviewed.

The [first paid results](../../reports/2026-11-01-called-helper-results.md) are
mixed: the helper arm recognizes two supported issues but loses another explicit
finding and adds a disproved warning. The unchanged grader does not qualify it
against either control. Cost, routing variation and unresolved claims are reported
separately; the experiment remains off by default.

### Compare bounded complete-module context

`--compare-module-context` tests whether complete small modules preserve useful
caller and error-handling context that selected functions can omit. It compares
the concise workflow, the same workflow with bounded module context, and the
standalone reference. Plan it without a provider or database:

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --compare-module-context --include-path-cases \
  --target anthropic-sonnet-5-5 --reps 1 \
  --analyze-concurrency 1 --calls-per-review 32 --dry-run
```

The candidate may include the complete primary module and one import-connected
module already present in the capture. Each must fit the 16 KiB raw-source cap;
the total rendered supplement, including labels and fences, must fit 32 KiB.
Larger or unavailable modules leave the scope-based context in place. This is
bounded captured context, not repository-wide retrieval or proof of runtime call
resolution. Extra source does not expand allowed finding locations or establish
that an unchanged test suite was searched. The module option is evaluation-only,
off by default and limited to the initial analysis pass.

`--include-path-cases` adds the existing receipt-verified Starlette 985 and
dbt-common 182/353 captures to the original four cases. Without it, the original
four-case dataset and input inventory remain selected. The expanded dataset is
`review-comparison-expanded-development-v1`: seven historical development-exposed
captures across three bug families. Their original labels, provenance and
per-case exposure strings are retained. Reuse does not make them held-out cases,
strengthen their oracle confidence or certify a counterpart PR as clean. Oracle
labels remain outside model requests.

Review the free plan's per-arm call bounds, request reservation headroom and
explicit dollar/call ceilings before fresh generation. `--calls-per-review` is a
workflow upper bound, not the expected call count or price; the standalone calls
also consume the batch budget. The current `baseline-first-module-headroom-v1`
policy reserves base file allocations first. Optional module text uses only
unused pool and per-file headroom; if it cannot fit, the base source and primary
scopes remain, with a bounded omission summary. It distinguishes source/context
limits from budget or per-file limits; detailed dependency omissions appear only
when the expanded inventory fits. Given the same triage result,
enabling modules cannot remove a file or primary scope from that base plan.
Different sampled triage responses can still change coverage across arms.

Judge raw and delivered findings independently, retaining incorrect and unresolved
claims. Report quality, actual usage cost, coverage, provider-call time and complete
review time together; keep cold/warm cache and concurrency differences explicit.
Local rendering tests cannot predict new model responses. Paired repeated runs
and broader checks on repositories and bug families kept aside during development
are required before considering release. This option establishes no quality or
cost advantage by itself.

The [first paid module pilot](../../reports/2026-11-01-module-context-results.md)
found three supported issues for $0.2364782, while standalone found a different
mix of three for $0.9096162. The module arm still missed a supported recovery issue and
the concise arm's bounded callback crash. It also changed which files were
reviewed under the earlier allocation policy. The new policy has only offline
coverage checks; those saved paid answers do not measure its quality. This is
progress within an exposed sample, not a parity or release pass; the option
remains disabled in production.

### Check saved assessments without model calls

The pure comparison in [review_floor.py](review_floor.py) also has an offline CLI:

```bash
.venv/bin/python -m tests.eval.review_floor \
  --inputs /path/to/frozen-assessments.json \
  --baseline direct-one-shot --candidate concise-workflow
```

The input contains an `arms` object with complete serialized `AdjudicatedArm`
records; the module's dataclasses define the fields. Prepare those assessments
from frozen claims, inspected delivery and independent source judgments first.
The command validates the supplied records and paired measurement identities;
it does not authenticate raw captures, their complete claim inventory or the
truth of the judgments. It needs no database, provider or credentials and writes
nothing to the inputs.

Exit `0` means no observed relative regression on this sample, `1` means a
regression, and `2` means inconclusive or invalid input. The report prints known
case checks separately and warns when the candidate has not demonstrated them.
Two silent methods can tie while missing a declared bug. No exit code certifies
general parity or authorizes a default change; follow the
[Claude baseline bar](../../docs/review-quality.md#the-claude-baseline-bar).

<a id="audit-recorded-parser-rejections"></a>

### Audit recorded admission and selection

Use [rejected_findings_audit.py](rejected_findings_audit.py) to review saved
proposals using persisted native candidate events or a separate parser replay.
Start with real recorded requests, replies, captured source and decisions. This
command makes no model calls and needs no database or credentials.

Pass a saved review comparison (`review-comparison-v1`) or captured review
diagnostic (`captured-review-diagnostics-v1`) directory to `--report`. Its
`receipts.json` must cover the plan, results, saved exchanges and adjudication
packet.

For a comparison with saved native candidate events, use those events directly.

```bash
.venv/bin/python -m tests.eval.rejected_findings_audit prepare \
  --report reports/review-comparison/20261102T030642Z-2b6d6416e278 \
  --limit 20 \
  --output .refactor/native-decisions-prepared
```

The observation reader in [rejection_audit_observations.py](rejection_audit_observations.py)
joins the saved review, exchange, response, proposal index/hash, file and source
side before using a native candidate event. It reads analysis-selected IDs
separately from completed draft findings. Failed runs and missing selection
records stay unknown. Completed draft membership establishes finding identity
selection, not unchanged wording, factual correctness or publication. A matching
content identity can establish a duplicate; an absent finding alone cannot
establish a cap loss. Aggregate drop counters remain separate observations.

For an older diagnostic with no recorded native decisions, supply its separately
recorded parser replay:

```bash
.venv/bin/python -m tests.eval.rejected_findings_audit prepare \
  --report reports/review-diagnostics/20261101T234842Z-87dc02b1f0fd \
  --admission-replay /path/to/recorded-parser-replay.json \
  --limit 20 \
  --output .refactor/rejected-findings-prepared
```

Choose a new output directory.
The parser replay has rows keyed by `case_id`, `rep`, `arm` and `exchange`, with
`admitted_findings`, `proposal_rejections`, `counters` and `response_rejection`.
Proposal hashes must join the exact saved response. Keep the replay command,
source version and supplied primary ranges beside that evidence. The audit
checks the join; a caller-supplied replay file alone does not prove execution.
Unjoinable or contradictory replay records fail visibly. Without a replay,
an occurrence with no native event remains unknown. Recorded native decisions
and counterfactual parser replay must be audited separately; the command rejects
both observation types on the same exchange.

Preparation freezes the complete raw occurrence roster before sampling by
reason and stable identity. It counts occurrences and unique claims separately.
Claim cards deduplicate exact wording within a captured case. Different wording
can describe the same defect, so card counts are not unique-bug counts.
Missing dispositions, unreadable content, incomplete rows and unsampled claims
remain visible. Sampling includes readable parser rejections and explicit
analysis or completed-draft selection gaps. Known draft inclusions and duplicates
stay in the roster without becoming loss claims. Original rejection reasons and
selection-stage audit reasons remain separate. A zero selected count does not
mean there was no evidence: admission, analysis and draft counts appear alongside
the unknowns. No factual judgment is needed when nothing is selected for review.

Give the reviewer only `blind/candidates.json` and its linked source files.
Copy `blind/review.json` outside the prepared directory, then record factual
truth, location, evidence, explanation and PR relevance separately. A valid
claim needs a concrete trigger, consequence and exact captured-source citations.
Use `valid`, `incorrect`, `conditional` or `unresolved` for the factual verdict.
Each accuracy field accepts `accurate`, `inaccurate`, `uncertain` or
`not_assessed`. Put qualifications in the rationale. PR relevance accepts
`introduced_or_worsened`, `fix_attempt`, `unrelated_preexisting` or `uncertain`.
Use `pending` for unfinished fields; completion reports them as incomplete.
Leave missing evidence unresolved. Keep identity-map and prior judgments closed
until this factual review is frozen. Wording and source can still reveal clues,
so this is partial blinding.

```bash
.venv/bin/python -m tests.eval.rejected_findings_audit complete \
  --audit .refactor/rejected-findings-prepared \
  --judgments /path/to/independent-review.json \
  --output .refactor/rejected-findings-completed
```

Completion validates and copies the external judgments before writing the
unblinded report. Original artifacts stay unchanged. Pending, conditional and
unresolved claims keep their counts. The report separates source truth from
parser rejection, analysis selection and recorded draft selection. It does not
infer that any rule was wrong. Policy
assessments remain unassessed. Establishing a faulty current gate also needs its
applicable rule, actual served evidence and a current-parser reproduction.
Neither source validity nor a matching quote proves delivery or authorizes
automatic restoration, category changes or default activation. The counts
describe the selected sample, not a population false-negative rate.

### Prepare native candidates without new model calls

Use [native_candidate_review.py](native_candidate_review.py) to inspect a completed
saved diagnostic report. It reuses native analyzer answers, checks the report's
receipts, fixtures and prepared requests, and retains every decoded proposal.
Unknown categories and malformed entries remain visible. It needs no credentials
or database and makes no model calls:

```bash
.venv/bin/python -m tests.eval.native_candidate_review \
  --report reports/review-diagnostics/20261101T234842Z-87dc02b1f0fd \
  --output .refactor/native-candidates-oauth
```

Choose a new output directory. The command writes a top-level `review.json`, then
per-exchange `ledger.json`, `summary.json` and prepared check requests. The ledger
keeps the exact response, each decoded entry and its identity. The readable source
path comes from the verified request, never an extra model-provided path. This
saved pilot retains all 34 proposals, including its 13 unknown categories.

Checks reuse the [bounded captured-source retrieval](#prepare-a-direct-first-review)
for possible helpers, callers and tests. Missing source stays unknown. A malformed
entry without readable text or a usable head location remains inspectable without
an invented check request. Invalid, unfinished or over-limit response envelopes
cannot produce investigation requests.

To exercise bookkeeping without spending, add `--scripted-assessments FILE` and
use another new output directory. The file is an array of objects with
`source_exchange` (such as `calls/000003.exchange.json`) and `assessment`:
`candidate_id`, `status`, `reason`, `citations`, and `suggested_finding_type`.
Copy candidate IDs from the prepared ledger. Status is `supports`, `refutes` or
`uncertain`; the suggested type is an existing `FindingType` or null.
Supports/refutes require exact complete-line citations from that prepared
request. See the [scripted example](test_native_candidate_review.py).

Assessments append without replacing the original claim, even when they refute
it. Category suggestions are separate and provisional; the recorded policy
assigns their severity. There is no arbitrary category or priority fallback.
Matching source quotes does not prove an assessment true, and scripted answers
do not measure model quality.

This command has no paid mode or provider construction. Its raw local artifacts
are outside production state and checkpoints; production retention, trace,
approval and publication are unchanged. A live investigation caller would need
an explicitly authorized budgeted provider. The
[design record](../../specs/2026-11-01-review-coverage-and-candidate-checks.md)
explains the remaining production work. Keeping candidates visible does not
establish parity with standalone Claude.

### Prepare a direct-first review

The experimental path begins with the unrestricted direct review, preserving its
exact response and every finding occurrence. A separate step can investigate each
candidate using captured source. Its assessments append to the original; even a
model refutation cannot remove or rewrite the candidate.

Prepare the saved OAuth reviews without credentials, a database or model calls:

```bash
.venv/bin/python -m tests.eval.direct_first \
  --report reports/review-comparison/20261101T162618Z-e1e80e4ab04d \
  --output .refactor/direct-first-oauth
```

Choose a new output directory each time. Repeat `--case` to select particular
cases; all saved repetitions for those cases are retained. The command checks the
saved plan and exchange receipts, fixture hashes and exact reconstructed baseline
request. These hashes bind the local inputs; they are not independent proof that
the captures or findings are true. It never executes archived source code.

Each review directory contains the original exchange, immutable seed inventory,
readable `review.md`, complete `review.json`, a numbered snapshot and prepared
follow-up requests. All original occurrences remain present, including duplicate
claims, malformed entries and categories the native policy cannot represent.
Malformed or unfinished discovery remains explicitly incomplete. An invalid
entry is retained rather than converted to a clean review. Existing outputs are
never reused or overwritten by another command run.

Follow-up context comes from the existing language parsers. It selects a primary
source window and possible definitions and callers, including test functions,
from frozen head files. Name matching is a heuristic. Requests declare unsupported
or oversized files, omitted snippets and capture limits; they cannot search an
uncaptured repository or establish test coverage. Source excerpts are capped at
eight snippets, 2,000 rendered characters each and 12,000 in total. The unchanged
candidate and retrieval-limit metadata are separate from that excerpt cap.

`run_direct_first()` exercises discovery and investigation with a supplied provider.
Scripted tests cover this path without spending. A future live caller must supply
an authorized budgeted provider and close it. It makes one discovery call shared
with the baseline, then at most `max_followups` investigation calls, without
retries. Skipped, invalid and failed assessments leave the original reviewable.
Assessment citations must match exact captured lines actually supplied to that
investigation. A matching quote does not prove the model's conclusion.

The command has no paid mode. Its output is a **local experimental draft**, not a
native review, dashboard approval item or posted GitHub comment. No production
prompt, category, severity, default model or delivery rule changes. All candidates
still require independent judgment; keeping them intact is a handling check,
not evidence that Outrider has matched Claude's review quality. Fresh generation,
false-alarm assessment and the final production delivery path remain to be tested.

## Diagnose a saved review miss

The [Starlette pilot](../../reports/2026-10-31-starlette-pilot-results.md) found a
real discovery miss: direct review caught the upload bug, while both workflow
modes returned no proposals. Use its exact saved requests to separate possible
causes before changing production instructions.

This plan is free, needs no keys or database, and makes no provider calls:

```bash
.venv/bin/python -m tests.eval.review_diagnostics \
  --report reports/review-comparison/20261101T030938Z-85310ba2326c \
  --file starlette/datastructures.py --reps 3 --dry-run
```

It schedules 36 single-call generations: two cases, three repetitions and six
arms. The four main arms pair current/concise system instructions with the
original selected-source request or that request plus full captured PR reference
data. Their native schema, model settings and primary finding bounds stay fixed.
The full-reference arms append source; they do not replace the existing context
or isolate every possible packaging change.

A fifth arm changes one general instruction to explicitly allow programming
language and standard library semantics while grounding repository claims in
supplied source. It adds no bug or API hints. The sixth repeats the generic direct
control, regenerated and checked against the original capture. All arms use the
saved model. Initial arm order rotates; provider cache state remains uncontrolled.

Source receipts, fixture bytes, exact transformations and request identities are
checked before provider construction and again during collection. There is no
silent source truncation. Paid collection requires `OUTRIDER_IS_EVAL=1`, resolved
provider credentials, both spend flags and explicit dollar/call caps. It uses
**one shared serial spending ceiling**, with equal scheduled call caps per arm.
This differs from the workflow collector's independent dollar allowances. Stops
and unattempted calls remain visible, and partial data cannot establish a winner.

Default plan output is short; `--json` exposes full identities. Each paid run
creates a fresh `reports/review-diagnostics/` directory with generation inputs,
original capture references, prepared requests, raw answers, quote/location
diagnostics, receipts and anonymous review cards. Original paid captures remain
unchanged. Quote checks preserve literal newlines and are separate from judging
whether a defect is real.

These are raw generation diagnostics. They do not run native admission, trace,
synthesis or the full workflow. A recovered raw proposal still needs admission
and factual review, followed by another full-workflow comparison, before changing
production prompts or defaults. No model-quality improvement follows from a
successful dry run or scripted collector test.

### Isolate discovery instructions on captured module context

Use `--compare-discovery` to vary instructions while keeping the exact source
from a completed `concise-with-modules` review. Plan it for free:

```bash
.venv/bin/python -m tests.eval.review_diagnostics --compare-discovery \
  --report reports/review-comparison/20261101T215045Z-46f4b9eaff87 \
  --file flask_security/oauth_glue.py --reps 3 --dry-run --json
```

This prepares 18 calls: two cases, three repetitions and three arms. It makes
none. `captured-control` keeps the original concise request; `simple-instructions`
replaces only its opening, asking for a specific trigger and consequence;
`open-categories` changes only the allowed-category guidance. The alternative
opening is not necessarily shorter. The category field already accepts strings
in the wire schema, so the category arm changes prompt wording, not decoding.

Every arm keeps the captured user source, manifests, messages, model settings,
output ceiling, wire schema and primary finding-location rules. The loader checks
receipts, exact request identity and the recorded provider profile; profile drift
or more than one matching request per case is an error. It appends no full-PR
reference and does not silently select a repetition. The original six-arm
diagnostic remains the default.

These controls test raw discovery only. Free-text categories remain proposals;
native admission, final delivery and review quality are not evaluated. No paid
calls, production prompts or defaults change when preparing this plan. Fresh
generation still requires a separately approved budget and independent judgment.

The [completed 18-call pilot](../../reports/2026-11-01-discovery-instructions-results.md)
cost $0.6594940. The alternative opening recovered the forms error-routing bug
in two of three tries against none for the control, while keeping three incorrect
redirect warnings. Its proposals pass offline parser replay. All 13 free-category
proposals fail the existing type enum. The alternative also maps the same
authentication defect to critical, medium and low priority across three answers.
These are discovery and parser results,
not a full-workflow win or a production prompt change.

### Compare the review opening through the workflow

`--compare-review-opening` compares the concise opening, the exact alternative
opening above, and standalone review. Both workflows enable the base supplement
and bounded complete-module references under `baseline-first-module-headroom-v1`.
Only eligible clean initial analysis requests receive the alternative opening;
category guidance, schema, native admission and later-stage instructions remain
the same. No production default changes.

Prepare the two-case pilot without credentials, a database or model calls:

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --compare-review-opening --target anthropic-sonnet-5-5 \
  --case flask-security-1216 --case flask-security-1183 --reps 1 \
  --analyze-concurrency 1 --calls-per-review 32 --dry-run
```

The upper bound is 130 calls: 64 per workflow and two standalone calls. Actual
counts depend on routing. Paid collection still requires both spending flags,
explicit dollar/call caps, provider credentials and the isolated local test
database. Dollar allowances split equally across the three methods; unused caps
do not transfer. Reservation headroom is not an expected bill.

Both workflows retain production planning overhead. Different sampled triage
responses can change later source selection, so use the saved request identities
and coverage when judging the result. The fixture data remain historical exposed
development examples. Inspect every raw and delivered warning, including false
alarms and conditional claims. Check category, policy priority and approval
placement, and keep prior known misses visible. The pilot
cannot qualify general parity or production latency.
The alternative's observed priority errors already block promotion; preparing
this runner does not by itself justify another paid trial.

### Compare bounded candidate checks through the workflow

Both methods use the same concise instructions, base context and bounded modules.
Only `concise-with-checks` enables up to two follow-up calls per review using
captured PR changes and HEAD dependencies. Those calls share the existing workflow dollar and call
caps. Both methods retain native candidates; standalone review is the third arm.

```bash
.venv/bin/python -m tests.eval.review_comparison \
  --compare-candidate-checks --target anthropic-sonnet-5-5 \
  --case flask-security-1216 --case flask-security-1183 --reps 1 \
  --analyze-concurrency 1 --calls-per-review 32 --dry-run
```

This free plan makes no model calls or database connection. Paid collection uses
the same explicit caps and spending flags described above. Saved results include
candidate references, provisional assessments, cost, timing and diagnostic events.
Exact citations validate supplied source, not semantic correctness. Unknown types
stay unclassified; assessments cannot rewrite or remove original findings.
Adjudicate every warning and hold before claiming a quality improvement.

The v3 checker requests separate current-code and PR-connection judgments.
A fix attempt alone does not set the combined verdict: supported current failures
remain supported; missing necessary premises remain uncertain. Definite current
judgments require HEAD citations; unrelated pre-existing concerns require both
BASE and HEAD. Those source checks do not prove model judgments correct.
Original findings, categories and priorities remain intact, and v1/v2 answers
keep their recorded decoder.

Three authored controls cover an incomplete zero guard, a complete guard and an
unprovided helper. Their shared API contract makes the named failure explicit:

```bash
.venv/bin/pytest tests/eval/test_checker_cases.py --is-eval -q
```

These free checks establish the source mechanisms, not model performance.
The [decision-contract report](../../reports/2026-11-02-checker-decision-contract.md)
records protocol/history checks and the remaining semantic limits.

### Retry only missing checks from saved discovery

The failed candidate-check pilot has reusable initial answers. Prepare only its
missing follow-ups, without paying for triage, analysis or standalone review again:

```bash
.venv/bin/python -m tests.eval.review_check_retry \
  --report reports/review-comparison/20261102T030642Z-2b6d6416e278 --dry-run
```

This command needs no keys or database. It validates original receipts and
generation files, including the complete call inventory, and prepares the current
fixed production questions. It refuses reports containing prior check attempts
or unaccounted exchanges. Captured source is read as data, never executed.

For this source there are three questions and six preserved original warnings.
The selected questions concern the PR1216 email warning and PR1183 example secrets
and initialization warning. The original two-check-per-review selection leaves
three warnings unassessed, including the PR1183 authentication classification.
Current v2 questions include bounded captured BASE/HEAD evidence and relevant
attribute assignments or possible constructors. Mapping validates the patch
against both sources; missing, inconsistent or oversized evidence stays explicit.
References identify source side, and citations validate exact lines on that side.
This does not establish every account constraint, runtime caller or PR defect.
No new selection or measured quality gain is implied.

Collection is separately gated by `OUTRIDER_IS_EVAL=1`, both spending flags and
explicit dollar/call caps. It performs only the prepared trace calls, without
automatic retries, summary generation, publication or full-workflow resumption.
Original findings, provisional assessments and computed approval holds remain
separate. Historical cost is recorded separately from new checking expense.

The existing conservative guard needs $2.52048 of serial headroom for each
2048-output Sonnet call; this is not expected spend. A $2.60 total ceiling and
three-call cap can complete normal small calls after settlement, but may stop
early. A 25-cent cap refuses before transport. Request-sized illustrations are
explicit token assumptions and never replace that guard. Preparing this command
does not authorize collection or weaken a user's lower cap.

Paid output goes to a deterministic `reports/review-check-retry/<retry-id>` directory
derived from the original candidate identities. An existing directory is refused
before provider construction, even after renaming the source report. Starts,
exchanges, assessments, results and receipts are saved separately; originals are
never overwritten. Inspect an interrupted or budget-stopped retry before any
further collection. It will not automatically repeat or resume billable calls.
The completed v1 retry already occupies this source's default output directory.
A v2 dry run is free; repeating its paid command still refuses that directory.
Do not remove it to bypass duplicate protection. A new paid experiment needs its
own reviewed collection plan rather than reusing the completed retry command.

### Compare revised checks with saved answers

Prepare only the revised questions against the completed check-only control:

```bash
.venv/bin/python -m tests.eval.review_check_comparison \
  --previous reports/review-check-retry/c8853a7b4eba9198b1049ad6 --dry-run
```

This validates the previous receipts, exact questions, source lineage and costs.
It requires the same claims and model with genuinely changed requests. Collection
needs evaluation mode, both spending flags and explicit USD/call caps. It pays only
for the revised checks; original discovery and answers remain unchanged. New
reports use `reports/review-check-comparison/`, with deterministic identities that
refuse repeat collection before provider construction. Renaming the control does
not create a new identity. The original retry directory guard remains intact.

Prior answers are historical controls. Context and instructions change together,
with uncontrolled provider timing/cache state. Judge each warning independently;
this is neither causal isolation nor a complete workflow or Claude-parity test.
The [pilot plan](../../reports/2026-11-02-revised-check-pilot-plan.md) freezes the
three warning criteria and separates prior expense from new checking cost.

## Reassess saved responses without spending

Keep the source used by the original run and a separate retained hash manifest
for its paid artifacts. Then create a new report:

```bash
uv run python -m tests.eval.model_refresh_regrade \
  --report reports/model-refresh/ORIGINAL_RUN/results.json \
  --original-source /path/to/the/original/source \
  --receipt-manifest /path/to/the/retained/paid-input-manifest.json \
  --output reports/model-refresh-regraded/NEW_REPORT
```

This command needs no API keys and makes no provider calls. It checks the old
source and saved artifacts, reproduces requests and admission with saved
responses, then applies the new measurement rules. Input or response drift
stops the reassessment. Only regenerated review/finding UUID metadata may differ.
Original usage and timing remain attributed to the original calls; they are not
new spending. Existing output directories and original reports are never replaced.
Reassessing already-seen examples is exploratory and does not replace a future
check on fresh examples.

### Grounded examples

For the separate historical public-PR set, use `--suite real-pr --phase analyze`.
The [real PR guide](corpus/real_pr_refresh/README.md) explains its narrow scope,
free capture/validation steps and the exact budget-limited paid command. Its
three source captures are already saved; no manual response collection is needed.

Use `--suite grounded --phase analyze` for the small development corpus in
[corpus/grounded](corpus/grounded). Each bug has a corresponding fix with the
same task and input. Local executable checks establish the narrow expected
behavior before any model answers are graded. These checks do not certify the
whole example as bug-free. Other findings still need review.

Start with a free plan, using the same examples for both models:

```bash
uv run python -m tests.eval.model_refresh \
  --suite grounded --phase analyze --reps 3 \
  --target anthropic-sonnet-5 --target anthropic-sonnet-5-5
```

The plan records the corpus, labels and procedure hashes. Normal spending limits
and both paid flags are still required to send requests. The compatibility checks
use the existing three compatibility examples. This suite measures analysis;
it does not measure triage, summaries, trace ranking or patch generation.

The [saved-response regressions](test_saved_review_cases.py) also replay three
actual paid responses through current admission and scoring without API calls.
They check a valid proposal, a reversed line range and a real quote with an
offset location but insufficient test context. Their [capture records](fixtures/saved_reviews)
retain the original response text and provenance. Replaying an answer does not
show how a model would answer today's prompt, and the answer is not its own
grading key.

When adding cases:

1. Establish the expected behavior from code, a reproducible failure or upstream
   evidence. State what the model must see to judge it. Leave missing premises
   unscored.
2. Prefer one bug/fix pair over many cosmetic copies. Keep closely related
   variants together; they are one evidence family, not independent discoveries.
3. Use saved JSON for response-shape and admission regressions. For review quality,
   vary the code and check the label independently. An edited response cannot
   prove that a real model finds the issue.
4. Keep examples used to develop prompts and scorers separate from a future
   untouched comparison set. This corpus is exposed development data, not a
   held-out benchmark. Do not promote a default from these scores alone.

Before reusing old files, check both the last Git change and local edits:
`git log -1 --format='%h %cI %s' -- path/to/file` and `stat path/to/file`.
A squash merge date can be later than the content edit. Dates are clues; verify
the source, prompt, measurement contract and captured hashes before reusing an
old result. Preserve historical evidence and create a new report when those
inputs change.

The [2026-10-31 role selections](baselines/model-refresh/2026-10-31.json) retain
the old baselines and record the measured trace/patch upgrade to Haiku 5.5.
Fireworks GLM-5.2 has retired from serverless and is rejected during the free
plan. Its attempted replacement comparison is incomplete. Output-limit failures
retain their paid exchange and cost; they cannot count as a quality or cost win.
The legacy cross-provider freeze/gate also rejects its retired Fireworks target
before spending. Its stored baselines remain readable; use explicit current
targets with `model_refresh` for new comparisons.

## Saved evidence and paid runs

The three captured patch/trace checks read local files without calling a model.
They skip if no capture exists and fail if a present capture is stale or invalid.
A failed capture remains a failed check; it must not be made valid by changing
its version, recorded verdicts or expected results.

The old GPT-5.6 experiment was parked after the refusal check failed. Revisit
admission when the model, endpoint or refusal behavior changes; repeating the
same experiment would not refresh its evidence. The
[recorded result](../../reports/2026-07-20-arc2-strict-schema-park-arc.md)
describes what was tested and its limits.

A new admission experiment needs observable refusal responses, a complete
[wire probe](../../spikes/openai/probe.py), and matching billing evidence.
The [wire contract](../../specs/2026-07-18-openai-native-host.md) explains those
requirements. The model-comparison command above measures candidates separately
and does not replace admission checks.

Request construction can be checked without a key or network:

```bash
uv run python spikes/openai/refusal_discovery.py --dry-run
uv run python spikes/openai/refusal_discovery.py --self-test
uv run python spikes/openai/probe.py --dry-run
```

Tests guarded by `OUTRIDER_EVAL_REAL_MODELS=1` make paid calls when enabled.
The full scorecard is separate from the wire probe and is not needed to run the
three checks over an existing capture.
