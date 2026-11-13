# Real PR comparison inputs

Three authentic public PRs were captured on October 31, 2026. They are historical
changes newly added to this comparison, not newly opened PRs or examples known
to be absent from model training. The original titles, comments, diffs, complete
changed files and license notices remain intact. Some upstream text names the
security issue, so this is an exposed comparison set, not a blind benchmark.

| Case | Upstream merge date | What it establishes |
| --- | --- | --- |
| [Starlette #985](https://github.com/Kludex/starlette/pull/985) | July 16, 2020 | A changed containment check accepts a sibling path that the base rejected. |
| [dbt-common #182](https://github.com/dbt-labs/dbt-common/pull/182) | August 19, 2024 | A new archive guard remains incomplete. This PR did not originate unrestricted extraction. |
| [dbt-common #353](https://github.com/dbt-labs/dbt-common/pull/353) | February 27, 2026 | Corrects the sibling-prefix mechanism, with separate modern and legacy paths. |

These cases cover one containment mechanism in two projects. They cannot
establish general review accuracy. The dbt fix is not labeled universally safe:
archive symlink behavior still needs separate review on the legacy path. All
additional model findings require adjudication, including findings inside a
counterpart's recorded scope. Source tests execute the captured functions to
check the named behavior; they do not execute whole upstream applications.

The [manifest](manifest.json) binds expectations, context review and the selected
source file to exact capture hashes. Each case directory contains `fixture.json`,
`receipt.json`, the verbatim `LICENSE`, and parsed GitHub REST JSON under `raw/`.
Receipts retain upstream creation/update/merge times separately from capture time.

## What the comparison runs

The existing capped runner reviews one selected changed Python file per PR.
It preserves the full captured PR context but assigns other changed files SKIP.
The model receives the production analyzer's actual selected scopes and hunks;
the free preflight verifies that this produces exactly one request per case.
The fixed STANDARD tier, medium risk and all review dimensions make routing
identical across models. This measures analysis, not triage or a full PR workflow.
The captured tests and other files are not automatically inserted into its prompt.

With three repetitions, each model makes nine analysis calls plus three existing
compatibility calls. Four explicit models therefore schedule at most 48 calls.
Compatibility is already included; no separate paid compatibility run is needed.

## When to run the commands

The captures are already saved. Start with the free plan from the repository root:

```bash
.venv/bin/python -m tests.eval.model_refresh \
  --suite real-pr --phase analyze --reps 3 \
  --target anthropic-sonnet-5 --target anthropic-sonnet-5-5 \
  --target anthropic-haiku-4-5 --target anthropic-haiku-5-5 \
  --max-cost-usd 3 --max-calls 48
```

It must show three cases, 48 planned calls and zero model calls made. Review the
dataset scope and readiness there before spending. The $3 limit is a ceiling,
not an expected charge. Sonnet's conservative per-call reservation requires
more than $2.58 of remaining headroom even for these small examples.

When ready to authorize that ceiling, run the same plan with credentials and
the two spending flags:

```bash
op run --env-file=.env -- .venv/bin/python -m tests.eval.model_refresh \
  --suite real-pr --phase analyze --reps 3 \
  --target anthropic-sonnet-5 --target anthropic-sonnet-5-5 \
  --target anthropic-haiku-4-5 --target anthropic-haiku-5-5 \
  --max-cost-usd 3 --max-calls 48 --paid --i-authorize-spend
```

Each run saves a new directory under `reports/model-refresh/`, containing the
plan, individual requests/responses, usage-based cost, timing, results and a
readable report. A budget stop or provider failure preserves partial evidence.
Share the printed report path afterward; no manual JSON collection is needed.
We then inspect the warnings and misses, compare each old/new pair, and decide
what additional evidence each task needs before changing a default.

## Add another public PR without spending

Use a new output directory; existing captures are never overwritten:

```bash
.venv/bin/python -m tests.eval.capture_pr \
  --repo OWNER/REPOSITORY --pr NUMBER --license-path LICENSE \
  --output reports/pr-captures/NEW_CASE
```

The importer uses `gh api` with its existing authentication and makes no model
calls. It records every changed text file at immutable commits, resolves the
diff merge-base, checks hunk totals and uses Git to reconstruct the complete head
from its base in a disposable directory. Incomplete or moving inputs are rejected.
The first version limits captures to 50 changed files and
1 MB per UTF-8 file. It supplies no expected answers and does not add the PR to
the comparison automatically. Review its source, behavior, license and actual
served context before adding byte-bound labels to a new manifest version.

GitHub's [file-list API](https://docs.github.com/en/rest/pulls/pulls#list-pull-requests-files)
and [content API](https://docs.github.com/en/rest/repos/contents#get-repository-content)
have pagination and file-size limits. Captures fail explicitly rather than
silently omit source that could affect the expected answer.
