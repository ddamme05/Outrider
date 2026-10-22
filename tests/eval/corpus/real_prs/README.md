# Real PR pilot corpus

Three authentic, forward aiohttp PRs provide two known path-traversal examples
and scoped clean regions in a security fix. All changed files, GitHub hunks, and
complete base/head UTF-8 contents are preserved. Inputs contain no scripted model
responses. `manifest.json` records immutable commits, source links, byte digests,
labels, and verbatim upstream Apache-2.0 notices; the 1327 notice includes the full
license text.

PR #8079's target-branch snapshot differs from its diff merge-base. Both are
recorded; the fixture uses the merge-base that produced the original PR hunks.
Only the installation identity is an eval placeholder.

The #8063 fixture includes unchanged `web_urldispatcher.py` and direct local
dependencies as head context. Their availability permits real trace fetches;
it does not assert that a model will request them or produce an INFERRED finding.

Labels were selected from upstream security evidence and source before model
execution. Findings outside the expected anchors and clean scopes need human
review. This small corpus measures these cases, not general reviewer precision.
Keep the original inputs and labels fixed when comparing runs; a model response
is an observation, never ground truth.

Validate captures without credentials or API calls:

```bash
uv run python -m tests.eval.pr_benchmark --output /tmp/outrider-pr-validation.json
```

Run one case with real models (incurs API charges), using the isolated test
Postgres container on port 5433:

```bash
op run --env-file=.env -- env OUTRIDER_IS_EVAL=1 OUTRIDER_EVAL_REAL_MODELS=1 \
  uv run python -m tests.eval.pr_benchmark --live --case aiohttp-1327 \
  --output /tmp/outrider-pr-results.json
```

Omit `--case` to run all three. The first version uses the Anthropic provider
and configured `OUTRIDER_MODEL_*` tiers. It drives intake, triage, analysis,
trace, synthesis and the HITL/publish boundary through the production graph;
GitHub reads use the captures, publishing is recorded locally, and gated
findings remain unapproved. Suggested patches are disabled.

Reports include known-issue matches and misses, false positives fully inside
adjudicated clean scopes, and additional findings needing review. Validation
marks every case `not_run`; failed live runs are `errored` and exit nonzero.
There is no overall PASS or inferred precision score. Model outputs vary
between runs even though the input bytes are frozen. Source/procedure hashes,
model configuration, prompt/policy/pricing versions, metrics and trace decisions
accompany live results. Zero findings alone do not prove a region was reviewed
or clean; inspect coverage metrics and trace decisions alongside the findings.
