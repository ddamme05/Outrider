# Mock GitHub fixtures

Static JSON files mirroring `githubkit` response shapes for hermetic
eval-harness execution per the eval-harness spec's "no live external
API or network calls" non-goal.

**Populated lazily:** scenario files reference files in this directory
by path (e.g., `tests/eval/fixtures/mock_github/pygoat_sql_injection.json`);
the JSON is created by the implementation commit that flips the scenario's
skip marker, NOT by the eval-harness spec itself. The eval-harness spec
ships this directory as scaffolding; the per-scenario JSON content lands
with each per-node spec when the scenario becomes executable.

**Schema validation:** until `api/webhooks/schemas.py` ships (deferred to
the webhook-receiver spec), the JSON shapes here match the `githubkit`
response shape by author discipline. When `api/webhooks/schemas.py`
exists, the eval-harness spec's held item adds a fixture-validation test
that flows these JSON files through the canonical input shapes.
