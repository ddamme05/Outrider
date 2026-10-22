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

| Question | Read |
| --- | --- |
| How does a PR fixture run through the graph? | [runner.py](runner.py), [eval driver](../../src/outrider/agent/eval_driver.py) |
| How are findings scored? | [grading.py](grading.py), [scorecard.py](scorecard.py) |
| What must saved probe evidence prove? | [probe_capture.py](probe_capture.py) |
| What rejects incomplete or inconsistent evidence? | [capture tests](test_probe_capture.py), [scorecard tests](test_openai_scorecard.py) |
| Do patch and trace responses satisfy their actual rules? | [patch admission](test_patch_admission.py), [trace admission](test_trace_admission.py) |

Capture validation follows four steps: verify the procedure and host, check the
declared model roles, hash-check every required fixture, then bind billing
evidence to those responses. Token classes must be nonnegative integers and
fit inside the prompt count. Synthetic captures test these rules locally; they
do not demonstrate real model behavior or establish real billing evidence.

## Saved evidence and paid runs

The three captured patch/trace checks read local files without calling a model.
They skip if no capture exists and fail if a present capture is stale or invalid.
A failed capture remains a failed check; it must not be made valid by changing
its version, recorded verdicts or expected results.

To refresh it, first establish structured refusal responses with
[refusal_discovery.py](../../spikes/openai/refusal_discovery.py). Freeze only
demonstrated prompts, then run the [wire probe](../../spikes/openai/probe.py).
The probe needs all required rows to pass. Capture admission also requires a
matching local raw billing export and its sanitized adjudication artifact.
The probe prints the required billing fields after a successful capture; the
[wire contract](../../specs/2026-07-18-openai-native-host.md) explains the admission rules.

Request construction can be checked without a key or network:

```bash
uv run python spikes/openai/refusal_discovery.py --dry-run
uv run python spikes/openai/refusal_discovery.py --self-test
uv run python spikes/openai/probe.py --dry-run
```

Tests guarded by `OUTRIDER_EVAL_REAL_MODELS=1` make paid calls when enabled.
The full scorecard is separate from the wire probe and is not needed to run the
three checks over an existing capture.
