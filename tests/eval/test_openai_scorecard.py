"""Opt-in, real-spend: Anthropic-vs-GPT-5.6 scorecard over the analyze node.

The openai-native-host candidate gate (specs/2026-07-18-openai-native-host.md).
Runs the SAME recall + precision scenarios as the Sonnet-vs-Haiku model-tier
comparison (`test_model_comparison.py`) and the GLM scorecard
(`test_glm_scorecard.py`), with TWO candidate columns matching the spec's
evidence-domain rule — the scorecard canonizes exactly the two ANALYZE fields:

  - deep candidate (default `gpt-5.6-sol`) vs the Anthropic DEEP-tier
    baseline (`cfg.analyze_model`)
  - standard candidate (default `gpt-5.6-luna`) vs the Anthropic
    STANDARD-tier baseline (`cfg.standard_analyze_model`)

Candidates are ENV-SELECTABLE so the Terra-swap workflow needs no code edit:
`OUTRIDER_SCORECARD_DEEP_CANDIDATE` (sol|terra) and
`OUTRIDER_SCORECARD_STANDARD_CANDIDATE` (luna|terra); accounting is keyed by
(candidate, baseline) column so a both-fields-Terra rerun stays two columns.

REPORT-ONLY BY DESIGN (the glm-scorecard precedent): pytest "passed" means the
run COMPLETED, not that a gate passed. ADJUDICATION RULE (frozen in the spec's
gates section): the operator reads the report and records the verdict + report
pointer in the spec's Actual Outcome. Canonizing a provisional default requires
BOTH (a) structured-output yield at the #059 bar (zero rejected responses
across the rows — json_object mode + prompt-named fields are the conformance
drivers here) and (b) the `grading.py` baseline recall floor against that
tier's incumbent. A miss on either swaps THAT field to `gpt-5.6-terra` and
reruns this scorecard — never a silent fallback — and a Terra swap first
inherits the full paid-wire probe matrix (spikes/openai/probe.py).

PRECONDITION (enforced, not advisory): a passing, coherent, CURRENT probe
capture (`spikes/openai/fixtures/manifest.json`) is REQUIRED — this test FAILS
without it. The gate verifies the verdict boolean AND the capture's provenance
(canonical base_url; the profile contract digest AND the probe's own
procedure/manifest versions — a wire-affecting profile change or a probe
prompt/matrix/predicate change both stale the capture), the EXACT expected row
set (refusal rows included: the refusal-normalization fixture is a pre-ship
gate per model; extra rows rejected; bare path-safe filenames only), each
fixture's sha256, the cold/warm conservation BOUNDS recomputed from fixture
bytes, and — because bounds cannot choose between the spec's two accounting
equations — the operator's sanitized billing-adjudication ARTIFACT
(`billing_adjudication.json`; raw exports stay local/gitignored, but the raw
export must EXIST under fixtures/raw/ and hash-match the artifact at
admission), which is closed-key at every level (extra keys refuse — that is
what keeps the sole committable file sanitized) and must be BOUND to this
capture (exact response IDs, a bounded window covering the fixtures'
`created` stamps, billed class counts consistent with the wire) and must
match the equation `read_usage()` ships. A conformance surprise is caught on
the probe's cheap capture, never on this ~128-call run.

Run (keys resolve from .env via 1Password):
  OUTRIDER_EVAL_REAL_MODELS=1 op run --env-file=.env -- \
    uv run pytest tests/eval/test_openai_scorecard.py --is-eval -v -s

Cost: 2 candidate columns x the FULL imported evidence catalog (22 recall + 10
safe fixtures at last count — counts are computed at runtime and printed before
any spend) = 64 comparisons, each a baseline + candidate call: ~128 small
analyze calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from outrider.llm.config import ModelConfig
from outrider.prompts.analyze import VERSION as _ANALYZE_PROMPT_VERSION

from .model_comparison import compare_models_on_scenario, state_from_eval_fixture
from .probe_capture import (
    _EXPECTED_ADJUDICATION_SCHEMA_VERSION,
    _EXPECTED_MANIFEST_SCHEMA_VERSION,
    _EXPECTED_PROBE_CONTRACT_VERSION,
    _LUNA,
    _PIN_CREATED_EPOCH,
    _PIN_WINDOW,
    _READ_USAGE_PINNED_EQUATION,
    _SOL,
    _TERRA,
    _expected_probe_rows,
    _require_probe_manifest,
)
from .runner import build_provenance
from .scorecard import Scorecard, ScorecardRow
from .test_model_comparison import (
    _GROUND_TRUTH_BY_FIXTURE,
    _SAFE_CODE_FIXTURES,
    _NoOpExchangePersister,
    _print_aggregate_metrics,
    _print_scenario_report,
    _run_scenario_isolating_transients,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .grading import ExpectedFinding, ModelComparison

# The APPROVED baselines: derived from the CANONICAL authority — ModelConfig
# FIELD DEFAULTS (the class defaults; ambient OUTRIDER_MODEL_* overrides
# influence instances, never the class) — so no second model authority exists
# (`model-strings-from-config-not-hardcoded`). The paid run must FAIL before
# spending when the ambient-resolved config diverges from these: an arbitrary
# valid Claude pair is not the approved comparison.
_APPROVED_BASELINES = {
    "deep": ModelConfig.model_fields["analyze_model"].default,
    "standard": ModelConfig.model_fields["standard_analyze_model"].default,
}


def _require_approved_baselines(deep_baseline: str, standard_baseline: str) -> None:
    resolved = {"deep": deep_baseline, "standard": standard_baseline}
    if resolved != _APPROVED_BASELINES:
        pytest.fail(
            f"resolved baselines {resolved} != approved {_APPROVED_BASELINES} — ambient "
            "OUTRIDER_MODEL_* overrides change the approved comparison; unset them (or "
            "amend the spec) before any scorecard spend"
        )


def _require_openai_billing_adjudicated() -> None:
    """Pre-spend billing preflight for the paid GPT-5.6 scorecard (FUP-247).

    The scorecard cannot produce evidence while openai cost is WITHHELD — analyze's
    `compute_cost_usd` sites (analyze.py:2821/3416) RAISE on `Unpriced(BILLING_PENDING)`
    AFTER the billed provider call, and that `ValueError` is not an `LLMProviderError`
    so the runner's transient isolation does not demote it — the run would spend, then
    abort. The caller has explicitly opted into real spend (the `OUTRIDER_EVAL_REAL_MODELS`
    skipif gates ordinary CI), so a missing prerequisite FAILS, never skips — a silent
    skip would read as a clean run (mirrors `_require_probe_manifest`). A LIVE module
    read, so billing adjudication (removing `"openai"` from the set) re-enables the run
    without touching this preflight; module-level so the zero-spend pins drive the exact
    branch the paid test uses."""
    import outrider.llm.pricing as _pricing_mod  # noqa: PLC0415

    if "openai" in _pricing_mod.COST_UNADJUDICATED_PROFILE_IDS:
        pytest.fail(
            "openai billing is unadjudicated (FUP-247) — the analyze compute_cost_usd "
            "sites raise Unpriced(BILLING_PENDING) after the paid call, so this scorecard "
            "cannot produce evidence. Bless the SHAPER v4 equation via a billing export "
            "and remove 'openai' from COST_UNADJUDICATED_PROFILE_IDS to enable it."
        )


def _require_openai_admission_credentials() -> tuple[str, str]:
    """Shared pre-spend credential preflight for ALL THREE OpenAI admission runners — the
    analyze scorecard here PLUS the triage/synthesize node-admission instruments in
    `test_openai_node_admission.py`. Returns the validated, whitespace-stripped
    `(anthropic_key, openai_key)`.

    The caller has opted into real spend (the `OUTRIDER_EVAL_REAL_MODELS` skipif gates
    ordinary CI), so a missing OR unresolved (`op://`) key for EITHER vendor FAILS, never
    skips — a skip on an opted-in run would read as a clean run that never happened
    (mirrors `_require_probe_manifest` / `_require_openai_billing_adjudicated`). Both keys
    are required BEFORE provider construction so no partial run can start."""

    def _resolved(name: str, purpose: str) -> str:
        value = (os.environ.get(name) or "").strip()
        if not value or value.startswith("op://"):
            pytest.fail(
                f"{name} (resolved — not empty, not an op:// ref) is required for {purpose}; "
                "run under `op run --env-file=.env -- ...`"
            )
        return value

    anthropic_key = _resolved("ANTHROPIC_API_KEY", "the Anthropic baseline")
    openai_key = _resolved("OPENAI_API_KEY", "the GPT-5.6 candidate")
    return anthropic_key, openai_key


def _require_candidates_wire_admitted(
    capture_manifest: dict[str, object], deep_candidate: str, standard_candidate: str
) -> None:
    """Per-model WIRE admission carries into the scorecard: a candidate may
    only run if THIS verified capture declared it as an analyze candidate — a
    default Sol/Luna capture must not authorize a paid Terra scorecard.
    Module-level so the zero-spend pin drives the exact branch the paid test
    uses."""
    wire_admitted = capture_manifest.get("analyze_models")
    assert isinstance(wire_admitted, list)  # gate validated the declaration
    for chosen in (deep_candidate, standard_candidate):
        if chosen not in wire_admitted:
            pytest.fail(
                f"candidate {chosen!r} is not among the verified capture's declared "
                f"analyze_models {wire_admitted} — re-run the probe declaring it "
                "(--analyze-models=...) before any scorecard spend"
            )


# --- Run artifacts: proportionate to a LOCAL, operator-controlled eval run -
# (per the 2026-07-18 de-scope: the operator is the trust anchor; artifacts
# need staleness protection and honest failure markers, not adversarial
# guarantees). Three pieces, no state machine:
#   - a pre-run `status.json` partial marker (collision-refusing directory),
#   - one append-only results.jsonl written DURING execution,
#   - one validated final manifest written atomically ONLY after the
#     completion assertion passes. On failure the partial marker and the raw
#     results remain, and nothing pretends otherwise.


class _FinalRunManifest(BaseModel):
    """The final manifest: models, commit provenance, counts, completion."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["complete"]
    columns: tuple[dict[str, str], ...]
    scheduled_gate_entries: int
    actual_gate_entries: int
    git_sha: str
    git_dirty: bool
    prompt_template_version: str


async def _with_providers(
    make_baseline: Callable[[], object],
    make_candidate: Callable[[], object],
    body: Callable[[object, object], Awaitable[None]],
) -> None:
    """Ordinary nested resource ownership, factored once so it is
    free-testable: each provider's ownership begins IMMEDIATELY after its
    construction — a candidate-constructor failure closes the
    already-constructed baseline and never runs the body."""
    baseline = make_baseline()
    try:
        candidate = make_candidate()
        try:
            await body(baseline, candidate)
        finally:
            await candidate.aclose()  # type: ignore[attr-defined]
    finally:
        await baseline.aclose()  # type: ignore[attr-defined]


def _start_run(base_dir: Path, run_id: str) -> Path:
    run_dir = base_dir / f"run-{run_id}"
    if run_dir.exists():
        msg = f"run dir {run_dir} already exists — run ids must be unique"
        raise FileExistsError(msg)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": run_id, "status": "partial"}, indent=2), encoding="utf-8"
    )
    return run_dir


def _append_result(run_dir: Path, record: dict[str, object]) -> None:
    with (run_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _finish_run(
    run_dir: Path,
    manifest: _FinalRunManifest,
) -> None:
    """Write the validated final manifest ATOMICALLY (tmp + rename) and flip
    the status marker — called only after the completion assertion passed."""
    tmp = run_dir / "final_manifest.json.tmp"
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp.rename(run_dir / "final_manifest.json")
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": manifest.run_id, "status": "complete"}, indent=2),
        encoding="utf-8",
    )


def _write_valid_capture(
    capture_dir: Path,
    full_models: tuple[str, ...] = (_SOL, _LUNA),
    trace_model: str = _LUNA,
    patch_model: str = _LUNA,
    analyze_models: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """A coherent fake capture for the zero-spend pins: full expected row set
    for the declared matrix, real sha256 over on-disk fixture bytes,
    conservation-consistent cold/warm usage, CURRENT profile provenance, and a
    fully BOUND sanitized adjudication artifact (matching response IDs, window
    covering `created`, billed classes consistent with the pinned equation).
    Returns the manifest dict (also written) so tests can perturb one
    dimension at a time."""
    from outrider.llm.host_profiles import OPENAI_PROFILE  # noqa: PLC0415

    if analyze_models is None:
        analyze_models = tuple(m for m in full_models if m != _TERRA) or (_SOL,)
    results: dict[str, dict[str, object]] = {}
    for tag in sorted(_expected_probe_rows(full_models, trace_model, patch_model, analyze_models)):
        kind = tag.rsplit(":", 1)[1]
        usage: dict[str, object] = {"prompt_tokens": 2000, "completion_tokens": 50}
        if kind == "cold":
            usage["prompt_tokens_details"] = {"cached_tokens": 0, "cache_write_tokens": 1500}
            # total = prompt + completion: the writes-inside shape, which
            # supports read_usage's pinned prompt_minus_cached_minus_writes
            # equation (the shape the real capture exhibited).
            usage["total_tokens"] = 2000 + 50
        elif kind == "warm":
            usage["prompt_tokens_details"] = {"cached_tokens": 1500, "cache_write_tokens": 0}
            usage["total_tokens"] = 2050
        payload = json.dumps(
            {"id": f"chatcmpl-pin-{tag}", "created": _PIN_CREATED_EPOCH, "usage": usage},
            indent=2,
        )
        fixture_name = tag.replace(":", "_") + ".json"
        (capture_dir / fixture_name).write_text(payload, encoding="utf-8")
        results[tag] = {
            "ok": True,
            "required": True,
            "note": "pin",
            "fixture": fixture_name,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
    # A real (fake-content) raw export on disk: the gate hashes these ACTUAL
    # bytes — "0"*64 with no file was exactly the round-13 bypass.
    (capture_dir / "raw").mkdir(exist_ok=True)
    raw_export_payload = json.dumps({"pin": "raw usage export stand-in"}, indent=2)
    (capture_dir / "raw" / "usage_export.json").write_text(raw_export_payload, encoding="utf-8")
    artifact: dict[str, object] = {
        "adjudication_schema_version": _EXPECTED_ADJUDICATION_SCHEMA_VERSION,
        "equation": _READ_USAGE_PINNED_EQUATION,
        "adjudicated_by": "test",
        "count_reconciliation": None,
        "raw_export_sha256": hashlib.sha256(raw_export_payload.encode("utf-8")).hexdigest(),
        "window_utc": dict(_PIN_WINDOW),
        "models": {
            model: {
                "cold_response_id": f"chatcmpl-pin-{model}:cold",
                "warm_response_id": f"chatcmpl-pin-{model}:warm",
                # prompt_minus_cached_minus_writes on the cold call:
                # fresh = 2000 - 0 - 1500.
                "billed_fresh_input_tokens": 500,
                "billed_cache_write_tokens": 1500,
            }
            for model in full_models
        },
    }
    manifest: dict[str, object] = {
        "schema_version": _EXPECTED_MANIFEST_SCHEMA_VERSION,
        "probe_contract_version": _EXPECTED_PROBE_CONTRACT_VERSION,
        "base_url": OPENAI_PROFILE.base_url,
        "profile_contract_digest": OPENAI_PROFILE.profile_contract_digest,
        "full_matrix_models": list(full_models),
        "trace_model": trace_model,
        "patch_model": patch_model,
        "analyze_models": list(analyze_models),
        "results": results,
        "missing_rows": [],
        "all_required_passed": True,
        "conservation_adjudication": {
            "adjudication_file": "billing_adjudication.json",
            "raw_export_file": "raw/usage_export.json",
        },
    }
    _rewrite_artifact(capture_dir, manifest, artifact)
    return {"manifest": manifest, "artifact": artifact}


def _rewrite_manifest(capture_dir: Path, manifest: dict[str, object]) -> None:
    (capture_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _rewrite_artifact(
    capture_dir: Path, manifest: dict[str, object], artifact: dict[str, object]
) -> None:
    """Write the adjudication artifact, update the manifest pointer's sha256,
    and rewrite the manifest — keeping pointer and bytes coherent so pins
    perturb exactly one dimension."""
    payload = json.dumps(artifact, indent=2)
    adjudication = manifest["conservation_adjudication"]
    assert isinstance(adjudication, dict)
    (capture_dir / str(adjudication["adjudication_file"])).write_text(payload, encoding="utf-8")
    adjudication["adjudication_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _rewrite_manifest(capture_dir, manifest)


@pytest.fixture
def probe_capture_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh, coherent local capture for each zero-spend admission case."""
    module = sys.modules[_require_probe_manifest.__module__]
    monkeypatch.setattr(module, "_PROBE_MANIFEST", tmp_path / "manifest.json")
    _write_valid_capture(tmp_path)
    return tmp_path


def _rewrite_cold_usage(
    capture_dir: Path,
    manifest: dict[str, object],
    model: str,
    usage: dict[str, object],
) -> None:
    """Keep fixture bytes and their manifest digest coherent while varying usage."""
    payload = json.dumps(
        {"id": f"chatcmpl-pin-{model}:cold", "created": _PIN_CREATED_EPOCH, "usage": usage},
        indent=2,
    )
    fixture_name = f"{model}:cold".replace(":", "_") + ".json"
    (capture_dir / fixture_name).write_text(payload, encoding="utf-8")
    manifest["results"][f"{model}:cold"]["sha256"] = hashlib.sha256(  # type: ignore[index, call-overload]
        payload.encode("utf-8")
    ).hexdigest()


@pytest.fixture
def indeterminate_probe_capture_dir(probe_capture_dir: Path) -> Path:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    for model in (_SOL, _LUNA):
        _rewrite_cold_usage(
            probe_capture_dir,
            manifest,
            model,
            {
                "prompt_tokens": 2000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500},
            },
        )
    _rewrite_manifest(probe_capture_dir, manifest)
    return probe_capture_dir


def test_probe_manifest_requires_an_existing_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = sys.modules[_require_probe_manifest.__module__]
    monkeypatch.setattr(module, "_PROBE_MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(pytest.fail.Exception, match="manifest missing"):
        _require_probe_manifest()


def test_probe_manifest_accepts_a_current_coherent_capture(probe_capture_dir: Path) -> None:
    _require_probe_manifest()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        pytest.param(
            "all_required_passed", False, "all_required_passed is not true", id="failed-verdict"
        ),
        pytest.param("schema_version", 99, "unknown manifest shape", id="unknown-schema"),
        pytest.param("probe_contract_version", 0, "probe PROCEDURE change", id="old-procedure"),
        pytest.param(
            "probe_contract_version", 3, "probe PROCEDURE change", id="superseded-trace-grader"
        ),
        pytest.param("base_url", "https://evil.example/v1", "wrong-host evidence", id="wrong-host"),
        pytest.param("profile_contract_digest", "0" * 64, "stale evidence", id="stale-profile"),
    ],
)
def test_probe_manifest_rejects_invalid_admission_metadata(
    probe_capture_dir: Path, field: str, value: object, error: str
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = value
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match=error):
        _require_probe_manifest()


@pytest.mark.parametrize("missing", [True, False], ids=["missing-matrix", "duplicate-model"])
def test_probe_manifest_rejects_a_malformed_matrix(probe_capture_dir: Path, missing: bool) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    if missing:
        del manifest["full_matrix_models"]
    else:
        manifest["full_matrix_models"] = [_SOL, _SOL]
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="not a supported capture matrix"):
        _require_probe_manifest()


def test_probe_manifest_requires_serving_models_in_the_matrix(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["full_matrix_models"] = [_SOL]
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match=r"per-field\s+serving role"):
        _require_probe_manifest()


@pytest.mark.parametrize(
    ("field", "model"),
    [
        pytest.param("trace_model", _SOL, id="sol-cannot-serve-trace"),
        pytest.param("patch_model", _TERRA, id="undeclared-patch-model"),
    ],
)
def test_probe_manifest_rejects_an_invalid_node_serving_role(
    probe_capture_dir: Path, field: str, model: str
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = model
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match=r"per-field\s+serving role"):
        _require_probe_manifest()


def test_probe_manifest_rejects_an_undeclared_analyze_candidate(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["analyze_models"] = [_SOL, _TERRA]
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="analyze-candidate roles"):
        _require_probe_manifest()


def test_probe_manifest_accepts_a_field_specific_model_swap(probe_capture_dir: Path) -> None:
    _write_valid_capture(
        probe_capture_dir,
        full_models=(_SOL, _LUNA, _TERRA),
        trace_model=_TERRA,
        patch_model=_LUNA,
        analyze_models=(_SOL, _LUNA),
    )
    _require_probe_manifest()


def test_probe_manifest_accepts_a_dropped_incumbent_and_a_restored_default(
    probe_capture_dir: Path,
) -> None:
    _write_valid_capture(
        probe_capture_dir,
        full_models=(_SOL, _TERRA),
        trace_model=_TERRA,
        patch_model=_TERRA,
        analyze_models=(_SOL, _TERRA),
    )
    _require_probe_manifest()
    _write_valid_capture(probe_capture_dir)
    _require_probe_manifest()


def test_probe_manifest_requires_the_refusal_row(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    del manifest["results"][f"{_SOL}:refusal"]
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="refusal.*absent from manifest"):
        _require_probe_manifest()


def test_probe_manifest_requires_the_refusal_row_to_pass(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["results"][f"{_LUNA}:refusal"]["ok"] = False
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="not a passing required row"):
        _require_probe_manifest()


def test_probe_manifest_rejects_unexpected_rows(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["results"]["gpt-5.6-terra:envelope"] = {"ok": True, "required": True}
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="unexpected result rows"):
        _require_probe_manifest()


def test_probe_manifest_rejects_fixture_path_traversal(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["results"][f"{_SOL}:envelope"]["fixture"] = "../evil.json"
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="bare filename"):
        _require_probe_manifest()


def test_probe_manifest_rejects_an_absolute_artifact_path(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["conservation_adjudication"]["adjudication_file"] = "/etc/hostname"
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="bare filename"):
        _require_probe_manifest()


def test_probe_manifest_requires_an_adjudication_pointer(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["conservation_adjudication"] = {"adjudication_file": None}
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="not adjudicated"):
        _require_probe_manifest()


def test_probe_adjudication_requires_relevant_capture_binding(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    generic = {"source": "some export", "billed": {"fresh": 500, "write": 1500}}
    _rewrite_artifact(probe_capture_dir, manifest, generic)
    with pytest.raises(pytest.fail.Exception, match="integrity, not relevance"):
        _require_probe_manifest()


def test_probe_adjudication_requires_the_artifact_file(probe_capture_dir: Path) -> None:
    (probe_capture_dir / "billing_adjudication.json").unlink()
    with pytest.raises(pytest.fail.Exception, match="adjudication artifact.*is missing"):
        _require_probe_manifest()


def test_probe_adjudication_rejects_tampered_bytes_and_accepts_restoration(
    probe_capture_dir: Path,
) -> None:
    artifact = probe_capture_dir / "billing_adjudication.json"
    original = artifact.read_text(encoding="utf-8")
    artifact.write_text(original + " ", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="do not match adjudication_sha256"):
        _require_probe_manifest()
    artifact.write_text(original, encoding="utf-8")
    _require_probe_manifest()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        pytest.param("equation", None, "unknown conservation equation", id="unknown-equation"),
        pytest.param(
            "equation",
            "prompt_minus_cached",
            "shipped accounting disagrees",
            id="superseded-equation",
        ),
        pytest.param(
            "raw_export_sha256", "not-a-hash", "raw_export_sha256", id="malformed-export-digest"
        ),
        pytest.param(
            "adjudicated_by",
            {"raw_export": {"project_id": "proj_secret", "rows": [1, 2]}},
            "single-line string",
            id="nested-export",
        ),
        pytest.param("adjudicated_by", "x" * 500, "single-line string", id="bulk-adjudicator"),
    ],
)
def test_probe_adjudication_rejects_invalid_scalar_values(
    probe_capture_dir: Path, field: str, value: object, error: str
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    artifact[field] = value
    _rewrite_artifact(probe_capture_dir, manifest, artifact)
    with pytest.raises(pytest.fail.Exception, match=error):
        _require_probe_manifest()


@pytest.mark.parametrize(
    "raw_file", [None, "raw/../../etc/hostname"], ids=["missing-pointer", "path-traversal"]
)
def test_probe_manifest_requires_a_confined_raw_export_pointer(
    probe_capture_dir: Path, raw_file: str | None
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["conservation_adjudication"]["raw_export_file"] = raw_file
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="raw/<filename>"):
        _require_probe_manifest()


def test_probe_adjudication_requires_the_raw_export_file(probe_capture_dir: Path) -> None:
    (probe_capture_dir / "raw/usage_export.json").unlink()
    with pytest.raises(pytest.fail.Exception, match="raw billing export.*is missing"):
        _require_probe_manifest()


def test_probe_adjudication_rejects_tampered_raw_bytes_and_accepts_restoration(
    probe_capture_dir: Path,
) -> None:
    raw_export = probe_capture_dir / "raw/usage_export.json"
    original = raw_export.read_text(encoding="utf-8")
    raw_export.write_text(original + " ", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="does not describe this export"):
        _require_probe_manifest()
    raw_export.write_text(original, encoding="utf-8")
    _require_probe_manifest()


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        pytest.param("artifact", "organization_id", "org-abc123", id="organization-identifier"),
        pytest.param("window", "project_id", "proj_secret", id="window-project-identifier"),
        pytest.param("model", "amount_usd", 0.42, id="model-dollar-amount"),
    ],
)
def test_probe_adjudication_rejects_extra_keys_at_each_level(
    probe_capture_dir: Path, scope: str, field: str, value: object
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    fields = artifact
    if scope == "window":
        fields = artifact["window_utc"]
    elif scope == "model":
        fields = artifact["models"][_SOL]
    fields[field] = value
    _rewrite_artifact(probe_capture_dir, manifest, artifact)
    with pytest.raises(pytest.fail.Exception, match="sanitization violation"):
        _require_probe_manifest()


def test_probe_adjudication_rejects_unearned_reconciliation(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    artifact["count_reconciliation"] = {_SOL: "wire_omitted_total_tokens"}
    _rewrite_artifact(probe_capture_dir, manifest, artifact)
    with pytest.raises(pytest.fail.Exception, match="unearned"):
        _require_probe_manifest()


def test_probe_adjudication_rejects_an_extra_model(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    artifact["models"]["gpt-5.6-terra"] = dict(artifact["models"][_SOL])
    _rewrite_artifact(probe_capture_dir, manifest, artifact)
    with pytest.raises(pytest.fail.Exception, match="exactly the declared full-matrix entries"):
        _require_probe_manifest()


@pytest.mark.parametrize(
    ("model", "field", "value", "error"),
    [
        pytest.param(
            _SOL,
            "cold_response_id",
            "chatcmpl-other-run",
            "not bound to THIS capture",
            id="wrong-response-id",
        ),
        pytest.param(
            _LUNA,
            "billed_cache_write_tokens",
            999,
            "billed cache-write count",
            id="wrong-billed-writes",
        ),
        pytest.param(
            _SOL,
            "billed_fresh_input_tokens",
            2000,
            "inconsistent with.*adjudicated equation",
            id="fresh-count-from-old-equation",
        ),
    ],
)
def test_probe_adjudication_rejects_model_evidence_that_disagrees_with_the_capture(
    probe_capture_dir: Path, model: str, field: str, value: object, error: str
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    artifact["models"][model][field] = value
    _rewrite_artifact(probe_capture_dir, manifest, artifact)
    with pytest.raises(pytest.fail.Exception, match=error):
        _require_probe_manifest()


@pytest.mark.parametrize(
    ("window", "error"),
    [
        pytest.param(
            {"start_epoch": _PIN_CREATED_EPOCH + 1000, "end_epoch": _PIN_CREATED_EPOCH + 2000},
            "outside the.*billing window",
            id="excludes-capture",
        ),
        pytest.param(
            {"start_epoch": _PIN_CREATED_EPOCH - 90_000, "end_epoch": _PIN_CREATED_EPOCH + 90_000},
            "cannot bind evidence",
            id="unbounded-window",
        ),
    ],
)
def test_probe_adjudication_requires_a_bounded_window_covering_the_capture(
    probe_capture_dir: Path, window: dict[str, int], error: str
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    artifact["window_utc"] = window
    _rewrite_artifact(probe_capture_dir, manifest, artifact)
    with pytest.raises(pytest.fail.Exception, match=error):
        _require_probe_manifest()


def test_probe_capture_rejects_usage_outside_conservation_bounds(probe_capture_dir: Path) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    _rewrite_cold_usage(
        probe_capture_dir,
        manifest,
        _SOL,
        {"prompt_tokens": 100, "prompt_tokens_details": {"cache_write_tokens": 1500}},
    )
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="conservation violated"):
        _require_probe_manifest()


def test_probe_capture_rejects_counts_contradicting_the_adjudicated_equation(
    probe_capture_dir: Path,
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    _rewrite_cold_usage(
        probe_capture_dir,
        manifest,
        _SOL,
        {
            "prompt_tokens": 2000,
            "completion_tokens": 50,
            "total_tokens": 3550,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500},
        },
    )
    _rewrite_manifest(probe_capture_dir, manifest)
    with pytest.raises(pytest.fail.Exception, match="CONTRADICT the adjudicated equation"):
        _require_probe_manifest()


@pytest.mark.parametrize(
    ("reconciliation", "error"),
    [
        pytest.param(None, "mapping EXACTLY the indeterminate", id="no-acknowledgment"),
        pytest.param(
            "wire_omitted_total_tokens", "mapping EXACTLY the indeterminate", id="global-scalar"
        ),
        pytest.param(
            {_SOL: "cold_warm_pair_incoherent", _LUNA: "cold_warm_pair_incoherent"},
            "evidence-derived",
            id="wrong-per-model-causes",
        ),
    ],
)
def test_probe_adjudication_rejects_incorrect_indeterminate_reconciliation(
    indeterminate_probe_capture_dir: Path, reconciliation: object, error: str
) -> None:
    manifest = json.loads(
        (indeterminate_probe_capture_dir / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = json.loads(
        (indeterminate_probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    artifact["count_reconciliation"] = reconciliation
    _rewrite_artifact(indeterminate_probe_capture_dir, manifest, artifact)
    with pytest.raises(pytest.fail.Exception, match=error):
        _require_probe_manifest()


def test_probe_adjudication_accepts_evidence_matching_reconciliation_for_both_models(
    indeterminate_probe_capture_dir: Path,
) -> None:
    manifest = json.loads(
        (indeterminate_probe_capture_dir / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = json.loads(
        (indeterminate_probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    artifact["count_reconciliation"] = {
        _SOL: "wire_omitted_total_tokens",
        _LUNA: "wire_omitted_total_tokens",
    }
    _rewrite_artifact(indeterminate_probe_capture_dir, manifest, artifact)
    _require_probe_manifest()


@pytest.mark.parametrize(
    "acknowledge_luna", [True, False], ids=["reject-extra-determinate-model", "accept-sol-only"]
)
def test_probe_adjudication_reconciliation_covers_only_indeterminate_models(
    probe_capture_dir: Path, acknowledge_luna: bool
) -> None:
    manifest = json.loads((probe_capture_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (probe_capture_dir / "billing_adjudication.json").read_text(encoding="utf-8")
    )
    _rewrite_cold_usage(
        probe_capture_dir,
        manifest,
        _SOL,
        {
            "prompt_tokens": 2000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500},
        },
    )
    reconciliation = {_SOL: "wire_omitted_total_tokens"}
    if acknowledge_luna:
        reconciliation[_LUNA] = "wire_omitted_total_tokens"
    artifact["count_reconciliation"] = reconciliation
    _rewrite_artifact(probe_capture_dir, manifest, artifact)
    if acknowledge_luna:
        with pytest.raises(pytest.fail.Exception, match="mapping EXACTLY the indeterminate"):
            _require_probe_manifest()
    else:
        _require_probe_manifest()


def test_candidate_wire_admission_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-spend pin for the manifest-to-candidate binding the paid test
    runs: a capture that declared (sol, luna) REFUSES a Terra candidate for
    either field, and admits the candidates it declared — per-model wire
    admission carried through the consumer, exercised without spend."""

    mod = sys.modules[_require_probe_manifest.__module__]
    monkeypatch.setattr(mod, "_PROBE_MANIFEST", tmp_path / "manifest.json")
    _write_valid_capture(tmp_path)
    manifest = _require_probe_manifest()

    _require_candidates_wire_admitted(manifest, _SOL, _LUNA)  # declared pair admits
    with pytest.raises(pytest.fail.Exception, match="not among the verified capture"):
        _require_candidates_wire_admitted(manifest, _TERRA, _LUNA)
    with pytest.raises(pytest.fail.Exception, match="not among the verified capture"):
        _require_candidates_wire_admitted(manifest, _SOL, _TERRA)

    # A swap capture that DECLARED Terra admits it.
    _write_valid_capture(
        tmp_path,
        full_models=(_SOL, _TERRA),
        trace_model=_TERRA,
        patch_model=_TERRA,
        analyze_models=(_SOL, _TERRA),
    )
    _require_candidates_wire_admitted(_require_probe_manifest(), _SOL, _TERRA)


def test_approved_baselines_binding() -> None:
    """Zero-spend pins for the approved-baseline gate: the canonical pair
    (ModelConfig FIELD defaults — the single model authority) is accepted;
    a divergent deep or standard baseline (what ambient OUTRIDER_MODEL_*
    overrides would resolve) refuses before any spend. The paid test calls
    this BEFORE provider construction."""
    deep_default = ModelConfig.model_fields["analyze_model"].default
    standard_default = ModelConfig.model_fields["standard_analyze_model"].default
    _require_approved_baselines(deep_default, standard_default)  # canonical admits
    with pytest.raises(pytest.fail.Exception, match="approved comparison"):
        _require_approved_baselines("claude-sonnet-4-6", standard_default)
    with pytest.raises(pytest.fail.Exception, match="approved comparison"):
        _require_approved_baselines(deep_default, "claude-sonnet-5")


def test_openai_billing_preflight_fails_when_unadjudicated() -> None:
    """Zero-spend pin (FUP-247): while openai is in COST_UNADJUDICATED_PROFILE_IDS the
    paid scorecard's pre-spend preflight FAILS (not skips), so an opted-in admission run
    cannot exit clean without ever producing evidence. Mirrors the manifest gate: once
    the operator opts into spend, a missing prerequisite fails loud."""
    import outrider.llm.pricing as _pricing_mod  # noqa: PLC0415

    assert "openai" in _pricing_mod.COST_UNADJUDICATED_PROFILE_IDS  # the current no-ship state
    with pytest.raises(pytest.fail.Exception, match="billing is unadjudicated"):
        _require_openai_billing_adjudicated()


def test_openai_billing_preflight_passes_when_adjudicated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-spend pin: once billing is adjudicated (openai removed from the set — the
    FUP-247 exit), the preflight passes and the paid scorecard is re-enabled. A LIVE
    module read, so no code edit is needed beyond dropping openai from the set."""
    monkeypatch.setattr("outrider.llm.pricing.COST_UNADJUDICATED_PROFILE_IDS", frozenset())
    _require_openai_billing_adjudicated()  # no raise


def test_openai_admission_credentials_fail_on_missing_or_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-spend pins for the SHARED preflight all three OpenAI admission runners use.
    An opted-in run FAILS (not skips) when EITHER vendor's key is absent OR left as an
    unresolved `op://` reference — the anthropic `op://` case is the round-3 hole (only
    the openai side rejected it before). Symmetric: both vendors, both failure shapes."""
    good = "sk-real-key-123456"
    # ANTHROPIC absent, then unresolved op:// (openai fine) — both FAIL, naming anthropic.
    monkeypatch.setenv("OPENAI_API_KEY", good)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(pytest.fail.Exception, match="ANTHROPIC_API_KEY"):
        _require_openai_admission_credentials()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "op://vault/anthropic/key")
    with pytest.raises(pytest.fail.Exception, match="ANTHROPIC_API_KEY"):
        _require_openai_admission_credentials()
    # OPENAI absent, then unresolved op:// (anthropic fine) — both FAIL, naming openai.
    monkeypatch.setenv("ANTHROPIC_API_KEY", good)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(pytest.fail.Exception, match="OPENAI_API_KEY"):
        _require_openai_admission_credentials()
    monkeypatch.setenv("OPENAI_API_KEY", "op://vault/openai/key")
    with pytest.raises(pytest.fail.Exception, match="OPENAI_API_KEY"):
        _require_openai_admission_credentials()


def test_openai_admission_credentials_pass_and_strip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-spend pin: with both keys resolved the preflight passes AND returns the
    whitespace-stripped values (surrounding whitespace is a common `op run` artifact)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-real-anthropic  ")
    monkeypatch.setenv("OPENAI_API_KEY", "\tsk-real-openai\n")
    assert _require_openai_admission_credentials() == ("sk-real-anthropic", "sk-real-openai")


def test_run_artifacts(tmp_path: Path) -> None:
    """Zero-spend pin for the simple run-artifact shape: a collision-refusing
    run dir with a partial marker; append-only raw results during execution;
    the validated final manifest written atomically only on completion — and
    a failed run leaves the partial marker + raw results, nothing more."""
    run_dir = _start_run(tmp_path, "r1")
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8")) == {
        "run_id": "r1",
        "status": "partial",
    }
    with pytest.raises(FileExistsError, match="already exists"):
        _start_run(tmp_path, "r1")

    _append_result(run_dir, {"scenario": "a", "ok": True})
    _append_result(run_dir, {"scenario": "b", "ok": False})
    raw = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(r)["scenario"] for r in raw] == ["a", "b"]

    # FAILED run: no final manifest is ever written — the partial marker and
    # the raw results ARE the record.
    assert not (run_dir / "final_manifest.json").exists()

    # INTERRUPTION mid-matrix: rows recorded so far survive because each is
    # written at record time, not replayed at the end.
    run2 = _start_run(tmp_path, "r2")
    with pytest.raises(ValueError, match="interrupted"):
        _append_result(run2, {"scenario": "first", "ok": True})
        raise ValueError("interrupted")  # the next comparison died
    survived = (run2 / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(r)["scenario"] for r in survived] == ["first"]
    assert not (run2 / "final_manifest.json").exists()
    assert json.loads((run2 / "status.json").read_text(encoding="utf-8"))["status"] == "partial"

    # COMPLETED run: the validated manifest lands atomically, marker flips.
    _finish_run(
        run_dir,
        _FinalRunManifest(
            run_id="r1",
            status="complete",
            columns=({"role": "deep", "candidate": _TERRA, "baseline": "claude-sonnet-5"},),
            scheduled_gate_entries=2,
            actual_gate_entries=2,
            git_sha="0" * 40,
            git_dirty=False,
            prompt_template_version="analyze-pin",
        ),
    )
    final = json.loads((run_dir / "final_manifest.json").read_text(encoding="utf-8"))
    assert final["status"] == "complete" and final["actual_gate_entries"] == 2
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "complete"
    assert not (run_dir / "final_manifest.json.tmp").exists()


@pytest.mark.asyncio
async def test_provider_nested_ownership() -> None:
    """Zero-spend pin for `_with_providers`: a candidate-constructor failure
    closes the already-constructed baseline exactly once, the candidate
    closer never runs, and the body (the paid comparisons) never starts; the
    clean path closes both."""
    closes: list[str] = []
    body_ran: list[bool] = []

    class _Fake:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            closes.append(self.name)

    async def _body(_b: object, _c: object) -> None:
        body_ran.append(True)

    def _boom() -> object:
        raise RuntimeError("candidate construction failed")

    with pytest.raises(RuntimeError, match="candidate construction failed"):
        await _with_providers(lambda: _Fake("baseline"), _boom, _body)
    assert closes == ["baseline"]  # exactly once; candidate closer never ran
    assert body_ran == []  # no comparison (no paid call) ever began

    closes.clear()
    await _with_providers(lambda: _Fake("baseline"), lambda: _Fake("candidate"), _body)
    assert body_ran == [True]
    assert closes == ["candidate", "baseline"]  # inner-out, both always


def test_provenance_captured_before_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-spend pin for the pre-spend provenance snapshot: git state is
    captured when the map is BUILT; a checkout change afterwards (mocked)
    must not leak into the emitted card — it describes the code that
    produced the calls."""
    from . import runner as runner_module  # noqa: PLC0415

    monkeypatch.setattr(runner_module, "_capture_git_state", lambda: ("a" * 40, False))
    captured = build_provenance(
        prompt_template_version="analyze-pin",
        scenario_labels=("fixtures/a.json",),
        baseline_model="claude-sonnet-5",
        candidate_models=(_TERRA,),
    )
    assert captured is not None
    # The tree "changes" mid-run:
    monkeypatch.setattr(runner_module, "_capture_git_state", lambda: ("b" * 40, True))
    card = Scorecard(
        rows=(
            ScorecardRow(
                node="analyze",
                model=_TERRA,
                scenario="fixtures/a.json",
                baseline_model="claude-sonnet-5",
                status="errored",
                error="pin",
            ),
        ),
        provenance=captured,
    )
    emitted = json.loads(card.to_json())
    assert emitted["provenance"]["git_sha"] == "a" * 40  # the PRE-SPEND snapshot
    assert emitted["provenance"]["git_dirty"] is False


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model GPT-5.6 scorecard spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1",
)
@pytest.mark.asyncio
async def test_gpt56_vs_anthropic_scorecard() -> None:
    """OPT-IN, real spend — the two-column GPT-5.6 candidate scorecard.

    REPORT-ONLY: asserts only that the run COMPLETED; the operator adjudicates
    per the spec's frozen rule (yield at the #059 bar AND the per-tier baseline
    recall floor; miss → Terra swap + rerun). Recall is TYPE-EXACT, so a delta
    can be a true miss OR a classification disagreement — read the printed
    `missed`/`extra` detail before acting. A non-conforming json_object
    response parses to no findings → recall 0, so the recall dimension also
    carries the yield signal.
    """
    # Pre-spend preflights: the operator opted into real spend (skipif gates ordinary
    # CI), so every missing prerequisite FAILS here (never skips — a skip would read as a
    # clean run that never happened) BEFORE any billed call / provider construction.
    anthropic_key, openai_key = _require_openai_admission_credentials()
    _require_openai_billing_adjudicated()
    capture_manifest = _require_probe_manifest()

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.host_profiles import OPENAI_PROFILE  # noqa: PLC0415
    from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider  # noqa: PLC0415

    cfg = ModelConfig()
    _require_approved_baselines(cfg.analyze_model, cfg.standard_analyze_model)
    persister = _NoOpExchangePersister()
    # Two candidate columns per the spec's evidence-domain rule: each analyze
    # field is judged against ITS incumbent, never one bar for both. The
    # candidates are ENV-SELECTABLE so the spec's Terra-swap workflow is
    # executable without a code edit: a field's miss reruns the scorecard
    # with that field's candidate swapped to Terra.
    deep_candidate = os.environ.get("OUTRIDER_SCORECARD_DEEP_CANDIDATE", _SOL)
    standard_candidate = os.environ.get("OUTRIDER_SCORECARD_STANDARD_CANDIDATE", _LUNA)
    if deep_candidate not in (_SOL, _TERRA):
        pytest.fail(f"OUTRIDER_SCORECARD_DEEP_CANDIDATE {deep_candidate!r} not in (Sol, Terra)")
    if standard_candidate not in (_LUNA, _TERRA):
        pytest.fail(
            f"OUTRIDER_SCORECARD_STANDARD_CANDIDATE {standard_candidate!r} not in (Luna, Terra)"
        )
    _require_candidates_wire_admitted(capture_manifest, deep_candidate, standard_candidate)
    # Columns carry the FIELD ROLE: ModelConfig permits independent baseline
    # overrides, so equal baselines + Terra in both fields would otherwise
    # collapse DEEP and STANDARD into one bucket.
    columns: tuple[tuple[str, str, str], ...] = (
        ("deep", deep_candidate, cfg.analyze_model),
        ("standard", standard_candidate, cfg.standard_analyze_model),
    )
    # Pre-spend plan: real counts from the imported catalog, printed BEFORE the
    # first paid call so the operator sees the true call volume, not a docstring
    # estimate. Exactly one gate entry per scheduled (scenario, column) — the
    # completion pin below holds this equality.
    recall_n = len(_GROUND_TRUTH_BY_FIXTURE)
    safe_n = len(_SAFE_CODE_FIXTURES)
    scheduled = len(columns) * (recall_n + safe_n)
    print(  # noqa: T201 — pre-spend operator plan
        f"\n[openai scorecard plan: {recall_n} recall + {safe_n} safe scenarios x "
        f"{len(columns)} candidate columns "
        f"({', '.join(f'{r}:{c} vs {b}' for r, c, b in columns)}) = {scheduled} comparisons "
        f"(~{2 * scheduled} paid calls incl. baselines)]"
    )

    # Pre-run partial marker: an interrupted run leaves this + the raw
    # results, and nothing pretends otherwise.
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = _start_run(Path("reports/scorecard"), run_id)

    # Provenance captured BEFORE any spend (the shared runner convention): if
    # the checkout changes during the ~128-call run, the cards must describe
    # the code that PRODUCED the calls, not the later tree.
    scenario_labels = tuple(_GROUND_TRUTH_BY_FIXTURE) + tuple(_SAFE_CODE_FIXTURES)
    provenance_by_column = {
        column: build_provenance(
            prompt_template_version=_ANALYZE_PROMPT_VERSION,
            scenario_labels=scenario_labels,
            baseline_model=column[2],
            candidate_models=(column[1],),
        )
        for column in columns
    }

    gate_results: list[tuple[str, str, bool, str]] = []

    def _record_gate(entry: tuple[str, str, bool, str]) -> None:
        """Append to memory AND immediately to results.jsonl — a process
        interruption keeps every completed row, not just the partial marker."""
        gate_results.append(entry)
        _append_result(
            run_dir,
            {"scenario": entry[0], "dimension": entry[1], "ok": entry[2], "label": entry[3]},
        )

    # Rows are kept PER COLUMN: `Scorecard` aggregates by (node, model), so a
    # both-fields-Terra rerun in one card would merge Terra-vs-Sonnet with
    # Terra-vs-Haiku — each column gets its own persisted card instead.
    rows_by_column: dict[tuple[str, str, str], list[ScorecardRow]] = {
        column: [] for column in columns
    }
    # Per-COLUMN comparison lists, keyed by (candidate, baseline): the
    # aggregate printer takes ONE candidate/baseline pair, each analyze field
    # is judged against ITS incumbent, and a both-fields-Terra rerun must not
    # collapse the two columns into one bucket.
    comparisons_by_column: dict[tuple[str, str, str], list[tuple[str, str, ModelComparison]]] = {
        column: [] for column in columns
    }
    aggregates: dict[tuple[str, str, str], dict[str, object] | None] = {}

    async def _compare_or_errored(
        fixture_path: str,
        ground_truth: tuple[ExpectedFinding, ...],
        dimension: str,
        *,
        candidate_model: str,
        baseline_model: str,
    ) -> ModelComparison | None:
        async def _compare() -> ModelComparison:
            return await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                ground_truth,
                baseline_provider=baseline_provider,
                baseline_model=baseline_model,
                candidate_provider=candidate_provider,
                candidate_model=candidate_model,
            )

        return await _run_scenario_isolating_transients(
            fixture_path, dimension, gate_results, _compare
        )

    async def _run_comparisons() -> None:
        for role, candidate_model, baseline_model in columns:
            for fixture_path, ground_truth in _GROUND_TRUTH_BY_FIXTURE.items():
                cmp = await _compare_or_errored(
                    fixture_path,
                    ground_truth,
                    # Column-qualified so an ERRORED gate entry names WHICH
                    # candidate's scenario needs the rerun.
                    f"recall:{role}:{candidate_model}-vs-{baseline_model}",
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                )
                if cmp is None:
                    # Transient ERRORED cell: the shared helper appended the
                    # gate entry to the list — mirror it to results.jsonl and
                    # materialize the errored row so the card stays complete.
                    _append_result(
                        run_dir,
                        {
                            "scenario": gate_results[-1][0],
                            "dimension": gate_results[-1][1],
                            "ok": gate_results[-1][2],
                            "label": gate_results[-1][3],
                        },
                    )
                    rows_by_column[(role, candidate_model, baseline_model)].append(
                        ScorecardRow.errored(
                            node="analyze",
                            scenario=fixture_path,
                            model=candidate_model,
                            baseline_model=baseline_model,
                            error=gate_results[-1][3],
                        )
                    )
                    continue
                _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
                rows_by_column[(role, candidate_model, baseline_model)].append(
                    ScorecardRow.from_comparison(
                        node="analyze",
                        scenario=fixture_path,
                        model=candidate_model,
                        baseline_model=baseline_model,
                        comparison=cmp,
                    )
                )
                recall_ok = cmp.recall_held and cmp.baseline_valid
                _record_gate(
                    (
                        fixture_path,
                        f"recall:{role}:{candidate_model}-vs-{baseline_model}",
                        recall_ok,
                        f"{candidate_model} recall < {baseline_model}",
                    )
                )
                comparisons_by_column[(role, candidate_model, baseline_model)].append(
                    (fixture_path, "recall", cmp)
                )
                assert cmp.baseline is not None  # the run completed
            for fixture_path in _SAFE_CODE_FIXTURES:
                cmp = await _compare_or_errored(
                    fixture_path,
                    (),
                    f"precision:{role}:{candidate_model}-vs-{baseline_model}",
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                )
                if cmp is None:
                    _append_result(
                        run_dir,
                        {
                            "scenario": gate_results[-1][0],
                            "dimension": gate_results[-1][1],
                            "ok": gate_results[-1][2],
                            "label": gate_results[-1][3],
                        },
                    )
                    rows_by_column[(role, candidate_model, baseline_model)].append(
                        ScorecardRow.errored(
                            node="analyze",
                            scenario=fixture_path,
                            model=candidate_model,
                            baseline_model=baseline_model,
                            error=gate_results[-1][3],
                        )
                    )
                    continue
                _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
                rows_by_column[(role, candidate_model, baseline_model)].append(
                    ScorecardRow.from_comparison(
                        node="analyze",
                        scenario=fixture_path,
                        model=candidate_model,
                        baseline_model=baseline_model,
                        comparison=cmp,
                    )
                )
                _record_gate(
                    (
                        fixture_path,
                        f"precision:{role}:{candidate_model}-vs-{baseline_model}",
                        cmp.fp_bounded,
                        f"{candidate_model} over-flags safe code",
                    )
                )
                comparisons_by_column[(role, candidate_model, baseline_model)].append(
                    (fixture_path, "precision", cmp)
                )
        for role, candidate_model, baseline_model in columns:
            aggregates[(role, candidate_model, baseline_model)] = _print_aggregate_metrics(
                comparisons_by_column[(role, candidate_model, baseline_model)],
                _GROUND_TRUTH_BY_FIXTURE,
                baseline_model,
                candidate_model,
            )

    # Nested ownership: each provider is owned the moment it exists — a
    # candidate-constructor failure closes the baseline and the comparisons
    # never start; results.jsonl was written per gate entry DURING the run,
    # so nothing is replayed here.
    async def _drive(baseline: object, candidate: object) -> None:
        nonlocal baseline_provider, candidate_provider
        baseline_provider = baseline
        candidate_provider = candidate
        await _run_comparisons()

    baseline_provider: object = None
    candidate_provider: object = None
    await _with_providers(
        lambda: AnthropicProvider(
            api_key=SecretStr(anthropic_key), model_config=cfg, persister=persister
        ),
        lambda: OpenAICompatibleProvider(
            api_key=SecretStr(openai_key),
            profile=OPENAI_PROFILE,
            persister=persister,
            models=tuple(dict.fromkeys(candidate for _role, candidate, _ in columns)),
        ),
        _drive,
    )

    assert len(gate_results) == scheduled, (
        f"scorecard incomplete: {len(gate_results)} gate entries for {scheduled} "
        f"scheduled (scenario, column) pairs — raw results + partial marker in {run_dir}"
    )

    # Success: per-column cards (each with the PRE-SPEND provenance snapshot),
    # aggregates, then the validated final manifest — written atomically.
    for (role, candidate, baseline), column_rows in rows_by_column.items():
        card = Scorecard(
            rows=tuple(column_rows),
            provenance=provenance_by_column[(role, candidate, baseline)],
        )
        stem = f"openai-gpt56-scorecard-{role}-{candidate}-vs-{baseline}"
        (run_dir / f"{stem}.json").write_text(card.to_json(), encoding="utf-8")
        (run_dir / f"{stem}.html").write_text(card.to_html(), encoding="utf-8")
        print(  # noqa: T201 — operator artifact pointer
            f"\n[scorecard column written to {run_dir}/{stem}.json + .html]"
        )
    for (role, candidate, baseline), aggregate in aggregates.items():
        if aggregate is not None:
            (run_dir / f"openai-gpt56-aggregate-{role}-{candidate}-vs-{baseline}.json").write_text(
                json.dumps(aggregate, indent=2), encoding="utf-8"
            )
    provenance_seed = provenance_by_column[columns[0]]
    assert provenance_seed is not None  # version is set, so build_provenance never None
    _finish_run(
        run_dir,
        _FinalRunManifest(
            run_id=run_id,
            status="complete",
            columns=tuple(
                {"role": role, "candidate": candidate, "baseline": baseline}
                for role, candidate, baseline in columns
            ),
            scheduled_gate_entries=scheduled,
            actual_gate_entries=len(gate_results),
            git_sha=provenance_seed.git_sha,
            git_dirty=provenance_seed.git_dirty,
            prompt_template_version=_ANALYZE_PROMPT_VERSION,
        ),
    )
    flagged = [(fx, dim, label) for fx, dim, ok, label in gate_results if not ok]
    print(  # noqa: T201 — operator gate summary (adjudication happens on the report)
        f"\n[openai scorecard: {len(gate_results) - len(flagged)} green / "
        f"{len(flagged)} flagged advisory gates — adjudicate per the spec's frozen rule]"
    )
