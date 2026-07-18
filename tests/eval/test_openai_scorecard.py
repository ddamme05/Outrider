"""Opt-in, real-spend: Anthropic-vs-GPT-5.6 scorecard over the analyze node.

The openai-native-host candidate gate (specs/2026-07-18-openai-native-host.md).
Runs the SAME recall + precision scenarios as the Sonnet-vs-Haiku model-tier
comparison (`test_model_comparison.py`) and the GLM scorecard
(`test_glm_scorecard.py`), with TWO candidate columns matching the spec's
evidence-domain rule — the scorecard canonizes exactly the two ANALYZE fields:

  - `gpt-5.6-sol`  vs the Anthropic DEEP-tier baseline (`cfg.analyze_model`)
  - `gpt-5.6-luna` vs the Anthropic STANDARD-tier baseline
    (`cfg.standard_analyze_model`)

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
gate per model; extra rows rejected), each fixture's sha256, the cold/warm
conservation BOUNDS recomputed from fixture bytes, and — because bounds cannot
choose between the spec's two accounting equations — the operator's
billing-verified `conservation_adjudication`, which must match the equation
`read_usage()` ships. A conformance surprise is caught on the probe's cheap
capture, never on this ~128-call run.

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
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from .model_comparison import compare_models_on_scenario, state_from_eval_fixture
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
    from .grading import ExpectedFinding, ModelComparison

_SOL = "gpt-5.6-sol"
_LUNA = "gpt-5.6-luna"

# The probe's success manifest — the enforced precondition for any paid run.
_PROBE_MANIFEST = (
    Path(__file__).resolve().parents[2] / "spikes" / "openai" / "fixtures" / "manifest.json"
)

# The probe matrix's tag set, hardcoded HERE deliberately (spikes/ is not
# importable from tests): a probe-matrix change without a gate update fails
# loud, which is the intended sync mechanism. Refusal rows are in the required
# set — the refusal-normalization fixture is a PRE-SHIP gate per model (spec
# "Gates before any production-shaped use"; wire admission is PER MODEL).
_EXPECTED_PROBE_ROWS: frozenset[str] = frozenset(
    f"{model}:{kind}" for model in (_SOL, _LUNA) for kind in ("envelope", "cold", "warm", "refusal")
) | {"gpt-5.6-terra:reasoning"}

# Pinned against the probe's PROBE_CONTRACT_VERSION / MANIFEST_SCHEMA_VERSION:
# a capture from an older probe PROCEDURE (different prompts, schema bytes,
# matrix, or predicates) or manifest shape must not admit, exactly as a
# stale profile digest must not.
_EXPECTED_PROBE_CONTRACT_VERSION = 1
_EXPECTED_MANIFEST_SCHEMA_VERSION = 1

# The conservation equation `read_usage()` currently implements for
# PROMPT_INCLUDES_CACHED_WRITES_REPORTED: input = prompt - cached, writes NOT
# subtracted (host_profiles.read_usage). The spec classifies the true equation
# as [probe] — count-undecidable from bounds, adjudicated by the operator
# against billed usage. If the adjudication lands on
# "prompt_minus_cached_minus_writes", read_usage + the pricing math change
# FIRST, then this pin with them — the gate refuses to admit a scorecard while
# the shipped accounting disagrees with the adjudicated wire.
_READ_USAGE_PINNED_EQUATION = "prompt_minus_cached"
_KNOWN_EQUATIONS = ("prompt_minus_cached", "prompt_minus_cached_minus_writes")


def _supported_by_counts(cold_usage: dict[str, object], warm_usage: dict[str, object]) -> str:
    """Which equation a model's hash-verified cold/warm fixture usage supports.
    Deliberately re-implements the probe's characterization (spikes/ is not
    importable from tests; independent recomputation is the point — the gate
    must not trust the manifest's own conservation_facts block). total_tokens
    is the disambiguator: == prompt + completion means writes ride INSIDE
    prompt_tokens (input must subtract them); == prompt + write + completion
    means writes are an additive class. Anything else — including an
    incoherent cold/warm prompt pair — is indeterminate, never a guess."""
    prompt = cold_usage.get("prompt_tokens")
    completion = cold_usage.get("completion_tokens")
    total = cold_usage.get("total_tokens")
    ptd = cold_usage.get("prompt_tokens_details")
    write = ptd.get("cache_write_tokens") if isinstance(ptd, dict) else None
    if (
        not all(isinstance(v, int) for v in (prompt, completion, total, write))
        or not write
        or warm_usage.get("prompt_tokens") != prompt
    ):
        return "indeterminate"
    assert isinstance(prompt, int) and isinstance(completion, int)  # narrowed above
    if total == prompt + completion:
        return "prompt_minus_cached_minus_writes"
    if total == prompt + write + completion:
        return "prompt_minus_cached"
    return "indeterminate"


def _require_probe_manifest() -> None:
    """FAIL (not skip) without a passing, coherent, CURRENT probe capture: the
    operator has explicitly opted into real spend, so a silent skip would read
    as a clean run. Beyond the probe's own verdict boolean, the gate verifies
    the capture's provenance (canonical base_url; profile digest + probe
    procedure/manifest versions), the EXACT expected row set, each fixture's
    existence + sha256, the cold/warm conservation bounds recomputed FROM THE
    FIXTURE BYTES, and the conservation adjudication BOUND to its evidence: a
    hash-pinned billing-export file, per-model count support recomputed from
    fixture bytes (contrary counts refuse outright; indeterminate counts need
    an explicit reconciliation), and equality with read_usage()'s shipped
    equation. (A determined forger can fabricate fixtures and hashes together;
    the gate's job is stale/partial/accidental artifacts, not adversarial
    operators — the operator IS the trust anchor.)"""
    from outrider.llm.host_profiles import OPENAI_PROFILE  # noqa: PLC0415

    def _fail(reason: str) -> NoReturn:
        pytest.fail(
            f"probe capture gate: {reason} — rerun the paid wire probe "
            "(op run --env-file=.env -- uv run python spikes/openai/probe.py) "
            f"and see {_PROBE_MANIFEST}"
        )

    if not _PROBE_MANIFEST.exists():
        _fail(f"success manifest missing at {_PROBE_MANIFEST}")
    manifest = json.loads(_PROBE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _EXPECTED_MANIFEST_SCHEMA_VERSION:
        _fail(
            f"manifest schema_version {manifest.get('schema_version')!r} != "
            f"{_EXPECTED_MANIFEST_SCHEMA_VERSION} — unknown manifest shape"
        )
    if manifest.get("probe_contract_version") != _EXPECTED_PROBE_CONTRACT_VERSION:
        _fail(
            f"probe_contract_version {manifest.get('probe_contract_version')!r} != "
            f"{_EXPECTED_PROBE_CONTRACT_VERSION} — the capture predates a probe "
            "PROCEDURE change (prompts/schema/matrix/predicates): stale evidence"
        )
    if manifest.get("all_required_passed") is not True:
        _fail("all_required_passed is not true")
    if manifest.get("base_url") != OPENAI_PROFILE.base_url:
        _fail(
            f"manifest base_url {manifest.get('base_url')!r} is not the canonical "
            f"{OPENAI_PROFILE.base_url!r} — wrong-host evidence"
        )
    if manifest.get("profile_contract_digest") != OPENAI_PROFILE.profile_contract_digest:
        _fail(
            "manifest profile_contract_digest does not match the CURRENT profile — "
            "the capture predates a wire-affecting profile change (stale evidence)"
        )
    results = manifest.get("results") or {}
    extra_rows = sorted(set(results) - _EXPECTED_PROBE_ROWS)
    if extra_rows:
        _fail(
            f"unexpected result rows {extra_rows} — the row set is exact; extras mean "
            "probe-procedure drift or a hand-edited manifest"
        )
    # Hash-verified usage stash for the post-loop equation recomputation: the
    # adjudication is judged against fixture BYTES, never against the
    # manifest's own (independently editable) conservation_facts block.
    usage_by_row: dict[str, dict[str, object]] = {}
    for tag in sorted(_EXPECTED_PROBE_ROWS):
        row = results.get(tag)
        if not isinstance(row, dict):
            _fail(f"expected probe row {tag!r} absent from manifest")
        if row.get("required") is not True or row.get("ok") is not True:
            _fail(f"probe row {tag!r} is not a passing required row: {row}")
        fixture_name = row.get("fixture")
        recorded_sha = row.get("sha256")
        if not fixture_name or not recorded_sha:
            _fail(f"probe row {tag!r} carries no fixture/sha256 provenance")
        fixture_path = _PROBE_MANIFEST.parent / str(fixture_name)
        if not fixture_path.exists():
            _fail(f"fixture {fixture_name!r} for row {tag!r} is missing")
        fixture_bytes = fixture_path.read_bytes()
        if hashlib.sha256(fixture_bytes).hexdigest() != recorded_sha:
            _fail(f"fixture {fixture_name!r} bytes do not match the manifest sha256")
        kind = tag.rsplit(":", 1)[1]
        if kind in ("cold", "warm"):
            usage = json.loads(fixture_bytes.decode("utf-8")).get("usage") or {}
            usage_by_row[tag] = usage
            prompt = usage.get("prompt_tokens")
            ptd = usage.get("prompt_tokens_details") or {}
            side = "cache_write_tokens" if kind == "cold" else "cached_tokens"
            value = ptd.get(side)
            if not isinstance(prompt, int) or not isinstance(value, int):
                _fail(f"fixture {fixture_name!r} lacks integer usage for {side}")
            if not (0 < value <= prompt):
                _fail(
                    f"conservation violated in {fixture_name!r}: {side}={value} "
                    f"vs prompt_tokens={prompt} (expected 0 < {side} <= prompt)"
                )
    # The bounds above catch malformed wire; they CANNOT choose between the
    # spec's two accounting equations. Admission requires the operator's
    # billing-verified adjudication, BOUND to its evidence three ways: a
    # hash-pinned billing-export file, per-model count support recomputed from
    # the (hash-verified) fixture bytes, and an explicit reconciliation when
    # any model's counts are indeterminate. Contrary counts refuse outright.
    adjudication = manifest.get("conservation_adjudication") or {}
    equation = adjudication.get("equation")
    if equation is None:
        _fail(
            "conservation equation not adjudicated — read conservation_facts, "
            "cross-check billed usage, and fill conservation_adjudication "
            "(equation/evidence_file/evidence_sha256/adjudicated_by) in the manifest"
        )
    if equation not in _KNOWN_EQUATIONS:
        _fail(f"unknown conservation equation {equation!r} (expected one of {_KNOWN_EQUATIONS})")
    if equation != _READ_USAGE_PINNED_EQUATION:
        _fail(
            f"adjudicated equation {equation!r} != read_usage()'s "
            f"{_READ_USAGE_PINNED_EQUATION!r} — the shipped accounting disagrees with "
            "the wire; change read_usage + pricing first, then update this pin"
        )
    evidence_file = adjudication.get("evidence_file")
    if not evidence_file or not adjudication.get("adjudicated_by"):
        _fail(
            "conservation_adjudication must carry non-empty evidence_file and "
            "adjudicated_by — a bare assertion is not billing evidence"
        )
    evidence_path = _PROBE_MANIFEST.parent / str(evidence_file)
    if not evidence_path.exists():
        _fail(f"billing-evidence file {evidence_file!r} is missing from the capture dir")
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != adjudication.get(
        "evidence_sha256"
    ):
        _fail(f"billing-evidence file {evidence_file!r} bytes do not match evidence_sha256")
    support_by_model = {
        model: _supported_by_counts(
            usage_by_row.get(f"{model}:cold") or {}, usage_by_row.get(f"{model}:warm") or {}
        )
        for model in (_SOL, _LUNA)
    }
    contrary = {m: s for m, s in support_by_model.items() if s not in ("indeterminate", equation)}
    if contrary:
        _fail(
            f"fixture counts CONTRADICT the adjudicated equation {equation!r}: {contrary} — "
            "billed evidence and wire counts disagree; re-probe or re-adjudicate before "
            "any scorecard run"
        )
    indeterminate = sorted(m for m, s in support_by_model.items() if s == "indeterminate")
    if indeterminate and not adjudication.get("count_reconciliation"):
        _fail(
            f"count support is indeterminate for {indeterminate} and "
            "conservation_adjudication.count_reconciliation is empty — state why the "
            "billing evidence alone settles the equation for these models"
        )


def _write_valid_capture(capture_dir: Path) -> dict[str, object]:
    """A coherent fake capture for the zero-spend pins: full expected row set,
    real sha256 over on-disk fixture bytes, conservation-consistent cold/warm
    usage, CURRENT profile provenance. Returns the manifest dict (also written)
    so tests can perturb one dimension at a time."""
    from outrider.llm.host_profiles import OPENAI_PROFILE  # noqa: PLC0415

    results: dict[str, dict[str, object]] = {}
    for tag in sorted(_EXPECTED_PROBE_ROWS):
        kind = tag.rsplit(":", 1)[1]
        usage: dict[str, object] = {"prompt_tokens": 2000, "completion_tokens": 50}
        if kind == "cold":
            usage["prompt_tokens_details"] = {"cached_tokens": 0, "cache_write_tokens": 1500}
            # total = prompt + write + completion: the additive shape, which
            # supports read_usage's pinned prompt_minus_cached equation.
            usage["total_tokens"] = 2000 + 1500 + 50
        elif kind == "warm":
            usage["prompt_tokens_details"] = {"cached_tokens": 1500, "cache_write_tokens": 0}
            usage["total_tokens"] = 2050
        payload = json.dumps({"usage": usage}, indent=2)
        fixture_name = tag.replace(":", "_") + ".json"
        (capture_dir / fixture_name).write_text(payload, encoding="utf-8")
        results[tag] = {
            "ok": True,
            "required": True,
            "note": "pin",
            "fixture": fixture_name,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
    evidence_payload = json.dumps(
        {"source": "zero-spend pin", "billed_input_class_tokens": {"fresh": 500, "write": 1500}},
        indent=2,
    )
    (capture_dir / "billing_evidence.json").write_text(evidence_payload, encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": _EXPECTED_MANIFEST_SCHEMA_VERSION,
        "probe_contract_version": _EXPECTED_PROBE_CONTRACT_VERSION,
        "base_url": OPENAI_PROFILE.base_url,
        "profile_contract_digest": OPENAI_PROFILE.profile_contract_digest,
        "results": results,
        "missing_rows": [],
        "all_required_passed": True,
        "conservation_adjudication": {
            "equation": _READ_USAGE_PINNED_EQUATION,
            "evidence_file": "billing_evidence.json",
            "evidence_sha256": hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest(),
            "adjudicated_by": "test",
            "count_reconciliation": None,
        },
    }
    (capture_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _rewrite_manifest(capture_dir: Path, manifest: dict[str, object]) -> None:
    (capture_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_probe_manifest_precondition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-spend pins for the paid-run gate, one per admission dimension: a
    coherent CURRENT capture admits; missing manifest, failed verdict, unknown
    manifest shape, stale probe-procedure version, wrong host, stale profile
    digest, a dropped required row (refusal included — the pre-ship gate), a
    non-passing row, an EXTRA row, a missing/tampered fixture, a conservation
    bounds violation, an unadjudicated / mismatched equation, a missing /
    tampered / assertion-only billing-evidence file, fixture counts that
    CONTRADICT the verdict, and indeterminate counts without reconciliation
    each FAIL (reconciled indeterminate admits). Fail (not skip) because the
    operator explicitly opted into spend — a silent skip would read as a clean
    run."""
    import copy  # noqa: PLC0415
    import sys  # noqa: PLC0415

    mod = sys.modules[_require_probe_manifest.__module__]
    monkeypatch.setattr(mod, "_PROBE_MANIFEST", tmp_path / "manifest.json")

    with pytest.raises(pytest.fail.Exception, match="manifest missing"):
        _require_probe_manifest()

    valid = _write_valid_capture(tmp_path)
    _require_probe_manifest()  # the coherent capture admits

    broken = copy.deepcopy(valid)
    broken["all_required_passed"] = False
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="all_required_passed is not true"):
        _require_probe_manifest()

    broken = copy.deepcopy(valid)
    broken["schema_version"] = 99
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="unknown manifest shape"):
        _require_probe_manifest()

    # A capture from an older probe PROCEDURE (prompts/matrix/predicates) is
    # stale evidence even when the profile digest still matches.
    broken = copy.deepcopy(valid)
    broken["probe_contract_version"] = 0
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="probe PROCEDURE change"):
        _require_probe_manifest()

    broken = copy.deepcopy(valid)
    broken["base_url"] = "https://evil.example/v1"
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="wrong-host evidence"):
        _require_probe_manifest()

    broken = copy.deepcopy(valid)
    broken["profile_contract_digest"] = "0" * 64
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="stale evidence"):
        _require_probe_manifest()

    # Refusal is a REQUIRED row: a capture missing it must not admit.
    broken = copy.deepcopy(valid)
    del broken["results"][f"{_SOL}:refusal"]  # type: ignore[index, arg-type]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="refusal.*absent from manifest"):
        _require_probe_manifest()

    broken = copy.deepcopy(valid)
    broken["results"][f"{_LUNA}:refusal"]["ok"] = False  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="not a passing required row"):
        _require_probe_manifest()

    # The row set is EXACT: an extra row means procedure drift or a hand edit.
    broken = copy.deepcopy(valid)
    broken["results"]["gpt-5.6-terra:envelope"] = {"ok": True, "required": True}  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="unexpected result rows"):
        _require_probe_manifest()

    # Bounds cannot choose the accounting equation: an unadjudicated capture
    # (equation null) and a mismatched adjudication both refuse admission.
    broken = copy.deepcopy(valid)
    broken["conservation_adjudication"]["equation"] = None  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="not adjudicated"):
        _require_probe_manifest()

    broken = copy.deepcopy(valid)
    broken["conservation_adjudication"]["equation"] = "prompt_minus_cached_minus_writes"  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="shipped accounting disagrees"):
        _require_probe_manifest()

    # The adjudication is BOUND to a durable, hash-pinned billing-evidence
    # file: a bare assertion string, a missing file, and tampered bytes all
    # refuse admission.
    broken = copy.deepcopy(valid)
    broken["conservation_adjudication"]["evidence_file"] = ""  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="non-empty evidence_file"):
        _require_probe_manifest()

    _rewrite_manifest(tmp_path, valid)
    evidence_file = tmp_path / "billing_evidence.json"
    evidence_original = evidence_file.read_text(encoding="utf-8")
    evidence_file.unlink()
    with pytest.raises(pytest.fail.Exception, match="billing-evidence file.*is missing"):
        _require_probe_manifest()
    evidence_file.write_text(evidence_original + " ", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="do not match evidence_sha256"):
        _require_probe_manifest()
    evidence_file.write_text(evidence_original, encoding="utf-8")
    _require_probe_manifest()  # restored evidence admits again

    _rewrite_manifest(tmp_path, valid)
    envelope_fixture = tmp_path / (f"{_SOL}:envelope".replace(":", "_") + ".json")
    original = envelope_fixture.read_text(encoding="utf-8")
    envelope_fixture.unlink()
    with pytest.raises(pytest.fail.Exception, match="is missing"):
        _require_probe_manifest()
    envelope_fixture.write_text(original + " ", encoding="utf-8")  # tampered bytes
    with pytest.raises(pytest.fail.Exception, match="do not match the manifest sha256"):
        _require_probe_manifest()
    envelope_fixture.write_text(original, encoding="utf-8")
    _require_probe_manifest()  # restored capture admits again

    def _set_cold_usage(manifest: dict[str, object], model: str, usage: dict[str, object]) -> None:
        payload = json.dumps({"usage": usage}, indent=2)
        (tmp_path / (f"{model}:cold".replace(":", "_") + ".json")).write_text(
            payload, encoding="utf-8"
        )
        manifest["results"][f"{model}:cold"]["sha256"] = hashlib.sha256(  # type: ignore[index, call-overload]
            payload.encode("utf-8")
        ).hexdigest()

    broken = copy.deepcopy(valid)
    _set_cold_usage(
        broken,
        _SOL,
        {"prompt_tokens": 100, "prompt_tokens_details": {"cache_write_tokens": 1500}},
    )
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="conservation violated"):
        _require_probe_manifest()

    # Counts that CONTRADICT the adjudicated equation refuse outright — the
    # writes-inside shape (total == prompt + completion) against the pinned
    # prompt_minus_cached verdict is exactly the billed-vs-wire disagreement
    # that must stop a paid run.
    broken = copy.deepcopy(valid)
    _set_cold_usage(
        broken,
        _SOL,
        {
            "prompt_tokens": 2000,
            "completion_tokens": 50,
            "total_tokens": 2050,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500},
        },
    )
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="CONTRADICT the adjudicated equation"):
        _require_probe_manifest()

    # Indeterminate counts (no total_tokens on the wire) demand an explicit
    # reconciliation naming why billing evidence alone settles it; with the
    # reconciliation present, the capture admits.
    broken = copy.deepcopy(valid)
    for model in (_SOL, _LUNA):
        _set_cold_usage(
            broken,
            model,
            {
                "prompt_tokens": 2000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500},
            },
        )
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="count_reconciliation is empty"):
        _require_probe_manifest()
    broken["conservation_adjudication"]["count_reconciliation"] = (  # type: ignore[index, call-overload]
        "wire omitted total_tokens; billed class breakdown alone supports the verdict"
    )
    _rewrite_manifest(tmp_path, broken)
    _require_probe_manifest()  # reconciled indeterminate capture admits


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
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the Anthropic baselines")
    if not openai_key or openai_key.startswith("op://"):
        pytest.skip(
            "OPENAI_API_KEY (resolved, not an op:// ref) is required for the GPT-5.6 "
            "candidates; run under `op run --env-file=.env -- ...`"
        )
    _require_probe_manifest()

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.host_profiles import OPENAI_PROFILE  # noqa: PLC0415
    from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider  # noqa: PLC0415

    cfg = ModelConfig()
    persister = _NoOpExchangePersister()
    baseline_provider = AnthropicProvider(
        api_key=SecretStr(anthropic_key), model_config=cfg, persister=persister
    )
    candidate_provider = OpenAICompatibleProvider(
        api_key=SecretStr(openai_key),
        profile=OPENAI_PROFILE,
        persister=persister,
        models=(_SOL, _LUNA),
    )

    # Two candidate columns per the spec's evidence-domain rule: each analyze
    # field is judged against ITS incumbent, never one bar for both.
    columns: tuple[tuple[str, str], ...] = (
        (_SOL, cfg.analyze_model),
        (_LUNA, cfg.standard_analyze_model),
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
        f"{len(columns)} candidate columns = {scheduled} comparisons "
        f"(~{2 * scheduled} paid calls incl. baselines)]"
    )

    gate_results: list[tuple[str, str, bool, str]] = []
    rows: list[ScorecardRow] = []
    # Per-column comparison lists: the aggregate printer takes ONE candidate/baseline
    # pair, so each column aggregates separately (the spec judges each analyze field
    # against ITS incumbent).
    comparisons_by_column: dict[str, list[tuple[str, str, ModelComparison]]] = {
        _SOL: [],
        _LUNA: [],
    }
    aggregates: dict[str, dict[str, object] | None] = {}

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

    try:
        for candidate_model, baseline_model in columns:
            for fixture_path, ground_truth in _GROUND_TRUTH_BY_FIXTURE.items():
                cmp = await _compare_or_errored(
                    fixture_path,
                    ground_truth,
                    # Column-qualified so an ERRORED gate entry names WHICH
                    # candidate's scenario needs the rerun.
                    f"recall:{candidate_model}",
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                )
                if cmp is None:
                    continue
                _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
                rows.append(
                    ScorecardRow.from_comparison(
                        node="analyze",
                        scenario=fixture_path,
                        model=candidate_model,
                        baseline_model=baseline_model,
                        comparison=cmp,
                    )
                )
                recall_ok = cmp.recall_held and cmp.baseline_valid
                gate_results.append(
                    (
                        fixture_path,
                        f"recall:{candidate_model}",
                        recall_ok,
                        f"{candidate_model} recall < {baseline_model}",
                    )
                )
                comparisons_by_column[candidate_model].append((fixture_path, "recall", cmp))
                assert cmp.baseline is not None  # the run completed
            for fixture_path in _SAFE_CODE_FIXTURES:
                cmp = await _compare_or_errored(
                    fixture_path,
                    (),
                    f"precision:{candidate_model}",
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                )
                if cmp is None:
                    continue
                _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
                rows.append(
                    ScorecardRow.from_comparison(
                        node="analyze",
                        scenario=fixture_path,
                        model=candidate_model,
                        baseline_model=baseline_model,
                        comparison=cmp,
                    )
                )
                gate_results.append(
                    (
                        fixture_path,
                        f"precision:{candidate_model}",
                        cmp.fp_bounded,
                        f"{candidate_model} over-flags safe code",
                    )
                )
                comparisons_by_column[candidate_model].append((fixture_path, "precision", cmp))
        for candidate_model, baseline_model in columns:
            aggregates[candidate_model] = _print_aggregate_metrics(
                comparisons_by_column[candidate_model],
                _GROUND_TRUTH_BY_FIXTURE,
                baseline_model,
                candidate_model,
            )
    finally:
        # Persist in `finally` (the glm-scorecard shape) so a paid run's partial
        # rows survive a mid-loop failure; nested so provider close always runs.
        try:
            report_dir = Path("reports/scorecard")
            if rows or gate_results or any(a is not None for a in aggregates.values()):
                report_dir.mkdir(parents=True, exist_ok=True)
            if gate_results:
                # The FULL gate table — ERRORED rows included, column-qualified —
                # persists alongside the scorecard so transiently errored
                # scenarios survive into the adjudication artifact instead of
                # existing only in stdout.
                (report_dir / "openai-gpt56-gates.json").write_text(
                    json.dumps(
                        [
                            {"scenario": fx, "dimension": dim, "ok": ok, "label": label}
                            for fx, dim, ok, label in gate_results
                        ],
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if rows:
                card = Scorecard(rows=tuple(rows))
                (report_dir / "openai-gpt56-scorecard.json").write_text(
                    card.to_json(), encoding="utf-8"
                )
                (report_dir / "openai-gpt56-scorecard.html").write_text(
                    card.to_html(), encoding="utf-8"
                )
                print(  # noqa: T201 — operator artifact pointer
                    f"\n[scorecard written to {report_dir}/openai-gpt56-scorecard.json + .html]"
                )
            for column, aggregate in aggregates.items():
                if aggregate is not None:
                    (report_dir / f"openai-gpt56-aggregate-{column}.json").write_text(
                        json.dumps(aggregate, indent=2), encoding="utf-8"
                    )
        finally:
            await baseline_provider.aclose()
            await candidate_provider.aclose()

    # Completion pin: EXACTLY one gate entry per scheduled (scenario, column) —
    # a completed comparison and a transient ERRORED each append one, so any
    # shortfall means a scenario was silently dropped (a bare truthiness check
    # would pass on a 1-of-64 run).
    assert len(gate_results) == scheduled, (
        f"scorecard incomplete: {len(gate_results)} gate entries for {scheduled} "
        f"scheduled (scenario, column) pairs — see reports/scorecard/openai-gpt56-gates.json"
    )
    flagged = [(fx, dim, label) for fx, dim, ok, label in gate_results if not ok]
    print(  # noqa: T201 — operator gate summary (adjudication happens on the report)
        f"\n[openai scorecard: {len(gate_results) - len(flagged)} green / "
        f"{len(flagged)} flagged advisory gates — adjudicate per the spec's frozen rule]"
    )
