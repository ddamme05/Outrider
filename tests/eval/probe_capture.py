"""Validate saved OpenAI probe evidence before an admission run spends money.

The manifest declares the model roles and fixture hashes. Billing adjudication
binds the usage counts to the same responses and a local raw export.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import NoReturn

import pytest

_SOL = "gpt-5.6-sol"
_LUNA = "gpt-5.6-luna"
_TERRA = "gpt-5.6-terra"
_PROBE_MANIFEST = (
    Path(__file__).resolve().parents[2] / "spikes" / "openai" / "fixtures" / "manifest.json"
)

_KNOWN_MODELS = (_SOL, _LUNA, _TERRA)
_ALLOWED_NODE_MODELS = (_LUNA, _TERRA)
_BASE_KINDS = ("envelope", "cold", "warm", "refusal")


def _expected_probe_rows(
    full_models: tuple[str, ...],
    trace_model: str = _LUNA,
    patch_model: str = _LUNA,
    analyze_models: tuple[str, ...] = (_SOL, _LUNA),
) -> frozenset[str]:
    """Require exactly the rows for the declared models and their serving roles."""
    rows = set()
    for model in full_models:
        for kind in _BASE_KINDS:
            rows.add(f"{model}:{kind}")
    rows.add(f"{trace_model}:trace")
    rows.add(f"{patch_model}:patch")
    for model in analyze_models:
        rows.add(f"{model}:trace_emission")
    rows.add(f"{_TERRA}:reasoning")
    return frozenset(rows)


# Independent procedure pins make probe changes invalidate old evidence.
_EXPECTED_PROBE_CONTRACT_VERSION = 5
_EXPECTED_MANIFEST_SCHEMA_VERSION = 4
_EXPECTED_ADJUDICATION_SCHEMA_VERSION = 2
_MAX_ADJUDICATION_WINDOW_SECONDS = 86_400


_ARTIFACT_KEYS = frozenset(
    {
        "adjudication_schema_version",
        "equation",
        "adjudicated_by",
        "count_reconciliation",
        "raw_export_sha256",
        "window_utc",
        "models",
    }
)
_WINDOW_KEYS = frozenset({"start_epoch", "end_epoch"})
_BINDING_KEYS = frozenset(
    {
        "cold_response_id",
        "warm_response_id",
        "billed_fresh_input_tokens",
        "billed_cache_write_tokens",
    }
)


_MAX_ADJUDICATED_BY_CHARS = 120
_INDETERMINATE_CAUSES = frozenset(
    {
        "wire_omitted_total_tokens",
        "cold_warm_pair_incoherent",
        "total_matches_neither_equation",
    }
)


def _is_bare_filename(name: object) -> bool:
    """A manifest filename must stay inside its capture directory."""
    return (
        isinstance(name, str)
        and name != ""
        and not Path(name).is_absolute()
        and "/" not in name
        and "\\" not in name
        and name not in (".", "..")
    )


# Must agree with the accounting equation shipped by read_usage().
_READ_USAGE_PINNED_EQUATION = "prompt_minus_cached_minus_writes"
_KNOWN_EQUATIONS = ("prompt_minus_cached", "prompt_minus_cached_minus_writes")


def _supported_by_counts(
    cold_usage: dict[str, object], warm_usage: dict[str, object]
) -> tuple[str, str | None]:
    """Derive the accounting equation or a reconciliation cause from wire counts."""
    prompt = cold_usage.get("prompt_tokens")
    completion = cold_usage.get("completion_tokens")
    total = cold_usage.get("total_tokens")
    ptd = cold_usage.get("prompt_tokens_details")
    write = ptd.get("cache_write_tokens") if isinstance(ptd, dict) else None
    if not all(type(value) is int for value in (prompt, completion, total, write)) or not write:
        return ("indeterminate", "wire_omitted_total_tokens")
    if warm_usage.get("prompt_tokens") != prompt:
        return ("indeterminate", "cold_warm_pair_incoherent")
    assert isinstance(prompt, int) and isinstance(completion, int)
    if total == prompt + completion:
        return ("prompt_minus_cached_minus_writes", None)
    if total == prompt + write + completion:
        return ("prompt_minus_cached", None)
    return ("indeterminate", "total_matches_neither_equation")


def _require_token_count(value: object, where: str) -> int:
    """Token counts are nonnegative integers; JSON booleans are not counts."""
    if type(value) is not int or value < 0:
        pytest.fail(f"probe capture gate: {where} must be a nonnegative integer token count")
    return value


def _require_probe_manifest() -> dict[str, object]:
    """Verify provenance, fixture bytes and billing before admitting paid work."""
    from outrider.llm.host_profiles import OPENAI_PROFILE  # noqa: PLC0415

    if not _PROBE_MANIFEST.exists():
        _fail(f"success manifest missing at {_PROBE_MANIFEST}")
    manifest = json.loads(_PROBE_MANIFEST.read_text(encoding="utf-8"))
    _require_current_provenance(
        manifest, OPENAI_PROFILE.base_url, OPENAI_PROFILE.profile_contract_digest
    )
    full_models, expected_rows = _require_model_roles(manifest)
    fixtures = _require_fixture_rows(manifest, expected_rows)
    _require_billing_adjudication(manifest, fixtures, full_models)
    return manifest


def _fail(reason: str) -> NoReturn:
    pytest.fail(
        f"probe capture gate: {reason} — rerun the paid wire probe "
        "(op run --env-file=.env -- uv run python spikes/openai/probe.py) "
        f"and see {_PROBE_MANIFEST}"
    )


def _require_exact_keys(obj: object, allowed: frozenset[str], where: str) -> None:
    """Only these keys may appear in a committable billing artifact."""
    if not isinstance(obj, dict):
        _fail(f"{where} must be a JSON object")
    extra = sorted(set(obj) - allowed)
    absent = sorted(allowed - set(obj))
    if extra or absent:
        _fail(
            f"{where} key set must be exactly {sorted(allowed)} — "
            f"extra={extra} missing={absent}; sanitization violation"
        )


def _require_current_provenance(
    manifest: dict[str, object], base_url: str, profile_digest: str
) -> None:
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
    if manifest.get("base_url") != base_url:
        _fail(
            f"manifest base_url {manifest.get('base_url')!r} is not the canonical "
            f"{base_url!r} — wrong-host evidence"
        )
    if manifest.get("profile_contract_digest") != profile_digest:
        _fail(
            "manifest profile_contract_digest does not match the CURRENT profile — "
            "the capture predates a wire-affecting profile change (stale evidence)"
        )


def _require_model_roles(manifest: dict[str, object]) -> tuple[tuple[str, ...], frozenset[str]]:
    declared = manifest.get("full_matrix_models")
    if (
        not isinstance(declared, list)
        or not declared
        or len(set(declared)) != len(declared)
        or any(m not in _KNOWN_MODELS for m in declared)
    ):
        _fail(
            f"full_matrix_models {declared!r} is not a supported capture matrix "
            f"(distinct members of {list(_KNOWN_MODELS)}) — the models being "
            "wire-admitted by this capture"
        )
    full_models = tuple(declared)
    trace_model = manifest.get("trace_model")
    patch_model = manifest.get("patch_model")
    for role_name, role_model in (("trace_model", trace_model), ("patch_model", patch_model)):
        if role_model not in _ALLOWED_NODE_MODELS or role_model not in full_models:
            _fail(
                f"{role_name} {role_model!r} must be a declared non-Sol full-matrix "
                f"model (allowed: {list(_ALLOWED_NODE_MODELS)}) — the per-field "
                "serving role (trace and patch swap independently)"
            )
    analyze_declared = manifest.get("analyze_models")
    if (
        not isinstance(analyze_declared, list)
        or not analyze_declared
        or len(set(analyze_declared)) != len(analyze_declared)
        or any(m not in full_models for m in analyze_declared)
    ):
        _fail(
            f"analyze_models {analyze_declared!r} must be distinct declared "
            "full-matrix models — the analyze-candidate roles the emission rows cover"
        )
    analyze_models = tuple(analyze_declared)
    assert isinstance(trace_model, str) and isinstance(patch_model, str)
    expected_rows = _expected_probe_rows(full_models, trace_model, patch_model, analyze_models)
    return full_models, expected_rows


def _require_fixture_rows(
    manifest: dict[str, object], expected_rows: frozenset[str]
) -> dict[str, dict[str, object]]:
    results = manifest.get("results") or {}
    extra_rows = sorted(set(results) - expected_rows)
    if extra_rows:
        _fail(
            f"unexpected result rows {extra_rows} — the row set is exact for the "
            "declared matrix; extras mean probe-procedure drift or a hand-edited manifest"
        )

    fixture_doc_by_row: dict[str, dict[str, object]] = {}
    for tag in sorted(expected_rows):
        row = results.get(tag)
        if not isinstance(row, dict):
            _fail(f"expected probe row {tag!r} absent from manifest")
        if row.get("required") is not True or row.get("ok") is not True:
            _fail(f"probe row {tag!r} is not a passing required row: {row}")
        fixture_name = row.get("fixture")
        recorded_sha = row.get("sha256")
        if not fixture_name or not recorded_sha:
            _fail(f"probe row {tag!r} carries no fixture/sha256 provenance")
        if not _is_bare_filename(fixture_name):
            _fail(f"fixture name {fixture_name!r} must be a bare filename inside the capture dir")
        fixture_path = _PROBE_MANIFEST.parent / str(fixture_name)
        if not fixture_path.exists():
            _fail(f"fixture {fixture_name!r} for row {tag!r} is missing")
        fixture_bytes = fixture_path.read_bytes()
        if hashlib.sha256(fixture_bytes).hexdigest() != recorded_sha:
            _fail(f"fixture {fixture_name!r} bytes do not match the manifest sha256")
        kind = tag.rsplit(":", 1)[1]
        if kind in ("cold", "warm"):
            doc = json.loads(fixture_bytes.decode("utf-8"))
            fixture_doc_by_row[tag] = doc
            usage = doc.get("usage") or {}
            prompt = usage.get("prompt_tokens")
            ptd = usage.get("prompt_tokens_details") or {}
            side = "cache_write_tokens" if kind == "cold" else "cached_tokens"
            value = ptd.get(side)
            if type(prompt) is not int or type(value) is not int:
                _fail(f"fixture {fixture_name!r} lacks integer usage for {side}")
            if not (0 < value <= prompt):
                _fail(
                    f"conservation violated in {fixture_name!r}: {side}={value} "
                    f"vs prompt_tokens={prompt} (expected 0 < {side} <= prompt)"
                )
            cached = ptd.get("cached_tokens")
            writes = ptd.get("cache_write_tokens")
            cached = _require_token_count(
                0 if cached is None else cached, f"{fixture_name} cached_tokens"
            )
            writes = _require_token_count(
                0 if writes is None else writes, f"{fixture_name} cache_write_tokens"
            )
            if cached + writes > prompt:
                _fail(
                    f"conservation violated in {fixture_name!r}: cached_tokens + "
                    f"cache_write_tokens exceeds prompt_tokens={prompt}; "
                    "fresh input tokens cannot be negative"
                )

    return fixture_doc_by_row


def _require_billing_adjudication(
    manifest: dict[str, object],
    fixture_doc_by_row: dict[str, dict[str, object]],
    full_models: tuple[str, ...],
) -> None:
    adjudication = manifest.get("conservation_adjudication") or {}
    adj_file = adjudication.get("adjudication_file")
    if not adj_file:
        _fail(
            "conservation equation not adjudicated — save the RAW billing export under "
            "fixtures/raw/ (gitignored), author the sanitized billing_adjudication.json, "
            "and point manifest.conservation_adjudication at it (probe success "
            "instructions walk through the fields)"
        )
    if not _is_bare_filename(adj_file):
        _fail(f"adjudication_file {adj_file!r} must be a bare filename inside the capture dir")
    adj_path = _PROBE_MANIFEST.parent / str(adj_file)
    if not adj_path.exists():
        _fail(f"adjudication artifact {adj_file!r} is missing from the capture dir")
    adj_bytes = adj_path.read_bytes()
    if hashlib.sha256(adj_bytes).hexdigest() != adjudication.get("adjudication_sha256"):
        _fail(f"adjudication artifact {adj_file!r} bytes do not match adjudication_sha256")
    artifact = json.loads(adj_bytes.decode("utf-8"))
    if artifact.get("adjudication_schema_version") != _EXPECTED_ADJUDICATION_SCHEMA_VERSION:
        _fail(
            "artifact is not a capture-bound billing adjudication "
            f"(adjudication_schema_version != {_EXPECTED_ADJUDICATION_SCHEMA_VERSION}) — "
            "a hash match proves integrity, not relevance"
        )

    _require_exact_keys(artifact, _ARTIFACT_KEYS, "adjudication artifact")
    equation = artifact.get("equation")
    if equation not in _KNOWN_EQUATIONS:
        _fail(f"unknown conservation equation {equation!r} (expected one of {_KNOWN_EQUATIONS})")
    if equation != _READ_USAGE_PINNED_EQUATION:
        _fail(
            f"adjudicated equation {equation!r} != read_usage()'s "
            f"{_READ_USAGE_PINNED_EQUATION!r} — the shipped accounting disagrees with "
            "the wire; change read_usage + pricing first, then update this pin"
        )
    adjudicated_by = artifact.get("adjudicated_by")
    if not (
        isinstance(adjudicated_by, str)
        and 0 < len(adjudicated_by) <= _MAX_ADJUDICATED_BY_CHARS
        and "\n" not in adjudicated_by
    ):
        _fail(
            f"adjudicated_by must be a non-empty single-line string of at most "
            f"{_MAX_ADJUDICATED_BY_CHARS} chars — bulk or nested content under an "
            "approved key is a sanitization violation"
        )
    reconciliation = artifact.get("count_reconciliation")
    raw_sha = artifact.get("raw_export_sha256")
    if not (isinstance(raw_sha, str) and re.fullmatch(r"[0-9a-f]{64}", raw_sha)):
        _fail(
            "raw_export_sha256 must be the sha256 hex of the LOCAL raw billing export "
            "(kept under fixtures/raw/, never committed)"
        )

    raw_file = adjudication.get("raw_export_file")
    raw_parts = str(raw_file).split("/") if isinstance(raw_file, str) else []
    if len(raw_parts) != 2 or raw_parts[0] != "raw" or not _is_bare_filename(raw_parts[1]):
        _fail(
            f"raw_export_file {raw_file!r} must name the local raw billing export as "
            "'raw/<filename>' (confined beneath the capture dir's raw/ subdirectory)"
        )
    raw_path = _PROBE_MANIFEST.parent / "raw" / raw_parts[1]
    if not raw_path.exists():
        _fail(
            f"raw billing export {raw_file!r} is missing — the independent billing "
            "source must exist locally at admission (gitignored, never committed)"
        )
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_sha:
        _fail(
            f"raw billing export {raw_file!r} bytes do not match the artifact's "
            "raw_export_sha256 — the adjudication does not describe this export"
        )
    window = artifact.get("window_utc")
    _require_exact_keys(window, _WINDOW_KEYS, "window_utc")
    assert isinstance(window, dict)
    start, end = window.get("start_epoch"), window.get("end_epoch")
    if not (type(start) is int and type(end) is int and 0 < start < end):
        _fail("window_utc must carry integer epochs with start_epoch < end_epoch")
    if end - start > _MAX_ADJUDICATION_WINDOW_SECONDS:
        _fail(
            f"billing window spans {end - start}s (max "
            f"{_MAX_ADJUDICATION_WINDOW_SECONDS}s) — a broad window cannot bind "
            "evidence to THIS capture"
        )
    bindings = artifact.get("models")
    if not isinstance(bindings, dict) or set(bindings) != set(full_models):
        _fail(
            f"artifact models must carry exactly the declared full-matrix entries "
            f"{sorted(full_models)} — extra or missing model blocks are a "
            "sanitization/coverage violation"
        )
    for model in full_models:
        binding = bindings.get(model)
        _require_exact_keys(binding, _BINDING_KEYS, f"models[{model}]")
        assert isinstance(binding, dict)
        cold_doc = fixture_doc_by_row.get(f"{model}:cold") or {}
        warm_doc = fixture_doc_by_row.get(f"{model}:warm") or {}
        for kind, doc in (("cold", cold_doc), ("warm", warm_doc)):
            if binding.get(f"{kind}_response_id") != doc.get("id") or not doc.get("id"):
                _fail(
                    f"{model} {kind} response id in the artifact does not match the "
                    "fixture — the billing evidence is not bound to THIS capture"
                )
            created = doc.get("created")
            if type(created) is not int or not (start <= created <= end):
                _fail(
                    f"{model} {kind} fixture created={created!r} falls outside the "
                    "artifact's billing window"
                )
        cold_usage = cold_doc.get("usage") or {}
        prompt = cold_usage.get("prompt_tokens")
        cold_ptd = cold_usage.get("prompt_tokens_details") or {}
        cached = cold_ptd.get("cached_tokens") or 0
        write = cold_ptd.get("cache_write_tokens")
        _require_token_count(
            binding.get("billed_cache_write_tokens"), f"{model} billed_cache_write_tokens"
        )
        _require_token_count(
            binding.get("billed_fresh_input_tokens"), f"{model} billed_fresh_input_tokens"
        )
        if binding.get("billed_cache_write_tokens") != write:
            _fail(
                f"{model} billed cache-write count "
                f"{binding.get('billed_cache_write_tokens')!r} != wire {write!r}"
            )
        if isinstance(prompt, int) and isinstance(cached, int):
            expected_fresh = (
                prompt - cached
                if equation == "prompt_minus_cached"
                else prompt - cached - (write if isinstance(write, int) else 0)
            )
            if binding.get("billed_fresh_input_tokens") != expected_fresh:
                _fail(
                    f"{model} billed fresh-input count "
                    f"{binding.get('billed_fresh_input_tokens')!r} is inconsistent with "
                    f"the adjudicated equation (wire counts imply {expected_fresh})"
                )
    contrary = {}
    indeterminate_causes = {}
    for model in full_models:
        cold = fixture_doc_by_row[f"{model}:cold"]
        warm = fixture_doc_by_row[f"{model}:warm"]
        support, cause = _supported_by_counts(
            cold.get("usage") or {},
            warm.get("usage") or {},  # type: ignore[arg-type]
        )
        if support == "indeterminate":
            indeterminate_causes[model] = cause
        elif support != equation:
            contrary[model] = support
    if contrary:
        _fail(
            f"fixture counts CONTRADICT the adjudicated equation {equation!r}: {contrary} — "
            "billed evidence and wire counts disagree; re-probe or re-adjudicate before "
            "any scorecard run"
        )

    if not indeterminate_causes:
        if reconciliation is not None:
            _fail(
                "count_reconciliation must be null when count support is determinate "
                "for every model — a reconciliation without an indeterminacy is unearned"
            )
    else:
        if not isinstance(reconciliation, dict) or set(reconciliation) != set(indeterminate_causes):
            _fail(
                f"count_reconciliation must be an object mapping EXACTLY the "
                f"indeterminate models {sorted(indeterminate_causes)} to their "
                "evidence-derived cause codes"
            )
        for m, cause in sorted(indeterminate_causes.items()):
            if reconciliation.get(m) != cause:
                _fail(
                    f"count_reconciliation[{m!r}] must equal the evidence-derived "
                    f"cause {cause!r} (got {reconciliation.get(m)!r}) — the operator "
                    "acknowledges what the fixture bytes show, not a chosen code"
                )


_PIN_CREATED_EPOCH = 1_789_000_100
_PIN_WINDOW = {"start_epoch": 1_789_000_000, "end_epoch": 1_789_003_600}


def verified_capture_fixture(tag: str) -> bytes | None:
    """Return verified bytes, or None if the capture declares no such row."""
    manifest = _require_probe_manifest()
    row = (manifest.get("results") or {}).get(tag)  # type: ignore[union-attr]
    if not isinstance(row, dict):
        return None
    data = (_PROBE_MANIFEST.parent / str(row["fixture"])).read_bytes()
    if hashlib.sha256(data).hexdigest() != str(row["sha256"]):
        pytest.fail(f"fixture bytes for {tag!r} changed between gate check and read")
    return data
