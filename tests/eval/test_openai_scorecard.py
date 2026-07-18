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
import re
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
_EXPECTED_PROBE_ROWS: frozenset[str] = (
    frozenset(
        f"{model}:{kind}"
        for model in (_SOL, _LUNA)
        for kind in ("envelope", "cold", "warm", "refusal")
    )
    # Node-admission rows (spec: one paid row each, folded into the probe) —
    # graded offline by test_trace_admission.py / test_patch_admission.py.
    | {f"{_LUNA}:trace", f"{_LUNA}:patch"}
    | {"gpt-5.6-terra:reasoning"}
)

# Pinned against the probe's PROBE_CONTRACT_VERSION / MANIFEST_SCHEMA_VERSION:
# a capture from an older probe PROCEDURE (different prompts, schema bytes,
# matrix, or predicates) or manifest shape must not admit, exactly as a
# stale profile digest must not.
# Procedure v2: the matrix gained the trace/patch node-admission rows.
_EXPECTED_PROBE_CONTRACT_VERSION = 2
_EXPECTED_MANIFEST_SCHEMA_VERSION = 3

# The operator-authored, sanitized billing-adjudication artifact (raw exports
# stay local/gitignored). A sha256 match proves INTEGRITY; the schema +
# capture-binding checks below prove RELEVANCE — response IDs, a bounded
# billing window covering the fixtures' `created` stamps, and billed class
# counts cross-checked against the wire under the adjudicated equation.
# v2: adjudicated_by narrowed to a bounded single-line string;
# count_reconciliation narrowed from free-form to a per-model mapping of
# evidence-derived closed cause codes — v1 named the permissive contract.
_EXPECTED_ADJUDICATION_SCHEMA_VERSION = 2
_MAX_ADJUDICATION_WINDOW_SECONDS = 86_400

# CLOSED key sets for the committable artifact, enforced as set EQUALITY at
# every object level. The artifact is the ONLY capture file .gitignore lets
# into the repo, and "sanitized" is a gate property, not a docstring claim:
# an extra key — a project/org/key identifier, a dollar amount, an embedded
# raw export — refuses admission instead of riding into git.
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

# Closed keys are not enough — approved keys must also carry BOUNDED, TYPED
# values, or an entire raw export can ride in nested under `adjudicated_by`.
_MAX_ADJUDICATED_BY_CHARS = 120
# The closed indeterminate-CAUSE vocabulary, one code per branch of
# _supported_by_counts. Reconciliation is EVIDENCE-DERIVED, not merely
# allowlisted: the artifact must map exactly the indeterminate models to
# exactly their derived causes (acknowledgment semantics — the operator types
# what the evidence shows), and must be null when every model is determinate.
_INDETERMINATE_CAUSES = frozenset(
    {
        "wire_omitted_total_tokens",  # total/completion operand absent or non-int
        "cold_warm_pair_incoherent",  # prompt counts differ across the pair
        "total_matches_neither_equation",  # coherent ints, neither identity holds
    }
)


def _is_bare_filename(name: object) -> bool:
    """Capture-dir files must be bare filenames: no separators, no traversal,
    no absolute paths — a manifest must not be able to reach outside the
    fixture directory it lives in."""
    return (
        isinstance(name, str)
        and name != ""
        and not Path(name).is_absolute()
        and "/" not in name
        and "\\" not in name
        and name not in (".", "..")
    )


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


def _supported_by_counts(
    cold_usage: dict[str, object], warm_usage: dict[str, object]
) -> tuple[str, str | None]:
    """Which equation a model's hash-verified cold/warm fixture usage supports,
    as `(support, indeterminate_cause)` — support is an equation or
    "indeterminate", and the cause is the TYPED reason (one of
    `_INDETERMINATE_CAUSES`) or None for determinate evidence, so the
    reconciliation can be required to match the evidence rather than merely
    name an allowlisted code. Deliberately re-implements the probe's
    characterization (spikes/ is not importable from tests; independent
    recomputation is the point — the gate must not trust the manifest's own
    conservation_facts block). total_tokens is the disambiguator:
    == prompt + completion means writes ride INSIDE prompt_tokens (input must
    subtract them); == prompt + write + completion means writes are an
    additive class."""
    prompt = cold_usage.get("prompt_tokens")
    completion = cold_usage.get("completion_tokens")
    total = cold_usage.get("total_tokens")
    ptd = cold_usage.get("prompt_tokens_details")
    write = ptd.get("cache_write_tokens") if isinstance(ptd, dict) else None
    if not all(isinstance(v, int) for v in (prompt, completion, total, write)) or not write:
        # prompt/write are pre-guaranteed by the bounds loop on real captures;
        # the realistic holes are total_tokens / completion_tokens.
        return ("indeterminate", "wire_omitted_total_tokens")
    if warm_usage.get("prompt_tokens") != prompt:
        return ("indeterminate", "cold_warm_pair_incoherent")
    assert isinstance(prompt, int) and isinstance(completion, int)  # narrowed above
    if total == prompt + completion:
        return ("prompt_minus_cached_minus_writes", None)
    if total == prompt + write + completion:
        return ("prompt_minus_cached", None)
    return ("indeterminate", "total_matches_neither_equation")


def _require_probe_manifest() -> None:
    """FAIL (not skip) without a passing, coherent, CURRENT probe capture: the
    operator has explicitly opted into real spend, so a silent skip would read
    as a clean run. Beyond the probe's own verdict boolean, the gate verifies
    the capture's provenance (canonical base_url; profile digest + probe
    procedure/manifest versions), the EXACT expected row set with path-safe
    bare filenames, each fixture's existence + sha256, the cold/warm
    conservation bounds recomputed FROM THE FIXTURE BYTES, and the sanitized
    billing-adjudication ARTIFACT bound to THIS capture: its own schema
    version, CLOSED key sets at every object level (sanitization is enforced,
    not asserted), the fixtures' exact response IDs, a bounded billing window
    covering their `created` stamps, billed fresh/write class counts
    cross-checked against the wire under the adjudicated equation, the LOCAL
    raw export located under fixtures/raw/ and hashed byte-for-byte against
    raw_export_sha256 (the independent billing source must exist, not merely
    be claimed), per-model count support recomputed from fixture bytes
    (contrary counts refuse outright; indeterminate counts need an explicit
    reconciliation), and equality with read_usage()'s shipped equation. A
    hash proves integrity; the binding proves relevance. (A determined forger
    can fabricate all of it together; the gate's job is
    stale/partial/accidental artifacts, not adversarial operators — the
    operator IS the trust anchor.)"""
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
    # Hash-verified fixture stash for the post-loop binding + recomputation:
    # the adjudication is judged against fixture BYTES (ids, created stamps,
    # usage), never against the manifest's own (independently editable)
    # conservation_facts block.
    fixture_doc_by_row: dict[str, dict[str, object]] = {}
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
            if not isinstance(prompt, int) or not isinstance(value, int):
                _fail(f"fixture {fixture_name!r} lacks integer usage for {side}")
            if not (0 < value <= prompt):
                _fail(
                    f"conservation violated in {fixture_name!r}: {side}={value} "
                    f"vs prompt_tokens={prompt} (expected 0 < {side} <= prompt)"
                )
    # The bounds above catch malformed wire; they CANNOT choose between the
    # spec's two accounting equations. Admission requires the operator's
    # billing-verified adjudication ARTIFACT, bound to THIS capture: a hash
    # match proves integrity, so the artifact must additionally prove
    # RELEVANCE — its schema, the fixtures' exact response IDs, a bounded
    # billing window covering their `created` stamps, and billed class counts
    # cross-checked against the wire under the adjudicated equation.
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

    def _require_exact_keys(obj: object, allowed: frozenset[str], where: str) -> None:
        """Closed-set EQUALITY: the artifact is the only committable capture
        file, so 'sanitized' is enforced, not asserted — an extra key (org/key
        identifier, dollar amount, embedded export) refuses admission."""
        if not isinstance(obj, dict):
            _fail(f"{where} must be a JSON object")
        extra = sorted(set(obj) - allowed)
        absent = sorted(allowed - set(obj))
        if extra or absent:
            _fail(
                f"{where} key set must be exactly {sorted(allowed)} — "
                f"extra={extra} missing={absent}; sanitization violation"
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
    # The independent billing source must EXIST, not merely be claimed: the
    # (local, gitignored) manifest names the raw export's path confined
    # beneath fixtures/raw/, and the gate hashes those ACTUAL bytes against
    # the artifact's raw_export_sha256. Without this, the "independent
    # source" collapses to a self-consistent operator-authored artifact.
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
    assert isinstance(window, dict)  # narrowed: _require_exact_keys raises otherwise
    start, end = window.get("start_epoch"), window.get("end_epoch")
    if not (isinstance(start, int) and isinstance(end, int) and 0 < start < end):
        _fail("window_utc must carry integer epochs with start_epoch < end_epoch")
    if end - start > _MAX_ADJUDICATION_WINDOW_SECONDS:
        _fail(
            f"billing window spans {end - start}s (max "
            f"{_MAX_ADJUDICATION_WINDOW_SECONDS}s) — a broad window cannot bind "
            "evidence to THIS capture"
        )
    bindings = artifact.get("models")
    if not isinstance(bindings, dict) or set(bindings) != {_SOL, _LUNA}:
        _fail(
            f"artifact models must carry exactly the two full-matrix entries "
            f"{sorted((_SOL, _LUNA))} — extra or missing model blocks are a "
            "sanitization/coverage violation"
        )
    for model in (_SOL, _LUNA):
        binding = bindings.get(model)
        _require_exact_keys(binding, _BINDING_KEYS, f"models[{model}]")
        assert isinstance(binding, dict)  # narrowed: _require_exact_keys raises otherwise
        cold_doc = fixture_doc_by_row.get(f"{model}:cold") or {}
        warm_doc = fixture_doc_by_row.get(f"{model}:warm") or {}
        for kind, doc in (("cold", cold_doc), ("warm", warm_doc)):
            if binding.get(f"{kind}_response_id") != doc.get("id") or not doc.get("id"):
                _fail(
                    f"{model} {kind} response id in the artifact does not match the "
                    "fixture — the billing evidence is not bound to THIS capture"
                )
            created = doc.get("created")
            if not isinstance(created, int) or not (start <= created <= end):
                _fail(
                    f"{model} {kind} fixture created={created!r} falls outside the "
                    "artifact's billing window"
                )
        cold_usage = cold_doc.get("usage") or {}
        prompt = cold_usage.get("prompt_tokens")
        cold_ptd = cold_usage.get("prompt_tokens_details") or {}
        cached = cold_ptd.get("cached_tokens") or 0
        write = cold_ptd.get("cache_write_tokens")
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
    support_by_model = {
        model: _supported_by_counts(
            (fixture_doc_by_row.get(f"{model}:cold") or {}).get("usage") or {},  # type: ignore[arg-type, union-attr]
            (fixture_doc_by_row.get(f"{model}:warm") or {}).get("usage") or {},  # type: ignore[arg-type, union-attr]
        )
        for model in (_SOL, _LUNA)
    }
    contrary = {
        m: s for m, (s, _cause) in support_by_model.items() if s not in ("indeterminate", equation)
    }
    if contrary:
        _fail(
            f"fixture counts CONTRADICT the adjudicated equation {equation!r}: {contrary} — "
            "billed evidence and wire counts disagree; re-probe or re-adjudicate before "
            "any scorecard run"
        )
    # Reconciliation is EVIDENCE-DERIVED, per model: null is REQUIRED when
    # every model's counts are determinate (a reconciliation without an
    # indeterminacy is unearned), and otherwise the artifact must map EXACTLY
    # the indeterminate models to EXACTLY their derived causes — Sol and Luna
    # can differ, so a single global scalar cannot express the evidence.
    indeterminate_causes = {
        m: cause for m, (s, cause) in support_by_model.items() if s == "indeterminate"
    }
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


# Fixed, deterministic capture epoch for the zero-spend pins.
_PIN_CREATED_EPOCH = 1_789_000_100
_PIN_WINDOW = {"start_epoch": 1_789_000_000, "end_epoch": 1_789_003_600}


def _write_valid_capture(capture_dir: Path) -> dict[str, object]:
    """A coherent fake capture for the zero-spend pins: full expected row set,
    real sha256 over on-disk fixture bytes, conservation-consistent cold/warm
    usage, CURRENT profile provenance, and a fully BOUND sanitized adjudication
    artifact (matching response IDs, window covering `created`, billed classes
    consistent with the pinned equation). Returns the manifest dict (also
    written) so tests can perturb one dimension at a time."""
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
                # prompt_minus_cached on the cold call: fresh = 2000 - 0.
                "billed_fresh_input_tokens": 2000,
                "billed_cache_write_tokens": 1500,
            }
            for model in (_SOL, _LUNA)
        },
    }
    manifest: dict[str, object] = {
        "schema_version": _EXPECTED_MANIFEST_SCHEMA_VERSION,
        "probe_contract_version": _EXPECTED_PROBE_CONTRACT_VERSION,
        "base_url": OPENAI_PROFILE.base_url,
        "profile_contract_digest": OPENAI_PROFILE.profile_contract_digest,
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


def test_probe_manifest_precondition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-spend pins for the paid-run gate, one per admission dimension: a
    coherent, fully BOUND capture admits; missing manifest, failed verdict,
    unknown manifest shape, stale probe-procedure version, wrong host, stale
    profile digest, a dropped required row (refusal included — the pre-ship
    gate), a non-passing row, an EXTRA row, a traversal-shaped fixture or
    artifact name, a missing/tampered fixture, a conservation bounds
    violation, an unadjudicated / mismatched equation, a missing / tampered /
    generic-unbound adjudication artifact, a wrong response ID, billed class
    counts that disagree with the wire, a window that misses the capture or is
    unboundedly wide, a missing / unconfined / tampered RAW export (the
    independent billing source must exist and be the hashed bytes), a
    sensitive extra key at any artifact level (closed-set sanitization),
    nested/bulk content under an APPROVED key (typed bounded scalars only), a
    free-form reconciliation (closed reason codes only), an extra model
    block, fixture counts that CONTRADICT the verdict, and indeterminate
    counts without reconciliation each FAIL (code-reconciled indeterminate
    admits). Fail (not skip) because the operator explicitly opted into
    spend — a silent skip would read as a clean run."""
    import copy  # noqa: PLC0415
    import sys  # noqa: PLC0415

    mod = sys.modules[_require_probe_manifest.__module__]
    monkeypatch.setattr(mod, "_PROBE_MANIFEST", tmp_path / "manifest.json")

    with pytest.raises(pytest.fail.Exception, match="manifest missing"):
        _require_probe_manifest()

    valid = _write_valid_capture(tmp_path)
    _require_probe_manifest()  # the coherent, fully bound capture admits

    def _perturbed_manifest() -> dict[str, object]:
        return copy.deepcopy(valid["manifest"])  # type: ignore[arg-type]

    def _perturbed_artifact() -> dict[str, object]:
        return copy.deepcopy(valid["artifact"])  # type: ignore[arg-type]

    def _restore_valid() -> None:
        # Full re-write (fixtures + raw export + artifact + manifest): pins
        # may have perturbed fixture FILES, not just the manifest/artifact.
        _write_valid_capture(tmp_path)

    broken = _perturbed_manifest()
    broken["all_required_passed"] = False
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="all_required_passed is not true"):
        _require_probe_manifest()

    broken = _perturbed_manifest()
    broken["schema_version"] = 99
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="unknown manifest shape"):
        _require_probe_manifest()

    # A capture from an older probe PROCEDURE (prompts/matrix/predicates) is
    # stale evidence even when the profile digest still matches.
    broken = _perturbed_manifest()
    broken["probe_contract_version"] = 0
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="probe PROCEDURE change"):
        _require_probe_manifest()

    broken = _perturbed_manifest()
    broken["base_url"] = "https://evil.example/v1"
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="wrong-host evidence"):
        _require_probe_manifest()

    broken = _perturbed_manifest()
    broken["profile_contract_digest"] = "0" * 64
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="stale evidence"):
        _require_probe_manifest()

    # Refusal is a REQUIRED row: a capture missing it must not admit.
    broken = _perturbed_manifest()
    del broken["results"][f"{_SOL}:refusal"]  # type: ignore[index, arg-type]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="refusal.*absent from manifest"):
        _require_probe_manifest()

    broken = _perturbed_manifest()
    broken["results"][f"{_LUNA}:refusal"]["ok"] = False  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="not a passing required row"):
        _require_probe_manifest()

    # The row set is EXACT: an extra row means procedure drift or a hand edit.
    broken = _perturbed_manifest()
    broken["results"]["gpt-5.6-terra:envelope"] = {"ok": True, "required": True}  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="unexpected result rows"):
        _require_probe_manifest()

    # Capture-dir names must be bare filenames — no traversal, no absolutes.
    broken = _perturbed_manifest()
    broken["results"][f"{_SOL}:envelope"]["fixture"] = "../evil.json"  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="bare filename"):
        _require_probe_manifest()

    broken = _perturbed_manifest()
    broken["conservation_adjudication"]["adjudication_file"] = "/etc/hostname"  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="bare filename"):
        _require_probe_manifest()

    # Unadjudicated: no artifact pointer at all.
    broken = _perturbed_manifest()
    broken["conservation_adjudication"] = {"adjudication_file": None}
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="not adjudicated"):
        _require_probe_manifest()

    # A hash match proves integrity, not relevance: a generic hash-matching
    # JSON (no schema, no capture binding) must NOT authorize the run.
    broken = _perturbed_manifest()
    generic = {"source": "some export", "billed": {"fresh": 500, "write": 1500}}
    _rewrite_artifact(tmp_path, broken, generic)
    with pytest.raises(pytest.fail.Exception, match="integrity, not relevance"):
        _require_probe_manifest()

    # Artifact missing / tampered bytes.
    _restore_valid()
    artifact_path = tmp_path / "billing_adjudication.json"
    artifact_original = artifact_path.read_text(encoding="utf-8")
    artifact_path.unlink()
    with pytest.raises(pytest.fail.Exception, match="adjudication artifact.*is missing"):
        _require_probe_manifest()
    artifact_path.write_text(artifact_original + " ", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="do not match adjudication_sha256"):
        _require_probe_manifest()
    artifact_path.write_text(artifact_original, encoding="utf-8")
    _require_probe_manifest()  # restored artifact admits again

    # Equation checks now live in the artifact.
    perturbed = _perturbed_artifact()
    perturbed["equation"] = None
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="unknown conservation equation"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["equation"] = "prompt_minus_cached_minus_writes"
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="shipped accounting disagrees"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["raw_export_sha256"] = "not-a-hash"
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="raw_export_sha256"):
        _require_probe_manifest()

    # The independent billing source must EXIST and be the hashed bytes — a
    # well-formed sha256 with no raw export (or the wrong one) must not admit.
    _restore_valid()
    broken = _perturbed_manifest()
    broken["conservation_adjudication"]["raw_export_file"] = None  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="raw/<filename>"):
        _require_probe_manifest()

    broken = _perturbed_manifest()
    broken["conservation_adjudication"]["raw_export_file"] = "raw/../../etc/hostname"  # type: ignore[index, call-overload]
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="raw/<filename>"):
        _require_probe_manifest()

    _rewrite_manifest(tmp_path, _perturbed_manifest())
    raw_export = tmp_path / "raw" / "usage_export.json"
    raw_original = raw_export.read_text(encoding="utf-8")
    raw_export.unlink()
    with pytest.raises(pytest.fail.Exception, match="raw billing export.*is missing"):
        _require_probe_manifest()
    raw_export.write_text(raw_original + " ", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="does not describe this export"):
        _require_probe_manifest()
    raw_export.write_text(raw_original, encoding="utf-8")
    _require_probe_manifest()  # restored raw export admits again

    # Closed key sets at every artifact level: a representative sensitive
    # extra (org identifier, dollar amount, embedded export) refuses
    # admission instead of riding into the sole committable file.
    perturbed = _perturbed_artifact()
    perturbed["organization_id"] = "org-abc123"
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="sanitization violation"):
        _require_probe_manifest()

    # Closed KEYS are not enough — approved keys must carry bounded scalars.
    # An entire raw export nested under adjudicated_by must refuse.
    perturbed = _perturbed_artifact()
    perturbed["adjudicated_by"] = {"raw_export": {"project_id": "proj_secret", "rows": [1, 2]}}
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="single-line string"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["adjudicated_by"] = "x" * 500  # bulk content under an approved key
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="single-line string"):
        _require_probe_manifest()

    # A reconciliation code on fully DETERMINATE evidence is unearned — null
    # is required when the counts already settle every model.
    perturbed = _perturbed_artifact()
    perturbed["count_reconciliation"] = {_SOL: "wire_omitted_total_tokens"}
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="unearned"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["window_utc"]["project_id"] = "proj_secret"  # type: ignore[index, call-overload]
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="sanitization violation"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["models"][_SOL]["amount_usd"] = 0.42  # type: ignore[index, call-overload]
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="sanitization violation"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["models"]["gpt-5.6-terra"] = dict(  # type: ignore[index, call-overload, arg-type]
        perturbed["models"][_SOL]  # type: ignore[index, call-overload, arg-type]
    )
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="exactly the two full-matrix entries"):
        _require_probe_manifest()

    # Capture binding: a wrong response ID is an export for some OTHER run.
    perturbed = _perturbed_artifact()
    perturbed["models"][_SOL]["cold_response_id"] = "chatcmpl-other-run"  # type: ignore[index, call-overload]
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="not bound to THIS capture"):
        _require_probe_manifest()

    # Billed class counts must agree with the wire under the adjudicated equation.
    perturbed = _perturbed_artifact()
    perturbed["models"][_LUNA]["billed_cache_write_tokens"] = 999  # type: ignore[index, call-overload]
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="billed cache-write count"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["models"][_SOL]["billed_fresh_input_tokens"] = 500  # type: ignore[index, call-overload]
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="inconsistent with.*adjudicated equation"):
        _require_probe_manifest()

    # The billing window must cover the capture and stay bounded.
    perturbed = _perturbed_artifact()
    perturbed["window_utc"] = {
        "start_epoch": _PIN_CREATED_EPOCH + 1000,
        "end_epoch": _PIN_CREATED_EPOCH + 2000,
    }
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="outside the.*billing window"):
        _require_probe_manifest()

    perturbed = _perturbed_artifact()
    perturbed["window_utc"] = {
        "start_epoch": _PIN_CREATED_EPOCH - 90_000,
        "end_epoch": _PIN_CREATED_EPOCH + 90_000,
    }
    _rewrite_artifact(tmp_path, _perturbed_manifest(), perturbed)
    with pytest.raises(pytest.fail.Exception, match="cannot bind evidence"):
        _require_probe_manifest()

    def _set_cold_usage(manifest: dict[str, object], model: str, usage: dict[str, object]) -> None:
        payload = json.dumps(
            {"id": f"chatcmpl-pin-{model}:cold", "created": _PIN_CREATED_EPOCH, "usage": usage},
            indent=2,
        )
        (tmp_path / (f"{model}:cold".replace(":", "_") + ".json")).write_text(
            payload, encoding="utf-8"
        )
        manifest["results"][f"{model}:cold"]["sha256"] = hashlib.sha256(  # type: ignore[index, call-overload]
            payload.encode("utf-8")
        ).hexdigest()

    _restore_valid()
    broken = _perturbed_manifest()
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
    _restore_valid()
    broken = _perturbed_manifest()
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

    # Indeterminate counts (no total_tokens on the wire) demand an
    # EVIDENCE-DERIVED, per-model reconciliation: null fails, a global scalar
    # fails, an allowlisted-but-wrong cause fails; only the mapping that
    # acknowledges exactly what the fixture bytes show admits.
    _restore_valid()
    no_total_usage = {
        "prompt_tokens": 2000,
        "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500},
    }
    broken = _perturbed_manifest()
    for model in (_SOL, _LUNA):
        _set_cold_usage(broken, model, dict(no_total_usage))
    _rewrite_manifest(tmp_path, broken)
    with pytest.raises(pytest.fail.Exception, match="mapping EXACTLY the indeterminate"):
        _require_probe_manifest()

    reconciled = _perturbed_artifact()
    reconciled["count_reconciliation"] = "wire_omitted_total_tokens"  # global scalar
    _rewrite_artifact(tmp_path, broken, reconciled)
    with pytest.raises(pytest.fail.Exception, match="mapping EXACTLY the indeterminate"):
        _require_probe_manifest()

    reconciled = _perturbed_artifact()
    reconciled["count_reconciliation"] = {  # allowlisted codes, wrong causes
        _SOL: "cold_warm_pair_incoherent",
        _LUNA: "cold_warm_pair_incoherent",
    }
    _rewrite_artifact(tmp_path, broken, reconciled)
    with pytest.raises(pytest.fail.Exception, match="evidence-derived"):
        _require_probe_manifest()

    reconciled = _perturbed_artifact()
    reconciled["count_reconciliation"] = {
        _SOL: "wire_omitted_total_tokens",
        _LUNA: "wire_omitted_total_tokens",
    }
    _rewrite_artifact(tmp_path, broken, reconciled)
    _require_probe_manifest()  # matching per-model acknowledgment admits

    # Mixed determinacy: only Sol indeterminate — the mapping must cover Sol
    # alone (a code for determinate Luna is an extra key), and the Sol-only
    # mapping admits. Per-model semantics, not one global verdict.
    _restore_valid()
    broken = _perturbed_manifest()
    _set_cold_usage(broken, _SOL, dict(no_total_usage))
    reconciled = _perturbed_artifact()
    reconciled["count_reconciliation"] = {
        _SOL: "wire_omitted_total_tokens",
        _LUNA: "wire_omitted_total_tokens",
    }
    _rewrite_artifact(tmp_path, broken, reconciled)
    with pytest.raises(pytest.fail.Exception, match="mapping EXACTLY the indeterminate"):
        _require_probe_manifest()
    reconciled["count_reconciliation"] = {_SOL: "wire_omitted_total_tokens"}
    _rewrite_artifact(tmp_path, broken, reconciled)
    _require_probe_manifest()  # Sol-only, evidence-matching reconciliation admits


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
