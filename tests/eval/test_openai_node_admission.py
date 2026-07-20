"""Triage + synthesize admission instruments for the openai host (openai-native-host spec).

The spec's per-node admission gates ("Gates before any production-shaped use")
require node-specific OpenAI-candidate evidence before Luna serves a default:
analyze has the two-column scorecard, trace + patch have their fixture-graded
instruments, and this file carries the two PAID runners those left open —
triage and synthesize. Each runs the env-selected candidate (Luna default,
field-specific Terra swap) against the APPROVED Anthropic Haiku baseline (the
ModelConfig FIELD default — the single model authority) through the real node
path:

  - TRIAGE — `build_triage_scorecard` over the shared hand-authored ground
    truth (`_triage_admission_specs`, single-sourced with the historical
    Claude flip instrument), the real triage node + parser, graded by
    `compare_triage` into `TriageScorecardRow`s. The GPT candidate validates
    through the builder's host-aware seam (`triage_preflight` with
    `_openai_candidate_validator`: native slug gate + pricing coverage —
    the seam's default Anthropic `ModelConfig` validator rejects every GPT
    slug), run BEFORE either provider is constructed. Frozen predicate
    (fully deterministic — a real grader): every row baseline-valid with the
    gate held (tier/risk/dimension) and ZERO candidate drops from analysis
    anywhere.
  - SYNTHESIZE — the real synthesize prompt (VERSION-stamped) over the six
    representative finding sets from the historical comparison; pairs are
    printed AND persisted for the operator. AUTOMATED FLOORS — the
    machine-checkable half of the spec's frozen predicate, deliberately
    incomplete (an invented finding CLAIM that names no path clears them):
    zero rejected/empty candidate responses, no material (critical/high)
    finding omitted (the finding's file basename appears in the summary), no
    invented finding/proof metadata (no path-like token outside the
    findings' file set, no query-match/64-hex proof tokens in prose). A
    floor FAIL makes admission impossible; a floor PASS is NOT admission —
    the operator adjudicates the prose, and the artifact persists
    `operator_verdict: "pending"`.

Both runners REFUSE before any spend unless the candidate is declared in the
VERIFIED capture's `full_matrix_models` — triage and synthesize have no
dedicated probe rows, so the model-level wire gate is the full matrix
(envelope/cold/warm/refusal), and a Terra swap therefore inherits the paid
wire probe before any instrument rerun. Both are REPORT-ONLY: pytest green
means the run COMPLETED. Triage's predicate verdict is printed whole (it has
a real grader); synthesize prints its automated floors plus OPERATOR
ADJUDICATION REQUIRED. In both cases the FINAL admission verdict is recorded
only in the spec's Actual Outcome. A predicate/floor FAIL swaps THAT field to
Terra (`OUTRIDER_TRIAGE_CANDIDATE` / `OUTRIDER_SYNTHESIZE_CANDIDATE`, no code
edit) and reruns that instrument — never a silent fallback.

Run (keys resolve from .env via 1Password):
  OUTRIDER_EVAL_REAL_MODELS=1 op run --env-file=.env -- \
    uv run pytest tests/eval/test_openai_node_admission.py --is-eval -v -s

Cost: triage 5 scenarios x (baseline + candidate) = 10 calls; synthesize
6 scenarios x (baseline + candidate) = 12 calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from outrider.llm.base import LLMRequest
from outrider.llm.config import ModelConfig
from outrider.llm.host_profiles import OPENAI_PROFILE
from outrider.llm.pricing import RATE_TABLE, pricing_key
from outrider.policy import FindingSeverity, FindingType
from outrider.prompts.synthesize import MAX_TOKENS, TEMPERATURE, render
from outrider.prompts.synthesize import VERSION as _SYNTHESIZE_PROMPT_VERSION
from outrider.schemas.triage_result import RiskLevel

from .runner import (
    TriageScenarioSpec,
    build_provenance,
    build_triage_scorecard,
    triage_preflight,
)
from .scorecard import TriageGateVerdict, TriageScorecardRow
from .test_model_comparison import _NoOpExchangePersister, _ScriptedProvider
from .test_openai_scorecard import (
    _LUNA,
    _SOL,
    _TERRA,
    _require_openai_admission_credentials,
    _require_probe_manifest,
    _with_providers,
    _write_valid_capture,
)
from .test_runner import _TRIAGE_EXPECTED, _build_state, _CountingProvider
from .test_scorecard import _triage_admission_specs
from .test_synthesize_summary_comparison import _finding, _metrics, _scenarios
from .test_triage_grading import _TRIAGE_DEEP

_NODE_INSTRUMENT_CANDIDATES = (_LUNA, _TERRA)
_MATERIAL_SEVERITIES = (FindingSeverity.CRITICAL, FindingSeverity.HIGH)
_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.py\b")
_PROOF_TOKEN_RE = re.compile(r"query_match_id|\b[0-9a-f]{64}\b")


def _require_node_candidate_wire_admitted(
    capture_manifest: dict[str, object], candidate: str, *, instrument: str
) -> None:
    """Per-model WIRE admission carried into the node instruments: triage and
    synthesize have no dedicated probe rows, so the model-level gate is the
    FULL wire matrix — a candidate may only spend if THIS verified capture
    declared it in `full_matrix_models`. A default Sol/Luna capture must not
    authorize a paid Terra instrument run."""
    wire_admitted = capture_manifest.get("full_matrix_models")
    assert isinstance(wire_admitted, list)  # gate validated the declaration
    if candidate not in wire_admitted:
        pytest.fail(
            f"{instrument} candidate {candidate!r} is not among the verified capture's "
            f"declared full_matrix_models {wire_admitted} — re-run the probe declaring "
            f"it (--full-models=...) before any {instrument} spend"
        )


def _require_approved_node_baseline(field: str, resolved: str) -> None:
    """The APPROVED node baseline is the CANONICAL authority — the ModelConfig
    FIELD default (ambient OUTRIDER_MODEL_* overrides influence instances,
    never the class). The paid run must FAIL before spending when the
    ambient-resolved config diverges: an arbitrary valid Claude model is not
    the approved comparison."""
    approved = ModelConfig.model_fields[field].default
    if resolved != approved:
        pytest.fail(
            f"resolved {field} {resolved!r} != approved baseline {approved!r} — ambient "
            "OUTRIDER_MODEL_* overrides change the approved comparison; unset them (or "
            "amend the spec) before any spend"
        )


def _openai_candidate_validator(model: str) -> None:
    """Host-aware candidate validation for `triage_preflight`'s seam: the
    openai host's native slug gate plus pricing coverage. The seam's default
    (the Anthropic `ModelConfig` field-validator) rejects every GPT slug, so
    without this the paid run would raise before its first call — after both
    providers were constructed."""
    OPENAI_PROFILE.validate_model_slug(model)
    key = pricing_key(OPENAI_PROFILE.host_id, model)
    if key not in RATE_TABLE:
        raise ValueError(
            f"model {model!r} has no RATE_TABLE entry under host "
            f"{OPENAI_PROFILE.host_id!r} — a candidate without pricing coverage would "
            "bill unpriceable calls; add the rate row before any instrument spend"
        )


def _triage_admission_verdict(
    rows: tuple[TriageScorecardRow, ...],
) -> tuple[bool, tuple[str, ...]]:
    """The frozen triage admission predicate over the card's rows: every row
    baseline-valid with the deterministic gate held (tier/risk/dimension) and
    ZERO candidate drops from analysis anywhere — a dropped expected-analyzed
    file is unsafe regardless of baseline validity. A baseline-invalid or
    errored row cannot support the comparison, so it blocks a PASS, named
    distinctly (ground-truth/baseline trouble, not candidate failure)."""
    failures: list[str] = []
    for row in rows:
        if row.status != "ok":
            failures.append(f"{row.scenario}: errored ({row.error})")
            continue
        assert row.gate is not None  # an 'ok' row populates every metric
        if row.n_dropped_from_analysis:
            failures.append(f"{row.scenario}: candidate dropped {row.dropped_files} from analysis")
        if not row.gate.baseline_valid:
            failures.append(
                f"{row.scenario}: baseline invalid — the row cannot support the comparison"
            )
        elif not row.gate.passes:
            held = {
                "drop_held": row.gate.drop_held,
                "risk_safety_held": row.gate.risk_safety_held,
                "overtier_bounded": row.gate.overtier_bounded,
                "dimension_recall_held": row.gate.dimension_recall_held,
            }
            failed = ", ".join(name for name, ok in held.items() if not ok)
            failures.append(f"{row.scenario}: gate failed ({failed})")
    if not rows:
        failures.append("no triage rows — the run produced nothing to admit on")
    return (not failures, tuple(failures))


def _material_omissions(summary: str, findings: tuple[Any, ...]) -> tuple[str, ...]:
    """Deterministic material-omission floor: every CRITICAL/HIGH finding's
    file BASENAME must appear in the summary (prose reliably names files;
    titles get paraphrased). Returns the omitted findings' full paths.
    Non-material findings are exempt — the compression scenario legitimately
    aggregates them."""
    omitted = []
    for finding in findings:
        if finding.severity in _MATERIAL_SEVERITIES:
            basename = finding.file_path.rsplit("/", 1)[-1]
            if basename not in summary:
                omitted.append(finding.file_path)
    return tuple(omitted)


def _token_matches_known(token: str, known_paths: frozenset[str]) -> bool:
    return any(
        token == path or path.endswith("/" + token) or token == path.rsplit("/", 1)[-1]
        for path in known_paths
    )


def _invented_references(summary: str, findings: tuple[Any, ...]) -> tuple[str, ...]:
    """Deterministic invention floor: no path-like token outside the findings'
    file set (full path, path suffix, or bare basename all count as known),
    and no proof-metadata token (query-match ids, 64-hex content hashes) —
    the summary writes bounded prose over already-final findings and has no
    business citing proof internals."""
    known = frozenset(finding.file_path for finding in findings)
    flagged = [t for t in _PATH_TOKEN_RE.findall(summary) if not _token_matches_known(t, known)]
    flagged.extend(_PROOF_TOKEN_RE.findall(summary))
    return tuple(dict.fromkeys(flagged))


async def _summary_or_rejected(
    provider: Any, model: str, parts: Any
) -> tuple[str | None, str | None]:
    """One synthesize call under `model`: (summary, None) on non-empty prose,
    (None, reason) for the predicate's rejected-response class — provider
    failure, a refusal surfaced as an exception, or empty text."""
    try:
        response = await provider.complete(
            LLMRequest(
                system_prompt=parts.system_prompt,
                user_prompt=parts.user_prompt,
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                review_id=uuid4(),
                node_id="synthesize",
                is_eval=True,
                prompt_template_version=_SYNTHESIZE_PROMPT_VERSION,
                degraded_mode=False,
            )
        )
    except Exception as exc:  # any provider failure is a rejected response here
        return None, f"{type(exc).__name__}: {exc}"
    text = response.text.strip()
    if not text:
        return None, "empty summary"
    return text, None


# --- Zero-spend pins --------------------------------------------------------


def test_node_candidate_wire_admission_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero-spend pin for the full-matrix wire-admission gate both node
    instruments run before spend: a capture that declared (Sol, Luna) admits
    Luna and REFUSES Terra for either instrument; a Terra-swap capture that
    declared Terra admits it — per-model wire admission carried through the
    consumer, exercised without spend."""
    mod = sys.modules[_require_probe_manifest.__module__]
    monkeypatch.setattr(mod, "_PROBE_MANIFEST", tmp_path / "manifest.json")
    _write_valid_capture(tmp_path)
    manifest = _require_probe_manifest()

    _require_node_candidate_wire_admitted(manifest, _LUNA, instrument="triage")
    _require_node_candidate_wire_admitted(manifest, _LUNA, instrument="synthesize")
    with pytest.raises(pytest.fail.Exception, match="full_matrix_models"):
        _require_node_candidate_wire_admitted(manifest, _TERRA, instrument="triage")
    with pytest.raises(pytest.fail.Exception, match="full_matrix_models"):
        _require_node_candidate_wire_admitted(manifest, _TERRA, instrument="synthesize")

    # A swap capture that DECLARED Terra (full matrix rerun) admits it.
    _write_valid_capture(
        tmp_path,
        full_models=(_SOL, _TERRA),
        trace_model=_TERRA,
        patch_model=_TERRA,
        analyze_models=(_SOL, _TERRA),
    )
    _require_node_candidate_wire_admitted(_require_probe_manifest(), _TERRA, instrument="triage")


def test_approved_node_baselines_binding() -> None:
    """Zero-spend pins for the node-baseline gate: the canonical Haiku FIELD
    defaults admit; a divergent ambient-resolved value (what OUTRIDER_MODEL_*
    overrides would produce) refuses before any spend."""
    _require_approved_node_baseline(
        "triage_model", ModelConfig.model_fields["triage_model"].default
    )
    _require_approved_node_baseline(
        "synthesize_model", ModelConfig.model_fields["synthesize_model"].default
    )
    with pytest.raises(pytest.fail.Exception, match="approved baseline"):
        _require_approved_node_baseline("triage_model", "claude-sonnet-5")
    with pytest.raises(pytest.fail.Exception, match="approved baseline"):
        _require_approved_node_baseline("synthesize_model", "claude-sonnet-4-6")


def _ok_row(
    scenario: str,
    *,
    n_dropped: int = 0,
    dropped: tuple[str, ...] = (),
    baseline_valid: bool = True,
    passes: bool = True,
    drop_held: bool = True,
    risk_safety_held: bool = True,
    overtier_bounded: bool = True,
    dimension_recall_held: bool = True,
) -> TriageScorecardRow:
    return TriageScorecardRow(
        model=_LUNA,
        scenario=scenario,
        baseline_model=ModelConfig.model_fields["triage_model"].default,
        tier_accuracy=1.0,
        n_dropped_from_analysis=n_dropped,
        n_deep_downgraded=0,
        n_overtiered=0,
        dimension_recall=1.0,
        dimension_precision=1.0,
        risk_correct=True,
        under_risked=False,
        gate=TriageGateVerdict(
            passes=passes,
            baseline_valid=baseline_valid,
            drop_held=drop_held,
            risk_safety_held=risk_safety_held,
            overtier_bounded=overtier_bounded,
            dimension_recall_held=dimension_recall_held,
            overtier_allowance=0,
            dimension_recall_tolerance=0.0,
        ),
        dropped_files=dropped,
    )


def test_triage_admission_verdict_predicate() -> None:
    """Zero-spend, per-variant pins for the frozen triage predicate: each
    failure class is detected ON ITS OWN, and the all-green card passes —
    reverting any single predicate clause fails its variant."""
    passes, failures = _triage_admission_verdict((_ok_row("clean"),))
    assert passes
    assert failures == ()

    # Unsafe drop is named with the dropped path.
    passes, failures = _triage_admission_verdict(
        (_ok_row("drop", n_dropped=1, dropped=("app/db.py",), passes=False, drop_held=False),)
    )
    assert not passes
    assert any("dropped" in f and "app/db.py" in f for f in failures)

    # A baseline-invalid row blocks PASS, named as baseline trouble.
    passes, failures = _triage_admission_verdict(
        (_ok_row("vacuous", baseline_valid=False, passes=False),)
    )
    assert not passes
    assert any("baseline invalid" in f for f in failures)

    # A held-gate failure on a valid baseline names the failed sub-condition.
    passes, failures = _triage_admission_verdict(
        (_ok_row("recall", passes=False, dimension_recall_held=False),)
    )
    assert not passes
    assert any("dimension_recall_held" in f for f in failures)

    # An errored row cannot admit.
    errored = TriageScorecardRow(
        model=_LUNA,
        scenario="boom",
        baseline_model=ModelConfig.model_fields["triage_model"].default,
        status="errored",
        error="transient",
    )
    passes, failures = _triage_admission_verdict((errored,))
    assert not passes
    assert any("errored" in f for f in failures)

    # An empty card cannot admit.
    passes, failures = _triage_admission_verdict(())
    assert not passes


def test_synthesize_admission_graders() -> None:
    """Zero-spend, per-variant pins for the deterministic synthesize floors —
    each floor fails on its own scripted summary and stays quiet on the clean
    one."""
    sqli = _finding(
        FindingType.SQL_INJECTION, file_path="app/orders.py", line=41, title="t", description="d"
    )
    unused = _finding(
        FindingType.UNUSED_IMPORT, file_path="app/util.py", line=3, title="t", description="d"
    )
    findings = (sqli, unused)
    # The floor keys on POLICY severity, so the fixture pair must actually
    # span material / non-material — a policy change that moves either
    # severity class fails here loud, not silently in the grader.
    assert sqli.severity in _MATERIAL_SEVERITIES
    assert unused.severity not in _MATERIAL_SEVERITIES

    clean = "Critical SQL injection in app/orders.py; minor cleanup also noted."
    assert _material_omissions(clean, findings) == ()
    assert _invented_references(clean, findings) == ()

    # Material omission: the material finding's file never named. The clean
    # summary above never names util.py either — non-material is exempt.
    omitting = "One critical issue and one minor cleanup were found."
    assert _material_omissions(omitting, findings) == ("app/orders.py",)

    # Invented path.
    invented = "SQL injection in app/orders.py and a bug in app/payments.py."
    assert _invented_references(invented, findings) == ("app/payments.py",)

    # Proof metadata never belongs in prose.
    proofy = f"Issue in app/orders.py (query_match_id set, hash {'a' * 64})."
    flagged = _invented_references(proofy, findings)
    assert "query_match_id" in flagged
    assert "a" * 64 in flagged

    # Known-path forms: bare basename and path suffix both count as known.
    assert _invented_references("orders.py has the bug", findings) == ()
    assert _invented_references("see app/orders.py and orders.py", findings) == ()


@pytest.mark.asyncio
async def test_summary_or_rejected_classes() -> None:
    """Zero-spend pin for the rejected-response classifier: provider failure
    and empty prose are REJECTED classes; non-empty prose passes through
    stripped."""
    parts = render(overall_risk=RiskLevel.LOW, findings=(), metrics=_metrics())

    class _Ok:
        async def complete(self, request: LLMRequest) -> Any:  # noqa: ARG002
            return SimpleNamespace(text="  a summary  ")

    class _Empty:
        async def complete(self, request: LLMRequest) -> Any:  # noqa: ARG002
            return SimpleNamespace(text="   ")

    class _Boom:
        async def complete(self, request: LLMRequest) -> Any:  # noqa: ARG002
            raise RuntimeError("refused")

    text, rejected = await _summary_or_rejected(_Ok(), "m", parts)
    assert text == "a summary"
    assert rejected is None
    text, rejected = await _summary_or_rejected(_Empty(), "m", parts)
    assert text is None
    assert rejected == "empty summary"
    text, rejected = await _summary_or_rejected(_Boom(), "m", parts)
    assert text is None
    assert rejected == "RuntimeError: refused"


def test_triage_builder_openai_candidate_exact_seam() -> None:
    """Zero-spend pin for the EXACT composed path the paid triage runner
    drives: the real shared `build_triage_scorecard` with a GPT slug and the
    OpenAI validator, scripted providers standing in for the paid ones. The
    run must reach the real triage node/parser and produce a graded row —
    helper-level pins cannot catch a builder that rejects the slug before
    the first call, which is precisely what the seam's default Anthropic
    validator does. The refusal twin proves the validator is actually wired
    through (a claude slug fails the openai slug gate) and refuses BEFORE
    any provider call."""
    spec = TriageScenarioSpec(scenario="example", state=_build_state(), expected=_TRIAGE_EXPECTED)
    card = build_triage_scorecard(
        [spec],
        baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
        candidate_provider=_ScriptedProvider(_TRIAGE_DEEP),
        baseline_model=ModelConfig.model_fields["triage_model"].default,
        candidate_models=[_LUNA],
        validate_candidate_model=_openai_candidate_validator,
    )
    assert len(card.triage_rows) == 1
    row = card.triage_rows[0]
    assert (row.model, row.status) == (_LUNA, "ok")
    assert row.gate is not None  # graded through the real node + compare_triage
    assert row.gate.passes is True

    # Refusal twin: the validator is consulted (not ignored) and fires before
    # any provider call — zero complete() invocations on either side.
    baseline = _CountingProvider(_TRIAGE_DEEP)
    candidate = _CountingProvider(_TRIAGE_DEEP)
    with pytest.raises(ValueError, match="slug pattern"):
        build_triage_scorecard(
            [spec],
            baseline_provider=baseline,
            candidate_provider=candidate,
            baseline_model=ModelConfig.model_fields["triage_model"].default,
            candidate_models=[ModelConfig.model_fields["triage_model"].default],
            validate_candidate_model=_openai_candidate_validator,
        )
    assert baseline.complete_calls == 0
    assert candidate.complete_calls == 0

    # The preflight is independently callable BEFORE provider construction —
    # the ordering the paid runner relies on to avoid constructing providers
    # it would leak on a deterministic input problem.
    triage_preflight([spec], (_LUNA,), validate_candidate_model=_openai_candidate_validator)
    with pytest.raises(ValueError, match="slug pattern"):
        triage_preflight(
            [spec],
            (ModelConfig.model_fields["triage_model"].default,),
            validate_candidate_model=_openai_candidate_validator,
        )


# --- Paid runners -----------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model triage admission spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1",
)
def test_gpt56_triage_admission() -> None:
    """OPT-IN, real spend — the OpenAI TRIAGE admission instrument: the
    env-selected candidate (Luna default, Terra swap) vs the APPROVED Haiku
    baseline through the real triage node, graded over the shared
    hand-authored ground truth.

    REPORT-ONLY: pytest green means the run COMPLETED; the frozen-predicate
    verdict is printed and lives in the JSON/HTML artifact, and the operator
    records the admission decision in the spec's Actual Outcome. Sync test:
    `build_triage_scorecard` owns its own event loop (same reason as the
    historical Claude instrument in test_scorecard.py)."""
    # Opted-in paid runner: every missing/unresolved prerequisite FAILS before spend
    # (shared preflight — never an inner skip that reads as a clean run).
    anthropic_key, openai_key = _require_openai_admission_credentials()
    capture_manifest = _require_probe_manifest()

    candidate = os.environ.get("OUTRIDER_TRIAGE_CANDIDATE", _LUNA)
    if candidate not in _NODE_INSTRUMENT_CANDIDATES:
        pytest.fail(f"OUTRIDER_TRIAGE_CANDIDATE {candidate!r} not in (Luna, Terra)")
    _require_node_candidate_wire_admitted(capture_manifest, candidate, instrument="triage")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider  # noqa: PLC0415
    from outrider.prompts.triage import VERSION as TRIAGE_PROMPT_VERSION  # noqa: PLC0415

    cfg = ModelConfig()
    baseline_model = cfg.triage_model
    _require_approved_node_baseline("triage_model", baseline_model)
    specs = _triage_admission_specs()
    # ALL deterministic preflight (slug + pricing coverage + ground-truth
    # coverage) BEFORE either provider exists: a raise inside
    # `build_triage_scorecard` lands after construction, where
    # `close_providers` cannot reach the already-built providers.
    triage_preflight(specs, (candidate,), validate_candidate_model=_openai_candidate_validator)
    print(  # noqa: T201 — pre-spend operator plan
        f"\n[openai triage admission plan: {len(specs)} scenarios x (baseline + candidate) "
        f"= {2 * len(specs)} paid calls ({candidate} vs {baseline_model})]"
    )

    persister = _NoOpExchangePersister()
    # Nested ownership at the constructor seam (the `_with_providers`
    # discipline, sync shape): a candidate-constructor failure closes the
    # already-constructed baseline and never spends.
    baseline_provider = AnthropicProvider(
        api_key=SecretStr(anthropic_key), model_config=cfg, persister=persister
    )
    try:
        candidate_provider = OpenAICompatibleProvider(
            api_key=SecretStr(openai_key),
            profile=OPENAI_PROFILE,
            persister=persister,
            models=(candidate,),
        )
    except BaseException:
        asyncio.run(baseline_provider.aclose())
        raise
    # `build_triage_scorecard` stamps provenance BEFORE the paid matrix (the
    # shared-runner convention), re-runs the (idempotent) preflight through
    # the same host-aware seam, and closes both providers inside its own loop.
    card = build_triage_scorecard(
        specs,
        baseline_provider=baseline_provider,
        candidate_provider=candidate_provider,
        baseline_model=baseline_model,
        candidate_models=[candidate],
        close_providers=True,
        prompt_template_version=TRIAGE_PROMPT_VERSION,
        validate_candidate_model=_openai_candidate_validator,
    )

    out_dir = Path("reports") / "scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"openai-triage-admission-{candidate}-vs-{baseline_model}-{stamp}"
    (out_dir / f"{stem}.json").write_text(card.to_json(), encoding="utf-8")
    (out_dir / f"{stem}.html").write_text(card.to_html(), encoding="utf-8")

    passes, failures = _triage_admission_verdict(card.triage_rows)
    print(f"\nTRIAGE ADMISSION — wrote {out_dir}/{stem}.{{json,html}}")  # noqa: T201
    verdict = "PASS" if passes else "FAIL"
    print(  # noqa: T201 — operator-visible frozen predicate
        f"TRIAGE ADMISSION VERDICT (frozen predicate): {verdict} — {candidate} vs {baseline_model}"
    )
    for failure in failures:
        print(f"  - {failure}")  # noqa: T201
    if not passes:
        print(  # noqa: T201
            "  a FAIL swaps THIS field to Terra (OUTRIDER_TRIAGE_CANDIDATE=gpt-5.6-terra) "
            "after Terra clears the full wire probe matrix"
        )
    # Report-only: assert only that the run produced a triage row per spec.
    assert len(card.triage_rows) == len(specs)


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model synthesize admission spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1",
)
@pytest.mark.asyncio
async def test_gpt56_synthesize_admission() -> None:
    """OPT-IN, real spend — the OpenAI SYNTHESIZE admission instrument: the
    env-selected candidate (Luna default, Terra swap) vs the APPROVED Haiku
    baseline over the real synthesize prompt and the six representative
    finding sets.

    REPORT-ONLY: pytest green means the run COMPLETED (every scenario
    produced a recorded pair). The AUTOMATED FLOORS are the machine-checkable
    half only — deliberately incomplete (an invented finding CLAIM that names
    no path clears them) — so a floor PASS is never the admission verdict:
    the artifact persists `operator_verdict: "pending"`, the operator reads
    the pairs, and the FINAL synthesize admission verdict is recorded only in
    the spec's Actual Outcome. A floor FAIL does make admission impossible on
    this evidence (without making pytest red — pytest answers "did the
    evidence run complete?")."""
    # Opted-in paid runner: every missing/unresolved prerequisite FAILS before spend
    # (shared preflight — never an inner skip that reads as a clean run).
    anthropic_key, openai_key = _require_openai_admission_credentials()
    capture_manifest = _require_probe_manifest()

    candidate = os.environ.get("OUTRIDER_SYNTHESIZE_CANDIDATE", _LUNA)
    if candidate not in _NODE_INSTRUMENT_CANDIDATES:
        pytest.fail(f"OUTRIDER_SYNTHESIZE_CANDIDATE {candidate!r} not in (Luna, Terra)")
    _require_node_candidate_wire_admitted(capture_manifest, candidate, instrument="synthesize")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider  # noqa: PLC0415

    cfg = ModelConfig()
    baseline_model = cfg.synthesize_model
    _require_approved_node_baseline("synthesize_model", baseline_model)
    scenarios = _scenarios()
    print(  # noqa: T201 — pre-spend operator plan
        f"\n[openai synthesize admission plan: {len(scenarios)} scenarios x "
        f"(baseline + candidate) = {2 * len(scenarios)} paid calls "
        f"({candidate} vs {baseline_model})]"
    )

    # Provenance captured BEFORE provider construction and spend: the
    # artifact must describe the code that PRODUCED the calls.
    provenance = build_provenance(
        prompt_template_version=_SYNTHESIZE_PROMPT_VERSION,
        scenario_labels=[name for name, _risk, _findings in scenarios],
        baseline_model=baseline_model,
        candidate_models=(candidate,),
    )
    assert provenance is not None  # version is set, so build_provenance never None

    persister = _NoOpExchangePersister()
    records: list[dict[str, object]] = []

    async def _drive(baseline: object, candidate_provider: object) -> None:
        for name, risk, findings in scenarios:
            parts = render(overall_risk=risk, findings=findings, metrics=_metrics())
            print(  # noqa: T201
                f"\n{'=' * 72}\nSCENARIO: {name} ({len(findings)} findings, risk={risk.value})"
            )
            b_text, b_rejected = await _summary_or_rejected(baseline, baseline_model, parts)
            c_text, c_rejected = await _summary_or_rejected(candidate_provider, candidate, parts)
            records.append(
                {
                    "scenario": name,
                    "risk": risk.value,
                    "n_findings": len(findings),
                    "baseline_summary": b_text,
                    "baseline_rejected": b_rejected,
                    "candidate_summary": c_text,
                    "candidate_rejected": c_rejected,
                    "baseline_material_omissions": (
                        _material_omissions(b_text, findings) if b_text else ()
                    ),
                    "candidate_material_omissions": (
                        _material_omissions(c_text, findings) if c_text else ()
                    ),
                    "candidate_invented_references": (
                        _invented_references(c_text, findings) if c_text else ()
                    ),
                }
            )
            for label, model, text, rejected in (
                ("baseline", baseline_model, b_text, b_rejected),
                ("candidate", candidate, c_text, c_rejected),
            ):
                if text is None:
                    print(f"\n--- {label} ({model}) REJECTED: {rejected} ---")  # noqa: T201
                else:
                    print(f"\n--- {label} ({model}) ---\n{text}")  # noqa: T201

    await _with_providers(
        lambda: AnthropicProvider(
            api_key=SecretStr(anthropic_key), model_config=cfg, persister=persister
        ),
        lambda: OpenAICompatibleProvider(
            api_key=SecretStr(openai_key),
            profile=OPENAI_PROFILE,
            persister=persister,
            models=(candidate,),
        ),
        _drive,
    )

    floor_failures: list[str] = []
    baseline_flags: list[str] = []
    for record in records:
        name = record["scenario"]
        if record["candidate_rejected"] is not None:
            floor_failures.append(f"{name}: candidate rejected — {record['candidate_rejected']}")
        if record["candidate_material_omissions"]:
            floor_failures.append(
                f"{name}: material finding omitted — {record['candidate_material_omissions']}"
            )
        if record["candidate_invented_references"]:
            floor_failures.append(
                f"{name}: invented reference — {record['candidate_invented_references']}"
            )
        if record["baseline_rejected"] is not None or record["baseline_material_omissions"]:
            baseline_flags.append(
                f"{name}: baseline flagged — the pair cannot discriminate; "
                "inspect before adjudicating"
            )
    automated_floor_passes = not floor_failures

    out_dir = Path("reports") / "scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"openai-synthesize-admission-{candidate}-vs-{baseline_model}-{stamp}"
    artifact = {
        "instrument": "synthesize-admission",
        "candidate": candidate,
        "baseline": baseline_model,
        "provenance": provenance.model_dump(mode="json"),
        "scenarios": records,
        # The floors are the machine-checkable HALF: their PASS is never the
        # admission verdict. The final verdict is the operator's, recorded in
        # the spec's Actual Outcome — until then it is pending here.
        "verdict": {
            "automated_floor_passes": automated_floor_passes,
            "automated_floor_failures": floor_failures,
            "baseline_flags": baseline_flags,
            "operator_verdict": "pending",
        },
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}\nSYNTHESIZE ADMISSION — wrote {out_dir}/{stem}.json")  # noqa: T201
    if automated_floor_passes:
        print(  # noqa: T201 — operator-visible floor result, NOT the admission verdict
            f"AUTOMATED FLOORS: PASS — OPERATOR ADJUDICATION REQUIRED "
            f"({candidate} vs {baseline_model}; floors are the machine-checkable half; "
            "prose-level invention/omission stays a human read)"
        )
    else:
        print(  # noqa: T201
            f"AUTOMATED FLOORS: FAIL — admission impossible on this evidence "
            f"({candidate} vs {baseline_model})"
        )
        for failure in floor_failures:
            print(f"  - {failure}")  # noqa: T201
        print(  # noqa: T201
            "  a floor FAIL swaps THIS field to Terra "
            "(OUTRIDER_SYNTHESIZE_CANDIDATE=gpt-5.6-terra) after Terra clears the full "
            "wire probe matrix"
        )
    for flag in baseline_flags:
        print(f"  ! {flag}")  # noqa: T201
    print(  # noqa: T201
        "REPORT ONLY: the artifact's operator_verdict is 'pending' — read the pairs and "
        "record the FINAL synthesize admission verdict only in the spec's Actual Outcome."
    )
    # Report-only: assert only that every scenario produced a recorded pair.
    assert len(records) == len(scenarios)
