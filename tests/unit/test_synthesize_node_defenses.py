# See specs/2026-05-28-synthesize-node.md §Severity policy.
"""Regression + defense tests for the synthesize node body.

Pins:
- **F1 regression** (prompt None-safety): `prompts/synthesize.py::render`
  must not crash when `ReviewMetrics` LLM-aggregate fields are `None`
  (the V1 placeholder shape). The original implementation used
  `{value:.4f}` format specs which raise `TypeError` against
  `NoneType`; the fix renders "unknown" for None.

- **H-1 defense** (policy_version smuggle): synthesize rejects findings
  carrying `policy_version != state.triage_result.policy_version` at
  node entry. The triage-captured snapshot is the trusted anchor (set
  upstream of analyze, immune to attacker control via forged finding[0]).
  `ReviewFinding._enforce_severity_matches_policy` short-circuits on
  non-active policy_version — a forged finding with arbitrary severity
  would survive the schema check and the audit row. Synthesize fails
  closed via `FindingForgeryDetectedError`.

- **H-2 defense** (original_severity smuggle): synthesize rejects
  findings carrying `original_severity != None` at node entry. HITL
  has not run; `original_severity` is set only by HITL after a
  reviewer override. A finding with the field pre-set indicates a
  forged HITL-override triplet attempting to bypass the gated set.

Companion to `test_synthesize_audit_events.py` (schema/discriminator
tests). This file covers the node body's entry-side invariants.
"""

from __future__ import annotations

from typing import Any

import pytest

from outrider.agent.nodes.synthesize import (
    FindingForgeryDetectedError,
    _enforce_synthesize_input_invariants,
)
from outrider.policy import FindingSeverity, FindingType
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts.synthesize import render
from outrider.schemas import ReviewMetrics
from outrider.schemas.triage_result import RiskLevel

# ---------------------------------------------------------------------------
# F1 — prompts/synthesize.py::render must be None-safe on LLM aggregates
# ---------------------------------------------------------------------------


def test_render_does_not_crash_on_none_llm_aggregates() -> None:
    """`ReviewMetrics` LLM-aggregate fields ship as `None` in V1 (audit-
    query helper not yet wired). `render` must not raise TypeError on
    `:.4f` format vs NoneType.

    Reproduces the sharp-edges F1 audit finding: original implementation
    used `f"${metrics.total_cost_usd:.4f}"` which crashes when
    total_cost_usd is None — every synthesize call would have failed
    before reaching the LLM provider.
    """
    metrics = ReviewMetrics(
        files_examined=3,
        files_traced_beyond_diff=1,
        wall_clock_seconds=12.5,
        # LLM aggregates default to None (V1 placeholder semantics).
    )
    # MUST NOT raise.
    parts = render(overall_risk=RiskLevel.MEDIUM, findings=(), metrics=metrics)
    # Verify the None values render as "unknown" rather than "None"
    # (more reader-friendly + signals the V1 placeholder semantics).
    assert "unknown" in parts.user_prompt
    assert "None" not in parts.user_prompt.split("Metrics:")[1].split("Wall clock:")[0]


def test_render_handles_concrete_llm_aggregates_when_present() -> None:
    """When LLM aggregates ARE populated (future post-FUP), render
    must format them correctly: `:.4f` on cost, `:d` on counts."""
    metrics = ReviewMetrics(
        files_examined=3,
        files_traced_beyond_diff=1,
        llm_calls_made=4,
        total_input_tokens=12_345,
        total_output_tokens=678,
        total_cost_usd=1.2345,
        wall_clock_seconds=12.5,
    )
    parts = render(overall_risk=RiskLevel.MEDIUM, findings=(), metrics=metrics)
    assert "LLM calls made: 4" in parts.user_prompt
    assert "12345 in / 678 out" in parts.user_prompt
    assert "$1.2345" in parts.user_prompt


def _non_active_policy_version() -> str:
    """Return a valid bare-semver guaranteed != `ACTIVE_POLICY_VERSION`.

    Bumps the major component of `ACTIVE_POLICY_VERSION`; the result is
    deterministically different from ACTIVE regardless of which patch /
    minor / major ACTIVE lands on, so forge-test paths don't age into
    equality + silently change semantics when policy versions bump in
    the future.
    """
    major = int(ACTIVE_POLICY_VERSION.split(".", 1)[0])
    return f"{major + 1}.0.0"


# ---------------------------------------------------------------------------
# H-1 defense — policy_version smuggle (synthesize entry rejects)
# ---------------------------------------------------------------------------


def _make_finding_stub(
    *,
    policy_version: str,
    original_severity: FindingSeverity | None = None,
    finding_type: FindingType | None = None,
) -> Any:
    """Lightweight stub that quacks like ReviewFinding for the entry
    invariant check. Avoids constructing a full ReviewFinding (which
    would trip its own validators — exactly the surfaces we're trying
    to bypass-test from the producer side)."""

    class _FindingStub:
        def __init__(self) -> None:
            self.policy_version = policy_version
            self.original_severity = original_severity
            self.severity = (
                FindingSeverity.MEDIUM
                if original_severity is None
                else FindingSeverity.LOW  # would-be downgrade smuggle
            )
            self.content_hash = "a" * 64
            self.finding_id = "stub-id"
            self.finding_type = finding_type or FindingType.SQL_INJECTION

    return _FindingStub()


def _make_state_stub(
    *,
    findings: list[Any],
    triage_policy_version: str = ACTIVE_POLICY_VERSION,
    rounds: list[list[Any]] | None = None,
) -> Any:
    """Build a minimal ReviewState-like stub for the synthesize defenses.

    `triage_policy_version` controls the snapshot anchor — synthesize's
    `_enforce_synthesize_input_invariants` reads it from
    `state.triage_result.policy_version`. Defaults to
    `ACTIVE_POLICY_VERSION` (the legitimate fresh-review path).
    `rounds=[[f1], [f2]]` lets callers exercise multi-round paths;
    when None, all findings land in a single round.
    """

    class _Round:
        def __init__(self, fs: list[Any]) -> None:
            self.findings = tuple(fs)

    class _Triage:
        def __init__(self, version: str) -> None:
            self.policy_version = version

    class _State:
        def __init__(self) -> None:
            if rounds is None:
                self.analysis_rounds = (_Round(findings),)
            else:
                self.analysis_rounds = tuple(_Round(r) for r in rounds)
            self.triage_result = _Triage(triage_policy_version)

    return _State()


def test_synthesize_admits_findings_matching_triage_snapshot() -> None:
    """The H-1 defense uses a triage-anchored snapshot: findings MUST
    match `state.triage_result.policy_version`. The snapshot is
    captured at triage entry, upstream of analyze, so it survives
    mid-deploy ACTIVE_POLICY_VERSION bumps AND defeats first-finding
    poisoning (an attacker who controls analyze cannot poison the
    snapshot).
    """
    # Happy path: findings match the triage snapshot (current active).
    legit = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
    state = _make_state_stub(findings=[legit])
    _enforce_synthesize_input_invariants(state)

    # Replay path: a historical review's triage carries the historical
    # version; the findings under that review match. Synthesize must
    # admit (not deny completion based on live ACTIVE_POLICY_VERSION).
    legit_historical = _make_finding_stub(policy_version="0.0.1")
    state = _make_state_stub(
        findings=[legit_historical],
        triage_policy_version="0.0.1",
    )
    _enforce_synthesize_input_invariants(state)


def test_synthesize_rejects_mid_batch_policy_version_drift() -> None:
    """H-1 detection fires when any finding's `policy_version` diverges
    from the triage-captured snapshot (`state.triage_result.policy_version`,
    set upstream of analyze by the triage node's Rule (d) gate). The
    triage snapshot is the trusted anchor; any divergent finding
    raises before the divergence detector + audit row emit."""
    anchor = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
    forged = _make_finding_stub(policy_version=_non_active_policy_version())
    state = _make_state_stub(findings=[anchor, forged])

    with pytest.raises(FindingForgeryDetectedError, match="policy_version"):
        _enforce_synthesize_input_invariants(state)


# ---------------------------------------------------------------------------
# H-2 defense — original_severity smuggle (synthesize entry rejects)
# ---------------------------------------------------------------------------


def test_synthesize_rejects_finding_with_preset_original_severity() -> None:
    """H-2: a finding carrying `original_severity != None` at synthesize
    entry indicates a forged HITL-override triplet (HITL has not run
    yet at synthesize). Fail-closed before the gated-set partition."""
    forged = _make_finding_stub(
        policy_version=ACTIVE_POLICY_VERSION,
        original_severity=FindingSeverity.CRITICAL,
    )
    state = _make_state_stub(findings=[forged])

    with pytest.raises(FindingForgeryDetectedError, match="original_severity"):
        _enforce_synthesize_input_invariants(state)


def test_synthesize_admits_finding_with_none_original_severity() -> None:
    """Negative pin: a finding with `original_severity=None` (the
    canonical pre-HITL shape) passes the entry check."""
    legit = _make_finding_stub(
        policy_version=ACTIVE_POLICY_VERSION,
        original_severity=None,
    )
    state = _make_state_stub(findings=[legit])
    # MUST NOT raise.
    _enforce_synthesize_input_invariants(state)


def test_synthesize_detects_forge_in_later_round() -> None:
    """Multi-round path: forge in `analysis_rounds[1]` with legitimate
    findings in `analysis_rounds[0]`. The outer round-loop must reach
    round_index=1 and trigger the snapshot mismatch.
    """
    legit = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
    forged = _make_finding_stub(policy_version=_non_active_policy_version())
    state = _make_state_stub(findings=[], rounds=[[legit], [forged]])

    with pytest.raises(FindingForgeryDetectedError, match="round_index=1"):
        _enforce_synthesize_input_invariants(state)


def test_synthesize_blocks_first_finding_poisoning() -> None:
    """The triage-anchored snapshot defeats the first-finding-poisoning
    DoS: an attacker who plants one forged finding in round 0 index 0
    cannot use it as the snapshot anchor — triage's captured
    policy_version is the trusted source. Legitimate findings then
    succeed; the SINGLE forged finding is detected.
    """
    forged_first = _make_finding_stub(policy_version="evil-snapshot")
    legit_second = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
    # Triage captured ACTIVE_POLICY_VERSION at review start (upstream
    # of the analyze compromise). The forged finding is detected as
    # the divergent one — NOT the legitimate finding.
    state = _make_state_stub(
        findings=[forged_first, legit_second],
        triage_policy_version=ACTIVE_POLICY_VERSION,
    )

    with pytest.raises(FindingForgeryDetectedError, match="evil-snapshot"):
        _enforce_synthesize_input_invariants(state)


def test_synthesize_requires_triage_result_to_anchor_snapshot() -> None:
    """Synthesize cannot derive the policy_version snapshot if triage
    has not run. Missing triage_result is itself a corruption signal
    (graph routed past triage somehow). Raises FindingForgeryDetectedError
    with a clear message naming the missing anchor.
    """

    class _RoundEmpty:
        def __init__(self) -> None:
            self.findings = ()

    class _StateNoTriage:
        def __init__(self) -> None:
            self.analysis_rounds = (_RoundEmpty(),)
            self.triage_result = None

    with pytest.raises(FindingForgeryDetectedError, match="triage_result"):
        _enforce_synthesize_input_invariants(_StateNoTriage())  # type: ignore[arg-type]


def test_synthesize_rejects_first_forge_when_mixed_with_legit() -> None:
    """When multiple findings are present and the original_severity
    forge is mixed with legit findings, synthesize raises on the FIRST
    forge encountered. Verifies the defense fires deterministically
    rather than silently filtering."""
    legit = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
    forged = _make_finding_stub(
        policy_version=ACTIVE_POLICY_VERSION,  # uniform snapshot
        original_severity=FindingSeverity.CRITICAL,  # forge axis
    )
    state = _make_state_stub(findings=[legit, forged])

    with pytest.raises(FindingForgeryDetectedError, match="original_severity"):
        _enforce_synthesize_input_invariants(state)


# ---------------------------------------------------------------------------
# Dead-branch defense: policy_version axis in _detect_and_report_divergence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_version_axis_divergence_emits_anomaly_and_raises() -> None:
    """`_detect_and_report_divergence` MUST fail-loud on the
    policy_version axis even when severity matches.

    This pins the dead-branch annotation at synthesize.py:298-322 against
    future deletion. The check `len(policy_versions) > 1` is structurally
    unreachable on the canonical path (because
    `_enforce_synthesize_input_invariants` upstream raises
    `FindingForgeryDetectedError` on mixed versions BEFORE this function
    runs). The check exists as defense-in-depth against future producer
    bypasses (direct ReviewState construction in tests, alternate
    dispatchers, ad-hoc tooling) — exactly what this test exercises:
    invoking `_detect_and_report_divergence` directly with mixed
    policy_versions, bypassing the upstream gate.

    If a future maintainer deletes the `len(policy_versions) > 1`
    clause from synthesize.py:323 ("structurally unreachable, drop it"),
    this test fails — proving the dead-branch is load-bearing for any
    non-canonical producer path.
    """
    import uuid
    from unittest.mock import AsyncMock

    from outrider.agent.nodes.synthesize import (
        SynthesizeAggregationError,
        _detect_and_report_divergence,
    )

    # Two findings: SAME content_hash, SAME severity, SAME finding_type,
    # DIFFERENT policy_version. The policy_version axis is the SOLE
    # divergence signal — the severity axis (the rule's namesake) is
    # uniform across both findings.
    f_v1 = _make_finding_stub(policy_version="1.0.0")
    f_v2 = _make_finding_stub(policy_version="0.0.1")
    assert f_v1.severity == f_v2.severity  # severity axis uniform
    assert f_v1.content_hash == f_v2.content_hash  # same content
    assert f_v1.policy_version != f_v2.policy_version  # only divergence

    # Synthesize state with both findings + a triage snapshot that
    # WOULD have caught this at the upstream gate (we intentionally
    # skip that gate in this test to exercise the dead branch).
    state = _make_state_stub(
        findings=[f_v1, f_v2],
        triage_policy_version="1.0.0",
    )
    # Stub the state's review_id + is_eval the helper doesn't carry.
    state.review_id = uuid.uuid4()  # noqa: SLF001
    state.is_eval = False  # noqa: SLF001

    anomaly_sink = AsyncMock()
    anomaly_sink.emit_anomaly = AsyncMock(return_value=None)

    # Bypassing _enforce_synthesize_input_invariants → directly invoke
    # the divergence detector. The dead-branch annotation says this
    # path is "structurally unreachable" on canonical execution — yet
    # any non-canonical producer can land here, which is exactly why
    # the check exists.
    with pytest.raises(SynthesizeAggregationError):
        await _detect_and_report_divergence(state=state, anomaly_sink=anomaly_sink)

    # Anomaly was emitted with both policy_versions in the details
    # payload — proves the emit-then-raise contract on the
    # policy_version axis specifically.
    assert anomaly_sink.emit_anomaly.await_count == 1
    emit_call = anomaly_sink.emit_anomaly.await_args
    details = emit_call.kwargs["details"]
    assert set(details["policy_versions"]) == {"1.0.0", "0.0.1"}, (
        f"expected both policy_versions in anomaly details, got {details!r}"
    )
    # is_eval kwarg pin: `AnomalySink.emit_anomaly` Protocol declares
    # `is_eval` as mandatory; the synthesize node MUST forward
    # `state.is_eval` verbatim. Per CodeRabbit catch, the prior assertion
    # block validated only severity / policy_versions / round_indices and
    # would have admitted a regression that dropped is_eval (silent
    # data-isolation breach for eval runs).
    assert "is_eval" in emit_call.kwargs, (
        f"emit_anomaly call missing is_eval kwarg, got {emit_call.kwargs!r}"
    )
    assert emit_call.kwargs["is_eval"] is False, (
        f"expected is_eval=False (stub set on state); got {emit_call.kwargs['is_eval']!r}"
    )


# ---------------------------------------------------------------------------
# Replay binding: summary_content_hash MUST hash the raw LLM response.text
# ---------------------------------------------------------------------------


def test_summary_content_hash_helper_binds_sha256_to_input_bytes() -> None:
    """Helper-level contract: `_compute_summary_content_hash` returns
    `sha256(text.encode("utf-8")).hexdigest()` over the input bytes.

    This is a tautological pin on the helper's identity — the call site
    contract (synthesize MUST pass raw `response.text` not stripped) is
    pinned by `test_synthesize_call_site_hashes_raw_response_text_not_stripped`
    below. Keeping the helper test for documentation; the call-site
    test is the non-vacuous regression gate.
    """
    import hashlib

    from outrider.agent.nodes.synthesize import _compute_summary_content_hash

    raw_response_text = '```json\n"This is the summary prose."\n```'
    stripped = '"This is the summary prose."'

    raw_hash = _compute_summary_content_hash(raw_response_text)
    stripped_hash = _compute_summary_content_hash(stripped)

    # The two MUST differ under fence-wrap — locks in that the helper
    # is sensitive to its input (catches a regression that, e.g.,
    # always returned a constant).
    assert raw_hash != stripped_hash
    expected_raw_hash = hashlib.sha256(raw_response_text.encode("utf-8")).hexdigest()
    assert raw_hash == expected_raw_hash


def test_synthesize_call_site_hashes_raw_response_text_not_stripped() -> None:
    """Call-site contract: `synthesize.py` MUST pass raw `response.text`
    to `_compute_summary_content_hash`, NOT `summary_text` (the
    post-`strip_outer_json_fence` form).

    Pins the regression Codex 2026-05-28 caught. Hashing stripped would
    break identity binding to `llm_call_content.completion` (which
    persists raw — see `audit/persister.py::_persist_llm_call_event`)
    the moment Anthropic wraps a response in ```json``` fences.

    Non-vacuous regression gate: source inspection catches a swap of
    `response.text` ↔ `summary_text` at the call site. A behavior-level
    test would require a 50-line state stub + mock provider just to
    invoke synthesize() once; this catches the same regression in two
    asserts. Pairs with the helper-level test above which pins the
    helper's identity. Both layers needed: helper proves
    "sha256 of bytes," call site proves "passed the right bytes."
    """
    import inspect

    from outrider.agent.nodes import synthesize as synthesize_module

    source = inspect.getsource(synthesize_module.synthesize)

    # Positive: the canonical call site MUST appear verbatim.
    assert "_compute_summary_content_hash(response.text)" in source, (
        "synthesize call site must hash RAW response.text (NOT "
        "summary_text). Hashing stripped text breaks identity with "
        "llm_call_content.completion under fence-wrap (Codex finding "
        "2026-05-28). See _compute_summary_content_hash docstring + "
        "DECISIONS.md#016 for replay-equivalence rationale."
    )
    # Negative: the regression pattern MUST NOT appear.
    assert "_compute_summary_content_hash(summary_text)" not in source, (
        "synthesize call site hashes summary_text — this is the Codex "
        "2026-05-28 regression. The hash MUST bind to raw response.text "
        "so the audit row's summary_content_hash matches the canon "
        "stored in llm_call_content.completion."
    )
