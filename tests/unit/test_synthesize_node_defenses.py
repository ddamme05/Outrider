# See specs/2026-05-28-synthesize-node.md §Severity policy.
"""Regression + defense tests for the synthesize node body.

Pins:
- **F1 regression** (prompt None-safety): `prompts/synthesize.py::render`
  must not crash when `ReviewMetrics` LLM-aggregate fields are `None`
  (the V1 placeholder shape). The original implementation used
  `{value:.4f}` format specs which raise `TypeError` against
  `NoneType`; the fix renders "unknown" for None.

- **H-1 defense** (policy_version smuggle): synthesize rejects findings
  carrying `policy_version != ACTIVE_POLICY_VERSION` at node entry.
  `ReviewFinding._enforce_severity_matches_policy` short-circuits on
  non-active policy_version (`review_finding.py:352`) — a forged
  finding with arbitrary severity would survive the schema check
  and the audit row. Synthesize fails closed via
  `FindingForgeryDetectedError`.

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
from outrider.schemas.review_report import ReviewMetrics
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


def _make_state_stub(*, findings: list[Any]) -> Any:
    class _Round:
        def __init__(self, fs: list[Any]) -> None:
            self.findings = tuple(fs)

    class _State:
        def __init__(self) -> None:
            self.analysis_rounds = (_Round(findings),)

    return _State()


def test_synthesize_admits_uniform_policy_version() -> None:
    """The H-1 defense uses a snapshot semantic: the first finding's
    `policy_version` becomes the review's "active policy" snapshot;
    subsequent findings MUST match it. A single finding (or all
    findings carrying the same version) admits, regardless of whether
    the value equals the live `ACTIVE_POLICY_VERSION` — operational
    mid-deploy bumps must not deny completion for in-flight reviews.

    `_enforce_severity_matches_policy` short-circuits on non-active
    versions, so the residual "all findings forged with coherent fake
    snapshot" case requires full analyze compromise; tracked as a
    snapshot-fortification FUP.
    """
    legit = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
    state = _make_state_stub(findings=[legit])
    # MUST NOT raise.
    _enforce_synthesize_input_invariants(state)

    # Also admits a non-active version when ALL findings share it
    # (legitimate replay path against a historical policy snapshot).
    legit_historical = _make_finding_stub(policy_version="0.0.1")
    state = _make_state_stub(findings=[legit_historical])
    _enforce_synthesize_input_invariants(state)


def test_synthesize_rejects_mid_batch_policy_version_drift() -> None:
    """H-1 detection fires on MIXED policy_version values across the
    finding batch. The first finding's version anchors the snapshot;
    any subsequent finding with a different value is a mid-batch
    forge attempt and raises before the divergence detector + audit
    row emit."""
    anchor = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
    forged = _make_finding_stub(policy_version="0.0.0")
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
    round_index=1 and trigger the snapshot mismatch."""

    class _MultiRoundState:
        def __init__(self) -> None:
            legit = _make_finding_stub(policy_version=ACTIVE_POLICY_VERSION)
            forged = _make_finding_stub(policy_version="0.0.0")

            class _R0:
                def __init__(self) -> None:
                    self.findings = (legit,)

            class _R1:
                def __init__(self) -> None:
                    self.findings = (forged,)

            self.analysis_rounds = (_R0(), _R1())

    with pytest.raises(FindingForgeryDetectedError, match="round_index=1"):
        _enforce_synthesize_input_invariants(_MultiRoundState())  # type: ignore[arg-type]


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
