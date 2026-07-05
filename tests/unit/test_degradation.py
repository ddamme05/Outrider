"""Unit tests for the pure analyze degradation decision (`agent/nodes/degradation.py`).

No LLM, no DB — these construct a `ParseResult` (+ optional `PatchedFile`) and
assert the typed `DegradationDecision`. The clean-no-error and degraded paths over
REAL parsed source are covered by the structural eval scenarios; here we pin the
typed-decision contract, the parser-skip caller-guard, and the no-scope skips.
"""

from __future__ import annotations

import pytest

from outrider.agent.nodes.degradation import DegradationDecision, decide_degradation
from outrider.ast_facts.models import ParseResult, SkipReason
from outrider.coordinates import lookup_patched_file

# ---------------------------------------------------------------------------
# DegradationDecision typed-contract guard (__post_init__)
# ---------------------------------------------------------------------------


def test_decision_skip_requires_skip_reason() -> None:
    with pytest.raises(ValueError, match="mode='skip' requires"):
        DegradationDecision(mode="skip", parse_status="clean")


def test_decision_degraded_requires_degradation_reason() -> None:
    with pytest.raises(ValueError, match="mode='degraded' requires"):
        DegradationDecision(mode="degraded", parse_status="degraded")


def test_decision_clean_must_not_carry_skip_reason() -> None:
    with pytest.raises(ValueError, match="must not carry skip_reason"):
        DegradationDecision(mode="clean", parse_status="clean", skip_reason=SkipReason.VENDORED)


def test_decision_clean_must_not_carry_degradation_reason() -> None:
    with pytest.raises(ValueError, match="must not carry degradation_reason"):
        DegradationDecision(mode="clean", parse_status="clean", degradation_reason="parse_failed")


def test_decision_valid_skip_constructs() -> None:
    d = DegradationDecision(
        mode="skip", parse_status="clean", skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS
    )
    assert d.mode == "skip"
    assert d.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
    assert d.included_scope_units == ()


# ---------------------------------------------------------------------------
# decide_degradation
# ---------------------------------------------------------------------------


def test_decide_degradation_raises_on_parser_skipped() -> None:
    # Parser-stage skips are the node's responsibility (handled BEFORE this is
    # called, because lookup_patched_file can raise on a skipped file's patch).
    # Passing one here is a caller-contract violation.
    skipped = ParseResult(parser_outcome="skipped", skip_reason=SkipReason.VENDORED)
    with pytest.raises(RuntimeError, match="parser-skipped result"):
        decide_degradation(skipped, None)


def test_decide_degradation_clean_no_patch_skips_no_changed_scope_units() -> None:
    decision = decide_degradation(ParseResult(parser_outcome="clean"), None)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
    assert decision.parse_status == "clean"


def test_decide_degradation_clean_no_scope_units_skips_no_changed_scope_units() -> None:
    # patched_file present but the parse has no scope units → empty intersection
    # → NO_CHANGED_SCOPE_UNITS (exercises _intersect_changed_scope_units's empty path).
    patched_file = lookup_patched_file("@@ -1 +1 @@\n-a\n+b\n", "x.py")
    assert patched_file is not None
    decision = decide_degradation(ParseResult(parser_outcome="clean"), patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS


# The "failed" parser outcome is V1-unreachable (intake gates invalid UTF-8) but
# decide_degradation handles it for the raw-bytes intake path (FUP-053); pin both
# branches directly since the node-level tests can't reach them in V1.


def test_decide_degradation_failed_no_patch_skips_no_reviewable_context() -> None:
    decision = decide_degradation(ParseResult(parser_outcome="failed"), None)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_REVIEWABLE_CONTEXT
    assert decision.parse_status == "failed"


def test_decide_degradation_failed_with_added_lines_degrades_parse_failed() -> None:
    # A failed parse WITH addable text → degraded(parse_failed), no scope context
    # (the failed path never intersects scopes).
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.py")
    assert patched_file is not None
    decision = decide_degradation(ParseResult(parser_outcome="failed"), patched_file)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "parse_failed"
    assert decision.parse_status == "failed"
    assert decision.included_scope_units == ()
    assert decision.included_clipped_hunks == ()


# ---------------------------------------------------------------------------
# decide_degradation — no-scope syntax-error degrade-don't-skip (DECISIONS#033)
# ---------------------------------------------------------------------------


def test_decide_degradation_no_scope_error_line_degrades() -> None:
    # clean parse, NO changed scope unit, but an ADDED line (2) intersects a tree
    # error line (error_lines={2}) → degrade with tree_has_error_no_scope, not skip.
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({2}))
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.py")  # adds line 2
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "tree_has_error_no_scope"
    assert decision.parse_status == "degraded"
    # No scope recovered → no scope context; the degraded prompt uses bounded hunks.
    assert decision.included_scope_units == ()


def test_decide_degradation_module_candidate_degrades_with_clean_parse_status() -> None:
    """Module-scope routing (DECISIONS.md#062):
    a module-only diff (clean parse, no changed scope units) with the
    precomputed eligible-match flag degrades as `module_level_observed_match`
    — and `parse_status` stays truthfully `clean` (a routing choice, not a
    parse defect)."""
    parse_result = ParseResult(parser_outcome="clean")
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.js")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file, module_level_observed_candidate=True)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "module_level_observed_match"
    assert decision.parse_status == "clean"
    assert decision.included_scope_units == ()


def test_decide_degradation_without_module_candidate_skip_unchanged() -> None:
    """Revert-the-fold control: the flag defaulting False keeps today's
    NO_CHANGED_SCOPE_UNITS skip for module-only diffs with no eligible match."""
    parse_result = ParseResult(parser_outcome="clean")
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.js")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS


def test_decide_degradation_parse_error_precedence_over_module_candidate() -> None:
    """Parse-error precedence: a syntax-error file (error_lines intersecting
    the added line) degrades as `tree_has_error_no_scope` even when the module
    candidate flag is set — the error branch runs first, so OBSERVED stays off
    error-recovered trees (revert-the-precedence: evaluating the module branch
    first fails this pin)."""
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({2}))
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.js")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file, module_level_observed_candidate=True)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "tree_has_error_no_scope"
    assert decision.parse_status == "degraded"


def test_decide_degradation_no_scope_deletion_only_stays_skip() -> None:
    # An error line exists but the change is a pure DELETION (no added target line)
    # → no addable line intersects error_lines → still NO_CHANGED_SCOPE_UNITS skip
    # (the addable-lines-only non-goal: we do not solve deletion-only changes).
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({2}))
    patched_file = lookup_patched_file("@@ -1,2 +1,1 @@\n a\n-b\n", "x.py")  # deletes line 2
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS


def test_decide_degradation_no_scope_added_line_not_an_error_line_stays_skip() -> None:
    # The added line (2) does NOT intersect error_lines (error on line 5) → skip.
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({5}))
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.py")  # adds line 2 only
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
