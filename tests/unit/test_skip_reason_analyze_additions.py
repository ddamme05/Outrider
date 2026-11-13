"""Check the three analysis skip reasons and their accepted event values.

Explicit names catch removals that a test iterating the enum could miss.
See specs/2026-05-19-analyze-foundation.md §0a and DECISIONS.md#018."""

from __future__ import annotations

from uuid import uuid4

from outrider.ast_facts.models import SkipReason
from outrider.audit.events import FileExaminationEvent


def test_skip_reason_cost_budget_exhausted_exists() -> None:
    """Analyze-stage skip: pre-flight budget gate refused the LLM call."""
    assert SkipReason.COST_BUDGET_EXHAUSTED.value == "COST_BUDGET_EXHAUSTED"


def test_skip_reason_no_reviewable_context_exists() -> None:
    """Analyze-stage skip: parse failed + no addable diff hunks (binary,
    pure deletion). Nothing for degraded-mode admission to anchor against."""
    assert SkipReason.NO_REVIEWABLE_CONTEXT.value == "NO_REVIEWABLE_CONTEXT"


def test_skip_reason_no_changed_scope_units_exists() -> None:
    """Analyze-stage skip: file parsed cleanly but diff hunks don't
    intersect any scope unit (comment-only, whitespace-only, module-level)."""
    assert SkipReason.NO_CHANGED_SCOPE_UNITS.value == "NO_CHANGED_SCOPE_UNITS"


def test_file_examination_event_admits_cost_budget_exhausted() -> None:
    """The audit-event cross-field validator accepts the new analyze-stage
    values just like the original five."""
    event = FileExaminationEvent(
        review_id=uuid4(),
        file_path="src/foo.py",
        examination_type="analyze",
        node_id="analyze",
        parse_status="skipped",
        skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
    )
    assert event.skip_reason == SkipReason.COST_BUDGET_EXHAUSTED


def test_file_examination_event_admits_no_reviewable_context() -> None:
    event = FileExaminationEvent(
        review_id=uuid4(),
        file_path="src/foo.py",
        examination_type="analyze",
        node_id="analyze",
        parse_status="skipped",
        skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
    )
    assert event.skip_reason == SkipReason.NO_REVIEWABLE_CONTEXT


def test_file_examination_event_admits_no_changed_scope_units() -> None:
    event = FileExaminationEvent(
        review_id=uuid4(),
        file_path="src/foo.py",
        examination_type="analyze",
        node_id="analyze",
        parse_status="skipped",
        skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS,
    )
    assert event.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS


def test_skip_reason_patch_head_misaligned_exists_and_admits() -> None:
    """Analyze-stage skip (FUP-217): detected patch/head-content
    misalignment — every coordinate anchor for the file is unsound, so
    the node skips with the data-integrity cause audit-visible."""
    assert SkipReason.PATCH_HEAD_MISALIGNED.value == "PATCH_HEAD_MISALIGNED"
    assert SkipReason.PATCH_HEAD_MISALIGNED.stage() == "analyze"
    event = FileExaminationEvent(
        review_id=uuid4(),
        file_path="src/foo.py",
        examination_type="analyze",
        node_id="analyze",
        parse_status="skipped",
        skip_reason=SkipReason.PATCH_HEAD_MISALIGNED,
    )
    assert event.skip_reason == SkipReason.PATCH_HEAD_MISALIGNED


def test_skip_reason_has_thirteen_total_values() -> None:
    """Six parser exclusions and seven analysis reasons, including BASE/HEAD misalignment."""
    assert len(list(SkipReason)) == 13
