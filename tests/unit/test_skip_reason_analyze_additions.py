# See specs/2026-05-19-analyze-foundation.md §0a + DECISIONS.md#018 Amended 2026-05-20.
"""Focused test on the three SkipReason values added in §0a.

The existing `test_file_examination_event.py` parameterizes over
`list(SkipReason)` so it automatically picks up the new values via
the audit-event cross-field validator. This file pins the §0a-specific
contract explicitly: the three new values EXIST with the documented
string values, and `FileExaminationEvent` accepts each one.

Without this test, a future contributor renaming or removing one of
the three would only see the parameterized test silently shift its
count — no `SkipReason.COST_BUDGET_EXHAUSTED` reference to grep for,
no specific assertion to break. This file is the deliberate grep
target for the §0a contract.
"""

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


def test_skip_reason_has_eight_total_values() -> None:
    """5 original parser-stage + 3 new analyze-stage = 8 total.

    Pins the count so a future addition or removal surfaces in the
    diff loud — adjusting this number is the canonical way to indicate
    intent to expand or contract the taxonomy.
    """
    assert len(list(SkipReason)) == 8
