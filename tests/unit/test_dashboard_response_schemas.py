"""Unit tests for the dashboard read-API response schemas.

Pins the cross-boundary model contract: `extra="forbid"`, `AwareDatetime`
(naive datetimes rejected), and that the file/wall-clock metric fields
accept `None` (the pending-not-zero edge case).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.api.dashboard.reviews import (
    ReviewDetail,
    ReviewListItem,
    ReviewMetricsView,
    SeverityCounts,
    StatusCounts,
    _hitl_decisions_from_events,
)
from outrider.audit.events import HITLDecisionEvent
from outrider.db.models._base import review_status_enum
from outrider.policy.canonical import compute_hitl_decision_content_hash
from outrider.policy.severity import FindingSeverity
from outrider.schemas.hitl import PerFindingDecision, PerFindingOutcome


def _metrics(**overrides: object) -> ReviewMetricsView:
    base: dict[str, object] = {
        "llm_calls_made": 1,
        "total_input_tokens": 10,
        "total_output_tokens": 5,
        "total_cost_usd": 0.01,
        "files_examined": 5,
        "files_traced_beyond_diff": 2,
        "wall_clock_seconds": 1.5,
    }
    base.update(overrides)
    return ReviewMetricsView(**base)  # type: ignore[arg-type]


def test_metrics_view_allows_none_file_and_wallclock_fields() -> None:
    m = _metrics(files_examined=None, files_traced_beyond_diff=None, wall_clock_seconds=None)
    assert m.files_examined is None
    assert m.files_traced_beyond_diff is None
    assert m.wall_clock_seconds is None
    # LLM aggregates are non-optional (always summable from the stream).
    assert m.llm_calls_made == 1


def test_metrics_view_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReviewMetricsView(
            llm_calls_made=1,
            total_input_tokens=10,
            total_output_tokens=5,
            total_cost_usd=0.01,
            files_examined=5,
            files_traced_beyond_diff=2,
            wall_clock_seconds=1.5,
            unexpected="x",  # type: ignore[call-arg]
        )


def test_list_item_rejects_naive_datetime() -> None:
    aware = datetime(2026, 5, 31, tzinfo=UTC)
    naive = datetime(2026, 5, 31)  # noqa: DTZ001 (intentionally naive for the test)
    common = {
        "id": uuid4(),
        "installation_id": 1,
        "repo_id": 1,
        "repo_full_name": None,
        "pr_number": 1,
        "pr_title": None,
        "head_sha": "sha1",
        "status": "running",
        "is_eval": False,
        "completed_at": None,
        "metrics": _metrics(),
        "severity_counts": None,
    }
    # Aware datetimes construct cleanly.
    ReviewListItem(created_at=aware, updated_at=aware, **common)  # type: ignore[arg-type]
    # Naive datetime is rejected by AwareDatetime.
    with pytest.raises(ValidationError):
        ReviewListItem(created_at=naive, updated_at=naive, **common)  # type: ignore[arg-type]


def test_severity_counts_fields_match_finding_severity() -> None:
    """`SeverityCounts` is built from a `GROUP BY severity` via a whitelist
    comprehension (`{sev: n for ... if sev in model_fields}`), which SILENTLY
    drops any severity not present as a field. Couple the closed key set to the
    canonical `FindingSeverity` so a new tier can't be silently under-counted
    (Class-10: centrally-pinned contract requires call-side registration)."""
    assert set(SeverityCounts.model_fields) == {s.value for s in FindingSeverity}


def test_status_counts_fields_match_review_status_enum() -> None:
    """`StatusCounts` is built from a `GROUP BY status` via the same whitelist
    comprehension, and the dashboard sums it for the "All N" headline — so a
    dropped status under-counts the queue, not just a chip. Couple the closed
    key set to `review_status_enum` so a new status can't be silently dropped."""
    assert set(StatusCounts.model_fields) == set(review_status_enum.enums)


def test_detail_forbids_extra_fields() -> None:
    aware = datetime(2026, 5, 31, tzinfo=UTC)
    with pytest.raises(ValidationError):
        ReviewDetail(
            id=uuid4(),
            installation_id=1,
            repo_id=1,
            pr_number=1,
            head_sha="sha1",
            status="completed",
            is_eval=False,
            created_at=aware,
            updated_at=aware,
            completed_at=aware,
            expires_at=None,
            metrics=_metrics(),
            policy_version="1.0.0",
            bogus=1,  # type: ignore[call-arg]
        )


def test_hitl_decisions_from_events_projects_only_from_passed_events() -> None:
    # DECISIONS#034 + single-snapshot consistency: the projection's ONLY source is the events it is
    # handed (reconstruct()'s VERIFIED stream), so a decision absent from those events cannot reach
    # findings[].hitl_decision — no fresh DB query, no other source.
    fid = uuid4()
    decision = PerFindingDecision(
        finding_id=fid,
        outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
        reason="downgraded: test-only path",
        original_severity=FindingSeverity.CRITICAL,
        override_severity=FindingSeverity.HIGH,
    )
    event = HITLDecisionEvent(
        review_id=uuid4(),
        reviewer_id="admin",
        decisions=(decision,),
        annotation=None,
        decided_at=datetime(2026, 6, 1, tzinfo=UTC),
        decision_latency_seconds=1.0,
        decisions_content_hash=compute_hitl_decision_content_hash(
            decisions=(decision,), annotation=None
        ),
    )
    # Present in the events → projected (stream-canonical, never the V1-null table columns).
    projected = _hitl_decisions_from_events((event,))
    assert projected[str(fid)].outcome == "severity_override"
    assert projected[str(fid)].original_severity == "critical"
    assert projected[str(fid)].override_severity == "high"
    assert projected[str(fid)].reviewer_id == "admin"
    # Absent from the events → cannot appear (the regression Codex asked for).
    assert _hitl_decisions_from_events(()) == {}
