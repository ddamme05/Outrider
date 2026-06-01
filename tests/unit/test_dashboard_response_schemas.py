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
)


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
        "pr_number": 1,
        "head_sha": "sha1",
        "status": "running",
        "is_eval": False,
        "completed_at": None,
        "metrics": _metrics(),
    }
    # Aware datetimes construct cleanly.
    ReviewListItem(created_at=aware, updated_at=aware, **common)  # type: ignore[arg-type]
    # Naive datetime is rejected by AwareDatetime.
    with pytest.raises(ValidationError):
        ReviewListItem(created_at=naive, updated_at=naive, **common)  # type: ignore[arg-type]


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
            bogus=1,  # type: ignore[call-arg]
        )
