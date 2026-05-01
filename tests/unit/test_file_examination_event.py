"""FileExaminationEvent skip_reason validator tests per DECISIONS.md#018.

Test list mirrors `specs/2026-04-29-audit-events-module.md`'s Actual
outcome supersession note for the audit-events module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.ast_facts.models import SkipReason
from outrider.audit.events import FileExaminationEvent


def _base_kwargs() -> dict[str, object]:
    """Common required fields for FileExaminationEvent construction."""
    return {
        "review_id": uuid4(),
        "timestamp": datetime.now(UTC),
        "file_path": "f.py",
        "examination_type": "structural",
        "node_id": "intake",
    }


@pytest.mark.parametrize(
    "parse_status",
    ["clean", "degraded", "failed", "skipped"],
)
def test_file_examination_event_parse_status_admits_four_canonical_values(
    parse_status: str,
) -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = parse_status
    if parse_status == "skipped":
        kwargs["skip_reason"] = SkipReason.VENDORED
    event = FileExaminationEvent(**kwargs)  # type: ignore[arg-type]
    assert event.parse_status == parse_status


def test_file_examination_event_parse_status_rejects_other_values() -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = "pending"  # not a canonical value
    with pytest.raises(ValidationError):
        FileExaminationEvent(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "reason",
    list(SkipReason),
)
def test_file_examination_event_skipped_admits_with_skip_reason(
    reason: SkipReason,
) -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = "skipped"
    kwargs["skip_reason"] = reason
    event = FileExaminationEvent(**kwargs)  # type: ignore[arg-type]
    assert event.skip_reason == reason


@pytest.mark.parametrize(
    "parse_status",
    ["clean", "degraded", "failed"],
)
def test_file_examination_event_non_skipped_admits_without_skip_reason(
    parse_status: str,
) -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = parse_status
    event = FileExaminationEvent(**kwargs)  # type: ignore[arg-type]
    assert event.skip_reason is None


def test_file_examination_event_skipped_without_skip_reason_raises() -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = "skipped"
    # skip_reason omitted -> defaults to None -> validator rejects
    with pytest.raises(ValidationError):
        FileExaminationEvent(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "parse_status",
    ["clean", "degraded", "failed"],
)
def test_file_examination_event_non_skipped_with_skip_reason_raises(
    parse_status: str,
) -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = parse_status
    kwargs["skip_reason"] = SkipReason.OVERSIZED
    with pytest.raises(ValidationError):
        FileExaminationEvent(**kwargs)  # type: ignore[arg-type]


def test_file_examination_event_round_trips_skipped_with_reason() -> None:
    """JSON round-trip preserves both parse_status and skip_reason —
    proves the str-enum serializes correctly and the cross-field
    validator runs on deserialization, not just at construction."""
    kwargs = _base_kwargs()
    kwargs["parse_status"] = "skipped"
    kwargs["skip_reason"] = SkipReason.OVERSIZED
    original = FileExaminationEvent(**kwargs)  # type: ignore[arg-type]
    payload = original.model_dump(mode="json")
    rehydrated = FileExaminationEvent.model_validate(payload)
    assert rehydrated.parse_status == "skipped"
    assert rehydrated.skip_reason == SkipReason.OVERSIZED
