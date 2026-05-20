"""FileExaminationEvent skip_reason validator tests per DECISIONS.md#018.

Test list mirrors `specs/2026-04-29-audit-events-module.md`'s Actual
outcome supersession note for the audit-events module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.ast_facts.models import SkipReason
from outrider.audit.events import FileExaminationEvent


def _base_kwargs() -> dict[str, Any]:
    """Common required fields for FileExaminationEvent construction.

    `Any` (not `object`) so callers can `**kwargs` into the constructor
    without `# type: ignore[arg-type]` per call site — the test file
    has many such constructions and `object` widens every value.
    """
    return {
        "review_id": uuid4(),
        "timestamp": datetime.now(UTC),
        "file_path": "f.py",
        # `examination_type` is bounded to the literal set actually emitted
        # by src/ — `intake_fetch` (agent/nodes/intake.py) and `analyze`
        # (sister analyze-node spec). A future stage that emits its own
        # examination_type widens the Literal in `audit/events.py` and
        # adds a row to the parametrized values test below.
        "examination_type": "intake_fetch",
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
    event = FileExaminationEvent(**kwargs)
    assert event.parse_status == parse_status


def test_file_examination_event_parse_status_rejects_other_values() -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = "pending"  # not a canonical value
    with pytest.raises(ValidationError):
        FileExaminationEvent(**kwargs)


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
    event = FileExaminationEvent(**kwargs)
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
    event = FileExaminationEvent(**kwargs)
    assert event.skip_reason is None


def test_file_examination_event_skipped_without_skip_reason_raises() -> None:
    kwargs = _base_kwargs()
    kwargs["parse_status"] = "skipped"
    # skip_reason omitted -> defaults to None -> validator rejects
    with pytest.raises(ValidationError):
        FileExaminationEvent(**kwargs)


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
        FileExaminationEvent(**kwargs)


def test_file_examination_event_round_trips_skipped_with_reason() -> None:
    """JSON round-trip preserves both parse_status and skip_reason —
    proves the str-enum serializes correctly and the cross-field
    validator runs on deserialization, not just at construction."""
    kwargs = _base_kwargs()
    kwargs["parse_status"] = "skipped"
    kwargs["skip_reason"] = SkipReason.OVERSIZED
    original = FileExaminationEvent(**kwargs)
    payload = original.model_dump(mode="json")
    rehydrated = FileExaminationEvent.model_validate(payload)
    assert rehydrated.parse_status == "skipped"
    assert rehydrated.skip_reason == SkipReason.OVERSIZED


@pytest.mark.parametrize("examination_type", ["intake_fetch", "analyze"])
def test_file_examination_event_admits_canonical_examination_types(
    examination_type: str,
) -> None:
    """The two values actually emitted by src/ admit cleanly."""
    kwargs = _base_kwargs()
    kwargs["examination_type"] = examination_type
    kwargs["parse_status"] = "clean"
    event = FileExaminationEvent(**kwargs)
    assert event.examination_type == examination_type


@pytest.mark.parametrize(
    "examination_type",
    ["deep", "structural", "shallow", "INTAKE_FETCH", ""],
)
def test_file_examination_event_rejects_unknown_examination_type(
    examination_type: str,
) -> None:
    """A non-canonical examination_type (typo, casing variant, made-up
    third stage) fails at the schema layer. Pre-PR-review-round-5 the
    field was `examination_type: str` and admitted any value; the
    Literal lock landed in round-5 to stop a future emission-site typo
    from drifting through the append-only audit log."""
    kwargs = _base_kwargs()
    kwargs["examination_type"] = examination_type
    kwargs["parse_status"] = "clean"
    with pytest.raises(ValidationError):
        FileExaminationEvent(**kwargs)
