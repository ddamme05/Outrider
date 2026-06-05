"""Unit tests for `ReplayVerdictEvent` (the replay-verdict-projection audit event).

Covers its validators (reason paired with inequivalence; the all-present-or-
all-absent reconstruction-metadata envelope), the `mode` constraint to the
replay modes, the all-absent envelope when reconstruct raised, the
`target_max_sequence_number >= 1` bound, the `extra="forbid"` contract, and the
discriminator round-trip through `AuditEventAdapter` (the union must select
`ReplayVerdictEvent` on `event_type`).
"""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from outrider.audit.events import AuditEventAdapter, ReplayVerdictEvent

_REVIEW_ID = UUID("11111111-1111-1111-1111-111111111111")


def _equivalent() -> ReplayVerdictEvent:
    return ReplayVerdictEvent(
        review_id=_REVIEW_ID,
        replay_equivalent=True,
        mode="full",
        event_count=10,
        finding_count=2,
        orphan_finding_count=0,
        target_max_sequence_number=10,
    )


def test_equivalent_verdict_constructs_with_no_reason() -> None:
    v = _equivalent()
    assert v.event_type == "replay_verdict"
    assert v.replay_equivalent is True
    assert v.reason is None
    assert v.target_max_sequence_number == 10


def test_inequivalent_verdict_requires_a_reason() -> None:
    # All-absent envelope (the reconstruct-raised shape) isolates the reason check.
    with pytest.raises(ValidationError, match="replay_equivalent=False requires a reason"):
        ReplayVerdictEvent(
            review_id=_REVIEW_ID,
            replay_equivalent=False,
            target_max_sequence_number=5,
        )


def test_equivalent_verdict_rejects_a_reason() -> None:
    with pytest.raises(ValidationError, match="replay_equivalent=True must have reason=None"):
        ReplayVerdictEvent(
            review_id=_REVIEW_ID,
            replay_equivalent=True,
            mode="full",
            event_count=1,
            finding_count=0,
            orphan_finding_count=0,
            reason="should not be here",
            target_max_sequence_number=5,
        )


def test_inequivalent_verdict_with_reason_constructs() -> None:
    # An assert_equivalent failure: reconstruction succeeded (full envelope) but the
    # verdict is inequivalent with a reason.
    v = ReplayVerdictEvent(
        review_id=_REVIEW_ID,
        replay_equivalent=False,
        mode="full",
        event_count=5,
        finding_count=2,
        orphan_finding_count=0,
        reason="finding_count mismatch: 5 vs 4",
        target_max_sequence_number=8,
    )
    assert v.replay_equivalent is False
    assert v.reason == "finding_count mismatch: 5 vs 4"


def test_equivalent_verdict_requires_full_envelope() -> None:
    # An equivalent verdict means reconstruction succeeded → mode + counts present.
    with pytest.raises(ValidationError, match="requires the full reconstruction metadata"):
        ReplayVerdictEvent(
            review_id=_REVIEW_ID,
            replay_equivalent=True,
            target_max_sequence_number=5,
        )


def test_partial_metadata_envelope_rejected() -> None:
    # mode + two counts present but orphan_finding_count missing → malformed.
    with pytest.raises(ValidationError, match="all-present or all-absent"):
        ReplayVerdictEvent(
            review_id=_REVIEW_ID,
            replay_equivalent=False,
            mode="full",
            event_count=3,
            finding_count=1,
            reason="reconstruct partial",
            target_max_sequence_number=5,
        )


def test_mode_constrained_to_replay_modes() -> None:
    # Drift guard: the event's `mode` Literal must enumerate exactly ReplayMode's
    # values (a bare Literal is used to avoid a circular import); if ReplayMode
    # changes, this fails and the Literal must be updated in lockstep.
    from outrider.audit.replay import ReplayMode

    assert {m.value for m in ReplayMode} == {"full", "metadata_only", "mixed"}
    with pytest.raises(ValidationError):
        ReplayVerdictEvent(
            review_id=_REVIEW_ID,
            replay_equivalent=True,
            mode="bogus",
            event_count=1,
            finding_count=0,
            orphan_finding_count=0,
            target_max_sequence_number=1,
        )


def test_all_absent_envelope_when_reconstruct_raised() -> None:
    # The absent envelope (mode + counts all None) is the reconstruct-RAISED case
    # (a corrupt row the reconstructor couldn't even read) — an inequivalent verdict
    # carrying the failure reason. NOT metadata-only mode: a legitimate metadata_only
    # replay SUCCEEDS, so it carries mode="metadata_only" + counts.
    v = ReplayVerdictEvent(
        review_id=_REVIEW_ID,
        replay_equivalent=False,
        mode=None,
        event_count=None,
        finding_count=None,
        orphan_finding_count=None,
        reason="reconstruct raised: corrupt payload",
        target_max_sequence_number=3,
    )
    assert v.event_count is None
    assert v.mode is None


def test_target_max_sequence_number_must_be_positive() -> None:
    # sequence_number is a BIGINT IDENTITY starting at 1 — 0 names no real row.
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            ReplayVerdictEvent(
                review_id=_REVIEW_ID,
                replay_equivalent=True,
                target_max_sequence_number=bad,
            )


def test_extra_fields_forbidden() -> None:
    # Full valid envelope so the ONLY error is the extra field — otherwise the
    # envelope validator (equivalent requires the envelope) would raise even if
    # extra="forbid" stopped working, making this test vacuous.
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ReplayVerdictEvent(
            review_id=_REVIEW_ID,
            replay_equivalent=True,
            mode="full",
            event_count=1,
            finding_count=0,
            orphan_finding_count=0,
            target_max_sequence_number=1,
            bogus="x",  # type: ignore[call-arg]
        )


def test_discriminator_round_trip_through_adapter() -> None:
    # The discriminated union selects ReplayVerdictEvent on event_type — so a
    # persisted verdict deserializes back to the right concrete type at replay.
    v = _equivalent()
    restored = AuditEventAdapter.validate_python(v.model_dump(mode="json"))
    assert isinstance(restored, ReplayVerdictEvent)
    assert restored == v
