"""AuditPersister.emit_observed_skip_shadow() — persist + replay the OBSERVED
skip-shadow telemetry (Cost Lever 3, DECISIONS.md#049).

Pins that the shadow event's coverage envelope (`covering_matches`) + `blockers`
+ `changed_regions` survive the JSONB round-trip and reconstruct via
`AuditEventAdapter` (replay-equivalence) with side + spans intact, and that
`is_eval` threads to the row's top-level column. Mirrors
`test_audit_persister_emit_file_examination.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from outrider.audit.events import (
    AuditEventAdapter,
    ObservedSkipChangedRegion,
    ObservedSkipCoveringMatch,
    ObservedSkipShadowEvent,
)

if TYPE_CHECKING:
    from uuid import UUID

    from tests.integration.conftest import PersisterTestSetup  # type: ignore[import-not-found]


def _not_eligible(review_id: UUID, *, is_eval: bool = False) -> ObservedSkipShadowEvent:
    region = ObservedSkipChangedRegion(side="head", line_start=10, line_end=14)
    return ObservedSkipShadowEvent(
        review_id=review_id,
        is_eval=is_eval,
        file_path="src/foo.py",
        outcome="not_eligible",
        changed_regions=(region,),
        blockers=(region,),
    )


def _would_skip(review_id: UUID, *, is_eval: bool = False) -> ObservedSkipShadowEvent:
    region = ObservedSkipChangedRegion(side="head", line_start=10, line_end=14)
    match = ObservedSkipCoveringMatch(
        query_match_id="python.q", side="head", line_start=8, line_end=16
    )
    return ObservedSkipShadowEvent(
        review_id=review_id,
        is_eval=is_eval,
        file_path="src/foo.py",
        outcome="would_skip",
        changed_regions=(region,),
        covering_matches=(match,),
    )


async def test_emit_observed_skip_shadow_not_eligible_persists_blockers(
    persister_setup: PersisterTestSetup,
) -> None:
    """not_eligible writes the audit row with changed_regions + blockers in the
    JSONB payload and an empty covering_matches."""
    event = _not_eligible(persister_setup.review_id)
    await persister_setup.persister.emit_observed_skip_shadow(event)

    async with persister_setup.engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT event_type, is_eval, payload FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
        ).one()
    assert row.event_type == "observed_skip_shadow"
    assert row.is_eval is False
    assert row.payload["outcome"] == "not_eligible"
    assert row.payload["node_id"] == "analyze"
    assert row.payload["file_path"] == "src/foo.py"
    assert row.payload["changed_regions"][0] == {"side": "head", "line_start": 10, "line_end": 14}
    assert row.payload["blockers"][0]["side"] == "head"
    assert row.payload["covering_matches"] == []
    assert "sequence_number" not in row.payload


async def test_emit_observed_skip_shadow_would_skip_replays_with_envelope(
    persister_setup: PersisterTestSetup,
) -> None:
    """would_skip persists the covering envelope, and the row reconstructs via the
    AuditEventAdapter (replay-equivalence) with the covering match's side + span
    intact — the promotion proof survives content retention."""
    event = _would_skip(persister_setup.review_id)
    await persister_setup.persister.emit_observed_skip_shadow(event)

    async with persister_setup.engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT sequence_number, payload FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
        ).one()
    reconstructed = AuditEventAdapter.validate_python(
        {**row.payload, "sequence_number": row.sequence_number}
    )
    assert isinstance(reconstructed, ObservedSkipShadowEvent)
    assert reconstructed.outcome == "would_skip"
    assert reconstructed.blockers == ()
    assert len(reconstructed.covering_matches) == 1
    covering = reconstructed.covering_matches[0]
    assert covering.query_match_id == "python.q"
    assert covering.side == "head"
    assert (covering.line_start, covering.line_end) == (8, 16)


async def test_emit_observed_skip_shadow_is_eval_propagation(
    persister_setup: PersisterTestSetup,
) -> None:
    """is_eval on the event lands in the row's top-level column (eval isolation)."""
    event = _not_eligible(persister_setup.review_id, is_eval=True)
    await persister_setup.persister.emit_observed_skip_shadow(event)

    async with persister_setup.engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT is_eval FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
        ).one()
    assert row.is_eval is True
