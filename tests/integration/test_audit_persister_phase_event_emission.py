"""AuditPersister.emit_phase() — phase event writes + denormalized phase_key.

Pins the C3 fix: persister populates the top-level `audit_events.phase_key`
column from `event.phase_key` so V1.5's per-file index queries
(`ix_audit_events_review_phase_key`) work the day they ship.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        PersisterTestSetup,
        ReviewPhaseEventFactory,
    )


async def test_emit_phase_writes_audit_row_with_correct_fields(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """Happy path: emit_phase() writes an audit_events row with event fields
    denormalized to top-level columns + payload JSONB."""
    event = review_phase_event_factory(persister_setup.review_id, marker="start")
    await persister_setup.persister.emit_phase(event)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text(
                "SELECT event_id, review_id, event_type, phase_key, is_eval, payload "
                "FROM audit_events WHERE event_id = :eid"
            ),
            {"eid": event.event_id},
        )
        result = row.one()
        assert result.event_id == event.event_id
        assert result.review_id == persister_setup.review_id
        assert result.event_type == "review_phase"
        assert result.is_eval is False
        # phase_key was None on the event — column is NULL.
        assert result.phase_key is None
        # Payload JSONB carries the full event dump (sans sequence_number).
        assert result.payload["phase_id"] == event.phase_id
        assert result.payload["marker"] == "start"
        assert result.payload["node_id"] == "triage"
        assert "sequence_number" not in result.payload


async def test_emit_phase_populates_top_level_phase_key_column(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """C3 regression test: when `event.phase_key` is non-None (V1.5 fanout
    case), persister writes it to the top-level `phase_key` column for
    index visibility, not just into the JSONB payload."""
    event = review_phase_event_factory(
        persister_setup.review_id, marker="start", phase_key="analyze:src/foo.py"
    )
    await persister_setup.persister.emit_phase(event)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text("SELECT phase_key, payload FROM audit_events WHERE event_id = :eid"),
            {"eid": event.event_id},
        )
        result = row.one()
        # Top-level column populated for the index.
        assert result.phase_key == "analyze:src/foo.py"
        # Payload also carries it (denormalized from payload).
        assert result.payload["phase_key"] == "analyze:src/foo.py"

        # The index query V1.5 will use returns the row.
        indexed = await conn.execute(
            text("SELECT COUNT(*) FROM audit_events WHERE review_id = :rid AND phase_key = :pk"),
            {"rid": persister_setup.review_id, "pk": "analyze:src/foo.py"},
        )
        assert indexed.scalar_one() == 1


async def test_emit_phase_is_idempotent_on_repeated_same_event(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """Re-emit the same constructed event → no-op (PK conflict + payload-
    equality verification passes)."""
    event = review_phase_event_factory(persister_setup.review_id, marker="start")
    await persister_setup.persister.emit_phase(event)
    await persister_setup.persister.emit_phase(event)

    async with persister_setup.engine.connect() as conn:
        count = await conn.execute(
            text("SELECT COUNT(*) FROM audit_events WHERE event_id = :eid"),
            {"eid": event.event_id},
        )
        assert count.scalar_one() == 1


async def test_emit_phase_start_and_end_pair(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """A start/end pair sharing the same phase_id lands as two distinct rows.

    Pins the contract that phase_id is the LOGICAL key but event_id is the PK;
    distinct events with the same phase_id are NOT deduped by the persister.
    """
    phase_id = "shared-phase-id"
    start_event = review_phase_event_factory(
        persister_setup.review_id, marker="start", phase_id=phase_id
    )
    end_event = review_phase_event_factory(
        persister_setup.review_id, marker="end", phase_id=phase_id
    )
    await persister_setup.persister.emit_phase(start_event)
    await persister_setup.persister.emit_phase(end_event)

    async with persister_setup.engine.connect() as conn:
        markers = await conn.execute(
            text(
                "SELECT payload->>'marker' AS marker FROM audit_events "
                "WHERE review_id = :rid AND payload->>'phase_id' = :pid "
                "ORDER BY sequence_number"
            ),
            {"rid": persister_setup.review_id, "pid": phase_id},
        )
        result = [r.marker for r in markers]
        assert result == ["start", "end"]
