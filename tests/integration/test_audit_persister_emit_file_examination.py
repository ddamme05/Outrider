"""Check persistence of file-examination events in PostgreSQL.

Cover the indexed file path, identical retries, changed-payload rejection, and
append-only protection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from outrider.ast_facts.models import SkipReason
from outrider.audit.persister import AuditPersisterIdempotencyConflict

if TYPE_CHECKING:
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        FileExaminationEventFactory,
        PersisterTestSetup,
    )


async def test_emit_file_examination_writes_audit_row_with_correct_fields(
    persister_setup: PersisterTestSetup,
    file_examination_event_factory: FileExaminationEventFactory,
) -> None:
    """Happy path: emit_file_examination writes an audit_events row with
    event fields landed in payload JSONB.

    Note: unlike `phase_key`, `file_path` is NOT a denormalized top-level
    column on `audit_events` — it lives only in the JSONB payload. The
    `phase_key` denormalization exists specifically because V1.5's
    per-file index queries need it; `file_path` queries can scan the
    JSONB directly (no equivalent index pressure).
    """
    event = file_examination_event_factory(
        persister_setup.review_id,
        file_path="src/foo.py",
        parse_status="clean",
    )
    await persister_setup.persister.emit_file_examination(event)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text(
                "SELECT event_id, review_id, event_type, is_eval, payload "
                "FROM audit_events WHERE event_id = :eid"
            ),
            {"eid": event.event_id},
        )
        result = row.one()
        assert result.event_id == event.event_id
        assert result.review_id == persister_setup.review_id
        assert result.event_type == "file_examination"
        assert result.is_eval is False
        # Payload JSONB carries the full event dump (sans sequence_number).
        assert result.payload["file_path"] == "src/foo.py"
        assert result.payload["parse_status"] == "clean"
        assert result.payload["node_id"] == "intake"
        assert "sequence_number" not in result.payload


async def test_emit_file_examination_skipped_with_skip_reason_landed(
    persister_setup: PersisterTestSetup,
    file_examination_event_factory: FileExaminationEventFactory,
) -> None:
    """The skipped variant — skip_reason populated, parse_status='skipped'.

    Pins that the persister writes through both the `parse_status` and
    `skip_reason` fields into the JSONB payload. Binary and oversized
    files currently share `SkipReason.OVERSIZED` pending FUP-033's
    canonical amendment to DECISIONS#018 for a separate `SkipReason.BINARY`.
    """
    event = file_examination_event_factory(
        persister_setup.review_id,
        file_path="src/blob.bin",
        parse_status="skipped",
        skip_reason=SkipReason.OVERSIZED,
    )
    await persister_setup.persister.emit_file_examination(event)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text("SELECT payload FROM audit_events WHERE event_id = :eid"),
            {"eid": event.event_id},
        )
        payload = row.scalar_one()
        assert payload["parse_status"] == "skipped"
        assert payload["skip_reason"] == "OVERSIZED"


async def test_emit_file_examination_idempotent_on_repeated_same_event(
    persister_setup: PersisterTestSetup,
    file_examination_event_factory: FileExaminationEventFactory,
) -> None:
    """Re-emit the same constructed event → no-op (PK conflict + payload-
    equality verification passes). Mirror of the emit_phase
    `test_emit_phase_is_idempotent_on_repeated_same_event` contract.
    """
    event = file_examination_event_factory(
        persister_setup.review_id,
        file_path="src/idempotent.py",
    )
    await persister_setup.persister.emit_file_examination(event)
    await persister_setup.persister.emit_file_examination(event)

    async with persister_setup.engine.connect() as conn:
        count = await conn.execute(
            text("SELECT COUNT(*) FROM audit_events WHERE event_id = :eid"),
            {"eid": event.event_id},
        )
        assert count.scalar_one() == 1


async def test_emit_file_examination_payload_mismatch_raises_conflict(
    persister_setup: PersisterTestSetup,
    file_examination_event_factory: FileExaminationEventFactory,
) -> None:
    """Producer bug: re-emit with same event_id but different payload
    field → AuditPersisterIdempotencyConflict raised.

    Divergence is on `file_path` — a field that should not differ
    between two emissions sharing the same event_id (if file_path
    legitimately changed, the producer should mint a new event_id, not
    re-emit). Same shape as the LLMCallEvent timestamp-divergence test.
    """
    event1 = file_examination_event_factory(persister_setup.review_id, file_path="src/original.py")
    await persister_setup.persister.emit_file_examination(event1)

    # Same event_id, different file_path → conflict.
    event2 = event1.model_copy(update={"file_path": "src/different.py"})
    assert event2.event_id == event1.event_id

    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.emit_file_examination(event2)

    exc = exc_info.value
    assert exc.event_id == event1.event_id
    assert "file_path" in exc.mismatched_fields


async def test_emit_file_examination_append_only_trigger_blocks_update(
    persister_setup: PersisterTestSetup,
    file_examination_event_factory: FileExaminationEventFactory,
) -> None:
    """Trigger pins: emit_file_examination's row is UPDATE/DELETE-protected
    by the audit_append_only_guard trigger applied at genesis migration.
    Cross-reference of `test_audit_append_only_trigger.py` against the
    file_examination event_type specifically.

    The UPDATE target is the JSONB `payload` column (where `file_path`
    actually lives) — same trigger semantics apply.
    """
    event = file_examination_event_factory(persister_setup.review_id, file_path="src/locked.py")
    await persister_setup.persister.emit_file_examination(event)

    # UPDATE attempt → trigger raises a PG exception → SQLAlchemy
    # surfaces as DBAPIError. The exact subclass varies by driver/version;
    # asserting the raise + that the row remains unchanged is enough.
    async with persister_setup.engine.connect() as conn:
        with pytest.raises(Exception, match=r"(?i)audit_events|append-only"):
            async with conn.begin():
                await conn.execute(
                    text(
                        "UPDATE audit_events SET payload = "
                        "jsonb_set(payload, '{file_path}', '\"mutated.py\"') "
                        "WHERE event_id = :eid"
                    ),
                    {"eid": event.event_id},
                )

    # Row content unchanged.
    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text(
                "SELECT payload->>'file_path' AS file_path FROM audit_events WHERE event_id = :eid"
            ),
            {"eid": event.event_id},
        )
        assert row.scalar_one() == "src/locked.py"


async def test_emit_file_examination_is_eval_propagation(
    persister_setup: PersisterTestSetup,
    file_examination_event_factory: FileExaminationEventFactory,
) -> None:
    """is_eval flag on the event lands in the row's top-level column,
    matching emit_phase's behavior. Catches a regression where the new
    persister method might forget to propagate the flag (the existing
    persister tests would not cover the new method's column write).
    """
    event = file_examination_event_factory(
        persister_setup.review_id,
        file_path="src/eval_run.py",
        is_eval=True,
    )
    await persister_setup.persister.emit_file_examination(event)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text("SELECT is_eval FROM audit_events WHERE event_id = :eid"),
            {"eid": event.event_id},
        )
        assert row.scalar_one() is True
