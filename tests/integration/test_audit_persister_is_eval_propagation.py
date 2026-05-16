"""AuditPersister is_eval propagation — flows from event through to DB rows.

The persister NEVER re-decides eval status; it propagates whatever the event
says. Both inserted rows (audit_events + llm_call_content) carry the event's
is_eval value. Producer-side discipline (state.is_eval flowing into event
construction) is the only gate against contamination; FUP-024 covers that.

**Boundary note**: `eval_db` integrity gate at `tests/eval/conftest.py:202-237`
catches the eval→production direction (eval test DB shouldn't have
is_eval=False rows). The production→eval contamination direction (real
review's events tagged is_eval=True) is FUP-024's territory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        LLMCallEventFactory,
        LLMRequestFactory,
        LLMResponseFactory,
        PersisterTestSetup,
        ReviewPhaseEventFactory,
    )


async def test_persist_propagates_is_eval_true(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """`is_eval=True` on the event flows into BOTH inserted rows."""
    event_obj = llm_call_event_factory(persister_setup.review_id, is_eval=True)
    request = llm_request_factory(persister_setup.review_id, is_eval=True)
    response = llm_response_factory()

    await persister_setup.persister.persist(event_obj, request, response)

    async with persister_setup.engine.connect() as conn:
        audit_row = await conn.execute(
            text("SELECT is_eval FROM audit_events WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        content_row = await conn.execute(
            text("SELECT is_eval FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        assert audit_row.scalar_one() is True
        assert content_row.scalar_one() is True


async def test_persist_propagates_is_eval_false(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """`is_eval=False` (the production default) flows into BOTH rows."""
    event_obj = llm_call_event_factory(persister_setup.review_id, is_eval=False)
    request = llm_request_factory(persister_setup.review_id, is_eval=False)
    response = llm_response_factory()

    await persister_setup.persister.persist(event_obj, request, response)

    async with persister_setup.engine.connect() as conn:
        audit_row = await conn.execute(
            text("SELECT is_eval FROM audit_events WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        content_row = await conn.execute(
            text("SELECT is_eval FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        assert audit_row.scalar_one() is False
        assert content_row.scalar_one() is False


async def test_emit_phase_propagates_is_eval(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """emit_phase honors is_eval the same way persist does."""
    event_eval = review_phase_event_factory(persister_setup.review_id, marker="start", is_eval=True)
    event_prod = review_phase_event_factory(
        persister_setup.review_id, marker="start", is_eval=False
    )
    await persister_setup.persister.emit_phase(event_eval)
    await persister_setup.persister.emit_phase(event_prod)

    async with persister_setup.engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT event_id, is_eval FROM audit_events "
                "WHERE review_id = :rid AND event_type = 'review_phase' "
                "ORDER BY sequence_number"
            ),
            {"rid": persister_setup.review_id},
        )
        result = {row.event_id: row.is_eval for row in rows}
        assert result[event_eval.event_id] is True
        assert result[event_prod.event_id] is False
