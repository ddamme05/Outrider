"""AuditPersister is_eval propagation + cross-check.

is_eval propagates from the event into both DB rows (audit_events + the content
row). At the two content-bearing sites that resolve the reviews row — `persist()`
(LLMCallEvent) and `emit_finding()` (FindingEvent) — the persister ALSO
cross-checks `event.is_eval` against `reviews.is_eval` and raises
`AuditPersisterIsEvalMismatchError` on divergence (FUP-130). Those two are the
dashboard's is_eval-sensitive read surfaces, and the read-API scopes its
metric/findings queries by review_id alone, trusting the per-event match.
Non-resolving emit paths (`emit_phase`) propagate WITHOUT the cross-check (out of
FUP-130 scope — they would each cost an extra SELECT and feed no is_eval-scoped
dashboard metric).

**Boundary note**: `eval_db` integrity gate at `tests/eval/conftest.py` catches
the eval→production direction (eval test DB shouldn't have is_eval=False rows).
The production→eval direction at construction time is FUP-024's territory; FUP-130
adds the persister-side backstop for the two resolving sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from outrider.audit.persister import AuditPersisterIsEvalMismatchError

if TYPE_CHECKING:
    from uuid import UUID

    from tests.integration.conftest import (  # type: ignore[import-not-found]
        LLMCallEventFactory,
        LLMRequestFactory,
        LLMResponseFactory,
        PersisterTestSetup,
        ReviewPhaseEventFactory,
    )


async def test_persist_propagates_is_eval_true(
    persister_setup: PersisterTestSetup,
    eval_review_id: UUID,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """`is_eval=True` on the event flows into BOTH inserted rows — emitted against
    a matching is_eval=True review (the FUP-130 cross-check requires the match)."""
    event_obj = llm_call_event_factory(eval_review_id, is_eval=True)
    request = llm_request_factory(eval_review_id, is_eval=True)
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


async def test_emit_phase_propagates_is_eval_without_cross_check(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """emit_phase propagates is_eval verbatim and does NOT cross-check it against
    the reviews row — it doesn't resolve the review, so it's outside FUP-130's
    two guarded sites. Both an is_eval=True and an is_eval=False phase event land
    against the same is_eval=False review (a mismatch persist()/emit_finding would
    reject), documenting the scope boundary."""
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


async def test_persist_raises_on_is_eval_mismatch(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """An is_eval=True event against the is_eval=False review is refused with
    AuditPersisterIsEvalMismatchError (FUP-130). Neither row lands — the guard
    fires before the audit/content INSERT, so eval data can't leak into the
    production review's review_id-scoped dashboard metrics."""
    event_obj = llm_call_event_factory(persister_setup.review_id, is_eval=True)
    request = llm_request_factory(persister_setup.review_id, is_eval=True)
    response = llm_response_factory()

    with pytest.raises(AuditPersisterIsEvalMismatchError):
        await persister_setup.persister.persist(event_obj, request, response)

    async with persister_setup.engine.connect() as conn:
        audit_count = (
            await conn.execute(
                text("SELECT COUNT(*) FROM audit_events WHERE event_id = :eid"),
                {"eid": event_obj.event_id},
            )
        ).scalar_one()
        content_count = (
            await conn.execute(
                text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :eid"),
                {"eid": event_obj.event_id},
            )
        ).scalar_one()
    assert audit_count == 0
    assert content_count == 0
