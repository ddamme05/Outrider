"""AuditPersister content-resurrection guard — purged-content re-emit is no-op.

Pins the spec's "Post-retention content-resurrection guard": when the
retention sweep has purged a `llm_call_content` row but the parent
`audit_events` row remains (per #014 append-only), a producer-side
re-emit MUST NOT resurrect raw prompt/completion content.

Threat model: retention contract guarantees content is purged after
TTL. A naive ON CONFLICT DO NOTHING re-insert (no resurrection guard)
would find no PK conflict on `llm_call_content` (the row was deleted)
and SUCCEED — resurrecting content the retention sweep deliberately
removed. The guard checks content-row existence on audit-row conflict
and returns as no-op if absent.
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
    )


async def test_persist_does_not_resurrect_purged_content_same_payload(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Persist event X (both rows land). Manually delete llm_call_content
    (simulates retention sweep). Re-persist X with same content. The
    persister returns no-op; content row stays absent."""
    event_obj = llm_call_event_factory(persister_setup.review_id)
    request = llm_request_factory(persister_setup.review_id, user_prompt="prompt A")
    response = llm_response_factory(text_value="completion A")

    await persister_setup.persister.persist(event_obj, request, response)

    # Simulate retention sweep purging the content row.
    async with persister_setup.engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )

    # Verify the audit row still exists (append-only) but content is gone.
    async with persister_setup.engine.connect() as conn:
        audit_count = await conn.execute(
            text("SELECT COUNT(*) FROM audit_events WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        content_count = await conn.execute(
            text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        assert audit_count.scalar_one() == 1
        assert content_count.scalar_one() == 0

    # Re-persist with the same payload. The guard returns as no-op.
    await persister_setup.persister.persist(event_obj, request, response)

    # Content row is STILL absent — no resurrection.
    async with persister_setup.engine.connect() as conn:
        audit_count = await conn.execute(
            text("SELECT COUNT(*) FROM audit_events WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        content_count = await conn.execute(
            text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        assert audit_count.scalar_one() == 1  # append-only; still here
        assert content_count.scalar_one() == 0  # purged; STILL gone


async def test_persist_does_not_resurrect_purged_content_different_content(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Variant: re-persist with DIFFERENT content after purge. The guard
    fires BEFORE the content compare (audit payload still matches), so
    no content-mismatch exception is raised — the no-op returns first.

    Pins the explicit ordering: resurrection-guard return is reached
    BEFORE any content-side comparison. The retention contract trumps
    content-conflict detection.
    """
    event_obj = llm_call_event_factory(persister_setup.review_id)
    request_a = llm_request_factory(persister_setup.review_id, user_prompt="prompt A")
    response_a = llm_response_factory(text_value="completion A")
    await persister_setup.persister.persist(event_obj, request_a, response_a)

    # Simulate purge.
    async with persister_setup.engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )

    # Re-persist same event with DIFFERENT content. Audit payload matches
    # (same event obj); content would differ (B vs A), but the guard returns
    # before the content compare.
    request_b = llm_request_factory(persister_setup.review_id, user_prompt="prompt B")
    response_b = llm_response_factory(text_value="completion B")
    # No exception; no-op return.
    await persister_setup.persister.persist(event_obj, request_b, response_b)

    async with persister_setup.engine.connect() as conn:
        content_count = await conn.execute(
            text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        assert content_count.scalar_one() == 0
