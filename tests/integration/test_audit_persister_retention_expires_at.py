"""AuditPersister retention TTL — explicit retention_expires_at write + override.

Pins:
  - Default TTL (90 days from DECISIONS#016) is applied to `retention_expires_at`.
  - `retention_expires_at = event.timestamp + ttl` (deterministic from event).
  - Override via constructor's `retention_settings` propagates end-to-end.
  - Aware-datetime semantics: tzinfo is UTC, never None.
"""

from __future__ import annotations

from datetime import UTC, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.audit.config import RetentionSettings
from outrider.audit.persister import AuditPersister

if TYPE_CHECKING:
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        LLMCallEventFactory,
        LLMRequestFactory,
        LLMResponseFactory,
        PersisterTestSetup,
    )


async def test_retention_expires_at_uses_default_90_day_ttl(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """`retention_expires_at = event.timestamp + 90 days` for the default TTL."""
    event_obj = llm_call_event_factory(persister_setup.review_id)
    request = llm_request_factory(persister_setup.review_id)
    response = llm_response_factory()

    await persister_setup.persister.persist(event_obj, request, response)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text("SELECT retention_expires_at FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        retention_db = row.scalar_one()

    expected = event_obj.timestamp + timedelta(days=90)
    # Within 1 second tolerance (DB roundtrip can drift microseconds).
    delta = abs((retention_db - expected).total_seconds())
    assert delta < 1.0


async def test_retention_expires_at_is_aware_utc(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Returned timestamp has tzinfo (aware datetime; never naive)."""
    event_obj = llm_call_event_factory(persister_setup.review_id)
    request = llm_request_factory(persister_setup.review_id)
    response = llm_response_factory()

    await persister_setup.persister.persist(event_obj, request, response)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            text("SELECT retention_expires_at FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        retention_db = row.scalar_one()
        assert retention_db.tzinfo is not None
        # Postgres returns UTC for TIMESTAMPTZ; assert that semantic.
        assert retention_db.utcoffset() == UTC.utcoffset(retention_db)


async def test_retention_expires_at_honors_operator_override(
    migrated_db: str,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Operator constructs the persister with `RetentionSettings(ttl=7 days)`;
    the persisted row carries the overridden TTL. Proves DECISIONS#012's
    operator-overridable property end-to-end."""
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        _seed_install_and_review,
    )

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id = await _seed_install_and_review(engine)

        override_settings = RetentionSettings(llm_content_retention_ttl=timedelta(days=7))
        persister = AuditPersister(
            session_factory=async_sessionmaker(engine, expire_on_commit=False),
            retention_settings=override_settings,
        )

        event_obj = llm_call_event_factory(review_id)
        request = llm_request_factory(review_id)
        response = llm_response_factory()
        await persister.persist(event_obj, request, response)

        async with engine.connect() as conn:
            row = await conn.execute(
                text("SELECT retention_expires_at FROM llm_call_content WHERE event_id = :eid"),
                {"eid": event_obj.event_id},
            )
            retention_db = row.scalar_one()

        expected = event_obj.timestamp + timedelta(days=7)
        delta = abs((retention_db - expected).total_seconds())
        assert delta < 1.0
    finally:
        await engine.dispose()
