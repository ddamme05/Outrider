"""Verify rolling upgrades and protect permanent audit metadata on downgrade."""

from collections.abc import Awaitable, Callable
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

_PRIOR = "c5a71b9de406"
_REVISION = "f29c7b41e3da"


async def test_migration_preserves_old_rows_and_refuses_to_drop_stored_receipts(
    fresh_db: str, alembic_runner: Callable[[str, str, str], Awaitable[None]]
) -> None:
    engine = create_async_engine(fresh_db, hide_parameters=True)
    old_write = text(
        "INSERT INTO audit_events (review_id, event_type, payload) "
        "VALUES (:review, 'llm_call', '{\"node_id\":\"triage\"}')"
    )
    new_write = text(
        "INSERT INTO audit_events (review_id, event_type, payload, triage_input) "
        "VALUES (:review, :kind, CAST(:payload AS jsonb), CAST(:coverage AS jsonb))"
    )
    coverage = '{"files":[],"omitted_files":0}'
    review = uuid4()
    try:
        await alembic_runner("upgrade", _PRIOR, fresh_db)
        async with engine.begin() as conn:
            await conn.execute(old_write, {"review": review})
        await alembic_runner("upgrade", _REVISION, fresh_db)
        async with engine.begin() as conn:
            await conn.execute(old_write, {"review": review})
            assert (
                await conn.scalar(
                    text("SELECT count(*) FROM audit_events WHERE triage_input IS NULL")
                )
                == 2
            )

        # Only historical rows exist. Returning to the old schema loses no evidence.
        await alembic_runner("downgrade", _PRIOR, fresh_db)
        await alembic_runner("upgrade", _REVISION, fresh_db)

        for kind, payload, value in (
            ("llm_call", '{"node_id":"analyze"}', coverage),
            ("llm_call", "{}", coverage),
            ("review_phase", '{"node_id":"triage"}', coverage),
            ("llm_call", '{"node_id":"triage"}', "null"),
        ):
            with pytest.raises(IntegrityError, match="ck_audit_events_triage_input"):
                async with engine.begin() as conn:
                    await conn.execute(
                        new_write, dict(review=review, kind=kind, payload=payload, coverage=value)
                    )

        async with engine.begin() as conn:
            await conn.execute(
                new_write,
                dict(
                    review=review,
                    kind="llm_call",
                    payload='{"node_id":"triage"}',
                    coverage=coverage,
                ),
            )
        with pytest.raises(RuntimeError, match="Stored triage input prevents this downgrade"):
            await alembic_runner("downgrade", _PRIOR, fresh_db)
        async with engine.connect() as conn:
            assert await conn.scalar(text("SELECT version_num FROM alembic_version")) == _REVISION
            assert await conn.scalar(
                text("SELECT triage_input FROM audit_events WHERE triage_input IS NOT NULL")
            ) == {"files": [], "omitted_files": 0}
    finally:
        await engine.dispose()
