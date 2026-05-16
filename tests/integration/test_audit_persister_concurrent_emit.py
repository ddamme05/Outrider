"""AuditPersister concurrent emission — fresh AsyncSession per call.

Pins the "fresh `AsyncSession` per call" contract from both Protocols'
docstrings. Fan out N concurrent `emit_phase()` calls from the same
persister instance; all N rows must land, no session-sharing exception.

V1.5 parallel-analyze fanout will issue concurrent persist()/emit_phase()
calls from N worker tasks — without per-call session acquisition, those
calls would share a single AsyncSession (which is NOT concurrent-safe per
SQLAlchemy docs) and silently corrupt transactions.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        PersisterTestSetup,
        ReviewPhaseEventFactory,
    )


async def test_concurrent_emit_phase_all_rows_land(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """Fan out 16 concurrent emit_phase() calls; all 16 rows must land.

    If the persister shared a single AsyncSession across coroutines (a
    naive impl), some calls would race and lose writes, OR raise
    `InvalidRequestError: This session is provisioning a new connection`
    deep inside SQLAlchemy.
    """
    n = 16
    events = [
        review_phase_event_factory(
            persister_setup.review_id, marker="start", phase_key=f"branch:{i:03d}"
        )
        for i in range(n)
    ]

    await asyncio.gather(*(persister_setup.persister.emit_phase(e) for e in events))

    async with persister_setup.engine.connect() as conn:
        count = await conn.execute(
            text(
                "SELECT COUNT(*) FROM audit_events "
                "WHERE event_type = 'review_phase' AND review_id = :rid"
            ),
            {"rid": persister_setup.review_id},
        )
        assert count.scalar_one() == n


async def test_concurrent_emit_phase_unique_event_ids(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """Every concurrent emit produced a distinct row (no silent drops).

    Stronger than just counting — distinct event_ids prove the persister
    isn't silently treating concurrent calls as same-event idempotency
    no-ops.
    """
    n = 16
    events = [
        review_phase_event_factory(
            persister_setup.review_id, marker="start", phase_key=f"branch:{i:03d}"
        )
        for i in range(n)
    ]
    expected_ids = {e.event_id for e in events}
    assert len(expected_ids) == n  # sanity: factory mints unique event_ids

    await asyncio.gather(*(persister_setup.persister.emit_phase(e) for e in events))

    async with persister_setup.engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT event_id FROM audit_events "
                "WHERE event_type = 'review_phase' AND review_id = :rid"
            ),
            {"rid": persister_setup.review_id},
        )
        actual_ids = {row.event_id for row in result}
        assert actual_ids == expected_ids
