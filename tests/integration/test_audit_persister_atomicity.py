"""AuditPersister.persist() atomicity — failure on second insert rolls back first.

Pins `DECISIONS.md#016` single-transaction-insert at the persister-public-method
layer (the raw-SQL test at `test_llm_content_single_transaction.py` only proves
the DB-layer shape; this proves the persister honors it via its real method).

Fault-injection approach: SQLAlchemy engine-level `before_execute` event
listener that raises a synthetic exception when the statement targets
`llm_call_content`. After C1's `installation_id`-via-SELECT design, naturally-
failing FK paths are unreachable — fault injection is the only realistic way
to test the rollback through the real public method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import event, text
from sqlalchemy.dialects.postgresql import Insert

if TYPE_CHECKING:
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        LLMCallEventFactory,
        LLMRequestFactory,
        LLMResponseFactory,
        PersisterTestSetup,
    )


class _InjectedFaultError(RuntimeError):
    """Synthetic exception raised by the fault-injection listener."""


def _install_content_insert_fault(engine_sync: Any) -> None:
    """Register a `before_execute` listener that raises on llm_call_content INSERTs.

    Listener inspects the `clauseelement`; if it's an Insert against the
    `llm_call_content` table, raises `_InjectedFaultError`. Other statements
    (the audit_events INSERT, the SELECTs) pass through unmodified.
    """

    def _listener(  # type: ignore[no-untyped-def]
        _conn,
        clauseelement,
        _multiparams,
        _params,
        _execution_options,
    ):
        if isinstance(clauseelement, Insert) and clauseelement.table.name == "llm_call_content":
            raise _InjectedFaultError("injected fault on llm_call_content INSERT")
        return clauseelement, _multiparams, _params

    event.listen(engine_sync, "before_execute", _listener, retval=True)


async def test_persister_rollback_on_content_insert_failure(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Inject failure on llm_call_content INSERT; assert both rows roll back.

    The audit_events INSERT happens first inside the transaction (succeeds);
    the SELECT-installation_id happens between (succeeds); the content
    INSERT raises the synthetic fault; the transaction rolls back; neither
    row is visible afterward.
    """
    _install_content_insert_fault(persister_setup.engine.sync_engine)

    event_obj = llm_call_event_factory(persister_setup.review_id)
    request = llm_request_factory(persister_setup.review_id)
    response = llm_response_factory()

    with pytest.raises(_InjectedFaultError):
        await persister_setup.persister.persist(event_obj, request, response)

    # Post-rollback: neither row exists.
    async with persister_setup.engine.connect() as conn:
        audit_count = await conn.execute(
            text("SELECT COUNT(*) FROM audit_events WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        content_count = await conn.execute(
            text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )
        assert audit_count.scalar_one() == 0
        assert content_count.scalar_one() == 0


async def test_persister_atomicity_does_not_swallow_injected_exception(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """The fault propagates out as the original exception — not wrapped,
    swallowed, or translated. Caller sees the real cause.
    """
    _install_content_insert_fault(persister_setup.engine.sync_engine)

    event_obj = llm_call_event_factory(persister_setup.review_id)
    request = llm_request_factory(persister_setup.review_id)
    response = llm_response_factory()

    with pytest.raises(_InjectedFaultError, match="injected fault"):
        await persister_setup.persister.persist(event_obj, request, response)
