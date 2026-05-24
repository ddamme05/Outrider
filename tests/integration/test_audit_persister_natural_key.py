"""AuditPersister natural-key idempotency — DB-touching contract pin.

Per specs/2026-05-23-trace-node.md M7 (b) + DECISIONS.md#026 (first
instance: trace's TraceDecisionEvent). Verifies the three load-bearing
paths through `_persist_keyed_by_natural_key`:

  1. **Insert path** — first emission with a fresh
     `(review_id, source_finding_id)` writes the row and returns the
     incoming event verbatim.
  2. **No-op path** — second emission with the SAME
     `(review_id, source_finding_id)` BUT differing per-emission fields
     (event_id, timestamp, reason, proposed_import_strings,
     resolved_candidate_paths) writes NO new row and returns the
     ORIGINALLY-PERSISTED event (identity-subset comparison succeeds).
     Pins the M7 (b) audit-first emission lockstep-recovery contract:
     state is built from the returned event, so state and audit stay
     aligned even when per-emission fields diverge across retries.
  3. **Conflict path** — second emission with the same natural-key but
     a DIFFERENT `target_file` (one of the identity-subset fields) raises
     `AuditPersisterNaturalKeyConflict` carrying both `event_id`s,
     `review_id`, `source_finding_id`, and `mismatched_fields=('target_file',)`.

Plus three companion checks:
  - emit_trace_decision routes through the helper (returns the same
    canonical event as a direct call would).
  - Mode-mixing: a `FindingEvent` with the same `(review_id,
    source_finding_id)` does NOT collide (partial-WHERE narrow scope).
  - is_eval divergence is treated as conflict per M7 (c) — invariant
    per review.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from outrider.audit.events import TraceDecisionEvent
from outrider.audit.persister import AuditPersisterNaturalKeyConflict

if TYPE_CHECKING:
    from tests.integration.conftest import PersisterTestSetup


def _build_trace_decision_event(
    review_id: UUID,
    source_finding_id: UUID,
    *,
    target_file: str | None = "src/middleware/auth.py",
    resolution_status: Literal["resolved", "unresolved", "ambiguous"] = "resolved",
    is_eval: bool = False,
    reason: str = "matches authentication call site",
    proposed_import_strings: tuple[str, ...] = ("middleware.auth",),
    resolved_candidate_paths: tuple[str, ...] = ("src/middleware/auth.py",),
    timestamp: datetime | None = None,
) -> TraceDecisionEvent:
    """Build a TraceDecisionEvent fixture for the natural-key tests."""
    return TraceDecisionEvent(
        review_id=review_id,
        source_finding_id=source_finding_id,
        timestamp=timestamp or datetime.now(UTC),
        target_file=target_file,
        reason=reason,
        resolution_status=resolution_status,
        proposed_import_strings=proposed_import_strings,
        resolved_candidate_paths=resolved_candidate_paths,
        is_eval=is_eval,
    )


async def _count_trace_decision_rows(
    persister_setup: PersisterTestSetup,
    review_id: UUID,
    source_finding_id: UUID,
) -> int:
    """Return how many trace_decision rows exist for the given natural-key
    in the test DB. Used to assert the no-op path doesn't insert a second
    row."""
    async with persister_setup.engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT count(*)::int FROM audit_events "
                "WHERE review_id = :rid "
                "  AND event_type = 'trace_decision' "
                "  AND payload->>'source_finding_id' = :sfid"
            ),
            {"rid": str(review_id), "sfid": str(source_finding_id)},
        )
        return int(result.scalar_one())


# ---------------------------------------------------------------------------
# Insert path
# ---------------------------------------------------------------------------


async def test_insert_path_returns_incoming_event_and_writes_row(
    persister_setup: PersisterTestSetup,
) -> None:
    """Fresh natural-key: helper inserts the row + returns the incoming
    event verbatim."""
    source_finding_id = uuid4()
    event = _build_trace_decision_event(persister_setup.review_id, source_finding_id)

    returned = await persister_setup.persister.emit_trace_decision(event)

    assert returned == event  # field-by-field equality on Pydantic models
    assert returned.event_id == event.event_id
    count = await _count_trace_decision_rows(
        persister_setup, persister_setup.review_id, source_finding_id
    )
    assert count == 1


# ---------------------------------------------------------------------------
# No-op path: identity-subset equality despite per-emission divergence
# ---------------------------------------------------------------------------


async def test_no_op_path_returns_existing_event_when_identity_subset_matches(
    persister_setup: PersisterTestSetup,
) -> None:
    """Second emission with the same natural-key but differing
    per-emission fields (event_id + timestamp + reason +
    proposed_import_strings + resolved_candidate_paths) returns the
    ORIGINALLY-persisted event (per M7 (b) lockstep-recovery). No new
    row is written."""
    source_finding_id = uuid4()
    first = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        reason="first-call reason",
        proposed_import_strings=("middleware.auth", "auth.middleware"),
        resolved_candidate_paths=("src/middleware/auth.py",),
    )
    persisted_first = await persister_setup.persister.emit_trace_decision(first)
    assert persisted_first == first

    # Build a second event with the SAME natural-key but a different
    # event_id (auto via default_factory), different timestamp, and
    # different per-emission fields. Identity-subset (source_finding_id,
    # target_file, resolution_status, is_eval) is identical.
    second = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        reason="retry — fresh Haiku ranking",  # per-emission divergence
        proposed_import_strings=("auth.middleware",),  # re-ranked
        resolved_candidate_paths=("src/middleware/auth.py",),
        timestamp=datetime.now(UTC) + timedelta(seconds=5),  # per-emission
    )
    assert second.event_id != first.event_id
    assert second.timestamp != first.timestamp
    assert second.reason != first.reason
    assert second.proposed_import_strings != first.proposed_import_strings

    persisted_second = await persister_setup.persister.emit_trace_decision(second)

    # The returned event is the ORIGINALLY-PERSISTED one (first), not the
    # incoming second — this is the load-bearing M7 (b) contract.
    assert persisted_second.event_id == first.event_id
    assert persisted_second.timestamp == first.timestamp
    assert persisted_second.reason == first.reason
    assert persisted_second.proposed_import_strings == first.proposed_import_strings
    assert persisted_second.resolved_candidate_paths == first.resolved_candidate_paths

    # No new row was inserted.
    count = await _count_trace_decision_rows(
        persister_setup, persister_setup.review_id, source_finding_id
    )
    assert count == 1


# ---------------------------------------------------------------------------
# Conflict path: identity-subset divergence raises
# ---------------------------------------------------------------------------


async def test_conflict_path_raises_on_target_file_divergence(
    persister_setup: PersisterTestSetup,
) -> None:
    """Producer-bug simulation: second emission with same natural-key
    but a DIFFERENT target_file (identity-subset member) raises
    AuditPersisterNaturalKeyConflict carrying both event_ids + the
    mismatched field name. No new row is written (conflict short-
    circuits before any INSERT could complete)."""
    source_finding_id = uuid4()
    first = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file="src/middleware/auth.py",
    )
    await persister_setup.persister.emit_trace_decision(first)

    diverging = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file="src/handlers/login.py",  # identity-subset divergence
        # Schema's cross-field validator: resolved → target_file ==
        # resolved_candidate_paths[0]. Update both in lockstep to
        # construct a valid event with a diverging target_file.
        resolved_candidate_paths=("src/handlers/login.py",),
    )

    with pytest.raises(AuditPersisterNaturalKeyConflict) as exc_info:
        await persister_setup.persister.emit_trace_decision(diverging)

    exc = exc_info.value
    assert exc.existing_event_id == first.event_id
    assert exc.incoming_event_id == diverging.event_id
    assert exc.review_id == persister_setup.review_id
    assert exc.source_finding_id == source_finding_id
    # Exact-equality on the contract: the helper sorts identity-subset
    # divergences into a tuple before raising. A target_file-only
    # divergence MUST surface as exactly ("target_file",) — a wider
    # tuple would mean the helper compared an extra field by accident
    # (membership-only assertion would mask that bug per the
    # vacuous-pass test anti-pattern in memory).
    assert exc.mismatched_fields == ("target_file",)

    # Still exactly one row — conflict short-circuited the INSERT.
    count = await _count_trace_decision_rows(
        persister_setup, persister_setup.review_id, source_finding_id
    )
    assert count == 1


async def test_conflict_path_raises_on_resolution_status_divergence(
    persister_setup: PersisterTestSetup,
) -> None:
    """`resolution_status` is in the identity-subset — divergence raises.
    Cross-field validator on TraceDecisionEvent forces target_file=None
    when resolution_status != 'resolved', so the second emission also
    diverges on target_file. mismatched_fields carries both."""
    source_finding_id = uuid4()
    first = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file="src/middleware/auth.py",
        resolution_status="resolved",
    )
    await persister_setup.persister.emit_trace_decision(first)

    diverging = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file=None,
        resolution_status="unresolved",
        resolved_candidate_paths=(),
    )

    with pytest.raises(AuditPersisterNaturalKeyConflict) as exc_info:
        await persister_setup.persister.emit_trace_decision(diverging)

    # Sorted tuple per the helper's `sorted(...)` discipline. Exact-
    # equality pins the cross-field validator behavior:
    # `resolution_status='unresolved'` forces `target_file=None`, so
    # the diverging event has BOTH fields different from the persisted
    # `resolved`/`src/middleware/auth.py` original.
    assert exc_info.value.mismatched_fields == ("resolution_status", "target_file")


async def test_conflict_path_raises_on_is_eval_divergence(
    persister_setup: PersisterTestSetup,
) -> None:
    """`is_eval` is invariant per review — cross-retry divergence is a
    config bug per M7 (c). The identity-subset comparison catches it."""
    source_finding_id = uuid4()
    first = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        is_eval=False,
    )
    await persister_setup.persister.emit_trace_decision(first)

    diverging = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        is_eval=True,
    )

    with pytest.raises(AuditPersisterNaturalKeyConflict) as exc_info:
        await persister_setup.persister.emit_trace_decision(diverging)

    # is_eval-only divergence (all other identity-subset fields match);
    # exact-equality pins the contract.
    assert exc_info.value.mismatched_fields == ("is_eval",)


# ---------------------------------------------------------------------------
# Mode-mixing: non-trace events with same natural-key tuple do NOT collide
# ---------------------------------------------------------------------------


async def test_finding_event_with_same_natural_key_does_not_collide(
    persister_setup: PersisterTestSetup,
) -> None:
    """Per DECISIONS.md#026 mode-mixing rule: a FindingEvent with the
    same `(review_id, source_finding_id)` tuple as an existing
    TraceDecisionEvent does NOT raise NaturalKeyConflict — the partial
    unique index's WHERE clause excludes it. Verified at the DB level
    via direct INSERT (a FindingEvent doesn't naturally carry
    `source_finding_id` in its payload, so this test exercises the
    partial-WHERE scope rather than the persister's emit_finding path)."""
    source_finding_id = uuid4()
    trace_event = _build_trace_decision_event(persister_setup.review_id, source_finding_id)
    await persister_setup.persister.emit_trace_decision(trace_event)

    # Direct INSERT of a non-trace_decision row with the same natural-key
    # tuple. The partial index's `event_type='trace_decision'` WHERE
    # excludes this row entirely, so no uniqueness check fires.
    async with persister_setup.engine.begin() as conn:
        metadata = sa.MetaData()
        await conn.run_sync(lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"]))
        audit_events_table = metadata.tables["audit_events"]
        await conn.execute(
            audit_events_table.insert().values(
                event_id=uuid4(),
                review_id=persister_setup.review_id,
                event_type="finding",
                timestamp=datetime.now(UTC),
                sequence_number=99,
                is_eval=False,
                payload={"source_finding_id": str(source_finding_id), "kind": "other"},
            )
        )
        # No IntegrityError raised — partial index scope held.


async def test_emit_trace_decision_succeeds_after_foreign_event_with_same_natural_key(
    persister_setup: PersisterTestSetup,
) -> None:
    """Reverse of the preceding test: a foreign-event row with the same
    `(review_id, payload->>'source_finding_id')` tuple already exists
    when `emit_trace_decision` fires. The partial-index WHERE excludes
    the foreign row, so the trace insert should take the INSERT path
    (NOT the no-op path) and return the incoming event. Pins the
    arbiter-binding semantics in the more dangerous direction (foreign
    row pre-existing — drift in `index_where` would cause the trace
    insert to mistakenly conflict against the foreign row)."""
    source_finding_id = uuid4()

    # Insert the foreign-mode row FIRST.
    async with persister_setup.engine.begin() as conn:
        metadata = sa.MetaData()
        await conn.run_sync(lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"]))
        audit_events_table = metadata.tables["audit_events"]
        await conn.execute(
            audit_events_table.insert().values(
                event_id=uuid4(),
                review_id=persister_setup.review_id,
                event_type="finding",
                timestamp=datetime.now(UTC),
                sequence_number=42,
                is_eval=False,
                payload={"source_finding_id": str(source_finding_id), "kind": "other"},
            )
        )

    # Now emit a TraceDecisionEvent with the SAME natural-key.
    # Insert path: returns the incoming event (not a conflict).
    trace_event = _build_trace_decision_event(persister_setup.review_id, source_finding_id)
    returned = await persister_setup.persister.emit_trace_decision(trace_event)
    assert returned.event_id == trace_event.event_id  # insert-path winner

    # Exactly one trace_decision row exists for this natural-key.
    count = await _count_trace_decision_rows(
        persister_setup, persister_setup.review_id, source_finding_id
    )
    assert count == 1
