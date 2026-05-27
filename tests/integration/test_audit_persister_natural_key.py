"""AuditPersister natural-key idempotency — DB-touching contract pin.

Per specs/2026-05-23-trace-node.md M7 (b) + DECISIONS.md#026 (first
instance: trace's TraceDecisionEvent). The identity subset is
`{source_finding_id, target_file, resolution_status, is_eval}` —
`resolved_candidate_paths` is EXCLUDED per #026 point 3 because it
is derived from LLM-ranking-order-variant `proposed_import_strings`
and would defeat the lockstep contract on legitimate retries.

Verifies the three load-bearing paths through
`_persist_keyed_by_natural_key`:

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
    # divergences into a tuple before raising. Per `DECISIONS.md#026`
    # `resolved_candidate_paths` is EXCLUDED from the identity subset
    # (it is derived from LLM-ranking-order-variant
    # `proposed_import_strings`), so even though the cross-field
    # validator forces the diverging event to update
    # `resolved_candidate_paths` in lockstep with `target_file`, only
    # `target_file` surfaces in `mismatched_fields`. The sorted-tuple
    # discipline pins the contract; a bare `in mismatched_fields`
    # membership check would mask a regression that started flagging
    # the excluded field too.
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

    # Sorted tuple per the helper's `sorted(...)` discipline. Per
    # `DECISIONS.md#026` `resolved_candidate_paths` is EXCLUDED from
    # the identity subset (LLM-ranking variance via its
    # `proposed_import_strings` derivation), so the cross-field
    # validator's lockstep update of resolved_candidate_paths to ()
    # does NOT surface here. Only the two identity-subset members
    # (`resolution_status` and `target_file`) appear in the divergence.
    assert exc_info.value.mismatched_fields == (
        "resolution_status",
        "target_file",
    )


async def test_ambiguous_resolved_paths_divergence_collapses_to_no_op(
    persister_setup: PersisterTestSetup,
) -> None:
    """For `resolution_status='ambiguous'` outcomes `target_file` is None
    (cross-field validator), and per `DECISIONS.md#026`
    `resolved_candidate_paths` is EXCLUDED from the identity subset
    (LLM-ranking-order variance via its `proposed_import_strings`
    derivation defeats lockstep). The identity-subset members
    (`source_finding_id`, `target_file=None`, `resolution_status=
    ambiguous`, `is_eval`) are identical across retries, so a second
    emission whose ambiguous candidate SET differs from the first
    no-ops back to the persisted original. The returned event carries
    the original's `resolved_candidate_paths`, preserving the lockstep
    contract between state-mirror and audit row across retries even
    though the LLM proposed a different ranking the second time."""
    source_finding_id = uuid4()
    first = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file=None,
        resolution_status="ambiguous",
        resolved_candidate_paths=("src/foo.py", "src/bar.py"),
    )
    persisted_first = await persister_setup.persister.emit_trace_decision(first)

    diverging = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file=None,
        resolution_status="ambiguous",
        resolved_candidate_paths=("src/foo.py", "src/baz.py"),
    )

    persisted_retry = await persister_setup.persister.emit_trace_decision(diverging)

    # No-op recovery path: returned event is the persisted ORIGINAL.
    # `resolved_candidate_paths` is OUT of the identity subset per
    # #026, so divergence here collapses to no-op rather than raising.
    assert persisted_retry.event_id == persisted_first.event_id
    assert persisted_retry.resolved_candidate_paths == first.resolved_candidate_paths

    # Still exactly one row.
    count = await _count_trace_decision_rows(
        persister_setup, persister_setup.review_id, source_finding_id
    )
    assert count == 1


async def test_no_conflict_when_ambiguous_resolved_paths_reordered(
    persister_setup: PersisterTestSetup,
) -> None:
    """Per `DECISIONS.md#026` `resolved_candidate_paths` is EXCLUDED
    from the identity subset entirely — any reorder OR set divergence
    collapses to a no-op (legitimate retry with different LLM ranking).
    This test pins the reorder-specific path; the set-divergence path
    is covered by
    `test_ambiguous_resolved_paths_divergence_collapses_to_no_op` above."""
    source_finding_id = uuid4()
    first = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file=None,
        resolution_status="ambiguous",
        resolved_candidate_paths=("src/foo.py", "src/bar.py"),
    )
    persisted_first = await persister_setup.persister.emit_trace_decision(first)

    reordered = _build_trace_decision_event(
        persister_setup.review_id,
        source_finding_id,
        target_file=None,
        resolution_status="ambiguous",
        resolved_candidate_paths=("src/bar.py", "src/foo.py"),
    )
    persisted_retry = await persister_setup.persister.emit_trace_decision(reordered)

    # No-op recovery path: returned event is the persisted ORIGINAL,
    # not the reordered incoming event. This is the audit-first
    # contract (`AuditFirstEmitSink.emit_trace_decision` returns the
    # canonical persisted event so state and audit stay in lockstep
    # across retries).
    assert persisted_retry.event_id == persisted_first.event_id

    # Still exactly one row.
    count = await _count_trace_decision_rows(
        persister_setup, persister_setup.review_id, source_finding_id
    )
    assert count == 1


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
                # `sequence_number` is `sa.Identity(always=False)` —
                # Postgres assigns from the table's identity sequence.
                # Don't hard-code it here; an inserted value would mask
                # a regression in the Identity column without surfacing
                # one in the test's actual assertion.
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
                # `sequence_number` is `sa.Identity(always=False)`;
                # Postgres auto-assigns. See sibling test above for
                # the rationale.
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
