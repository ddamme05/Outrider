"""trace_decision natural-key partial unique index (M7 a + DECISIONS.md#026).

Revision ID: 8f2a4c1e7b3d
Revises: 3d03bca7f2be
Create Date: 2026-05-24

Per `specs/2026-05-23-trace-node.md` M7 (a) + `DECISIONS.md#026` natural-key
idempotency mode (first instance: trace's TraceDecisionEvent). Adds a
partial unique index on `audit_events` that closes the race between
concurrent emissions of the same logical trace decision:

    CREATE UNIQUE INDEX uq_audit_events_trace_decision_natural_key
        ON audit_events (review_id, (payload->>'source_finding_id'))
        WHERE event_type = 'trace_decision';

The DB-level constraint is load-bearing for M7's audit-first emission
contract: trace emits the TraceDecisionEvent BEFORE returning the state
delta from the node function, and the persister-side
`_persist_keyed_by_natural_key` helper (Group 4) relies on this index
firing `IntegrityError(UniqueViolation)` (caught and translated to a
no-op return via `postgresql_insert(...).on_conflict_do_nothing(...)`)
when a retry / replay / V1.5-parallel-analyze invocation produces a
second TraceDecisionEvent with the same `(review_id, source_finding_id)`.

Partial-index design choices:

  - `event_type = 'trace_decision'` WHERE clause restricts the
    constraint to trace-decision rows only. Other event types
    (FindingEvent, LLMCallEvent, PublishRoutingEvent, etc.) stay
    under the existing `event_id`-PK idempotency mode per #026; their
    `(review_id, source_finding_id)` tuples are NOT subject to this
    uniqueness rule. Mixing modes per event type is supported by
    design (#026 point 4).

  - `(payload->>'source_finding_id')` expression — the trace decision's
    natural-key component is stored in the JSONB payload. PostgreSQL
    supports expression-on-jsonb-key in unique indexes; the field is
    a UUID-shaped string per `TraceDecisionEvent.source_finding_id: UUID`
    serialization. `payload->>'...'` returns text (UUIDs serialize as
    canonical-form strings), so the index ordering is lexicographic on
    the string representation — fine for uniqueness, irrelevant for
    range queries (this index is consulted only for natural-key
    conflict detection on INSERT, not for ordered scans).

  - `review_id` is the index's first column (not in the WHERE clause):
    it's the natural-key tuple's first component AND a real top-level
    column on `audit_events`. The follow-up SELECT after conflict
    filters on `(review_id, payload->>'source_finding_id')` — both
    components are in the index lookup, so the SELECT uses this index
    too.

No new event-type discriminator (TraceDecisionEvent.event_type='trace_decision'
already exists per the audit-events module). No JSONB-schema constraint
needed (Outrider's audit-payload pattern per #014 / #016 doesn't use
DB-level JSON constraints). No backfill required: per #024 + #025 + #026
Migration sections, no production TraceDecisionEvent rows exist yet
(trace hasn't shipped); the partial unique index applies to the
currently-empty event-type partition cleanly.

The index is IF NOT EXISTS-guarded so a partial-failure rerun lands
cleanly (sibling pattern to 3d03bca7f2be's idempotent CHECK + trigger
creations).

See:
  - specs/2026-05-23-trace-node.md M7 (a)
  - DECISIONS.md#026 (natural-key idempotency mode — first instance: trace)
  - DECISIONS.md#017 × #024 amendment (one decision per source_finding_id)
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f2a4c1e7b3d"
down_revision: str | Sequence[str] | None = "3d03bca7f2be"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the trace_decision natural-key partial unique index.

    IF NOT EXISTS-guarded — a partial-failure rerun lands cleanly. The
    index applies to the trace-decision event-type partition only;
    other event types stay under event_id-PK idempotency per #026.
    """
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_audit_events_trace_decision_natural_key
            ON audit_events (review_id, (payload->>'source_finding_id'))
            WHERE event_type = 'trace_decision';
        """
    )


def downgrade() -> None:
    """Drop the partial unique index."""
    op.execute("DROP INDEX IF EXISTS uq_audit_events_trace_decision_natural_key;")
