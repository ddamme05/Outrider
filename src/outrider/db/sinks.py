"""Review-status sink + reader Protocols owned by the `db/` subsystem.

Mirrors the `audit/sinks.py` precedent (one Protocol per node responsibility,
`@runtime_checkable`, durable + recording implementations) but for the
`reviews` table lifecycle writes the HITL node owns: status flips +
JSONB column writes paired in single transactions, plus the read surface
the dashboard `/decide` endpoint consumes at interrupt time.

Why a sink Protocol rather than direct `db_factory: async_sessionmaker`
injection (intake's pattern): the HITL writes pair a STATUS flip with a
JSONB column write atomically, and the resume path is a SECOND such write
at a SEPARATE moment. A typed Protocol with three methods documents the
single-transaction contract cleanly and lets recording test doubles
capture both transitions for assertion. Per the
`nodes-receive-deps-via-closure` invariant, both Protocols are
constructor-injected at `build_graph(...)` time.

Per `DECISIONS.md#019`: cross-boundary models with a clear owner live
with their owner. `ReviewDecidePreflight` lives here because `db/` owns
the read-path shape; the dashboard endpoint + the
`ReviewStatusReader.fetch_for_decide` implementation both consume it.
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict

from outrider.policy import FindingSeverity
from outrider.schemas.hitl import HITLDecision, HITLRequest


class ReviewDecidePreflight(BaseModel):
    """Snapshot of `reviews` row state the `/decide` endpoint needs before
    admitting a reviewer decision.

    Returned by `ReviewStatusReader.fetch_for_decide`. Frozen + extras-
    forbidden per the cross-boundary-model convention. Four fields:

    - `status` — string form of `reviews.status` (Postgres enum value).
      Endpoint compares against `{'awaiting_approval',
      'awaiting_approval_expired'}` for the state gate.
    - `hitl_request` — Pydantic-deserialized `reviews.hitl_request` JSONB
      cache. Canonical at interrupt time (the audit row is canonical
      post-resume). `None` when the column is NULL — endpoint returns
      409 Conflict.
    - `hitl_decision` — Pydantic-deserialized `reviews.hitl_decision`
      JSONB cache. Non-None means a decision already landed (single-shot
      per spec Non-goals); endpoint returns 409 Conflict.
    - `gated_finding_severities` — finding_id -> FindingSeverity map for
      every finding in `hitl_request.findings_requiring_approval`,
      sourced from the persisted-at-admit-time `FindingEvent.severity`
      (carries the policy version per
      `severity-policy-versioned-for-replay`). Endpoint uses this to
      derive server-side `original_severity` for `PerFindingDecision`
      construction. Empty when `hitl_request is None`. Bounded by gated-
      set size <= 256.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    hitl_request: HITLRequest | None
    hitl_decision: HITLDecision | None
    gated_finding_severities: Mapping[UUID, FindingSeverity]


@runtime_checkable
class ReviewStatusSink(Protocol):
    """Sink for lifecycle status transitions + paired JSONB writes the
    HITL node owns. Three methods, all idempotent against target state.

    Single-transaction contract: each method opens ONE `AsyncSession`,
    runs ONE atomic `UPDATE reviews SET ...` statement covering status +
    JSONB column + (where applicable) `expires_at`, commits.

    Idempotency contract: every method MUST be a no-op when the row is
    already in the target state. A `rowcount=0` UPDATE returns
    successfully (NOT an error). This is load-bearing for the resume-
    body re-run path: `mark_awaiting_approval` fires twice in the happy
    case (once on first body invocation, once on resume body re-run);
    the second call must not error. Same for `mark_running` against a
    row already in `running`, and `mark_awaiting_approval_expired`
    against a row already in `awaiting_approval_expired`.

    Source-of-truth contract: on cross-persister divergence (JSONB
    column vs `audit_events` row), the audit row is canonical per
    `audit-events-append-only` + `DECISIONS.md#016`. The JSONB column
    is a convenience cache for the dashboard read path. Replay
    reconstruction always reads `audit_events`, never the JSONB.
    """

    async def mark_awaiting_approval(
        self,
        *,
        review_id: UUID,
        expires_at: AwareDatetime,
        hitl_request_payload: dict[str, Any],
    ) -> None:
        """Flip `reviews.status` from `running` -> `awaiting_approval`,
        atomically writing `expires_at` + `hitl_request` JSONB.

        Predicate: `WHERE id=:r AND status='running' AND hitl_request IS NULL`.
        The `hitl_request IS NULL` discriminator makes the method
        first-write-only: subsequent calls (re-emit on resume body
        re-run; concurrent second background task post-first-completion
        when `status='running'` again) see `hitl_request` already
        populated and no-op (rowcount=0). Without this discriminator,
        a post-completion replay (status flipped back to `running` by
        the first task's `mark_running`) would REGRESS to
        `awaiting_approval`, corrupting the lifecycle.
        """
        ...

    async def mark_running(
        self,
        *,
        review_id: UUID,
        hitl_decision_payload: dict[str, Any],
    ) -> None:
        """Flip `reviews.status` back to `running` on resume, atomically
        writing `hitl_decision` JSONB.

        Predicate: `WHERE id=:r AND status IN ('awaiting_approval',
        'awaiting_approval_expired', 'running')`. Admits
        `awaiting_approval_expired` so a reviewer's late decision
        against an expired review progresses correctly; admits
        `running` for the idempotent re-fire no-op semantic on resume
        body re-run.

        `reviews.expires_at` is LEFT IN PLACE (NOT cleared to NULL).
        The sweep filter is `status='awaiting_approval' AND expires_at
        < NOW()`; once status moves past `awaiting_approval`, the row
        is no longer in the sweep's query result regardless of
        `expires_at` value. Keeping the value preserves forensic
        visibility (when did HITL fire? when would it have expired?)
        at zero correctness cost.
        """
        ...

    async def mark_awaiting_approval_expired(self, *, review_id: UUID) -> None:
        """Flip `reviews.status` -> `awaiting_approval_expired` (the
        sweep-job entrypoint).

        Predicate: `WHERE id=:r AND status IN ('awaiting_approval',
        'awaiting_approval_expired')`. First clause is the canonical
        transition; second admits idempotent re-fire when a sweep tick
        re-processes an already-expired row (the partial unique index
        on the anomaly side makes the paired anomaly emit no-op via
        `on_conflict_do_nothing`).
        """
        ...


@runtime_checkable
class ReviewStatusReader(Protocol):
    """Read surface for the dashboard `/decide` endpoint at interrupt
    time.

    Endpoint reads the canonical HITLRequest snapshot from
    `reviews.hitl_request` JSONB, NOT from `graph.aget_state(...)`. The
    state delta `{"hitl_request": ..., "hitl_decision": ...}` only
    returns AFTER resume completes (step 13 of the HITL node body), so
    at interrupt time the state-snapshot view of `hitl_request` is
    empty; the only place the canonical snapshot exists is the JSONB
    column the node body's `mark_awaiting_approval` wrote atomically
    BEFORE `interrupt()`.

    `gated_finding_severities` is populated via a sibling SELECT against
    `audit_events` filtered on `event_type='finding'` for the gated set,
    reading `FindingEvent.severity` (the persisted-at-admit-time
    severity carrying the policy version under which the finding was
    admitted). Bounded by gated-set size <= 256.
    """

    async def fetch_for_decide(self, *, review_id: UUID) -> ReviewDecidePreflight | None:
        """Return the `/decide` endpoint preflight snapshot, or `None`
        if the review row does not exist.

        Runs TWO SELECTs in one session: (1) `SELECT status,
        hitl_request, hitl_decision FROM reviews WHERE id=:r`; (2) -
        only when `hitl_request is not None` - a JSONB-filtered SELECT
        against `audit_events` to populate `gated_finding_severities`.
        Both JSONB columns deserialize via `HITLRequest.model_validate`
        and `HITLDecision.model_validate` respectively.
        """
        ...


__all__ = [
    "ReviewDecidePreflight",
    "ReviewStatusReader",
    "ReviewStatusSink",
]
