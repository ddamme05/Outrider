"""Review lifecycle writes and decision-preflight reads.

The graph receives these protocols through dependency injection
(nodes-receive-deps-via-closure). Each write atomically pairs a status transition
with its request, decision, or timestamp fields. The dashboard reads the pending
request from this cache because HITL has not yet returned its graph-state delta.

Audit events remain canonical on divergence (DECISIONS.md#016,
audit-events-append-only). ReviewDecidePreflight belongs to the database read
boundary (DECISIONS.md#019).
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict

from outrider.policy import FindingSeverity
from outrider.schemas.hitl import HITLDecision, HITLRequest


class ReviewDecidePreflight(BaseModel):
    """Review cache and audit-derived severity baselines needed before a decision.

    The endpoint gates on status and any prior decision. Gated severities come from
    persisted FindingEvents, preserving severity-policy-versioned-for-replay even
    after the active policy changes. An absent request yields an empty severity map.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    hitl_request: HITLRequest | None
    hitl_decision: HITLDecision | None
    gated_finding_severities: Mapping[UUID, FindingSeverity]


@runtime_checkable
class ReviewStatusSink(Protocol):
    """Atomically persist each lifecycle transition and its paired fields.

    Each method uses one session and UPDATE. A nonmatching predicate is a successful
    no-op, allowing graph replay. Request and decision writes are first-write-only;
    completion accepts only running reviews. The JSONB cache serves dashboard reads;
    audit rows remain canonical (DECISIONS.md#016, audit-events-append-only).
    """

    async def mark_awaiting_approval(
        self,
        *,
        review_id: UUID,
        expires_at: AwareDatetime,
        hitl_request_payload: dict[str, Any],
    ) -> None:
        """Store the first approval request, expiry, and awaiting_approval status.

        Require running and hitl_request IS NULL. Replay after resume must not replace
        the original request or regress the review to awaiting approval.
        """
        ...

    async def mark_running(
        self,
        *,
        review_id: UUID,
        hitl_decision_payload: dict[str, Any],
    ) -> None:
        """Store the first decision and resume an awaiting or expired review.

        Require hitl_decision IS NULL so a conflicting replay cannot replace the cache
        with a decision the audit stream did not accept (DECISIONS.md#016). Keep
        expires_at for forensic visibility; the expiry sweep filters on awaiting status.
        Expired reviews use the remediation path in docs/spec.md §4.1.6.
        """
        ...

    async def mark_awaiting_approval_expired(self, *, review_id: UUID) -> None:
        """Mark an awaiting review expired; an already-expired row is also admitted."""
        ...

    async def mark_completed(self, *, review_id: UUID) -> None:
        """Complete a running review and stamp completed_at at terminal publish success.

        This includes successful graph paths that post nothing. Replays after completion
        match no rows. Completion makes the review eligible for retention purge
        (docs/spec.md §3.3 step 10; docs/architecture.md step 10).
        """
        ...


@runtime_checkable
class ReviewStatusReader(Protocol):
    """Read the persisted approval request before admitting a human decision.

    HITL writes the request before interrupt; its graph-state delta returns only
    after resume. Read gated severities from the canonical audit stream, bounded by
    the request's 256-finding cap.
    """

    async def fetch_for_decide(self, *, review_id: UUID) -> ReviewDecidePreflight | None:
        """Return None for an absent review, otherwise deserialize its request and decision.

        Load gated severities in the same session only when the request exists.
        """
        ...


__all__ = [
    "ReviewDecidePreflight",
    "ReviewStatusReader",
    "ReviewStatusSink",
]
