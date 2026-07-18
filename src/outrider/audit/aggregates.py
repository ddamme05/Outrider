# See DECISIONS.md#030 — this helper populates the (nullable-for-read-compat)
# ReviewMetrics / SynthesizeCompletedEvent LLM aggregates that #030 governs.
"""Shared read-through aggregation of a review's LLM-call metrics.

The single place that "sums this review's `LLMCallEvent` rows", consumed by BOTH the
read side (dashboard `_aggregate_metrics`) AND the write side (the synthesize node, via
`AuditPersister.query_review_llm_aggregates`, populating its `SynthesizeCompletedEvent`
aggregates — FUP-093). Because both go through this one function, the persisted audit
row and the dashboard badge are computed by one aggregation path and cannot diverge —
the FUP-093 "single-source it" requirement.

The function is session-taking (not session-opening) so each caller supplies its own
session model: the persister method opens a per-call session and delegates here; the
dashboard passes its request-scoped session.

**V1 aggregation is a naive `COUNT`/`SUM` — correct only while the dispatcher is
non-durable.** Exactly one `llm_call` row lands per logical call under
`BackgroundTasksDispatcher` (it never replays a node body; HITL-resume re-enters
at the hitl node, after the LLM-calling nodes). When durable retry lands (V2
Celery + Redis), a crash-recovery re-emit mints a fresh `event_id` and lands a
SECOND `llm_call` row — and `event_id`-PK dedup cannot catch a fresh-UUID re-emit.
At that point this SUM double-counts cost/tokens/calls and MUST dedup to one row
per logical call. The dedup key is a contract decision (the V2 `llm_call_event_id`
binding, `DECISIONS.md#029`) deferred to V2 and tracked in FUP-145 — NOT part of
FUP-093's close; decide it once HERE and both the dashboard read and the synthesize
write inherit it automatically (single source). Until then, this is the documented
V1 floor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID  # noqa: TC003  (runtime: function param type)

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Integer, Numeric, case, cast, func, select

from outrider.audit.events import LLMCallEvent
from outrider.db.models.audit_events import AuditEvent

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Pulled from the schema declaration, not hardcoded, so a future rename of the
# discriminator Literal cannot silently disable the filter (mirrors
# `AuditPersister.query_prior_publish_event`).
_LLM_CALL_EVENT_TYPE: str = LLMCallEvent.model_fields["event_type"].default


class ReviewLLMAggregates(BaseModel):
    """A review's LLM-call totals, summed from its `LLMCallEvent` audit rows.

    Completeness-aware since the openai-native-host arc: `cost_usd` is nullable
    on genuinely unpriceable completed calls (typed-reason coupled), SQL `SUM`
    skips those NULLs, so `total_cost_usd` is the KNOWN SUBTOTAL — a lower
    bound whenever `cost_complete` is False. Consumers must render "at least
    $X" / "incomplete" from these two fields rather than presenting the
    subtotal as exact (#016/#039 honest data)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_calls_made: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    unpriced_call_count: int = Field(default=0, ge=0)
    cost_complete: bool = True


async def aggregate_review_llm_metrics(
    session: AsyncSession, *, review_id: UUID, is_eval: bool
) -> ReviewLLMAggregates:
    """Sum a review's `llm_call` rows: count + input/output tokens + cost.

    Every event is scoped by BOTH `review_id` AND `is_eval` (the FUP-130 read-side
    defense): a divergent eval `llm_call` row must not leak into a production
    review's totals. `is_eval` is passed by the caller (it is the review's own
    `is_eval`), never inferred from the events.

    Read-only `SELECT` — respects the audit append-only boundary. `COALESCE(..., 0)`
    makes a review with zero `llm_call` rows return zeros (not NULL). See the module
    docstring for the V1-naive-SUM / V2-dedup contract.
    """
    stmt = select(
        func.count().label("calls"),
        func.coalesce(func.sum(cast(AuditEvent.payload["input_tokens"].astext, Integer)), 0).label(
            "input_tokens"
        ),
        func.coalesce(func.sum(cast(AuditEvent.payload["output_tokens"].astext, Integer)), 0).label(
            "output_tokens"
        ),
        func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0).label(
            "cost_usd"
        ),
        # Unpriced rows carry payload cost_usd=null (typed-reason coupled; pre-field
        # historical rows are always numeric) — SUM above skips them, this counts them
        # so the subtotal is never silently presented as complete.
        func.coalesce(
            func.sum(case((AuditEvent.payload["cost_usd"].astext.is_(None), 1), else_=0)), 0
        ).label("unpriced_calls"),
    ).where(
        AuditEvent.review_id == review_id,
        AuditEvent.is_eval == is_eval,
        AuditEvent.event_type == _LLM_CALL_EVENT_TYPE,
    )
    row = (await session.execute(stmt)).one()
    return ReviewLLMAggregates(
        llm_calls_made=row.calls,
        total_input_tokens=row.input_tokens,
        total_output_tokens=row.output_tokens,
        total_cost_usd=float(row.cost_usd),
        unpriced_call_count=row.unpriced_calls,
        cost_complete=row.unpriced_calls == 0,
    )
