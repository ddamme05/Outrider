# See DECISIONS.md#030 — this helper populates the (nullable-for-read-compat)
# ReviewMetrics / SynthesizeCompletedEvent LLM aggregates that #030 governs.
"""One query for dashboard and synthesis LLM-call totals.

`aggregate_review_llm_metrics` sums a review's audit rows in the caller's
session. The dashboard supplies its request session; `AuditPersister`
opens a session for synthesis. Both use the same aggregation rules, though
a later read can include calls made after an earlier synthesis snapshot.

V1 counts stored rows. Durable node retries that mint new event IDs would
double-count a logical call; V2 must choose a logical-call dedup key here
so both consumers inherit it (`DECISIONS.md#029`, FUP-145).
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
    """Sum call count, tokens, and known cost for a review and its eval scope.

    The caller supplies the review's `is_eval`; mismatched event rows are excluded.
    This read-only query respects the append-only audit boundary. Empty reviews
    return zeros; unpriced calls make the cost subtotal incomplete.
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
