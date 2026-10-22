"""Typed, closure-injected seam for durable anomaly emission.

`AnomalyPersister` owns idempotency through per-rule partial unique indexes.
The enums catch producer typos; mandatory `is_eval` keeps eval emissions
out of the production queue. Text columns avoid database enum migrations,
but every new rule requires a matching partial-index migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

    from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity


@runtime_checkable
class AnomalySink(Protocol):
    """Emit an anomaly once per `(review_id, rule_name)`.

    Sweep callers hold `SWEEP_LOCK_ID` for their surrounding status changes
    (`sweep-jobs-use-advisory-locks`). Graph callers use database idempotency
    without a sweep lock; concurrent graph invocations are not serialized by
    LangGraph (`DECISIONS.md#027`).

    Return only after persistence or raise. Callers supply typed severity,
    JSON-native details, and explicit eval scope. The persister starts rows
    with `status='open'`; it exposes no status option on this protocol.
    """

    async def emit_anomaly(
        self,
        *,
        review_id: UUID,
        rule_name: AnomalyRuleName,
        severity: AnomalySeverity,
        details: dict[str, Any],
        is_eval: bool,
    ) -> None:
        """Persist one anomaly; a matching retry returns successfully.

        `is_eval` is required even when a sweep already filtered to production.
        """
        ...


__all__ = ["AnomalySink"]
