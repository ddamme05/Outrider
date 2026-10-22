"""Coordinate Slack cards with the durable review journey.

For HITL, check prior posts, build the approval card, post it, then record its
message id. Reviews without a prior approval card receive a compact posted notice.
Publication can update the original approval card with one terminal rendering;
concurrent writers must agree because chat.update has no compare-and-set (FUP-252).

Ordinary transport and audit failures are logged and contained (degrades-gracefully);
cancellation propagates. Posting before recording leaves a crash window that can duplicate a
notification on replay. Dependencies come from the per-install resolver and stay
outside ReviewState (nodes-receive-deps-via-closure, state-is-pure-data).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from outrider.audit.events import SlackNotificationEvent
from outrider.notify.base import SlackNotifyError
from outrider.notify.deeplink import build_review_deeplink
from outrider.notify.messages import (
    build_hitl_pending_message,
    build_review_posted_message,
    build_status_mirror_message,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from uuid import UUID

    from outrider.audit.sinks import SlackEventSink
    from outrider.notify.base import SlackNotifier, SlackPostResult
    from outrider.schemas.review_finding import ReviewFinding

__all__ = ["SlackNotificationOrchestrator", "SlackNotifyTarget", "SlackTargetResolver"]

logger = logging.getLogger(__name__)


class SlackNotificationOrchestrator:
    """Check deduplication, build, post, and record without raising ordinary failures.

    Post methods return SlackPostResult or None; mirror_status returns whether the
    existing card was updated. These calls are awaited and cancellation propagates.
    """

    def __init__(
        self,
        *,
        notifier: SlackNotifier,
        sink: SlackEventSink,
        dashboard_base_url: str,
    ) -> None:
        self._notifier = notifier
        self._sink = sink
        self._dashboard_base_url = dashboard_base_url

    async def notify_hitl_pending(
        self,
        *,
        review_id: UUID,
        is_eval: bool,
        channel_id: str,
        repo: str,
        pr_number: int,
        pr_title: str,
        findings: Sequence[ReviewFinding],
    ) -> SlackPostResult | None:
        """Post the rich HITL-pending card when a review enters awaiting_approval.

        Skips (returns None) if a `hitl_pending` notification already exists for
        `(review_id, channel_id)` — the common replay case.
        """
        try:
            if await self._already_posted(review_id, channel_id, "hitl_pending"):
                return None
            deep_link = build_review_deeplink(self._dashboard_base_url, review_id)
            message = build_hitl_pending_message(
                repo=repo,
                pr_number=pr_number,
                pr_title=pr_title,
                findings=findings,
                deep_link=deep_link,
            )
            result = await self._notifier.post_message(
                channel=channel_id, text=message.text, blocks=message.blocks
            )
        except SlackNotifyError as exc:
            logger.warning(
                "Slack hitl_pending post failed (%s); degrading to dashboard",
                type(exc).__name__,
                extra={"review_id": str(review_id), "channel_id": channel_id},
            )
            return None
        except Exception:
            logger.exception(
                "Unexpected error building/posting Slack hitl_pending notification",
                extra={"review_id": str(review_id), "channel_id": channel_id},
            )
            return None
        await self._record(review_id, is_eval, channel_id, result.ts, "hitl_pending")
        return result

    async def notify_review_posted(
        self,
        *,
        review_id: UUID,
        is_eval: bool,
        channel_id: str,
        repo: str,
        pr_number: int,
        posted_count: int,
        dashboard_only_count: int,
    ) -> SlackPostResult | None:
        """Post the compact review-posted FYI for a review that published without gating.

        Skips (returns None) if a `review_posted` row already exists (replay) OR a
        `hitl_pending` row exists (the review gated -> its terminal is the status
        mirror, not an FYI; the two message classes are mutually exclusive).
        """
        try:
            if await self._already_posted(review_id, channel_id, "review_posted"):
                return None
            if await self._already_posted(review_id, channel_id, "hitl_pending"):
                return None
            deep_link = build_review_deeplink(self._dashboard_base_url, review_id)
            message = build_review_posted_message(
                repo=repo,
                pr_number=pr_number,
                posted_count=posted_count,
                dashboard_only_count=dashboard_only_count,
                deep_link=deep_link,
            )
            result = await self._notifier.post_message(
                channel=channel_id, text=message.text, blocks=message.blocks
            )
        except SlackNotifyError as exc:
            logger.warning(
                "Slack review_posted post failed (%s); degrading to dashboard",
                type(exc).__name__,
                extra={"review_id": str(review_id), "channel_id": channel_id},
            )
            return None
        except Exception:
            logger.exception(
                "Unexpected error building/posting Slack review_posted notification",
                extra={"review_id": str(review_id), "channel_id": channel_id},
            )
            return None
        await self._record(review_id, is_eval, channel_id, result.ts, "review_posted")
        return result

    async def mirror_status(
        self,
        *,
        review_id: UUID,
        channel_id: str,
        repo: str,
        pr_number: int,
        pr_title: str,
        findings: Sequence[ReviewFinding],
        reviewer_id: str | None = None,
        approved_count: int = 0,
        gated_count: int = 0,
        posted_count: int = 0,
        dashboard_only_count: int = 0,
    ) -> bool:
        """`chat.update` the review's HITL card to its terminal state. Returns whether
        the edit was applied.

        Targets the `message_ts` recorded on the original `hitl_pending` event — the
        mirror edits the card that exists, and never posts a new one. No matching row
        is a legitimate no-op, not an error: the review never gated, the original post
        failed, or the install's channel was rotated since the card was posted (the
        lookup is channel-scoped, so a rotated config finds nothing rather than editing
        an unrelated message in the new channel).

        Emits NO audit event (spec: Audit events emitted) — the decision and publish
        facts are already captured by `HITLDecisionEvent` / `PublishEvent`, and
        `audit-events-append-only` forbids mutating the original notification row.
        `chat.update` is idempotent for a given rendering, so replay needs no dedup.
        """
        try:
            existing = await self._sink.query_slack_notification(
                review_id=review_id, channel_id=channel_id, kind="hitl_pending"
            )
            if existing is None:
                return False
            deep_link = build_review_deeplink(self._dashboard_base_url, review_id)
            message = build_status_mirror_message(
                repo=repo,
                pr_number=pr_number,
                pr_title=pr_title,
                findings=findings,
                deep_link=deep_link,
                reviewer_id=reviewer_id,
                approved_count=approved_count,
                gated_count=gated_count,
                posted_count=posted_count,
                dashboard_only_count=dashboard_only_count,
            )
            await self._notifier.update_message(
                channel=existing.channel_id,
                ts=existing.message_ts,
                text=message.text,
                blocks=message.blocks,
            )
        except SlackNotifyError as exc:
            logger.warning(
                "Slack status mirror failed (%s); card left at its prior state",
                type(exc).__name__,
                extra={"review_id": str(review_id), "channel_id": channel_id},
            )
            return False
        except Exception:
            logger.exception(
                "Unexpected error building/posting Slack status mirror",
                extra={"review_id": str(review_id), "channel_id": channel_id},
            )
            return False
        return True

    async def _already_posted(
        self, review_id: UUID, channel_id: str, kind: Literal["hitl_pending", "review_posted"]
    ) -> bool:
        """Best-effort pre-post dedup: True if a matching notification row exists."""
        existing = await self._sink.query_slack_notification(
            review_id=review_id, channel_id=channel_id, kind=kind
        )
        return existing is not None

    async def _record(
        self,
        review_id: UUID,
        is_eval: bool,
        channel_id: str,
        message_ts: str,
        kind: Literal["hitl_pending", "review_posted"],
    ) -> None:
        """Append the `SlackNotificationEvent` after a successful post (post-then-record).

        A failure here — event construction OR emit — is logged, not raised: the
        post already happened but the audit row did not, so replay may re-post once
        (the V1 crash-window residual). Construction is inside the try so a schema
        violation (e.g. an over-length `channel_id`) degrades like an emit failure
        rather than propagating into the graph; the orchestrator upholds its own
        no-raise contract without trusting its collaborators' inputs.
        """
        try:
            event = SlackNotificationEvent(
                review_id=review_id,
                is_eval=is_eval,
                channel_id=channel_id,
                message_ts=message_ts,
                kind=kind,
                posted_at=datetime.now(UTC),
            )
            await self._sink.emit_slack_notification(event)
        except Exception:
            logger.exception(
                "Slack notification posted but audit record failed; replay may re-post once",
                extra={"review_id": str(review_id), "channel_id": channel_id, "kind": kind},
            )


@dataclass(frozen=True)
class SlackNotifyTarget:
    """A resolved per-install Slack destination: the channel + an orchestrator bound
    to that install's bot token.

    Produced by the composition-root resolver (lifespan), consumed by the hitl /
    publish nodes — which call `orchestrator.notify_*(channel_id=channel_id, ...)`
    and hold NO token / install-config logic (FUP-186). The orchestrator is a real
    `SlackNotificationOrchestrator`, so its notifier-method presence is structurally
    guaranteed (no build_graph member-presence guard needed)."""

    channel_id: str
    orchestrator: SlackNotificationOrchestrator


# installation_id -> resolved Slack target, or None when the install has no ACTIVE
# Slack config. The graph sees only this callable; the implementation (installations
# read + token decrypt + notifier construction) lives in the lifespan composition
# root, keeping `cryptography` / `slack_sdk` out of `agent/` (import-lint enforced).
SlackTargetResolver = Callable[[int], Awaitable["SlackNotifyTarget | None"]]
