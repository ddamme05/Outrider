"""Slack transport protocol, result, and typed failures.

SDK calls stay in notify's wrappers (vendor-sdks-only-in-wrappers). The orchestrator
awaits posts and contains ordinary notification failures so HITL and publication
can continue through the dashboard. The graph does not retry Slack failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# Message composition lives in messages.py; transport passes blocks through unchanged.
SlackBlocks = Sequence[Mapping[str, Any]]


class SlackPostResult(BaseModel):
    """A successful post's outcome: the channel + the message `ts` that the
    status-mirror `chat.update` later targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: str
    ts: str


class SlackNotifyError(Exception):
    """Base for every Slack notifier failure. V1 callers catch this and degrade
    to the dashboard fallback (best-effort, fire-and-forget)."""


class SlackAuthError(SlackNotifyError):
    """Token / auth failure (invalid_auth, not_authed, token_revoked,
    account_inactive) — operator-actionable: the install is misconfigured."""


class SlackChannelError(SlackNotifyError):
    """The bot cannot post to the configured channel (channel_not_found,
    not_in_channel). The V1 install precondition — the bot must be a member of
    the channel — is unmet. Distinct from auth so the operator gets the right fix.
    """


class SlackRateLimitError(SlackNotifyError):
    """Slack rate-limited the request (HTTP 429 / `ratelimited`). `retry_after`
    is the server's Retry-After hint in seconds, when provided."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SlackTransientError(SlackNotifyError):
    """Transient transport failure (network error, timeout, HTTP 5xx) — the post
    may succeed on a later attempt."""


class SlackApiResponseError(SlackNotifyError):
    """An `ok=false` Slack response not covered by the more specific subclasses
    (e.g. msg_too_long, invalid_blocks). `error_code` is Slack's error string."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@runtime_checkable
class SlackNotifier(Protocol):
    """Transport Protocol for posting + mirroring Slack notifications.

    The notify subsystem consumes this; the `slack_sdk`-backed implementation is
    in `notify/slack.py`. Mirrors `llm/base.py`'s `LLMProvider`.
    """

    async def post_message(
        self, *, channel: str, text: str, blocks: SlackBlocks | None = None
    ) -> SlackPostResult:
        """Post a message; return the channel + message `ts`. `text` is the
        notification fallback; `blocks` is the optional Block Kit layout."""
        ...

    async def update_message(
        self, *, channel: str, ts: str, text: str, blocks: SlackBlocks | None = None
    ) -> None:
        """Update an existing bot message in place (the status mirror) — targets
        the `ts` returned by an earlier `post_message`."""
        ...

    async def aclose(self) -> None:
        """Release transport resources (the SDK's HTTP session). Wired into the
        FastAPI lifespan teardown; idempotent."""
        ...
