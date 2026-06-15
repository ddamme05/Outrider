"""slack_sdk-backed `SlackNotifier` — the ONLY module that imports slack_sdk.

Confines `import slack_sdk` per the `vendor-sdks-only-in-wrappers` boundary
(enforced by `scripts/check_import_boundaries.py` + the `check-trust-boundaries`
skill, both allowlisting `notify/`). Translates slack_sdk's `SlackApiError` into
the project's typed `SlackNotifyError` hierarchy from `notify/base.py`; the
notify subsystem consumes the `SlackNotifier` Protocol, never this class or the
SDK directly. V1 posting is best-effort: the caller catches `SlackNotifyError`
and degrades to the dashboard fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from outrider.notify.base import (
    SlackApiResponseError,
    SlackAuthError,
    SlackChannelError,
    SlackNotifyError,
    SlackPostResult,
    SlackRateLimitError,
    SlackTransientError,
)

if TYPE_CHECKING:
    from slack_sdk.web.async_slack_response import AsyncSlackResponse

    from outrider.notify.base import SlackBlocks

# Slack error codes meaning the install's auth is broken (operator-actionable).
_AUTH_ERROR_CODES = frozenset(
    {"invalid_auth", "not_authed", "account_inactive", "token_revoked", "token_expired"}
)
# Codes meaning the bot can't reach the configured channel (the V1 membership precondition).
_CHANNEL_ERROR_CODES = frozenset({"channel_not_found", "not_in_channel", "is_archived"})


class SlackWebClientNotifier:
    """`SlackNotifier` backed by slack_sdk's `AsyncWebClient`.

    Constructed with a bot token (`chat:write`). For testing, an `AsyncWebClient`
    (or any object exposing `chat_postMessage` / `chat_update`) may be injected.
    """

    def __init__(self, *, token: str, client: AsyncWebClient | None = None) -> None:
        if not token:
            raise SlackAuthError("Slack bot token is empty")
        self._client = client if client is not None else AsyncWebClient(token=token)

    async def post_message(
        self, *, channel: str, text: str, blocks: SlackBlocks | None = None
    ) -> SlackPostResult:
        try:
            # blocks bridges the project's SlackBlocks (Sequence[Mapping]) to the SDK's
            # stricter dict|Block type — the wrapper is the one place this mismatch lives.
            response = await self._client.chat_postMessage(
                channel=channel,
                text=text,
                blocks=blocks,  # type: ignore[arg-type]
            )
        except SlackApiError as exc:
            raise self._translate(exc) from exc
        except (TimeoutError, OSError) as exc:  # aiohttp ClientError subclasses OSError
            raise SlackTransientError(f"Slack request failed: {exc}") from exc
        ts = response.get("ts")
        if not ts:
            raise SlackApiResponseError("chat.postMessage returned no ts", error_code="missing_ts")
        return SlackPostResult(channel=channel, ts=str(ts))

    async def update_message(
        self, *, channel: str, ts: str, text: str, blocks: SlackBlocks | None = None
    ) -> None:
        try:
            await self._client.chat_update(channel=channel, ts=ts, text=text, blocks=blocks)  # type: ignore[arg-type]
        except SlackApiError as exc:
            raise self._translate(exc) from exc
        except (TimeoutError, OSError) as exc:
            raise SlackTransientError(f"Slack request failed: {exc}") from exc

    async def aclose(self) -> None:
        # AsyncWebClient (no injected session) opens and closes an aiohttp session
        # per request — there is no persistent transport to release. Kept for
        # SlackNotifier / FastAPI-lifespan compliance; idempotent.
        return None

    @staticmethod
    def _translate(exc: SlackApiError) -> SlackNotifyError:
        """Map a slack_sdk `SlackApiError` to the project exception hierarchy."""
        response: AsyncSlackResponse | None = exc.response
        code: str = (response.get("error") or "") if response is not None else ""
        status: int | None = getattr(response, "status_code", None)
        if status == 429 or code == "ratelimited":
            return SlackRateLimitError(
                "Slack rate-limited the request", retry_after=_retry_after(response)
            )
        if code in _AUTH_ERROR_CODES:
            return SlackAuthError(f"Slack auth error: {code}")
        if code in _CHANNEL_ERROR_CODES:
            return SlackChannelError(f"Slack channel error: {code}")
        if status is not None and status >= 500:
            return SlackTransientError(f"Slack server error (HTTP {status})")
        return SlackApiResponseError(
            f"Slack API error: {code or 'unknown'}", error_code=code or "unknown"
        )


def _retry_after(response: AsyncSlackResponse | None) -> float | None:
    """Parse the `Retry-After` header (seconds) from a rate-limited response."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
