"""SlackWebClientNotifier — the slack_sdk-backed provider.

Pins the post→ts / update flow against an injected fake AsyncWebClient, the
empty-token guard, and the SlackApiError → typed-exception translation
(auth / channel / rate-limit / transient / api-response). Importing slack_sdk's
SlackApiError here is the documented surface-tier test exception (wrapper test).
See specs/2026-06-15-slack-dashboard-in-slack.md.
"""

from __future__ import annotations

from typing import Any

import pytest
from slack_sdk.errors import SlackApiError

from outrider.notify import (
    SlackApiResponseError,
    SlackAuthError,
    SlackChannelError,
    SlackNotifier,
    SlackNotifyError,
    SlackPostResult,
    SlackRateLimitError,
    SlackTransientError,
)
from outrider.notify.slack import SlackWebClientNotifier


class _FakeResp:
    """Minimal stand-in for slack_sdk's SlackResponse (dict-like + status/headers)."""

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._data = data or {}
        self.status_code = status_code
        self.headers = headers or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class _FakeClient:
    """Fake AsyncWebClient: returns a canned response or raises a canned error."""

    def __init__(
        self, *, response: _FakeResp | None = None, error: Exception | None = None
    ) -> None:
        self._response = response or _FakeResp({"ok": True, "ts": "1718500000.000100"})
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def chat_postMessage(self, **kwargs: Any) -> _FakeResp:  # noqa: N802 (SDK method name)
        self.calls.append({"method": "chat_postMessage", **kwargs})
        if self._error is not None:
            raise self._error
        return self._response

    async def chat_update(self, **kwargs: Any) -> _FakeResp:  # noqa: N802 (SDK method name)
        self.calls.append({"method": "chat_update", **kwargs})
        if self._error is not None:
            raise self._error
        return self._response


def _notifier(client: _FakeClient) -> SlackWebClientNotifier:
    return SlackWebClientNotifier(token="xoxb-test", client=client)  # type: ignore[arg-type]  # noqa: S106


async def test_post_message_returns_ts() -> None:
    client = _FakeClient(response=_FakeResp({"ok": True, "ts": "1718500000.000100"}))
    result = await _notifier(client).post_message(channel="C0123", text="hi", blocks=[{"x": 1}])
    assert isinstance(result, SlackPostResult)
    assert result.channel == "C0123"
    assert result.ts == "1718500000.000100"
    assert client.calls[0]["channel"] == "C0123"
    assert client.calls[0]["blocks"] == [{"x": 1}]


async def test_post_message_missing_ts_raises() -> None:
    client = _FakeClient(response=_FakeResp({"ok": True}))  # no ts
    with pytest.raises(SlackApiResponseError):
        await _notifier(client).post_message(channel="C0123", text="hi")


async def test_update_message_calls_chat_update() -> None:
    client = _FakeClient()
    await _notifier(client).update_message(channel="C0123", ts="1.2", text="done")
    assert client.calls[0]["method"] == "chat_update"
    assert client.calls[0]["ts"] == "1.2"


def test_empty_token_rejected() -> None:
    with pytest.raises(SlackAuthError):
        SlackWebClientNotifier(token="")


async def test_runtime_checkable_conformance() -> None:
    assert isinstance(_notifier(_FakeClient()), SlackNotifier)


@pytest.mark.parametrize(
    ("error", "status", "headers", "expected"),
    [
        ({"error": "invalid_auth"}, 401, None, SlackAuthError),
        ({"error": "token_revoked"}, 401, None, SlackAuthError),
        ({"error": "channel_not_found"}, 200, None, SlackChannelError),
        ({"error": "not_in_channel"}, 200, None, SlackChannelError),
        ({"error": "ratelimited"}, 429, {"Retry-After": "12"}, SlackRateLimitError),
        ({"error": "internal_error"}, 503, None, SlackTransientError),
        ({"error": "msg_too_long"}, 200, None, SlackApiResponseError),
    ],
)
async def test_api_error_translation(
    error: dict[str, Any],
    status: int,
    headers: dict[str, str] | None,
    expected: type[SlackNotifyError],
) -> None:
    resp = _FakeResp(error, status_code=status, headers=headers)
    client = _FakeClient(error=SlackApiError("boom", resp))  # type: ignore[arg-type]
    with pytest.raises(expected):
        await _notifier(client).post_message(channel="C0123", text="hi")


async def test_rate_limit_carries_retry_after() -> None:
    resp = _FakeResp({"error": "ratelimited"}, status_code=429, headers={"Retry-After": "12"})
    client = _FakeClient(error=SlackApiError("boom", resp))  # type: ignore[arg-type]
    with pytest.raises(SlackRateLimitError) as exc_info:
        await _notifier(client).post_message(channel="C0123", text="hi")
    assert exc_info.value.retry_after == 12.0


async def test_transient_on_network_error() -> None:
    client = _FakeClient(error=TimeoutError("network down"))
    with pytest.raises(SlackTransientError):
        await _notifier(client).post_message(channel="C0123", text="hi")


async def test_aclose_is_noop() -> None:
    await _notifier(_FakeClient()).aclose()  # does not raise
