"""SlackNotifier boundary contract — result shape, exception tree, Protocol.

Pins SlackPostResult (frozen, extra=forbid), the typed exception hierarchy
(every failure is a SlackNotifyError; rate-limit carries retry_after; api-response
carries error_code), and that SlackNotifier is runtime_checkable. The concrete
slack_sdk-backed provider is tested separately. See
specs/2026-06-15-slack-dashboard-in-slack.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.notify import (
    SlackApiResponseError,
    SlackAuthError,
    SlackBlocks,
    SlackChannelError,
    SlackNotifier,
    SlackNotifyError,
    SlackPostResult,
    SlackRateLimitError,
    SlackTransientError,
)


def test_post_result_shape() -> None:
    r = SlackPostResult(channel="C0123", ts="1718500000.123456")
    assert r.channel == "C0123"
    assert r.ts == "1718500000.123456"


def test_post_result_frozen() -> None:
    r = SlackPostResult(channel="C0123", ts="1.2")
    with pytest.raises(ValidationError):
        r.ts = "9.9"  # type: ignore[misc]


def test_post_result_forbids_extra() -> None:
    # extra="forbid" keeps a stray message body out of the result object.
    with pytest.raises(ValidationError):
        SlackPostResult(channel="C0123", ts="1.2", text="leak")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "exc",
    [
        SlackAuthError,
        SlackChannelError,
        SlackRateLimitError,
        SlackTransientError,
        SlackApiResponseError,
    ],
)
def test_every_failure_is_a_slack_notify_error(exc: type[SlackNotifyError]) -> None:
    assert issubclass(exc, SlackNotifyError)


def test_rate_limit_carries_retry_after() -> None:
    assert SlackRateLimitError("slow down", retry_after=30.0).retry_after == 30.0
    assert SlackRateLimitError("no hint").retry_after is None


def test_api_response_error_carries_code() -> None:
    assert SlackApiResponseError("bad blocks", error_code="invalid_blocks").error_code == (
        "invalid_blocks"
    )


def test_notifier_is_runtime_checkable() -> None:
    class _Conforming:
        async def post_message(
            self, *, channel: str, text: str, blocks: SlackBlocks | None = None
        ) -> SlackPostResult:
            assert text and blocks is None or blocks is not None
            return SlackPostResult(channel=channel, ts="1.2")

        async def update_message(
            self, *, channel: str, ts: str, text: str, blocks: SlackBlocks | None = None
        ) -> None:
            assert channel and ts and text and (blocks is None or blocks is not None)

        async def aclose(self) -> None:
            return None

    assert isinstance(_Conforming(), SlackNotifier)
    assert not isinstance(object(), SlackNotifier)
