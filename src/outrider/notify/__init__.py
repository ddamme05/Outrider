"""Slack notification subsystem.

Public surface: the `SlackNotifier` Protocol + result + typed exceptions from
`notify/base.py`. The concrete `slack_sdk`-backed provider (`SlackWebClientNotifier`)
lives in `notify/slack.py`; consumers import the Protocol from here, never the SDK.
"""

from outrider.notify.base import (
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

__all__ = [
    "SlackApiResponseError",
    "SlackAuthError",
    "SlackBlocks",
    "SlackChannelError",
    "SlackNotifier",
    "SlackNotifyError",
    "SlackPostResult",
    "SlackRateLimitError",
    "SlackTransientError",
]
