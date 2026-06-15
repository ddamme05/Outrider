"""Slack notification subsystem.

Public surface: the `SlackNotifier` Protocol + result + typed exceptions from
`notify/base.py`. The concrete `slack_sdk`-backed provider lives in
`notify/slack.py` (a later commit); consumers import from here, never the SDK.
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
