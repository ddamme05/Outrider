"""Expose review notifications and their transport contract.

Slack SDK calls belong in notify/slack.py. Other modules build messages and
coordinate delivery through the injected notifier.
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
