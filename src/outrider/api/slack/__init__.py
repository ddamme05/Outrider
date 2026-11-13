"""Routes that connect a GitHub installation to a Slack channel.

oauth.py starts the admin-authorized connection and handles Slack's callback.
The callback verifies signed state before exchanging and encrypting the bot
token. The app mounts these routes through slack_oauth_router.
"""

from outrider.api.slack.oauth import router as slack_oauth_router

__all__ = ["slack_oauth_router"]
