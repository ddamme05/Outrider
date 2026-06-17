"""Slack OAuth install-flow HTTP surface.

`oauth.py` ships the two routes that connect a GitHub App installation to a Slack
workspace: admin-authed `GET /slack/install` (mint signed `state` → redirect to
Slack) and the public `GET /slack/oauth/callback` (verify state → exchange code →
encrypt + persist the per-install bot token). Exported as `slack_oauth_router` for
the app to mount via `app.include_router(...)`.
"""

from outrider.api.slack.oauth import router as slack_oauth_router

__all__ = ["slack_oauth_router"]
