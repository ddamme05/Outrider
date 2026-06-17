# See DECISIONS.md#051-slack-bot-tokens-are-encrypted-at-rest
"""Slack OAuth install-flow endpoints (commit 6.3e).

Two routes connect a GitHub App installation to a Slack workspace:

  - `GET /slack/install` (ADMIN-authed) — the operator passes `installation_id` +
    `channel_id`; we validate the channel shape, mint an HMAC-signed `state` binding
    (installation_id, admin principal, channel) + expiry, and 302-redirect to Slack's
    `oauth/v2/authorize` with `scope=chat:write`.
  - `GET /slack/oauth/callback` (PUBLIC — Slack redirects the browser here, so no
    bearer auth is possible; the signed `state` IS the CSRF defense) — verify the
    state, exchange the `code` for a bot token server-side, encrypt it, and persist
    it per-install. The installation_id / channel / admin are read from the VERIFIED
    STATE, never from the callback query params (trust-boundaries.md §5: GitHub/Slack
    redirect params are attacker-controllable).

Trust boundaries: input (§5 — signed-state provenance, no shell/format-string use of
params), token encryption (DECISIONS.md#051 — `encrypt_token` before persist; the
bot token only ever flows as `SecretStr` / ciphertext, never logged), install
retention (DECISIONS.md#012 — `set_slack_config` refuses tombstoned/absent installs).
The Slack SDK stays confined to `notify/` (this module calls the `notify/` wrappers,
never `slack_sdk` directly).
"""

from __future__ import annotations

import html
import logging
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.db.models.installations import set_slack_config
from outrider.notify.oauth_state import (
    SlackStateError,
    sign_state,
    validate_channel_id,
    verify_state,
)
from outrider.notify.slack_oauth import SlackOAuthError, exchange_code
from outrider.notify.token_crypto import TokenCryptoError, encrypt_token

if TYPE_CHECKING:
    from outrider.notify.config import SlackOAuthSettings

_LOGGER = logging.getLogger("outrider.api.slack.oauth")

_SLACK_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
_SLACK_SCOPE = "chat:write"
# The admin bearer key is a single shared credential with no per-user identity; the
# signed-state admin principal is "the operator who holds the admin key".
_ADMIN_PRINCIPAL = "admin"

router = APIRouter(prefix="/slack", tags=["slack-oauth"])


def _require_oauth_settings(request: Request) -> SlackOAuthSettings:
    """The OAuth flow is opt-in: when `OUTRIDER_SLACK_CLIENT_ID` is unset the lifespan
    binds `slack_oauth_settings = None` and both routes are disabled with a uniform
    503 (never a 500/AttributeError)."""
    settings: SlackOAuthSettings | None = getattr(request.app.state, "slack_oauth_settings", None)
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Slack OAuth is not configured",
        )
    return settings


@router.get("/install", dependencies=[Depends(require_admin_api_key)])
async def slack_install(
    request: Request,
    installation_id: Annotated[int, Query()],
    channel_id: Annotated[str, Query()],
) -> RedirectResponse:
    """Admin-authed install start: validate channel → sign state → redirect to Slack."""
    settings = _require_oauth_settings(request)
    try:
        channel = validate_channel_id(channel_id)
    except SlackStateError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    try:
        state = sign_state(
            installation_id=installation_id,
            admin_id=_ADMIN_PRINCIPAL,
            channel_id=channel,
        )
    except SlackStateError as exc:
        # Channel already validated above, so this is the state-secret deploy
        # misconfiguration path (missing / placeholder OUTRIDER_SLACK_STATE_SECRET).
        _LOGGER.error("slack_install_state_signing_failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Slack OAuth state signing failed",
        ) from exc
    query = urlencode(
        {
            "client_id": settings.client_id,
            "scope": _SLACK_SCOPE,
            "redirect_uri": settings.redirect_uri,
            "state": state,
        }
    )
    return RedirectResponse(
        url=f"{_SLACK_AUTHORIZE_URL}?{query}",
        status_code=status.HTTP_302_FOUND,
    )


@router.get("/oauth/callback")
async def slack_oauth_callback(
    request: Request,
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    """Public OAuth callback: verify state → exchange code → encrypt → persist.

    Identity is read from the VERIFIED state, never from the query params. Failure
    modes fail closed: denied/bad/expired/forged → 4xx with nothing persisted.
    """
    settings = _require_oauth_settings(request)
    if error:
        # Slack-side error or the admin denied the install; nothing to persist.
        _LOGGER.info("slack_oauth_callback_error: %s", error)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slack authorization was denied or failed",
        )
    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="missing code or state",
        )
    try:
        verified = verify_state(state)
    except SlackStateError as exc:
        # Forged / expired / tampered state (or a rotated-away secret). Reject.
        _LOGGER.warning("slack_oauth_callback_bad_state: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid or expired state",
        ) from exc
    try:
        result = await exchange_code(
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            code=code,
            redirect_uri=settings.redirect_uri,
        )
    except SlackOAuthError as exc:
        # Bad/expired code, Slack API error, or malformed response. The error code is
        # surfaced in the exception (never the token — see notify/slack_oauth.py).
        _LOGGER.warning("slack_oauth_exchange_failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slack OAuth exchange failed",
        ) from exc
    try:
        ciphertext = encrypt_token(result.bot_token)
    except TokenCryptoError as exc:
        # Missing / malformed OUTRIDER_TOKEN_ENC_KEY — deploy misconfiguration.
        _LOGGER.error("slack_oauth_token_encrypt_failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="token encryption failed",
        ) from exc
    session_factory = request.app.state.session_factory
    async with session_factory() as session, session.begin():
        updated = await set_slack_config(
            session,
            installation_id=verified.installation_id,
            team_id=result.team_id,
            bot_token_ciphertext=ciphertext,
            channel_id=verified.channel_id,
            configured_by=verified.admin_id,
        )
    if not updated:
        # No active install matched (absent or tombstoned-in-grace per #012).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="installation not found or inactive",
        )
    return HTMLResponse(_success_html(result.team_name or result.team_id))


def _success_html(team: str) -> str:
    """Minimal confirmation page for the browser landing on the callback. `team` comes
    from Slack's verified response; escape it defensively before interpolation."""
    safe = html.escape(team)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Slack connected</title>"
        "</head><body><h2>Slack connected</h2>"
        f"<p>Outrider is now connected to <strong>{safe}</strong>. You can close this tab.</p>"
        "</body></html>"
    )
