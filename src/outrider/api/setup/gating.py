"""Block credential-dependent routes until GitHub App setup is complete.

require_credentials_configured returns 503 when the selected provider is missing
or unconfigured. Setup, read-only dashboard, health and privacy routes remain
available. Environment credential mode is already configured; demo mode does
not mount the affected routes. See DECISIONS.md#070.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from outrider.github.credentials import GitHubCredentialProvider

__all__ = ["require_credentials_configured"]


async def require_credentials_configured(request: Request) -> None:
    """FastAPI dependency: raise `503` unless the credential provider reports `CONFIGURED`.

    Reads `app.state.credential_provider` (set in every non-demo boot). A `None` provider (should
    not occur on a mounted gated route — demo mode does not mount these) or an unconfigured
    `database`-mode provider fails closed. The webhook path returning 503 marks the delivery
    failed on GitHub's side; GitHub does NOT auto-retry, so the operator redelivers missed
    events from the App's Recent Deliveries page after setup completes.
    """
    provider: GitHubCredentialProvider | None = getattr(
        request.app.state, "credential_provider", None
    )
    if provider is None or not await provider.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="setup incomplete"
        )
