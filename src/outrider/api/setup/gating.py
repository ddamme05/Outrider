# See DECISIONS.md#070 — setup-only route gating (credential-dependent surfaces fail closed).
"""Setup-only route gating (spec F6): fail closed while not `CONFIGURED`.

While the credential provider is not `CONFIGURED` — a `database`-mode instance still onboarding
(`UNCONFIGURED`/`AWAITING_CALLBACK`/`CONVERTING`/`ORPHANED`), or a missing provider — every
credential-dependent / side-effecting route returns `503 Service Unavailable`. The `/setup*` routes,
the read-only dashboard, `GET /health`, and `GET /privacy` stay up (they need no App credentials).

`env` mode is always configured, so the gate is a no-op there; demo mode mounts none of the gated
routers at all, so the gate is structurally absent on the demo box. Applied at mount time via
`include_router(..., dependencies=[Depends(require_credentials_configured)])` on the webhook, HITL,
and Slack-OAuth routers (`main.py::_include_routers`).
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
