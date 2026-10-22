"""Authenticate dashboard admin and read-only agent scopes with separate keys.

Compare bearer-token bytes with hmac.compare_digest. Authentication precedes
review-state inspection on /decide (M12: auth → state → mismatch), so rejected
callers cannot enumerate review state. See docs/trust-boundaries.md §5.
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from pydantic import SecretStr


_BEARER_PREFIX = "Bearer "


async def require_admin_api_key(request: Request) -> None:
    """Require the admin bearer key before reading review state.

    Missing configuration, malformed headers, and mismatched tokens share one 401
    response. Compare UTF-8 bytes so submitted Unicode does not trigger str's
    ASCII-only comparison restriction.
    """
    expected: SecretStr | None = getattr(request.app.state, "admin_api_key", None)
    if expected is None:
        # Lifespan didn't wire the credential — operator misconfiguration.
        # 401 (not 500) so the response surface is uniform; the missing
        # state is logged separately at startup so operators see it
        # there, not via 401 inspection.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": 'Bearer realm="outrider-admin"'},
        )

    auth_header = request.headers.get("Authorization")
    if auth_header is None or not auth_header.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": 'Bearer realm="outrider-admin"'},
        )

    submitted = auth_header[len(_BEARER_PREFIX) :].encode("utf-8")
    expected_bytes = expected.get_secret_value().encode("utf-8")
    if not hmac.compare_digest(submitted, expected_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": 'Bearer realm="outrider-admin"'},
        )


async def require_agent_api_key(request: Request) -> None:
    """Require the separate read-only agent key for /agent-view.

    There is no admin fallback. An unset key disables the surface with the same
    uniform 401 response used for invalid tokens.
    """
    expected: SecretStr | None = getattr(request.app.state, "agent_api_key", None)
    if expected is None:
        # Agent surface not configured → disabled. Uniform 401 (not 500).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": 'Bearer realm="outrider-agent"'},
        )

    auth_header = request.headers.get("Authorization")
    if auth_header is None or not auth_header.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": 'Bearer realm="outrider-agent"'},
        )

    submitted = auth_header[len(_BEARER_PREFIX) :].encode("utf-8")
    expected_bytes = expected.get_secret_value().encode("utf-8")
    if not hmac.compare_digest(submitted, expected_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": 'Bearer realm="outrider-agent"'},
        )
