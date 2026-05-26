"""Bearer-token auth for the dashboard endpoints.

`require_admin_api_key` is a FastAPI dependency that validates the
incoming `Authorization: Bearer <key>` header against
`app.state.admin_api_key` (wired at lifespan startup from
`DashboardSettings`). HMAC `compare_digest` is used to prevent the
timing-oracle class of attack where character-by-character comparison
leaks the key length / prefix.

Per the M12 step-order contract on the `/decide` endpoint (auth ->
state -> mismatch), this dependency fires FIRST so an unauthenticated
caller cannot enumerate review state by probing the endpoint with
various review_ids and observing the 200/409/422 response shape.

Same `compare_digest` idiom as `api/webhooks/signature.py`. Trust
boundary classification per `docs/trust-boundaries.md` §5 (input
boundary, sub-rule 4).
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from pydantic import SecretStr


_BEARER_PREFIX = "Bearer "


async def require_admin_api_key(request: Request) -> None:
    """Validate `Authorization: Bearer <key>` against the lifespan-bound
    admin key. Raises HTTP 401 on any failure mode.

    Failure modes that surface as 401 (uniform response shape so the
    error surface doesn't disclose which check failed):
      - No Authorization header
      - Header doesn't start with `Bearer `
      - The token after `Bearer ` doesn't match the configured key

    `hmac.compare_digest` requires both arguments to be the same type
    AND the same length; mismatched lengths return False without
    leaking via timing. The comparison runs against the *bytes*
    encoding so a unicode-shaped key submission can't change the
    comparison surface.
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
        )

    auth_header = request.headers.get("Authorization")
    if auth_header is None or not auth_header.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )

    submitted = auth_header[len(_BEARER_PREFIX) :].encode("utf-8")
    expected_bytes = expected.get_secret_value().encode("utf-8")
    if not hmac.compare_digest(submitted, expected_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )
