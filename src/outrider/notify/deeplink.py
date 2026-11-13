"""Build review-dashboard links from the configured public base URL.

Links contain review identifiers, never dashboard access tokens. The dashboard
still applies its normal authentication checks when a link is opened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from outrider.policy.output_sanitizer import is_safe_link_url

if TYPE_CHECKING:
    from uuid import UUID

__all__ = ["build_review_deeplink"]


def build_review_deeplink(
    base_url: str, review_id: UUID, finding_id: UUID | None = None
) -> str | None:
    """`{base_url}/reviews/{review_id}` (+ `?finding={finding_id}` when given), or
    None when `base_url` is malformed (per the shared `is_safe_link_url` gate, which
    the publish review-body renderer shares) — the caller then degrades to a no-link
    Slack message rather than embedding a broken mrkdwn link. The base URL is
    operator/per-install config, so the threat is misconfiguration, not attacker
    input. A trailing slash on `base_url` is tolerated.
    """
    if not is_safe_link_url(base_url):
        return None
    url = f"{base_url.rstrip('/')}/reviews/{review_id}"
    if finding_id is not None:
        url = f"{url}?finding={finding_id}"
    return url
