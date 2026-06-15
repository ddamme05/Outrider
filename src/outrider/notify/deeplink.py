"""Dashboard deep-link builder for Slack notifications.

A pure helper: `base_url` (the deployed dashboard's public URL) is passed in by
the composition root, so this module stays free of config/IO. The review_id /
finding_id are internal UUIDs (unguessable, no path validation needed — the
deep-link is an entry point, not a credential; auth is the dashboard's job).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

__all__ = ["build_review_deeplink"]


def build_review_deeplink(base_url: str, review_id: UUID, finding_id: UUID | None = None) -> str:
    """`{base_url}/reviews/{review_id}` (+ `?finding={finding_id}` when given).

    Lands the reviewer on the decision UI for a review, optionally focused on one
    finding. A trailing slash on `base_url` is tolerated.
    """
    url = f"{base_url.rstrip('/')}/reviews/{review_id}"
    if finding_id is not None:
        url = f"{url}?finding={finding_id}"
    return url
