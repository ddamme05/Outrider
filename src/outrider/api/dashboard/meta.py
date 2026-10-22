"""Expose the public demo-mode flag before the SPA's token gate.

The SPA uses it for the demo banner and disabled decision controls. This route
is read-only and requires no credentials. The /api prefix keeps it covered by
both the demo proxy and the Vite proxy.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

__all__ = ["router"]

router = APIRouter(prefix="/api/meta", tags=["dashboard"])


class MetaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    demo_mode: bool


@router.get("", response_model=MetaResponse)
async def get_meta(request: Request) -> MetaResponse:
    """Deployment-shape flags. `demo_mode` mirrors `app.state.demo_mode`."""
    return MetaResponse(demo_mode=bool(getattr(request.app.state, "demo_mode", False)))
