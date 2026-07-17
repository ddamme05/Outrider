"""GET /api/meta — deployment-shape flags the SPA needs before it renders.

One unauthenticated, side-effect-free boolean surface: `demo_mode`. The SPA uses
it to render the read-only-demo banner and disable HITL decision controls —
without it, a public demo box shows live-looking Submit controls whose POST can
never reach FastAPI (the demo Caddyfile proxies no mutation route, and the hitl
router is not even mounted). Unauthenticated is deliberate and matches
`/health`'s posture: the flag reveals only what the deployment shape already
makes observable, and being auth-free lets the SPA mount the read-only-demo
banner ABOVE its token gate (`main.tsx`), so a demo viewer sees it before
entering the admin key.

Lives under `/api/*` so both hand-maintained route mirrors — the demo Caddyfile
and the Vite dev proxy — cover it with their existing prefix rules (the
route-mirror bug class shipped three times on non-/api prefixes: /privacy twice,
/setup once).
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
