"""Dashboard router — FastAPI surface for HITL approvals.

Sub-modules:
  - `config` — pydantic-settings `DashboardSettings` (`OUTRIDER_ADMIN_API_KEY`).
  - `auth` — `require_admin_api_key` FastAPI dependency. HMAC-constant-time
    compare against the lifespan-bound admin key; raises HTTP 401 on
    mismatch. Same `compare_digest` idiom as
    `api/webhooks/signature.py` per the input-boundary invariant.
  - `hitl` — `POST /reviews/{review_id}/decide` endpoint. M12 step-order
    is load-bearing: auth -> state -> mismatch. Enqueues
    `graph.ainvoke(Command(resume=...))` via FastAPI BackgroundTasks
    behind the failure wrapper that catches divergent-content concurrent
    decide races without flipping `reviews.status`.

This module does NOT import vendor SDKs — auth uses `hmac.compare_digest`
from stdlib; resume dispatch reads `app.state.compiled_graph` from the
lifespan-bound graph (the checkpointer that backs cross-process resume
is wired at `api/lifespan.py:Step 7b`).
"""

from outrider.api.dashboard.hitl import router as hitl_router

__all__ = ["hitl_router"]
