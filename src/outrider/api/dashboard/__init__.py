"""Dashboard routers — FastAPI surface for HITL approvals + the read-API.

Sub-modules:
  - `config` — pydantic-settings `DashboardSettings` (`OUTRIDER_ADMIN_API_KEY`).
  - `auth` — `require_admin_api_key` FastAPI dependency. HMAC-constant-time
    compare against the lifespan-bound admin key; raises HTTP 401 on
    mismatch. Same `compare_digest` idiom as
    `api/webhooks/signature.py` per the input-boundary invariant.
  - `hitl` — `POST /reviews/{review_id}/decide` endpoint (the one write
    path). M12 step-order is load-bearing: auth -> state -> mismatch.
    Enqueues `graph.ainvoke(Command(resume=...))` via FastAPI
    BackgroundTasks behind the failure wrapper that catches
    divergent-content concurrent decide races without flipping
    `reviews.status`.
  - `reviews` — the read-only dashboard read-API under `/api/reviews`
    (queue + detail + findings + replay + events; metrics computed
    read-through from the audit stream, never the zeroed `reviews.*`
    columns). Pure consumer — no mutation.
  - `policy` — `GET /api/policy/{version}` exposing the versioned
    `FindingType` → severity table via `load_policy_for_version` (the
    STORED versioned policy, never the active in-code `SEVERITY_POLICY`).
    Read-only.

This module does NOT import vendor SDKs — auth uses `hmac.compare_digest`
from stdlib; resume dispatch reads `app.state.compiled_graph` from the
lifespan-bound graph (the checkpointer that backs cross-process resume
is wired at `api/lifespan.py:Step 7b`).
"""

from outrider.api.dashboard.agent_view import router as agent_view_router
from outrider.api.dashboard.hitl import router as hitl_router
from outrider.api.dashboard.metrics import router as metrics_router
from outrider.api.dashboard.policy import router as policy_router
from outrider.api.dashboard.reviews import router as reviews_router

__all__ = [
    "agent_view_router",
    "hitl_router",
    "metrics_router",
    "policy_router",
    "reviews_router",
]
