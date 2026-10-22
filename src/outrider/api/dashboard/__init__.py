"""Dashboard reads, human decisions, and scoped agent review data.

reviews.py reads queue, detail, findings, and audit replay; metrics.py aggregates
that history; policy.py loads the stored severity version. hitl.py accepts one
human decision and queues graph resume after auth, state, and mismatch checks.
The lifespan supplies the durable graph and database dependencies.

Admin bearer auth protects review reads and decisions. agent_view.py has a separate
read-only key; meta.py exposes only public deployment flags. Vendor SDKs stay
outside these routers.
"""

from outrider.api.dashboard.agent_view import router as agent_view_router
from outrider.api.dashboard.hitl import router as hitl_router
from outrider.api.dashboard.meta import router as meta_router
from outrider.api.dashboard.metrics import router as metrics_router
from outrider.api.dashboard.policy import router as policy_router
from outrider.api.dashboard.reviews import router as reviews_router

__all__ = [
    "agent_view_router",
    "hitl_router",
    "meta_router",
    "metrics_router",
    "policy_router",
    "reviews_router",
]
