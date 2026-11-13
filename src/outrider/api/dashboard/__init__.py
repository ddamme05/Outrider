"""Dashboard routes for reading reviews and submitting approval decisions.

reviews.py serves review history; findings.py builds finding details; metrics.py
serves totals. hitl.py validates decisions and queues review resume. Admin
credentials protect these routes. agent_view.py uses a separate read-only key;
meta.py exposes public deployment flags.
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
