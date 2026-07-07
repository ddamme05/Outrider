# Per specs/2026-06-21-demo-deployment.md — the DEMO_MODE route allowlist.
"""`DEMO_MODE` mounts an EXACT, default-deny read-only allowlist on the real app.

The guard is exact `(path, methods)` equality against the actual `create_app()`
output — NOT negative pattern matching on a helper-built app. So even a future
side-effecting `GET /api/*` route fails this test until a human deliberately adds
it to the allowlist below (and thinks about whether it's safe on a public box).
The Slack OAuth callback is a GET that persists config, which is why a method
denylist would be insufficient and the allowlist is the only correct shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.routing import APIRoute

from outrider.main import create_app

if TYPE_CHECKING:
    from fastapi import FastAPI

# FastAPI auto-mounts these read-only schema/docs routes; they are framework-level
# (not our surface) and excluded from the exact comparison.
_FASTAPI_BUILTINS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}

# The EXACT read-only allowlist served in demo mode (all GET, no side effects).
_DEMO_ALLOWLIST = {
    ("/health", frozenset({"GET"})),
    ("/api/metrics", frozenset({"GET"})),
    ("/api/metrics/replay", frozenset({"GET"})),
    ("/api/policy/{version}", frozenset({"GET"})),
    ("/api/reviews", frozenset({"GET"})),
    ("/api/reviews/{review_id}", frozenset({"GET"})),
    ("/api/reviews/{review_id}/agent-view", frozenset({"GET"})),
    ("/api/reviews/{review_id}/events", frozenset({"GET"})),
    ("/api/reviews/{review_id}/findings", frozenset({"GET"})),
    ("/api/reviews/{review_id}/replay-timeline", frozenset({"GET"})),
    # B3: public, unauthenticated privacy page — always mounted (it must be
    # reachable pre-install), read-only, so it belongs in the demo allowlist.
    ("/privacy", frozenset({"GET"})),
}

# Production adds EXACTLY these mutation + side-effecting routers on top.
_PRODUCTION_EXTRA = {
    ("/reviews/{review_id}/decide", frozenset({"POST"})),  # HITL decide
    ("/slack/install", frozenset({"GET"})),  # side-effecting GET
    ("/slack/oauth/callback", frozenset({"GET"})),  # side-effecting GET
    ("/webhooks/github", frozenset({"POST"})),  # intake
}


def _surface(app: FastAPI) -> set[tuple[str, frozenset[str]]]:
    """Our `(path, methods)` routes — APIRoutes minus FastAPI's built-in docs."""
    return {
        (r.path, frozenset(r.methods or ()))
        for r in app.routes
        if isinstance(r, APIRoute) and r.path not in _FASTAPI_BUILTINS
    }


def test_demo_mode_surface_is_exactly_the_readonly_allowlist() -> None:
    # EXACT equality on the real app — a new route (even a GET /api/*) fails this
    # until added to _DEMO_ALLOWLIST, forcing a deliberate is-it-safe decision.
    assert _surface(create_app(demo_mode=True)) == _DEMO_ALLOWLIST


def test_demo_mode_has_no_mutating_or_side_effecting_method() -> None:
    methods = {m for _, ms in _surface(create_app(demo_mode=True)) for m in ms}
    assert methods <= {"GET", "HEAD"}, f"non-GET method reachable in demo mode: {methods}"


def test_production_surface_is_allowlist_plus_mutation_routes() -> None:
    assert _surface(create_app(demo_mode=False)) == _DEMO_ALLOWLIST | _PRODUCTION_EXTRA
