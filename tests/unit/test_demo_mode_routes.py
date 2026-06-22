# Per specs/2026-06-21-demo-deployment.md — the DEMO_MODE route allowlist.
"""`DEMO_MODE` mounts a default-deny, read-only allowlist: only the dashboard GET
surface, never the webhook intake, HITL `decide`, or the Slack OAuth flow.

The Slack OAuth callback is a GET that exchanges a code and persists config, so a
method-based "block non-GET" denylist would leave it reachable — the allowlist is
the only correct shape. The guard is the **route enumeration**, not a hand-kept
name list, so a future side-effecting route can't silently slip into the public
demo box.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from outrider.main import _include_routers

_MUTATING = {"POST", "PUT", "DELETE", "PATCH"}


def _paths_and_methods(app: FastAPI) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    methods: set[str] = set()
    for route in app.routes:
        paths.add(getattr(route, "path", ""))
        methods |= set(getattr(route, "methods", set()) or ())
    return paths, methods


def _demo_app(demo_mode: bool) -> Any:
    app = FastAPI()
    _include_routers(app, demo_mode=demo_mode)
    return app


def test_demo_mode_mounts_only_readonly_get_allowlist() -> None:
    paths, methods = _paths_and_methods(_demo_app(demo_mode=True))

    # The real guard: NO mutating method is reachable anywhere in demo mode.
    assert not (methods & _MUTATING), f"mutating routes mounted in demo mode: {methods & _MUTATING}"

    # Known side-effecting surfaces are structurally absent — including the
    # GET-with-side-effects Slack OAuth flow a method denylist would miss.
    assert not any("/webhooks" in p for p in paths), "webhook intake reachable in demo mode"
    assert not any(p.startswith("/slack") for p in paths), "Slack OAuth flow reachable in demo mode"
    assert not any(p.endswith("/decide") for p in paths), "HITL decide reachable in demo mode"

    # The read-only dashboard surface IS present.
    assert "/api/reviews" in paths
    assert any(p.startswith("/api/") for p in paths)


def test_production_mounts_mutation_and_side_effecting_routers() -> None:
    paths, methods = _paths_and_methods(_demo_app(demo_mode=False))

    assert any("/webhooks" in p for p in paths), "webhook intake missing in production"
    assert any(p.startswith("/slack") for p in paths), "Slack OAuth flow missing in production"
    assert any(p.endswith("/decide") for p in paths), "HITL decide missing in production"
    assert "POST" in methods, "no mutation method mounted in production"


def test_demo_and_production_share_the_readonly_surface() -> None:
    # The allowlist (read-only dashboard) is identical in both modes; production
    # only ADDS the mutation/side-effecting routers on top.
    demo_paths, _ = _paths_and_methods(_demo_app(demo_mode=True))
    prod_paths, _ = _paths_and_methods(_demo_app(demo_mode=False))
    assert demo_paths <= prod_paths
    api_demo = {p for p in demo_paths if p.startswith("/api/")}
    api_prod = {p for p in prod_paths if p.startswith("/api/")}
    assert api_demo == api_prod
