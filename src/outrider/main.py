"""Build the FastAPI app and mount the dashboard, webhook, and setup routes.

Run with `uv run uvicorn outrider.main:app --host 0.0.0.0 --port 8000`.
`api/lifespan.py` constructs shared dependencies at startup and closes them at
shutdown. Routers read those bindings from `app.state`.

Production requires a database and an admitted LLM host's API key
(DECISIONS.md#056). GitHub credentials come from the selected env or database
provider (DECISIONS.md#070). Database mode can boot before onboarding; its
credential-dependent routes return 503 until setup reaches CONFIGURED.

Demo mode builds read-side dependencies and mounts the read-only route allowlist.
It needs DATABASE_URL and OUTRIDER_ADMIN_API_KEY, but no model or GitHub keys.
API docs are opt-in. `/health` reports process liveness without probing external
services; mount the SPA last so API routes take precedence.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI

from outrider.api import lifespan
from outrider.api.dashboard import (
    agent_view_router,
    hitl_router,
    meta_router,
    metrics_router,
    policy_router,
    reviews_router,
)
from outrider.api.privacy import router as privacy_router
from outrider.api.setup.gating import require_credentials_configured
from outrider.api.setup.mount import mount_setup_router
from outrider.api.slack import slack_oauth_router
from outrider.api.spa import mount_spa_if_configured
from outrider.api.webhooks.router import router as webhook_router
from outrider.github.credentials import resolve_credential_source


def _demo_mode_from_env() -> bool:
    """Read the demo flag before app construction, ahead of lifespan settings.

    See specs/2026-06-21-demo-deployment.md.
    """
    return os.environ.get("OUTRIDER_DEMO_MODE", "") == "1"


def _docs_enabled_from_env() -> bool:
    """Enable API docs only for OUTRIDER_ENABLE_DOCS=1 (FUP-229).

    Read before FastAPI construction; lifespan settings load later.
    """
    return os.environ.get("OUTRIDER_ENABLE_DOCS", "") == "1"


def _include_routers(app: FastAPI, *, demo_mode: bool) -> None:
    """Mount read-only routes first, then production's side-effecting routes.

    Demo mode uses an allowlist: Slack's OAuth callback is a GET with side effects,
    so HTTP method alone cannot identify safe routes. New mutation routers belong
    after the demo return. See specs/2026-06-21-demo-deployment.md.
    """
    # Demo allowlist: these routers only read stored review data.
    app.include_router(reviews_router)
    app.include_router(policy_router)
    app.include_router(metrics_router)
    # Agent reads use a separate key from the admin dashboard.
    app.include_router(agent_view_router)
    # Public disclosure must be available before installation, including in demo mode.
    app.include_router(privacy_router)
    # The SPA needs the public demo flag before its token gate.
    app.include_router(meta_router)
    if demo_mode:
        return
    # Production writes require configured credentials (DECISIONS.md#070, spec F6).
    # Mount specific API routes before setup and the SPA fallback.
    setup_gate = [Depends(require_credentials_configured)]
    app.include_router(webhook_router, dependencies=setup_gate)
    app.include_router(hitl_router, dependencies=setup_gate)  # POST /reviews/{id}/decide
    # These GET routes exchange credentials and persist configuration.
    app.include_router(slack_oauth_router, dependencies=setup_gate)
    # Database-mode onboarding resolves startup dependencies lazily in setup/mount.py.
    # Register before the SPA catch-all (DECISIONS.md#069).
    if resolve_credential_source(os.environ) == "database":
        mount_setup_router(app)


def create_app(*, demo_mode: bool, enable_docs: bool = False) -> FastAPI:
    """Construct the app for production or the read-only demo.

    API docs default off (FUP-229); scripts/gen_openapi.py opts in for generation.
    Their paths remain reserved when disabled, so the SPA cannot serve them.
    """
    app = FastAPI(
        title="Outrider",
        description=(
            "Agentic PR review (intake → triage → analyze ⇄ trace → synthesize → hitl → publish)."
        ),
        lifespan=lifespan,
        # FUP-229: OFF by default (production-safe); the developer opt-in restores all three.
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )
    _include_routers(app, demo_mode=demo_mode)
    # Read by the lifespan to select the keyless boot (no provider/github/graph/
    # slack/sweep — the demo box serves precomputed seeds and runs no reviews).
    app.state.demo_mode = demo_mode

    @app.get("/health")
    async def health() -> dict[str, str]:
        """**Liveness only**, not readiness.

        Returns 200 once the lifespan reaches its `yield` (i.e., the
        process booted and dependency CONSTRUCTORS ran without raising).
        It does NOT probe DB connectivity, LLM-host reachability, or
        GitHub-API health — the lifespan's construction may have built an
        engine pointed at an unreachable Postgres host and this endpoint
        would still return 200.

        Useful as a "did uvicorn boot" smoke test behind smee.io /
        cloudflared / nginx. NOT a substitute for an actual readiness
        probe; layer that on top if you need it.
        """
        return {"status": "ok"}

    # See DECISIONS.md#069 — serve the built dashboard SPA (production image) AFTER all
    # API routers so specific API routes always win; the catch-all is registration-last.
    # No-op unless OUTRIDER_SERVE_SPA=1 (demo / API-only image leaves it unset); an invalid
    # flag value or an OUTRIDER_SERVE_SPA=1 image with no baked dist/ fails startup here.
    mount_spa_if_configured(app)

    return app


app = create_app(demo_mode=_demo_mode_from_env(), enable_docs=_docs_enabled_from_env())
