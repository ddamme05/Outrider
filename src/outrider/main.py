# See src/outrider/api/__init__.py — this module ships the production FastAPI
# entry the lifespan docstring at api/lifespan.py:505 promises.
"""Outrider FastAPI application entry.

Constructs the production app: `FastAPI(lifespan=lifespan)` plus the
webhook router mounted at its canonical `/webhooks/github` path.

Run with uvicorn:

    uv run uvicorn outrider.main:app --host 0.0.0.0 --port 8000

The lifespan (`outrider.api.lifespan`) handles all dependency construction
at startup (Anthropic provider, audit persister, GitHub client factory,
compiled graph) and LIFO teardown at shutdown via AsyncExitStack. The
webhook router (`outrider.api.webhooks.router`) reads its dependencies
from `app.state` bindings the lifespan installs (engine, session_factory,
retention_settings, persister, provider, github_app_settings,
github_factory, compiled_graph, run_graph).

**DEMO_MODE.** Under `create_app(demo_mode=True)` (env `OUTRIDER_DEMO_MODE`,
the public read-only demo box) the lifespan takes a keyless early-return
path: it builds only the read-side deps (engine, session, retention,
persister) and serves precomputed reviews through the dashboard allowlist.
The Anthropic provider, GitHub App, graph, checkpointer, Slack, and sweeps
are NOT constructed, and the review/write half of `app.state` is `None` — so
the env vars listed below are NOT required in demo mode (only
`OUTRIDER_ADMIN_API_KEY` + `DATABASE_URL` are). See `api/lifespan.py`'s
`if demo_mode:` branch.

**Startup failure modes (production mode).** Required env vars (`OUTRIDER_GITHUB_APP_ID`,
`OUTRIDER_GITHUB_APP_PRIVATE_KEY`, `OUTRIDER_GITHUB_WEBHOOK_SECRET`,
`ANTHROPIC_API_KEY`, `DATABASE_URL`) read by their respective
`BaseSettings` subclasses with `extra="forbid"` — a missing or typoed
name raises `ValidationError` at lifespan step 6 (GitHubAppSettings) or
step 1 (database engine). Uvicorn fails to start; the error surfaces
in the logs with the field name that's missing. The `OUTRIDER_GITHUB_*`
prefix is required — unprefixed `GITHUB_APP_ID` is silently ignored
by `pydantic-settings` and produces "field required" at startup.

**`/health` is a liveness probe, not a readiness probe.** It returns
200 as soon as the lifespan reaches its `yield` (i.e., construction
finished), but it does NOT probe DB connectivity, Anthropic
reachability, or GitHub-API health. A proper readiness probe would
`SELECT 1` against the engine and a no-op vendor call — out of scope
for V1. Operators who need readiness semantics should layer that on
top (k8s readiness probe, an `/api/readiness` endpoint, etc.).
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from outrider.api import lifespan
from outrider.api.dashboard import (
    agent_view_router,
    hitl_router,
    metrics_router,
    policy_router,
    reviews_router,
)
from outrider.api.privacy import router as privacy_router
from outrider.api.slack import slack_oauth_router
from outrider.api.webhooks.router import router as webhook_router


def _demo_mode_from_env() -> bool:
    """Demo deployment toggle (`specs/2026-06-21-demo-deployment.md`). A plain
    env read, NOT a `BaseSettings` field, because it gates module-level route
    mounting that happens BEFORE the lifespan loads pydantic-settings."""
    return os.environ.get("OUTRIDER_DEMO_MODE", "") == "1"


def _include_routers(app: FastAPI, *, demo_mode: bool) -> None:
    """Mount the route allowlist.

    `demo_mode` is a default-deny **allowlist**, NOT a method-based denylist: a
    "block non-GET" rule would leave Slack's GET `/oauth/callback` — which
    exchanges an OAuth code and persists config — reachable on a public demo box.
    So the read-only dashboard GET surface is ALWAYS mounted; every mutation AND
    side-effecting router (webhook intake, HITL `decide`, the Slack OAuth GET
    flow) mounts ONLY in production. A new side-effecting route added later stays
    off in demo mode by default. See `specs/2026-06-21-demo-deployment.md`.
    """
    # Read-only dashboard surface — the demo allowlist (all four routers are GET-only).
    app.include_router(reviews_router)
    app.include_router(policy_router)
    app.include_router(metrics_router)
    # Feature 3 / S2: read-only agent-view endpoint on its own require_agent_api_key
    # router (separate scope from the admin-gated routers above).
    app.include_router(agent_view_router)
    # B3: PUBLIC, unauthenticated privacy page (GET /privacy) — the App-listing
    # privacy-policy URL + the dashboard footer target. Always mounted (incl. demo
    # mode): it must be readable before any install exists, and it is read-only.
    app.include_router(privacy_router)
    if demo_mode:
        return
    # Production-only: mutation + side-effecting routers.
    app.include_router(webhook_router)
    app.include_router(hitl_router)  # POST /reviews/{id}/decide
    # Slack OAuth install flow (commit 6.3e): admin-authed GET /slack/install +
    # public GET /slack/oauth/callback (both side-effecting). Disabled (uniform
    # 503) unless OUTRIDER_SLACK_CLIENT_ID is set — but in demo mode it's not
    # mounted at all, so the GET-with-side-effects surface is structurally absent.
    app.include_router(slack_oauth_router)


def create_app(*, demo_mode: bool) -> FastAPI:
    """Build the production FastAPI app for the given mode.

    `demo_mode` selects the route allowlist (see `_include_routers`). Factored out
    of module scope so the route-mount surface is testable EXACTLY per mode against
    the real app (not a helper-built one) — `app` below uses the env flag.
    """
    app = FastAPI(
        title="Outrider",
        description=(
            "Agentic PR review (intake → triage → analyze ⇄ trace → synthesize → hitl → publish)."
        ),
        lifespan=lifespan,
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
        It does NOT probe DB connectivity, Anthropic reachability, or
        GitHub-API health — the lifespan's construction may have built an
        engine pointed at an unreachable Postgres host and this endpoint
        would still return 200.

        Useful as a "did uvicorn boot" smoke test behind smee.io /
        cloudflared / nginx. NOT a substitute for an actual readiness
        probe; layer that on top if you need it.
        """
        return {"status": "ok"}

    return app


app = create_app(demo_mode=_demo_mode_from_env())
