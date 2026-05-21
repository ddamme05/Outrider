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

**Startup failure modes.** Required env vars (`OUTRIDER_GITHUB_APP_ID`,
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

from fastapi import FastAPI

from outrider.api import lifespan
from outrider.api.webhooks.router import router as webhook_router

app = FastAPI(
    title="Outrider",
    description=(
        "Agentic PR review (intake → triage → analyze; trace/synthesize/hitl/publish post-V1)."
    ),
    lifespan=lifespan,
)
app.include_router(webhook_router)


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
