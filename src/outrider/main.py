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
from `app.state` bindings the lifespan installs.

A `GET /health` endpoint returns 200 when the lifespan has finished
startup. Useful as a smoke test behind smee.io / cloudflared / nginx —
operators can curl it before testing real PR webhook delivery.
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
    """Liveness probe. Returns 200 once the lifespan finishes constructing
    deps. Operators behind a tunnel (smee.io, cloudflared) can curl this
    after `uvicorn` starts to confirm startup completed."""
    return {"status": "ok"}
