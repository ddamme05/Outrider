"""HTTP / FastAPI surface for Outrider.

V1 ships two surfaces:

  - `lifespan` (and `build_lifespan` for test-injection composition) —
    wires up the durable `AuditPersister`, the `AnthropicProvider`, the
    `RejectLLMContentFilter` re-registration step, the compiled graph,
    and the per-installation `GitHubAppSettings` at app startup; drains
    them on shutdown via `AsyncExitStack` LIFO teardown.
  - `webhooks/router.py` — `POST /webhooks/github` FastAPI router with
    signature verification, idempotency, active-membership check, and
    transactional review + audit row INSERT. Exported as
    `outrider.api.webhooks.router` for the consumer to mount via
    `app.include_router(router)` at production wire-up time.

The production FastAPI app entry point lives in `outrider/main.py`, which
constructs `FastAPI(lifespan=lifespan)` and mounts the webhook router for
operator deployment.

Dashboard endpoints under `dashboard/` ship behind bearer auth
(`dashboard/auth.py`) and mount on the same app (split at the router level
so the dashboard API can later be extracted into its own service).
"""

from outrider.api.lifespan import build_lifespan, lifespan

__all__ = ["build_lifespan", "lifespan"]
