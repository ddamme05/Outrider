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

The production FastAPI app entry point (e.g., `outrider/main.py` or a
similar module that constructs `FastAPI(lifespan=lifespan)` and mounts
the webhook router) is NOT in this package yet — see the spec backlog
for the operator-facing app shape. Today the production composition is
implicit in `tests/integration/test_webhook_router_integration.py`'s
fixture and any operator wishing to deploy must construct it themselves.

Dashboard endpoints (`dashboard/`) are future-spec work.
"""

from outrider.api.lifespan import build_lifespan, lifespan

__all__ = ["build_lifespan", "lifespan"]
