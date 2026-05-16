"""HTTP / FastAPI surface for Outrider.

V1 ships a single surface: the `lifespan` context manager wires up the
durable `AuditPersister`, the `AnthropicProvider`, and the
`RejectLLMContentFilter` re-registration step at app startup, and drains
them on shutdown via `AsyncExitStack` LIFO teardown.

Future specs add the webhook receiver (`webhooks/`) and dashboard
endpoints (`dashboard/`); they will share this lifespan.
"""

from outrider.api.lifespan import build_lifespan, lifespan

__all__ = ["build_lifespan", "lifespan"]
