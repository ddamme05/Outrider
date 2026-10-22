"""HTTP entry points and shared dependency construction.

`main.py` constructs the app. `lifespan.py` builds its providers, persisters,
graph, and background workers. `webhooks/` accepts signed GitHub deliveries;
`dashboard/` reads reviews and accepts human decisions. `setup/` onboards a
GitHub App in database credential mode (DECISIONS.md#070); `slack/` connects channels.
Dashboard review data uses bearer auth; deployment metadata is public.
"""

from outrider.api.lifespan import build_lifespan, lifespan

__all__ = ["build_lifespan", "lifespan"]
