"""HTTP routes and startup wiring for the application.

main.py mounts the routes; lifespan.py builds their shared services. Webhooks
start reviews, the dashboard displays results and accepts decisions, and setup
and Slack routes connect integrations. GitHub App setup follows DECISIONS.md#070.
"""

from outrider.api.lifespan import build_lifespan, lifespan

__all__ = ["build_lifespan", "lifespan"]
