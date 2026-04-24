"""Minimal FastAPI webhook receiver for the GitHub App spike.

One route: POST /webhooks/github. No dispatcher, no database, no agent —
just the primitives spec §6.3 says must work before anything downstream:

1. Read the raw body (bytes, pre-JSON-decode) so signature verification
   operates on the exact bytes GitHub signed.
2. Verify X-Hub-Signature-256 using time-constant comparison (see
   webhook-signature-constant-time-compare invariant in
   docs/invariants.md). 401 on any failure.
3. Parse the body into a githubkit Pydantic model keyed on X-GitHub-Event.
4. Log the event shape (action, key identifiers) and ACK 202 with a
   correlation ID.

The ACK is the operational contract from spec §6.3: under 10 seconds,
ideally well under a second, with downstream work dispatched separately.
In the spike we don't dispatch — Q5's demo asserts the ACK is fast; the
full dispatcher seam is Month 2 work.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from githubkit.webhooks import parse, verify

WEBHOOK_SECRET_ENV = "OUTRIDER_SPIKE_WEBHOOK_SECRET"

# Emit a visible shape-summary record on every accepted delivery. Runbook
# step 7 captures these log lines to diff real payloads against the octokit
# fixtures. Use the stdlib logger rather than structlog so the spike has
# no dependency we don't already control in pyproject.toml.
logger = logging.getLogger("outrider.spike.github_app.receiver")


def _get_secret() -> str:
    secret = os.environ.get(WEBHOOK_SECRET_ENV)
    if not secret:
        raise RuntimeError(
            f"{WEBHOOK_SECRET_ENV} must be set. Spike intentionally does not "
            "fall back to a default — a webhook receiver running without a "
            "secret is a misconfigured receiver, not a development shortcut."
        )
    return secret


def create_app() -> FastAPI:
    app = FastAPI(title="outrider-github-app-spike", version="0.1.0")

    @app.post("/webhooks/github", status_code=status.HTTP_202_ACCEPTED)
    async def receive(request: Request) -> dict[str, Any]:
        body_bytes = await request.body()
        signature = request.headers.get("X-Hub-Signature-256")
        event_name = request.headers.get("X-GitHub-Event")
        delivery_id = request.headers.get("X-GitHub-Delivery", "unknown")

        if not signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing X-Hub-Signature-256",
            )
        if not event_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing X-GitHub-Event",
            )

        # githubkit.webhooks.verify is documented as time-constant. It wraps
        # the hmac.compare_digest primitive that the
        # webhook-signature-constant-time-compare invariant names.
        if not verify(_get_secret(), body_bytes, signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="signature verification failed",
            )

        # Parse into a typed event — raises on schema mismatch. In production
        # this is where api/webhooks/schemas.py translates into PRContext.
        try:
            event = parse(event_name, body_bytes)
        except Exception as exc:  # noqa: BLE001 — spike-level catch-all
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"parse failed for event {event_name!r}: {exc}",
            ) from exc

        # Correlation ID for log + ACK body. Echo GitHub's delivery ID back
        # so a reviewer can trace a webhook through logs by one identifier.
        correlation_id = str(uuid.uuid4())

        # Shape summary — pull only the identifier-shaped fields a future
        # PRContext builder will need. This is Q4's reference payload.
        summary: dict[str, Any] = {
            "event": event_name,
            "action": getattr(event, "action", None),
            "delivery_id": delivery_id,
            "correlation_id": correlation_id,
        }
        if event_name == "pull_request":
            pr = event.pull_request
            summary.update(
                {
                    "pr_number": pr.number,
                    "pr_head_sha": pr.head.sha,
                    "pr_base_sha": pr.base.sha,
                    "repo_full_name": event.repository.full_name,
                    "repo_id": event.repository.id,
                    "installation_id": (
                        event.installation.id if event.installation else None
                    ),
                }
            )
        elif event_name == "installation":
            summary.update(
                {
                    "installation_id": event.installation.id,
                    "account_login": event.installation.account.login
                    if event.installation.account
                    else None,
                    "app_slug": event.installation.app_slug,
                }
            )

        # Emit the shape summary at INFO so the runbook can capture it by
        # following uvicorn's log output, without any extra configuration.
        # structlog-style key=value pairs keep the line machine-greppable.
        logger.info("webhook_received " + " ".join(
            f"{k}={v!r}" for k, v in summary.items()
        ))
        return summary

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
