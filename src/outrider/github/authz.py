# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Live GitHub authorization check (DECISIONS.md#065).

The local install tables are a cache; GitHub is the authorization authority, checked
LIVE at intake. As the App (App-JWT), this module confirms — for a given
`(installation_id, owner, repo)` — that:

  1. the installation still exists and is NOT suspended
     (`GET /app/installations/{id}` → 200 with `suspended_at` null; 404 → uninstalled), and
  2. the target repo is still accessible
     (`POST /app/installations/{id}/access_tokens` with a repo-scoped body → 201).

Per #065 the two checks fail CLOSED: any non-affirmative result — suspended, uninstalled,
repo-removed, OR a network error / uncertainty — collapses to a single `authorized == False`
(the caller drives the review to `skipped`). The local cache is never consulted for the
answer. It is the GitHub-authority counterpart to the DB-cache `active_repo_membership()`
gate on the webhook path.

Lives under `github/` (the only subsystem that may touch githubkit). It calls the App-level
endpoints through the injected `AppGitHubClient`'s `arequest` escape hatch — githubkit has no
confirmed generated method for these `/app/*` endpoints (`docs/mcp-usage.md`), and the raw
call is the same pattern `github/publisher.py` uses. Consumers (intake) receive an
`InstallationAuthorizer` closure via `build_graph`/`lifespan` and never import githubkit.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from outrider.github.auth import AppGitHubClient

logger = logging.getLogger(__name__)

# Pin the REST API version to match the rest of the GitHub surface (publisher /
# fetch send the same header).
_API_VERSION_HEADER: Final[dict[str, str]] = {"X-GitHub-Api-Version": "2026-03-10"}


class LiveAuthOutcome(StrEnum):
    """The live-check result. Only `AUTHORIZED` proceeds; every other value fails closed
    (#065). The non-authorized variants exist for log/observability detail, not for
    differentiated caller behavior — the caller branches on `LiveAuthResult.authorized`."""

    AUTHORIZED = "authorized"
    SUSPENDED = "suspended"
    UNINSTALLED = "uninstalled"
    REPO_INACCESSIBLE = "repo_inaccessible"
    UNCERTAIN = "uncertain"  # network error / 5xx / unparseable → fail closed


@dataclass(frozen=True)
class LiveAuthResult:
    """Outcome of the live authorization check. `detail` is a short human/log string
    (never carries a minted token or response body)."""

    outcome: LiveAuthOutcome
    detail: str

    @property
    def authorized(self) -> bool:
        return self.outcome is LiveAuthOutcome.AUTHORIZED


# The closure intake receives (built by `make_installation_authorizer` over an
# `AppGitHubClient`). Kept githubkit-free at the type level so `agent/` never imports
# the SDK: `(installation_id, owner, repo) -> LiveAuthResult`.
InstallationAuthorizer = Callable[[int, str, str], Awaitable[LiveAuthResult]]


def _status_code(exc: Exception) -> int | None:
    """Extract the HTTP status from a githubkit `RequestFailed` (raised on non-2xx).
    Mirrors `github/publisher.py`'s extraction; `None` means no response reached us
    (network error / pre-response failure) → the caller treats it as uncertainty."""
    return getattr(getattr(exc, "response", None), "status_code", None)


async def check_installation_authorization(
    app_client: AppGitHubClient,
    *,
    installation_id: int,
    owner: str,
    repo: str,
) -> LiveAuthResult:
    """Live App-JWT authorization check for `(installation_id, owner, repo)`. Fail-closed
    on any non-affirmative result (#065). Never consults the local cache.

    `owner` is accepted for symmetry/logging but the App-level endpoints key on
    `installation_id` + the repo NAME; the repo-scoped token body uses `repo` (a name
    string, per `POST /app/installations/{id}/access_tokens` `repositories: [name]`).
    """
    # Check 1 — installation exists AND is not suspended. GET returns 200 for an existing
    # install (with `suspended_at` null when active) and 404 when uninstalled; suspension
    # is signalled by the body field, NOT a status code (per the REST contract).
    try:
        inst_resp = await app_client.arequest(
            "GET",
            f"/app/installations/{installation_id}",
            headers=_API_VERSION_HEADER,
        )
    except Exception as exc:
        status = _status_code(exc)
        if status == 404:
            return _denied(
                LiveAuthOutcome.UNINSTALLED,
                installation_id,
                repo,
                f"GET /app/installations/{installation_id} → 404 (uninstalled)",
            )
        return _denied(
            LiveAuthOutcome.UNCERTAIN,
            installation_id,
            repo,
            f"install check errored: {type(exc).__name__} status={status}",
        )

    try:
        installation = json.loads(inst_resp.text)
        suspended = installation.get("suspended_at") is not None
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        # Can't confirm active → fail closed (uncertainty).
        return _denied(
            LiveAuthOutcome.UNCERTAIN,
            installation_id,
            repo,
            f"install response unparseable: {type(exc).__name__}",
        )
    if suspended:
        return _denied(
            LiveAuthOutcome.SUSPENDED,
            installation_id,
            repo,
            "installation suspended (suspended_at is set)",
        )

    # Check 2 — repo accessible. Mint a token scoped to THIS repo; 201 means the install
    # still covers it. Any non-201 (403 suspended, 404 gone, 422 repo-not-covered, network
    # error) fails closed. We discard the minted token — the mint attempt IS the probe.
    try:
        await app_client.arequest(
            "POST",
            f"/app/installations/{installation_id}/access_tokens",
            json={"repositories": [repo]},
            headers=_API_VERSION_HEADER,
        )
    except Exception as exc:
        status = _status_code(exc)
        if status == 403:
            outcome = LiveAuthOutcome.SUSPENDED
        elif status == 404:
            outcome = LiveAuthOutcome.UNINSTALLED
        elif status is None:
            outcome = LiveAuthOutcome.UNCERTAIN
        else:
            # 422 (repo not covered), 401 (our JWT), any other 4xx → repo not accessible
            # under this install. Fail closed regardless; the status is in `detail`.
            outcome = LiveAuthOutcome.REPO_INACCESSIBLE
        return _denied(
            outcome,
            installation_id,
            repo,
            f"repo-scoped token mint failed: status={status}",
        )

    logger.info(
        "live_auth authorized",
        extra={"installation_id": installation_id, "repo": repo},
    )
    return LiveAuthResult(LiveAuthOutcome.AUTHORIZED, "installation active and repo accessible")


def _denied(
    outcome: LiveAuthOutcome, installation_id: int, repo: str, detail: str
) -> LiveAuthResult:
    """Build a fail-closed result and log it at WARNING (there is no audit event for the
    denial per #065 — the log + the terminal `skipped` transition carry it)."""
    logger.warning(
        "live_auth denied",
        extra={
            "installation_id": installation_id,
            "repo": repo,
            "outcome": outcome.value,
            "detail": detail,
        },
    )
    return LiveAuthResult(outcome, detail)


def make_installation_authorizer(app_client: AppGitHubClient) -> InstallationAuthorizer:
    """Bind an `AppGitHubClient` into an `InstallationAuthorizer` closure for injection
    into `build_graph` → intake. `lifespan` constructs the app client once
    (`make_app_client`) and calls this; intake calls the returned closure and stays
    githubkit-free."""

    async def authorize(installation_id: int, owner: str, repo: str) -> LiveAuthResult:
        return await check_installation_authorization(
            app_client, installation_id=installation_id, owner=owner, repo=repo
        )

    return authorize
