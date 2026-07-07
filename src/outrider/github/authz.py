# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Live GitHub authorization check (DECISIONS.md#065).

The local install tables are a cache; GitHub is the authorization authority, checked
LIVE at intake. As the App (App-JWT), this module confirms — for a given
`(installation_id, repo_id)` — that:

  1. the installation still exists and is NOT suspended
     (`GET /app/installations/{id}` → 200 with `suspended_at` null; 404 → uninstalled), and
  2. the target repo is still accessible
     (`POST /app/installations/{id}/access_tokens` scoped to the repo → 201).

Authorization keys on the IMMUTABLE `repo_id` (via `repository_ids`), not the mutable repo
name: a rename must not silently deny (or, worse, mis-authorize) a still-covered repo.

Per #065 the two checks fail CLOSED: any non-affirmative result — suspended, uninstalled,
repo-removed, OR a network error / uncertainty — collapses to `authorized == False` (the
caller drives the review to `skipped`). The local cache is never consulted for the answer.
It is the GitHub-authority counterpart to the DB-cache `active_repo_membership()` gate on
the webhook path.

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

from outrider.github.auth import make_app_client

if TYPE_CHECKING:
    from outrider.github.auth import AppGitHubClient
    from outrider.github.config import GitHubAppSettings

logger = logging.getLogger(__name__)

# Pin the REST API version to match the rest of the GitHub surface (publisher /
# fetch send the same header).
_API_VERSION_HEADER: Final[dict[str, str]] = {"X-GitHub-Api-Version": "2026-03-10"}


class LiveAuthOutcome(StrEnum):
    """The live-check result. Only `AUTHORIZED` proceeds; every other value fails closed
    (#065). The non-authorized variants exist for log/observability detail, not for
    differentiated caller behavior — the caller branches on `LiveAuthResult.authorized`."""

    AUTHORIZED = "authorized"
    # SUSPENDED / UNINSTALLED are only produced by the GET install-check, where the
    # pinned 2026-03-10 REST contract makes them reliable (`suspended_at` body field /
    # 404). The token-mint step never claims a specific cause — see UNCERTAIN.
    SUSPENDED = "suspended"
    UNINSTALLED = "uninstalled"
    # Fail-closed catch-all: any token-mint failure (the contract does not reliably
    # attribute 403/404/422/401/429/5xx to a cause), a network error, or an unparseable
    # install response. Authorization is denied; we simply do not over-label the reason.
    UNCERTAIN = "uncertain"


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
# `AppGitHubClient`). Kept githubkit-free at the type level so `agent/` never imports the
# SDK: `(installation_id, repo_id) -> LiveAuthResult`.
InstallationAuthorizer = Callable[[int, int], Awaitable[LiveAuthResult]]


def _status_code(exc: Exception) -> int | None:
    """Extract the HTTP status from a githubkit `RequestFailed` (raised on non-2xx).
    Mirrors `github/publisher.py`'s extraction; `None` means no response reached us
    (network error / pre-response failure) → the caller treats it as uncertainty."""
    return getattr(getattr(exc, "response", None), "status_code", None)


async def check_installation_authorization(
    app_client: AppGitHubClient,
    *,
    installation_id: int,
    repo_id: int,
) -> LiveAuthResult:
    """Live App-JWT authorization check for `(installation_id, repo_id)`. Fail-closed on any
    non-affirmative result (#065). Never consults the local cache.

    ENTERS `app_client` as an async context manager so the GET + POST for this one
    authorization share ONE underlying httpx client that closes on exit. githubkit's
    `reusing-client` guidance (0.15.3) warns that the un-entered path creates a new client
    per request and that "repeatedly creating new HTTP clients may lead to memory leaks".
    `make_installation_authorizer` passes a FRESH, un-entered client per invocation (a
    githubkit context manager cannot be entered twice), so this single enter is safe.

    Error model — precedence, HIGHEST first:
      1. `CancelledError` from ANY stage (enter / probe body / exit) always PROPAGATES — a task
         being cancelled must never be turned into a result. (Exit cancellation wins even over a
         pending body defect: the outer handler is `except Exception`, not `BaseException`, so a
         `__aexit__` cancellation propagates before the body defect is re-raised.)
      2. Otherwise, an UNEXPECTED probe-body exception (a defect) PROPAGATES → intake's
         `failed`/fail-loud path, beating an ordinary `__aexit__` (close) failure. The probe
         converts every EXPECTED network/parse failure into a result, so a raise is a bug.
      3. Otherwise, a lifecycle failure (`__aenter__` / `__aexit__`; constructor handled in
         `make_installation_authorizer`) is live-check uncertainty → `UNCERTAIN` (→ `skipped`).
    """
    # Enter the client under `async with` (creates the shared httpx client, closes on exit).
    # The probe body's exception (defect or cancellation) is captured INSIDE the block, so the
    # `async with` exits CLEANLY and an ORDINARY `__aexit__` (close) `Exception` cannot replace
    # the load-bearing body exception — it is re-raised preferentially below. (An exit
    # `CancelledError` intentionally DOES win, per precedence rule 1: it is a `BaseException`, so
    # it bypasses the `except Exception` handler and propagates before the re-raise.)
    body_result: LiveAuthResult | None = None
    body_error: BaseException | None = None
    try:
        async with app_client:
            try:
                body_result = await _probe_installation(
                    app_client, installation_id=installation_id, repo_id=repo_id
                )
            except BaseException as exc:  # noqa: BLE001 — captured to re-raise preferentially
                body_error = exc
    except Exception as ctx_exc:
        # `__aenter__` (before the probe ran → both vars still None) or `__aexit__` (close)
        # failed — lifecycle uncertainty. But a captured body exception is LOAD-BEARING
        # (re-raised below) and takes precedence over a close failure, so report lifecycle
        # uncertainty ONLY when none is pending. `except Exception`, NOT `BaseException`: a
        # `CancelledError` (graph cancellation / shutdown) MUST propagate.
        if body_error is None:
            return _denied(
                LiveAuthOutcome.UNCERTAIN,
                installation_id,
                repo_id,
                f"live-auth client lifecycle error: {type(ctx_exc).__name__}",
            )

    # A captured body defect / cancellation beats an ORDINARY close failure (an exit
    # `CancelledError` already propagated above via the `except Exception` bypass — precedence
    # rule 1). Re-raise it: an unexpected probe defect → intake's `failed`/fail-loud path; a
    # body `CancelledError` → propagates.
    if body_error is not None:
        raise body_error
    if body_result is None:  # defensive: body succeeded ⇒ result was set; fail closed if not
        return _denied(
            LiveAuthOutcome.UNCERTAIN, installation_id, repo_id, "live-auth produced no result"
        )
    return body_result


async def _probe_installation(
    app_client: AppGitHubClient,
    *,
    installation_id: int,
    repo_id: int,
) -> LiveAuthResult:
    """The two ordered App-JWT probes, run against an ALREADY-ENTERED `app_client` (its
    httpx client is live for the block). Split from the context-manager wrapper so the GET
    and POST reuse one client rather than creating one per request."""
    # Check 1 — installation exists AND is not suspended. GET returns 200 for an existing
    # install and 404 when uninstalled; suspension is signalled by the body's `suspended_at`
    # field (non-null), NOT a status code, so the field's PRESENCE is required — a response
    # missing it is treated as uncertainty (fail closed), never silently "active".
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
                repo_id,
                f"GET /app/installations/{installation_id} → 404 (uninstalled)",
            )
        return _denied(
            LiveAuthOutcome.UNCERTAIN,
            installation_id,
            repo_id,
            f"install check errored: {type(exc).__name__} status={status}",
        )

    try:
        installation = json.loads(inst_resp.text)
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        return _denied(
            LiveAuthOutcome.UNCERTAIN,
            installation_id,
            repo_id,
            f"install response not JSON: {type(exc).__name__}",
        )
    if not isinstance(installation, dict) or "suspended_at" not in installation:
        # Can't confirm active — the affirmative signal (an explicit `suspended_at`) is
        # absent. Fail closed.
        return _denied(
            LiveAuthOutcome.UNCERTAIN,
            installation_id,
            repo_id,
            "install response missing the suspended_at field",
        )
    if installation["suspended_at"] is not None:
        return _denied(
            LiveAuthOutcome.SUSPENDED,
            installation_id,
            repo_id,
            "installation suspended (suspended_at is set)",
        )

    # Check 2 — repo accessible. Mint a token scoped to this repo by IMMUTABLE id; 201 means
    # the install still covers it. We discard the minted token — the mint attempt IS the probe.
    # (Publish, per #065, needs the token itself; it mints+consumes its own via a separate
    # wrapper op — slice 3 — because this probe is authorize-only.) The GET above already
    # settled existence + suspension reliably; a mint failure here is a non-affirmative repo
    # probe, but the pinned 2026-03-10 REST contract does NOT reliably attribute a cause to
    # 403 ("Forbidden"), 404, 422 ("Validation failed or spammed"), 401, 429, or 5xx — so we
    # do not claim suspended / repo-inaccessible. Any failure fails closed as UNCERTAIN; the
    # status is in `detail` for ops.
    try:
        await app_client.arequest(
            "POST",
            f"/app/installations/{installation_id}/access_tokens",
            json={"repository_ids": [repo_id]},
            headers=_API_VERSION_HEADER,
        )
    except Exception as exc:
        return _denied(
            LiveAuthOutcome.UNCERTAIN,
            installation_id,
            repo_id,
            f"repo-scoped token mint failed: status={_status_code(exc)}",
        )

    logger.info(
        "live_auth authorized",
        extra={"installation_id": installation_id, "repo_id": repo_id},
    )
    return LiveAuthResult(LiveAuthOutcome.AUTHORIZED, "installation active and repo accessible")


def _denied(
    outcome: LiveAuthOutcome, installation_id: int, repo_id: int, detail: str
) -> LiveAuthResult:
    """Build a fail-closed result and log it at WARNING (there is no audit event for the
    denial per #065 — the log + the terminal `skipped` transition carry it)."""
    logger.warning(
        "live_auth denied",
        extra={
            "installation_id": installation_id,
            "repo_id": repo_id,
            "outcome": outcome.value,
            "detail": detail,
        },
    )
    return LiveAuthResult(outcome, detail)


def make_installation_authorizer(settings: GitHubAppSettings) -> InstallationAuthorizer:
    """Bind `GitHubAppSettings` into an `InstallationAuthorizer` closure for injection into
    `build_graph` → intake.

    Each invocation constructs a FRESH App-JWT client (`make_app_client`) that
    `check_installation_authorization` async-with-scopes for its GET + POST pair — per
    githubkit's reusing-client guidance, ONE context-managed client per authorization,
    closed on exit (the un-entered path creates a client per request and may leak). A single
    shared long-lived client is deliberately NOT used: githubkit keeps its httpx client in a
    task-local ContextVar, and intake runs in a per-review task, so a lifespan-entered client
    would be invisible here. `lifespan` passes the validated settings once; intake calls the
    returned closure and stays githubkit-free. The PEM is read per authorization (once per
    review), matching the per-installation client's brief-PEM-window pattern."""

    async def authorize(installation_id: int, repo_id: int) -> LiveAuthResult:
        try:
            app_client = make_app_client(settings)
        except Exception as exc:
            # App-JWT client CONSTRUCTION failed (e.g. malformed settings / PEM). Live-check
            # uncertainty → fail closed (UNCERTAIN → intake `skipped`), not intake's `failed`
            # path. `except Exception`, NOT `BaseException`: cancellation must propagate.
            return _denied(
                LiveAuthOutcome.UNCERTAIN,
                installation_id,
                repo_id,
                f"App-JWT client construction failed: {type(exc).__name__}",
            )
        return await check_installation_authorization(
            app_client, installation_id=installation_id, repo_id=repo_id
        )

    return authorize


# `GET /app/installations` list-page size (GitHub caps per_page at 100) + a safety cap on pages
# for the reconcile janitor. A self-host serves one org's installs (#066), so pagination is
# defensive — a real deployment has a handful of installs, well under a single page.
_LIST_PAGE_SIZE: Final[int] = 100
_MAX_LIST_PAGES: Final[int] = 100


async def list_installation_ids(settings: GitHubAppSettings) -> set[int]:
    """Return the installation ids GitHub currently lists for this App (App-JWT
    `GET /app/installations`, paginated). The reconcile janitor (`#065` / `#012` / `#067`) uses
    this as the AUTHORITY to catch MISSED lifecycle events: a local install absent from this set is
    a missed `installation.deleted` (→ tombstone); a tombstoned local install present here is a
    live-confirmed reinstall (→ clear the tombstone).

    RAISES on any failure (network error, non-2xx, non-list body, missing `id`, page-cap exceeded)
    — the janitor MUST NOT reconcile against a partial or empty-by-error list, or it would wrongly
    tombstone live installs. A fresh App-JWT client is `async with`-scoped for the paginated GET
    (githubkit reusing-client guidance, same shape as `make_installation_authorizer`)."""
    ids: set[int] = set()
    async with make_app_client(settings) as app_client:
        for page in range(1, _MAX_LIST_PAGES + 1):
            resp = await app_client.arequest(
                "GET",
                f"/app/installations?per_page={_LIST_PAGE_SIZE}&page={page}",
                headers=_API_VERSION_HEADER,
            )
            batch = json.loads(resp.text)
            if not isinstance(batch, list):
                raise TypeError(
                    f"GET /app/installations returned {type(batch).__name__}, expected list"
                )
            for item in batch:
                ids.add(int(item["id"]))
            if len(batch) < _LIST_PAGE_SIZE:
                return ids
    raise RuntimeError(
        f"GET /app/installations exceeded the {_MAX_LIST_PAGES}-page cap; refusing to "
        "reconcile against a truncated list"
    )
