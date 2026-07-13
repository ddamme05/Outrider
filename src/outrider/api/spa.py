# See DECISIONS.md#069 — the FastAPI app serves the built dashboard SPA in the V1
# production image (static assets + index.html history-fallback), behind the API
# routers. The demo / API-only image leaves OUTRIDER_SERVE_SPA unset and this module
# is a no-op there (Caddy serves the SPA on the demo box).
"""Serve the built dashboard SPA from the FastAPI app (production image).

`OUTRIDER_SERVE_SPA` is a strict tri-state contract (`DECISIONS.md#069`):

- **absent** — deliberate demo / API-only image; no SPA mount, no ``dist/`` required.
- **exactly** ``"1"`` — SPA required; a missing ``dist/index.html`` fails startup
  (`RuntimeError`), so a broken production build never boots UI-less.
- **any other present value** (``"0"``, ``""``, a typo) — configuration error; fails
  startup rather than silently downgrading to "not declared" (which would let a
  mistyped override boot without its UI).

Route precedence (`DECISIONS.md#069`): the SPA fallback is registered AFTER every API
router (GET/HEAD only). For a path that matched no API route it serves, in order: a real
built file if present (any ``Accept``); else 404 if the path looks like a file (its last
segment has an extension) — a missing asset never becomes the app shell; else, for a
route-shaped path with ``Accept: text/html``, ``index.html`` (the SPA's own ``/reviews``
client routes land here). Reserved backend namespaces are excluded root-and-descendant
(404 for unknown sub-paths).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, Response
from fastapi.responses import FileResponse

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import FastAPI

_SERVE_SPA_ENV = "OUTRIDER_SERVE_SPA"
_DIST_DIR_ENV = "OUTRIDER_SPA_DIST_DIR"
_DEFAULT_DIST_DIR = "/app/dashboard_dist"

# Reserved backend namespaces (root-and-descendant). The SPA fallback returns 404 for
# any unknown sub-path under these — never index.html. `/reviews` is deliberately ABSENT:
# it is the method/path-shared exception (backend owns POST /reviews/{id}/decide; the SPA
# owns GET /reviews, /reviews/:id, /reviews/:id/replay). Kept in sync with the real router
# set by tests/unit/test_spa.py::test_reserved_prefixes_match_backend_namespaces.
RESERVED_PREFIXES: tuple[str, ...] = (
    "/api",
    "/webhooks",
    "/slack",
    "/health",
    "/privacy",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def resolve_spa_dist_dir(env: Mapping[str, str] | None = None) -> Path | None:
    """Tri-state resolution of the SPA dist directory from `OUTRIDER_SERVE_SPA`.

    Returns the validated ``dist/`` Path when serving is enabled, or ``None`` when the
    variable is absent (demo / API-only image). Raises `RuntimeError` on an enabled flag
    with a missing build, or on any invalid flag value (fail-loud, `DECISIONS.md#069`).
    """
    env = os.environ if env is None else env
    raw = env.get(_SERVE_SPA_ENV)
    if raw is None:
        return None
    if raw != "1":
        raise RuntimeError(
            f"{_SERVE_SPA_ENV}={raw!r} is invalid: set exactly '1' to serve the SPA, or "
            f"leave it unset for an API-only image. Any other value is a configuration error."
        )
    dist = Path(env.get(_DIST_DIR_ENV, _DEFAULT_DIST_DIR))
    if not (dist / "index.html").is_file():
        raise RuntimeError(
            f"{_SERVE_SPA_ENV}=1 but no built dashboard at {dist / 'index.html'}. The "
            f"production image must bake the Vite build; a missing dist is a broken build."
        )
    return dist


def _is_reserved(path: str) -> bool:
    """True if `path` is a reserved backend namespace root or one of its descendants."""
    return any(path == prefix or path.startswith(prefix + "/") for prefix in RESERVED_PREFIXES)


def _safe_static_file(dist: Path, rel: str) -> Path | None:
    """Resolve `rel` under `dist`, rejecting traversal escapes; ``None`` if outside `dist`."""
    if not rel:
        return None
    candidate = (dist / rel).resolve()
    dist_resolved = dist.resolve()
    if candidate == dist_resolved or dist_resolved in candidate.parents:
        return candidate
    return None


def mount_spa_if_configured(app: FastAPI, *, env: Mapping[str, str] | None = None) -> bool:
    """Mount the SPA fallback when ``OUTRIDER_SERVE_SPA=1``; returns True if mounted.

    Must be called AFTER all API routers are registered — the catch-all is
    registration-order-last, so specific API routes always win (`DECISIONS.md#069`).
    """
    dist = resolve_spa_dist_dir(env)
    if dist is None:
        return False
    index = dist / "index.html"

    async def spa_fallback(full_path: str, request: Request) -> Response:
        path = "/" + full_path
        # Unknown sub-path under a reserved backend namespace -> 404 (never index.html).
        if _is_reserved(path):
            raise HTTPException(status_code=404)
        # A real built file (hashed asset, favicon, robots.txt, ...) -> serve it, any Accept.
        candidate = _safe_static_file(dist, full_path)
        if candidate is not None and candidate.is_file():
            return FileResponse(candidate)
        # A path that LOOKS like a file (its last segment has an extension) but isn't on disk
        # is a MISSING ASSET -> 404, never the app shell — so a broken build (or a stray
        # /robots.txt, /favicon.ico, /assets/*.js) fails loud instead of masquerading as a
        # 200. Only route-shaped (extension-less) paths reach the history fallback below.
        if "." in full_path.rsplit("/", 1)[-1]:
            raise HTTPException(status_code=404)
        # History fallback: browser navigations (Accept: text/html) to a client route get
        # the app shell (React Router renders the view / its own 404).
        if "text/html" in request.headers.get("accept", ""):
            return FileResponse(index)
        raise HTTPException(status_code=404)

    app.add_api_route(
        "/{full_path:path}",
        spa_fallback,
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    return True
