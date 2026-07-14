"""Unit tests for the production SPA-serving layer (`outrider.api.spa`, DECISIONS.md#069).

Three concerns:
1. The tri-state `OUTRIDER_SERVE_SPA` contract (absent / "1" / invalid) is fail-loud.
2. Route precedence: the SPA fallback serves the app shell for its own GET client
   routes (incl. `/reviews/*`), 404s unknown sub-paths under reserved backend
   namespaces, never shadows `POST /reviews/{id}/decide`, and never serves the shell
   for a missing hashed asset or a path traversal.
3. `RESERVED_PREFIXES` stays in sync with the real router set (drift guard).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from outrider.api.spa import (
    RESERVED_DESCENDANT_PREFIXES,
    RESERVED_PREFIXES,
    _safe_static_file,
    mount_spa_if_configured,
    resolve_spa_dist_dir,
)
from outrider.main import create_app

if TYPE_CHECKING:
    from pathlib import Path

_HTML = {"accept": "text/html,application/xhtml+xml"}


def _make_dist(tmp_path: Path) -> Path:
    """A minimal Vite-style build tree: index.html + a hashed asset + a root file."""
    (tmp_path / "index.html").write_text("<!doctype html><div id=root>APP SHELL</div>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app-abc123.js").write_text("console.log('spa')")
    (tmp_path / "favicon.ico").write_text("icon-bytes")
    return tmp_path


def _env(dist: Path, flag: str = "1") -> dict[str, str]:
    return {"OUTRIDER_SERVE_SPA": flag, "OUTRIDER_SPA_DIST_DIR": str(dist)}


# ---------------------------------------------------------------------------
# 1. Tri-state OUTRIDER_SERVE_SPA contract (fail-loud)
# ---------------------------------------------------------------------------


def test_serve_spa_absent_returns_none(tmp_path: Path) -> None:
    """Absent variable → demo / API-only image; no dist required, no mount."""
    assert resolve_spa_dist_dir(env={}) is None


def test_serve_spa_one_with_build_returns_dist(tmp_path: Path) -> None:
    """`=1` with a real build → the validated dist Path."""
    dist = _make_dist(tmp_path)
    resolved = resolve_spa_dist_dir(env=_env(dist))
    assert resolved == dist


def test_serve_spa_one_without_build_fails(tmp_path: Path) -> None:
    """`=1` but no baked `dist/index.html` → fail-loud (a broken production build)."""
    with pytest.raises(RuntimeError, match="broken build"):
        resolve_spa_dist_dir(env=_env(tmp_path))


def test_serve_spa_rejects_index_symlink_escape(tmp_path: Path) -> None:
    """A symlinked index.html escaping dist fails startup — the app shell is served on EVERY
    HTML route (bypassing _safe_static_file), so a symlink escape would leak a host file."""
    dist = tmp_path / "dist"
    dist.mkdir()
    secret = tmp_path / "outside_index.html"
    secret.write_text("<html>SECRET OUTSIDE DIST</html>")
    (dist / "index.html").symlink_to(secret)
    with pytest.raises(RuntimeError, match="symlink escape"):
        resolve_spa_dist_dir(env=_env(dist))


@pytest.mark.parametrize("bad", ["0", "", "true", "yes", "1 ", " 1"])
def test_serve_spa_invalid_value_fails(bad: str, tmp_path: Path) -> None:
    """Any present-but-not-`1` value is a configuration error — fail startup, never
    silently treated as 'not declared' (which would boot a mistyped image UI-less)."""
    dist = _make_dist(tmp_path)
    with pytest.raises(RuntimeError, match="invalid"):
        resolve_spa_dist_dir(env=_env(dist, flag=bad))


# ---------------------------------------------------------------------------
# 2. Route precedence — SPA fallback over a representative backend router set
# ---------------------------------------------------------------------------


@pytest.fixture
def spa_client(tmp_path: Path) -> TestClient:
    """A FastAPI app with representative backend routes registered FIRST, then the SPA
    fallback mounted — mirroring create_app's ordering without the lifespan/DB deps."""
    dist = _make_dist(tmp_path)
    app = FastAPI()

    @app.get("/api/reviews")
    async def _reviews() -> dict[str, str]:
        return {"surface": "api"}

    @app.get("/privacy")
    async def _privacy() -> dict[str, str]:
        return {"surface": "privacy"}

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    # The one shared-namespace backend route: POST /reviews/{id}/decide (HITL).
    @app.post("/reviews/{review_id}/decide")
    async def _decide(review_id: str) -> dict[str, str]:
        return {"surface": "hitl", "review_id": review_id}

    mounted = mount_spa_if_configured(app, env=_env(dist))
    assert mounted is True
    return TestClient(app)


@pytest.mark.parametrize(
    "path", ["/", "/reviews", "/reviews/123", "/reviews/123/replay", "/settings", "/setup"]
)
def test_spa_serves_shell_for_client_routes(spa_client: TestClient, path: str) -> None:
    """Browser navigations to SPA client routes (incl. the whole GET `/reviews/*` space + the exact
    `/setup` F5 page — the failed-callback redirect target) get the app shell. `/reviews` is NOT
    reserved for GET; `/setup` is descendant-only reserved, so its EXACT path still serves the
    shell."""
    resp = spa_client.get(path, headers=_HTML)
    assert resp.status_code == 200
    assert "APP SHELL" in resp.text


def test_spa_does_not_shadow_hitl_post(spa_client: TestClient) -> None:
    """`POST /reviews/{id}/decide` still reaches the backend — the GET-only fallback
    never shadows it (the crux of the `/reviews` shared namespace)."""
    resp = spa_client.post("/reviews/123/decide")
    assert resp.status_code == 200
    assert resp.json() == {"surface": "hitl", "review_id": "123"}


def test_api_route_still_served(spa_client: TestClient) -> None:
    """A real API route wins over the fallback (registration order)."""
    resp = spa_client.get("/api/reviews", headers=_HTML)
    assert resp.status_code == 200
    assert resp.json() == {"surface": "api"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/unknown",
        "/privacy/unknown",
        "/health/x",
        "/webhooks/github",
        "/slack/install",
        "/docs/x",
        "/redoc/x",
        "/openapi.json/x",  # the one reserved entry with a dot / exact-path shape
        "/setup/reset",  # descendant-only: POST-only backend route, GET must not be the shell
        "/setup/unknown",  # descendant-only reserved: unknown sub-path 404s, not the shell
    ],
)
def test_reserved_namespace_unknown_subpath_404s(spa_client: TestClient, path: str) -> None:
    """Unknown sub-paths under reserved backend namespaces 404 — never the app shell, even for a
    text/html request. Covers both root-and-descendant prefixes and the descendant-only `/setup`
    (whose exact path IS the shell, but whose sub-paths are not)."""
    resp = spa_client.get(path, headers=_HTML)
    assert resp.status_code == 404
    assert "APP SHELL" not in resp.text


def test_real_hashed_asset_is_served(spa_client: TestClient) -> None:
    resp = spa_client.get("/assets/app-abc123.js")
    assert resp.status_code == 200
    assert "console.log" in resp.text


def test_missing_hashed_asset_404s_not_shell(spa_client: TestClient) -> None:
    """A missing asset fails loud (404), never falls back to index.html — even if the
    request happens to accept text/html — so a broken build is visible."""
    resp = spa_client.get("/assets/does-not-exist.js", headers=_HTML)
    assert resp.status_code == 404
    assert "APP SHELL" not in resp.text


def test_root_static_file_is_served(spa_client: TestClient) -> None:
    resp = spa_client.get("/favicon.ico")
    assert resp.status_code == 200
    assert "icon-bytes" in resp.text


def test_non_html_get_to_client_route_404s(spa_client: TestClient) -> None:
    """A non-navigation GET (Accept: application/json) to an SPA path gets 404, not the
    shell — the fallback is text/html-only (`DECISIONS.md#069`)."""
    resp = spa_client.get("/reviews/123", headers={"accept": "application/json"})
    assert resp.status_code == 404


def test_missing_root_asset_404s_not_shell(spa_client: TestClient) -> None:
    """A missing ROOT-level file (e.g. /robots.txt) with Accept: text/html returns 404,
    not the app shell — the missing-asset rule covers root files (extension = file-shaped),
    not just /assets/. A missing root asset must fail loud like a missing hashed asset."""
    resp = spa_client.get("/robots.txt", headers=_HTML)
    assert resp.status_code == 404
    assert "APP SHELL" not in resp.text


def test_post_to_fallback_path_is_not_shell(spa_client: TestClient) -> None:
    """A non-GET/HEAD request to a path with no backend route never serves the app shell —
    the fallback is GET/HEAD-only; Starlette answers 404/405."""
    resp = spa_client.post("/some/client/route")
    assert resp.status_code in (404, 405)
    assert "APP SHELL" not in resp.text


def test_head_request_on_client_route(spa_client: TestClient) -> None:
    resp = spa_client.head("/reviews/123", headers=_HTML)
    assert resp.status_code == 200


# _safe_static_file is the traversal guard; test it DIRECTLY. A request-level `..` test is
# vacuous — httpx/Starlette normalize `/../x` to `/x` before the request is sent, so the
# guard is never reached through the client. (This replaced exactly such a vacuous test.)


def test_safe_static_file_accepts_in_tree(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path)
    resolved = _safe_static_file(dist, "assets/app-abc123.js")
    assert resolved is not None and resolved.is_file()


@pytest.mark.parametrize("rel", ["../outside_secret.txt", "../../etc/passwd", "a/../../b"])
def test_safe_static_file_rejects_traversal(tmp_path: Path, rel: str) -> None:
    """`..` escapes out of dist resolve to a path outside the tree and are rejected."""
    dist = _make_dist(tmp_path)
    (tmp_path.parent / "outside_secret.txt").write_text("SECRET")
    assert _safe_static_file(dist, rel) is None


def test_safe_static_file_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink INSIDE dist pointing outside resolves (via .resolve()) to a path outside
    dist and is rejected — a baked build with a stray symlink can't leak host files."""
    dist = _make_dist(tmp_path)
    secret = tmp_path.parent / "outside_secret.txt"
    secret.write_text("SECRET")
    (dist / "leak.txt").symlink_to(secret)
    assert _safe_static_file(dist, "leak.txt") is None


# ---------------------------------------------------------------------------
# 2b. mount_spa_if_configured — no-op when unset, fail-loud when enabled without a build
# ---------------------------------------------------------------------------


def test_mount_spa_noop_when_unset() -> None:
    app = FastAPI()
    before = len(app.routes)
    assert mount_spa_if_configured(app, env={}) is False
    assert len(app.routes) == before  # no catch-all route added


def test_mount_spa_raises_when_enabled_without_build(tmp_path: Path) -> None:
    """The fail-loud contract propagates through the mount, not just the leaf resolver:
    OUTRIDER_SERVE_SPA=1 with no baked dist raises at mount time (→ app boot fails)."""
    app = FastAPI()
    with pytest.raises(RuntimeError, match="broken build"):
        mount_spa_if_configured(app, env=_env(tmp_path))


# ---------------------------------------------------------------------------
# 3. Drift guard — RESERVED_PREFIXES vs the real backend router set
# ---------------------------------------------------------------------------


def test_reserved_prefixes_match_backend_namespaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """`RESERVED_PREFIXES` (root+descendant) plus `RESERVED_DESCENDANT_PREFIXES` (exact path is the
    SPA, sub-paths reserved) must together cover every backend top-level namespace MINUS the
    deliberate `/reviews` shared exception. Fails if a new router adds a namespace the SPA fallback
    would wrongly swallow, or if a reserved prefix goes stale."""
    # Hermetic: neutralize any ambient OUTRIDER_SERVE_SPA so create_app's SPA mount is a
    # no-op here (a `=1` in the runner's env would else fail create_app with no baked dist).
    monkeypatch.delenv("OUTRIDER_SERVE_SPA", raising=False)
    monkeypatch.delenv("OUTRIDER_SPA_DIST_DIR", raising=False)
    # database credential mode so the /setup onboarding namespace mounts (#070) — its sub-paths are
    # backend, its exact GET is the SPA page, so it must be covered by RESERVED_DESCENDANT_PREFIXES.
    monkeypatch.setenv("OUTRIDER_GITHUB_CREDENTIAL_SOURCE", "database")
    monkeypatch.setenv("OUTRIDER_PUBLIC_BASE_URL", "https://drift.example")
    monkeypatch.setenv("OUTRIDER_SETUP_STATE_SECRET", "drift-guard-secret-long-enough-abcdef123")
    # enable_docs=True so /docs, /redoc, /openapi.json are registered: they stay in
    # RESERVED_PREFIXES even when the prod default (FUP-229) disables them, because a GET to a
    # disabled /docs must still 404 (reserved), not fall through to the SPA shell.
    app = create_app(demo_mode=False, enable_docs=True)
    segments: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path or path == "/":
            continue
        first = path.split("/")[1]
        if first.startswith("{"):  # a catch-all path param, not a namespace
            continue
        segments.add("/" + first)
    expected = segments - {"/reviews"} - set(RESERVED_DESCENDANT_PREFIXES)
    assert set(RESERVED_PREFIXES) == expected, (
        f"RESERVED_PREFIXES drifted from the router set. "
        f"missing={expected - set(RESERVED_PREFIXES)} stale={set(RESERVED_PREFIXES) - expected}"
    )
    # The /setup onboarding namespace IS present (database mode) and descendant-only reserved.
    assert "/setup" in segments
    assert "/setup" in RESERVED_DESCENDANT_PREFIXES
