"""Caddy/FastAPI demo-surface parity guard.

`deploy/Caddyfile` (the keyless demo topology: Caddy serves the SPA and proxies
the API) is a hand-maintained mirror of the FastAPI demo-mode route surface —
the same mirror class as `dashboard/vite.config.ts`, which shipped broken twice
(`/privacy` 587f4a9, `/setup` FUP-230) before its contract-sweep rule landed.
This test makes the demo mirror mechanical: every route `create_app(demo_mode=True)`
serves must be covered by a `handle` matcher that proxies to the app, or the SPA
catch-all swallows it and returns index.html with a 200.

Parsing is deliberately minimal (this is a ~25-line Caddyfile): `handle <path> {`
blocks only. Named matchers (`handle @x`) or a reshaped file fail loud rather
than silently skipping — the vacuous-pass guard below.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.routing import APIRoute

from outrider.main import create_app

_CADDYFILE = Path(__file__).resolve().parents[2] / "deploy" / "Caddyfile"

# FastAPI auto-mounts these; framework-level, not part of the demo surface
# (mirrors tests/unit/test_demo_mode_routes.py).
_FASTAPI_BUILTINS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _proxied_matchers(src: str) -> list[str]:
    return re.findall(r"^\s*handle\s+(/[^\s{]*)\s*\{", src, re.MULTILINE)


def _covered(path: str, exact: set[str], prefixes: set[str]) -> bool:
    # Caddy v2 path-matcher semantics: a trailing `*` is a prefix match on
    # everything before it, so `/api/*` matches `/api/<anything>` (but NOT bare
    # `/api`) while `/privacy*` matches `/privacy` AND `/privacy/`. A matcher with
    # no trailing `*` is exact-match only. `prefixes` already has the `*` stripped.
    return path in exact or any(path.startswith(pre) for pre in prefixes)


def test_every_demo_route_is_proxied_by_the_caddyfile() -> None:
    src = _CADDYFILE.read_text(encoding="utf-8")
    matchers = _proxied_matchers(src)

    # Fail-loud guards against a vacuous pass on a reshaped Caddyfile:
    assert matchers, "no proxied handle blocks parsed — deploy/Caddyfile reshaped?"
    assert re.search(r"^\s*handle\s+@", src, re.MULTILINE) is None, (
        "named matchers appeared in deploy/Caddyfile — this parser does not "
        "understand them; extend the test before using them"
    )
    # Each parsed matcher block must actually proxy (and nothing else does).
    assert src.count("reverse_proxy") == len(matchers), (
        "reverse_proxy count diverges from parsed handle matchers — deploy/Caddyfile reshaped?"
    )

    exact = {m for m in matchers if not m.endswith("*")}
    prefixes = {m[:-1] for m in matchers if m.endswith("*")}

    app = create_app(demo_mode=True)
    backend = {
        r.path for r in app.routes if isinstance(r, APIRoute) and r.path not in _FASTAPI_BUILTINS
    }
    assert backend, "demo app exposed no routes — create_app reshaped?"

    uncovered = sorted(p for p in backend if not _covered(p, exact, prefixes))
    assert not uncovered, (
        f"demo routes not proxied by deploy/Caddyfile (the SPA catch-all will "
        f"swallow them): {uncovered} — add a handle block per route"
    )
