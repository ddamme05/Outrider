"""Check that Caddy forwards every demo API route instead of serving the SPA shell.

Read the real demo app routes and the checked-in Caddyfile. The small parser
rejects unsupported matcher shapes so a changed file cannot pass unchecked."""

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
