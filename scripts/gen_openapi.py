#!/usr/bin/env python
"""Regenerate `dashboard/openapi.json` + `dashboard/src/api/schema.d.ts` from the FastAPI app.

The single, deterministic regen target for the dashboard API contract (FUP-200). Run it after ANY
change to a FastAPI route or a response / audit-event model the dashboard surfaces:

    uv run python scripts/gen_openapi.py

It builds the PRODUCTION app (`create_app(demo_mode=False)` — the full route surface: webhooks +
mutations + slack, matching what the dashboard talks to), dumps `app.openapi()` to
`dashboard/openapi.json` with stable formatting (indent=2, `ensure_ascii`, trailing newline —
byte-for-byte reproducible for a fixed code version), then regenerates the TypeScript types via the
dashboard's `npm run gen:types` (openapi-typescript).

Before this target existed there was no scripted way to regenerate, so the checked-in `openapi.json`
silently drifted from `events.py` / the route surface (FUP-200). Running this in CI (or before
committing an audit-event field / route change) keeps the source↔mirror sync from drifting again.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from outrider.main import create_app

_REPO = Path(__file__).resolve().parent.parent
_OPENAPI = _REPO / "dashboard" / "openapi.json"


def main() -> int:
    spec = create_app(demo_mode=False).openapi()
    with _OPENAPI.open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    n_paths = len(spec.get("paths", {}))
    n_schemas = len(spec.get("components", {}).get("schemas", {}))
    print(f"wrote {_OPENAPI.relative_to(_REPO)} ({n_paths} paths, {n_schemas} schemas)")

    npm = shutil.which("npm")
    if npm is None:
        print(
            "WARNING: npm not found; schema.d.ts NOT regenerated. "
            "Run `cd dashboard && npm run gen:types` once npm is available.",
            file=sys.stderr,
        )
        return 1
    try:
        # Fixed command (npm resolved via shutil.which), no untrusted input — build-script
        # subprocess, not a GitHub-input path (the §5 shell-exec boundary is src/outrider/ only).
        subprocess.run([npm, "run", "gen:types"], cwd=_REPO / "dashboard", check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: `npm run gen:types` failed ({exc}); schema.d.ts not regenerated.",
            file=sys.stderr,
        )
        return 1
    print("regenerated dashboard/src/api/schema.d.ts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
