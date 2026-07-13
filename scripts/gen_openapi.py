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
_SCHEMA_DTS = _REPO / "dashboard" / "src" / "api" / "schema.d.ts"


def _restore(path: Path, backup: bytes | None) -> None:
    """Restore `path` to its pre-run bytes, or remove it if it did not exist before the run."""
    if backup is not None:
        path.write_bytes(backup)
    else:
        path.unlink(missing_ok=True)


def main() -> int:
    spec = create_app(demo_mode=False, enable_docs=True).openapi()
    new_bytes = (json.dumps(spec, indent=2) + "\n").encode("utf-8")
    n_paths = len(spec.get("paths", {}))
    n_schemas = len(spec.get("components", {}).get("schemas", {}))

    # ATOMIC: update BOTH mirrors or NEITHER. Writing openapi.json and then failing to regenerate
    # schema.d.ts (npm missing / gen:types error) would leave a fresh JSON with a stale .d.ts — a
    # half-update the openapi.json freshness test alone cannot catch. So: refuse to touch
    # openapi.json if npm is unavailable, and revert it if gen:types fails. A fresh openapi.json
    # then reliably implies schema.d.ts was regenerated from it.
    npm = shutil.which("npm")
    if npm is None:
        print(
            "ERROR: npm not found — refusing to update openapi.json without also regenerating "
            "schema.d.ts (both mirrors or neither). Install npm and rerun.",
            file=sys.stderr,
        )
        return 1

    # Back up BOTH mirrors: gen:types may truncate / partially write schema.d.ts before erroring,
    # so on failure we revert BOTH to the pre-run bytes (reverting only openapi.json would leave a
    # half-written .d.ts — the gap that breaks the "both mirrors or neither" guarantee).
    openapi_backup = _OPENAPI.read_bytes() if _OPENAPI.exists() else None
    schema_backup = _SCHEMA_DTS.read_bytes() if _SCHEMA_DTS.exists() else None
    _OPENAPI.write_bytes(new_bytes)
    try:
        # Fixed command (npm resolved via shutil.which), no untrusted input — build-script
        # subprocess, not a GitHub-input path (the §5 shell-exec boundary is src/outrider/ only).
        subprocess.run([npm, "run", "gen:types"], cwd=_REPO / "dashboard", check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        _restore(_OPENAPI, openapi_backup)
        _restore(_SCHEMA_DTS, schema_backup)
        print(
            f"ERROR: `npm run gen:types` failed ({exc}); reverted BOTH openapi.json and "
            "schema.d.ts so the two mirrors stay in lockstep.",
            file=sys.stderr,
        )
        return 1
    print(
        f"wrote dashboard/openapi.json ({n_paths} paths, {n_schemas} schemas) "
        "+ regenerated dashboard/src/api/schema.d.ts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
