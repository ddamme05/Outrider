"""Freshness guard for the checked-in dashboard API contract (FUP-200).

`dashboard/openapi.json` is a generated artifact the dashboard's `schema.d.ts` is built from.
Before this guard it had no scripted regen target and silently drifted from the route surface +
`events.py` models. This test fails if the checked-in file no longer matches what the app produces
— the fix is to run `uv run python scripts/gen_openapi.py` (which also regenerates `schema.d.ts`).

Pure: `app.openapi()` builds the schema offline (no lifespan, no DB).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from outrider.main import create_app

_DASHBOARD = Path(__file__).resolve().parents[2] / "dashboard"
_OPENAPI = _DASHBOARD / "openapi.json"


def test_dashboard_openapi_json_is_fresh() -> None:
    expected = json.dumps(create_app(demo_mode=False, enable_docs=True).openapi(), indent=2) + "\n"
    actual = _OPENAPI.read_text(encoding="utf-8")
    assert actual == expected, (
        "dashboard/openapi.json is stale vs create_app(demo_mode=False).openapi(); "
        "regenerate with `uv run python scripts/gen_openapi.py` (FUP-200)."
    )


def test_dashboard_schema_d_ts_is_fresh(tmp_path: Path) -> None:
    """schema.d.ts must equal `openapi-typescript(openapi.json)` — its codegen is deterministic
    (static header, no timestamp), so a direct compare is airtight. Closes the gap where a stale
    schema.d.ts slips past the openapi.json check (the openapi.json test can't see the .ts mirror).

    Skipped where the dashboard's openapi-typescript binary isn't installed (python-only CI); there
    the atomic `gen_openapi.py` (never updates the JSON without regenerating the .ts) plus the
    openapi.json freshness test together prevent a half-update.
    """
    binary = _DASHBOARD / "node_modules" / ".bin" / "openapi-typescript"
    if not binary.exists():
        pytest.skip("openapi-typescript not installed (dashboard/node_modules absent)")
    out = tmp_path / "schema.d.ts"
    # Build-tool subprocess, fixed args (the local binary), no untrusted input.
    subprocess.run([str(binary), "openapi.json", "-o", str(out)], cwd=_DASHBOARD, check=True)  # noqa: S603
    expected = out.read_text(encoding="utf-8")
    actual = (_DASHBOARD / "src" / "api" / "schema.d.ts").read_text(encoding="utf-8")
    assert actual == expected, (
        "dashboard/src/api/schema.d.ts is stale vs openapi.json; "
        "regenerate with `uv run python scripts/gen_openapi.py` (FUP-200)."
    )
