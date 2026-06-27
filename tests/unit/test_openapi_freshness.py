"""Freshness guard for the checked-in dashboard API contract (FUP-200).

`dashboard/openapi.json` is a generated artifact the dashboard's `schema.d.ts` is built from.
Before this guard it had no scripted regen target and silently drifted from the route surface +
`events.py` models. This test fails if the checked-in file no longer matches what the app produces
— the fix is to run `uv run python scripts/gen_openapi.py` (which also regenerates `schema.d.ts`).

Pure: `app.openapi()` builds the schema offline (no lifespan, no DB).
"""

from __future__ import annotations

import json
from pathlib import Path

from outrider.main import create_app

_OPENAPI = Path(__file__).resolve().parents[2] / "dashboard" / "openapi.json"


def test_dashboard_openapi_json_is_fresh() -> None:
    expected = json.dumps(create_app(demo_mode=False).openapi(), indent=2) + "\n"
    actual = _OPENAPI.read_text(encoding="utf-8")
    assert actual == expected, (
        "dashboard/openapi.json is stale vs create_app(demo_mode=False).openapi(); "
        "regenerate with `uv run python scripts/gen_openapi.py` (FUP-200)."
    )
