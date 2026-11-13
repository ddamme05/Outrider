"""Check dashboard API schema freshness and generator rollback.

Generate it offline without startup or a database. If it differs, run
``uv run python scripts/gen_openapi.py`` to update it and schema.d.ts. See FUP-200."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX shebang handling")
def test_openapi_generator_restores_both_mirrors_when_npm_cannot_start(tmp_path: Path) -> None:
    generator = tmp_path / "scripts" / "gen_openapi.py"
    generator.parent.mkdir()
    shutil.copyfile(_DASHBOARD.parent / "scripts" / "gen_openapi.py", generator)

    openapi = tmp_path / "dashboard" / "openapi.json"
    schema = tmp_path / "dashboard" / "src" / "api" / "schema.d.ts"
    schema.parent.mkdir(parents=True)
    previous_spec = json.loads(_OPENAPI.read_bytes())
    previous_spec["info"]["title"] += " previous"
    previous_openapi = (json.dumps(previous_spec, indent=2) + "\n").encode("utf-8")
    previous_schema = (_DASHBOARD / "src" / "api" / "schema.d.ts").read_bytes()
    openapi.write_bytes(previous_openapi)
    schema.write_bytes(previous_schema)

    npm = tmp_path / "bin" / "npm"
    npm.parent.mkdir()
    npm.write_text(f"#!{tmp_path / 'missing-node'}\n", encoding="utf-8")
    npm.chmod(0o755)
    result = subprocess.run(  # noqa: S603 - copied local generator with a test-only PATH
        [sys.executable, str(generator)],
        cwd=tmp_path,
        env={
            "PATH": str(npm.parent),
            "PYTHONPATH": str(_DASHBOARD.parent / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C.UTF-8",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert openapi.read_bytes() == previous_openapi, "npm launch failure rewrote openapi.json"
    assert schema.read_bytes() == previous_schema
    assert "reverted BOTH openapi.json and schema.d.ts" in result.stderr
