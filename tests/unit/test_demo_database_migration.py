"""Keep the demo API behind seed restoration and schema migration."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_demo_waits_for_migrations_after_the_seed_has_restored() -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose CLI unavailable")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "POSTGRES_PASSWORD": "offline-test",
        "OUTRIDER_ADMIN_API_KEY": "offline-test",
        "DEMO_DOMAIN": ":80",
    }
    version = subprocess.run([docker, "compose", "version"], capture_output=True, env=env)  # noqa: S603
    if version.returncode:
        pytest.skip("Docker Compose plugin unavailable")
    result = subprocess.run(  # noqa: S603 - fixed configuration check; no Docker daemon is needed
        [
            docker,
            "compose",
            "--env-file",
            os.devnull,
            "-f",
            str(ROOT / "deploy/docker-compose.demo.yml"),
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        check=True,
        env=env,
    )
    services = json.loads(result.stdout)["services"]
    app, migration = services["app"], services["migrate"]
    assert migration["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert migration["command"] == ["/app/.venv/bin/alembic", "upgrade", "head"]
    assert app["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert app["image"] == migration["image"]
    assert migration["environment"] == {"DATABASE_URL": app["environment"]["DATABASE_URL"]}
