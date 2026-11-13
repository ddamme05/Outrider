"""Alembic must preserve URL escapes when reading its migration configuration."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import URL

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(
            URL.create(
                "postgresql+psycopg",
                username="migration_test",
                password="p@ss%word",  # noqa: S106 — offline fixture value.
                host="127.0.0.1",
                port=5433,
                database="outrider_test",
            ).render_as_string(hide_password=False),
            id="encoded-password",
        ),
        pytest.param(
            URL.create(
                "postgresql+psycopg",
                username="migration_test",
                database="outrider_test",
                query={"host": "/tmp/outrider_test_socket", "port": "5433"},  # noqa: S108 — URL data only.
            ).render_as_string(hide_password=False),
            id="encoded-socket-query",
        ),
    ],
)
def test_offline_migration_preserves_database_url(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", url)
    config = Config(output_buffer=StringIO())
    config.set_main_option("script_location", str(_ROOT / "db/migrations"))

    command.upgrade(config, "base:base", sql=True)

    assert config.get_main_option("sqlalchemy.url") == url
    assert config.get_section(config.config_ini_section)["sqlalchemy.url"] == url
