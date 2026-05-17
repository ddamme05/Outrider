"""Unit tests for `_default_engine_factory` — driver-allowlist gate.

The lifespan's engine factory must fail-loud at startup if `DATABASE_URL`
uses a sync driver scheme; otherwise `create_async_engine` constructs
lazily and the failure surfaces deep in the first `persister.persist()`
call as `InvalidRequestError: The asyncio extension requires an async
driver` — far from the configuration source.

Pins the round-38 sharp-edges fold: explicit driver-allowlist gate
alongside `hide_parameters=True` so misconfigured `.env` values fail at
lifespan startup, not in production request handling.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from outrider.api.lifespan import _default_engine_factory

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def clean_database_url_env() -> Iterator[None]:
    """Save and restore the DATABASE_URL env var around each test."""
    original = os.environ.get("DATABASE_URL")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original


def test_default_engine_factory_requires_database_url_env_var(
    clean_database_url_env: None,
) -> None:
    """Missing `DATABASE_URL` env var fails-loud with a clear error."""
    os.environ.pop("DATABASE_URL", None)
    with pytest.raises(RuntimeError, match="DATABASE_URL env var is required"):
        _default_engine_factory()


def test_default_engine_factory_rejects_bare_postgresql_scheme(
    clean_database_url_env: None,
) -> None:
    """`postgresql://...` resolves to the SYNC psycopg2 driver — would
    crash `create_async_engine` on first use deep in a request. Reject at
    startup."""
    os.environ["DATABASE_URL"] = "postgresql://user:pw@host:5432/db"
    with pytest.raises(RuntimeError, match="async driver"):
        _default_engine_factory()


def test_default_engine_factory_rejects_psycopg2_scheme(
    clean_database_url_env: None,
) -> None:
    """`postgresql+psycopg2://...` is the explicit SYNC driver — same
    failure mode as bare `postgresql://`. Reject at startup."""
    os.environ["DATABASE_URL"] = "postgresql+psycopg2://user:pw@host:5432/db"
    with pytest.raises(RuntimeError, match="async driver"):
        _default_engine_factory()


async def test_default_engine_factory_accepts_psycopg_async_scheme(
    clean_database_url_env: None,
) -> None:
    """`postgresql+psycopg://` is the psycopg3 async driver scheme used in
    production. Construction succeeds (no DB connection happens yet —
    `create_async_engine` is lazy) AND the `hide_parameters=True`
    contract round-trips from the factory."""
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pw@host:5432/db"
    engine = _default_engine_factory()
    try:
        assert engine.sync_engine.hide_parameters is True
    finally:
        await engine.dispose()


def test_default_engine_factory_rejects_asyncpg_scheme(
    clean_database_url_env: None,
) -> None:
    """`postgresql+asyncpg://` is rejected — the asyncpg driver is not a
    project dependency (DECISIONS.md#001 standardizes on psycopg3 async),
    so accepting the scheme would advertise a URL shape that crashes
    `create_async_engine` with `ModuleNotFoundError` at construction.
    Reject at lifespan startup with the same gate that catches sync URLs."""
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://user:pw@host:5432/db"
    with pytest.raises(RuntimeError, match="canonical async driver"):
        _default_engine_factory()


def test_default_engine_factory_rejects_completely_unrelated_scheme(
    clean_database_url_env: None,
) -> None:
    """A typo or copy-paste from another project (`mysql://`, `sqlite:///`)
    fails-loud with the same async-driver message."""
    os.environ["DATABASE_URL"] = "sqlite:///tmp/foo.db"
    with pytest.raises(RuntimeError, match="async driver"):
        _default_engine_factory()
