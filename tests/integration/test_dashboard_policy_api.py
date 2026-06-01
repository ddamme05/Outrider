"""Integration tests for `GET /api/policy/{version}` (FUP-132).

The load-bearing test is `stored-not-active`: the endpoint must return the policy
as STORED for the requested version (via `load_policy_for_version`), never the
active in-code `SEVERITY_POLICY` — otherwise a historical review's table would
silently re-render under today's policy (`severity-policy-versioned-for-replay`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from outrider.api.dashboard import policy_router
from outrider.policy import FindingSeverity, FindingType
from outrider.policy.severity import SEVERITY_POLICY

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}


async def _seed_policy(engine: AsyncEngine, version: str, mapping: dict[str, str]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO severity_policies (version, policy) VALUES (:v, CAST(:p AS jsonb))"),
            {"v": version, "p": json.dumps(mapping)},
        )


@pytest_asyncio.fixture
async def client(migrated_db: str) -> AsyncGenerator[tuple[TestClient, AsyncEngine]]:
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app = FastAPI()
    app.include_router(policy_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    try:
        yield TestClient(app), engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_returns_stored_policy_not_active(client: tuple[TestClient, AsyncEngine]) -> None:
    """The stored value wins over the active in-code policy."""
    api, engine = client
    # A COMPLETE mapping (load_policy_for_version requires every FindingType), but with
    # SQL_INJECTION flipped to LOW — the active policy classifies it CRITICAL.
    mapping = {ft.value: sev.value for ft, sev in SEVERITY_POLICY.items()}
    mapping["sql_injection"] = "low"
    assert SEVERITY_POLICY[FindingType.SQL_INJECTION] == FindingSeverity.CRITICAL  # active differs
    await _seed_policy(engine, "9.9.9", mapping)

    resp = api.get("/api/policy/9.9.9", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "9.9.9"
    entry = next(e for e in body["entries"] if e["finding_type"] == "sql_injection")
    # STORED value, not the active CRITICAL.
    assert entry["severity"] == "low"
    assert entry["severity"] != FindingSeverity.CRITICAL.value
    # Dimension always resolves (lockstep, #021).
    assert entry["dimension"] == "security"


@pytest.mark.asyncio
async def test_unknown_version_404(client: tuple[TestClient, AsyncEngine]) -> None:
    api, _ = client
    assert api.get("/api/policy/0.0.0", headers=_AUTH).status_code == 404


@pytest.mark.asyncio
async def test_corrupt_stored_row_is_loud_500(client: tuple[TestClient, AsyncEngine]) -> None:
    """An undecodable finding_type key surfaces as a structured 500, not a partial table."""
    api, engine = client
    await _seed_policy(engine, "9.9.8", {"not_a_real_finding_type": "low"})
    resp = api.get("/api/policy/9.9.8", headers=_AUTH)
    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == "policy_version_shape"


@pytest.mark.asyncio
async def test_auth_required(client: tuple[TestClient, AsyncEngine]) -> None:
    api, _ = client
    assert api.get("/api/policy/1.0.0").status_code == 401
