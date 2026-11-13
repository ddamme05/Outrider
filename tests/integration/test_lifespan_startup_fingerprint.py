"""Check the active policy against its stored PostgreSQL row at startup.

Cover missing versions, changed mappings, and a matching policy. The tests use
the live active version rather than fixing the check to a particular release.
See specs/2026-05-19-analyze-foundation.md §0c."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from outrider.api.lifespan import StartupError, build_lifespan
from outrider.policy.severity import SEVERITY_POLICY, FindingSeverity, FindingType
from outrider.policy.versions import PolicyVersionShapeError


@pytest.fixture(autouse=True)
def _activate_github_app_env(github_app_env: None) -> None:  # noqa: ARG001
    """Lifespan hard-requires `GitHubAppSettings()` at startup."""


def _build_engine_factory(db_url: str):
    """Return an engine_factory that hands back a real async engine.

    `hide_parameters=True` per lifespan's production gate.
    """

    def _factory():
        return create_async_engine(db_url, hide_parameters=True)

    return _factory


async def test_happy_path_matches_active_version_seed(
    migrated_db: str,
    make_stub_llm_provider: type,
    in_memory_checkpointer_factory: object,
) -> None:
    """Lifespan starts when the migrated active policy matches the live mapping."""
    stub_provider = make_stub_llm_provider()
    lifespan = build_lifespan(
        engine_factory=_build_engine_factory(migrated_db),
        provider_factory=lambda _persister, _model_config, _host, _reasoning: stub_provider,
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    # No StartupError raised; lifespan body yields normally.
    async with lifespan(app):
        # Sanity: deps wired through to app.state after the fingerprint passes.
        assert app.state.engine is not None
        assert app.state.provider is stub_provider


async def test_missing_row_raises_startup_error(
    migrated_db: str,
    make_stub_llm_provider: type,
) -> None:
    """ACTIVE_POLICY_VERSION pointing at a version with no DB row raises."""
    stub_provider = make_stub_llm_provider()

    with patch("outrider.api.lifespan.ACTIVE_POLICY_VERSION", "9.9.9"):
        lifespan = build_lifespan(
            engine_factory=_build_engine_factory(migrated_db),
            provider_factory=lambda _persister, _model_config, _host, _reasoning: stub_provider,
        )

        app = FastAPI()
        with pytest.raises(StartupError, match="has no row in severity_policies"):
            async with lifespan(app):
                pass


@pytest.mark.parametrize(
    "extra_key",
    [pytest.param(False, id="value-drift"), pytest.param(True, id="extra-key")],
)
async def test_policy_mismatch_refuses_startup(migrated_db: str, extra_key: bool) -> None:
    """Check the loader's unknown-key error and the fingerprint's value comparison."""
    stored_policy = {kind.value: severity.value for kind, severity in SEVERITY_POLICY.items()}
    if extra_key:
        stored_policy["phantom_type"] = FindingSeverity.MEDIUM.value
        expected_error: type[Exception] = PolicyVersionShapeError
        expected_detail = "has key 'phantom_type' that is not a valid FindingType"
    else:
        finding_type = FindingType.UNUSED_IMPORT
        live_severity = SEVERITY_POLICY[finding_type]
        stored_severity = (
            FindingSeverity.LOW if live_severity != FindingSeverity.LOW else FindingSeverity.INFO
        )
        stored_policy[finding_type.value] = stored_severity.value
        difference = {finding_type: (live_severity, stored_severity)}
        expected_error = StartupError
        expected_detail = f"only-in-live=[], only-in-db=[], value-mismatches={difference!r}"

    setup_engine = create_async_engine(migrated_db)
    try:
        async with setup_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO severity_policies (version, policy) "
                    "VALUES ('9.9.9', CAST(:policy AS jsonb))"
                ),
                {"policy": json.dumps(stored_policy)},
            )
    finally:
        await setup_engine.dispose()

    provider_factory = MagicMock(side_effect=AssertionError("policy mismatch reached the provider"))
    with patch("outrider.api.lifespan.ACTIVE_POLICY_VERSION", "9.9.9"):
        lifespan = build_lifespan(
            engine_factory=_build_engine_factory(migrated_db),
            provider_factory=provider_factory,
        )

        app = FastAPI()
        with pytest.raises(expected_error) as caught:
            async with lifespan(app):
                pass
    assert expected_detail in str(caught.value)


async def test_fingerprint_runs_before_provider_construction(
    migrated_db: str,
) -> None:
    """When the fingerprint check raises, downstream wiring (persister,
    provider) is NOT constructed. Confirms ordering: the check is Step 1b,
    BEFORE Step 4 (persister) and Step 5b (provider factory).

    Detection: a provider_factory that records its invocation. If the
    fingerprint check raises first, the factory is never called.
    """
    provider_factory_calls = MagicMock()
    provider_factory_calls.return_value = MagicMock(aclose=AsyncMock())

    with patch("outrider.api.lifespan.ACTIVE_POLICY_VERSION", "9.9.9"):
        lifespan = build_lifespan(
            engine_factory=_build_engine_factory(migrated_db),
            provider_factory=provider_factory_calls,
        )

        app = FastAPI()
        with pytest.raises(StartupError):
            async with lifespan(app):
                pass

    # Provider factory never invoked — fingerprint short-circuited the lifespan.
    provider_factory_calls.assert_not_called()
