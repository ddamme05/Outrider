"""Lifespan severity-policy fingerprint check — §0c of analyze-foundation spec.

Pins the three drift windows §0c closes:

  (a) `ACTIVE_POLICY_VERSION` points at a version with no DB row.
  (b) DB row exists but its content differs from `dict(SEVERITY_POLICY)`.
  (c) Happy path: `ACTIVE_POLICY_VERSION='1.0.0'` matches the genesis-seeded
      DB row AND the live SEVERITY_POLICY mapping. Lifespan starts cleanly.

The test injects a real `migrated_db` engine via `build_lifespan(
engine_factory=...)`. The fingerprint check runs against the real DB row
seeded by genesis migration; happy path passes by construction, drift
cases are exercised by either patching `ACTIVE_POLICY_VERSION` (allowed —
the conftest autouse guard protects only `SEVERITY_POLICY`) or
inserting an extra row with diverging content.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from outrider.api.lifespan import StartupError, build_lifespan
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


async def test_happy_path_matches_genesis_seed(
    migrated_db: str,
    make_stub_llm_provider: type,
    in_memory_checkpointer_factory: object,
) -> None:
    """Lifespan starts cleanly when ACTIVE_POLICY_VERSION='1.0.0' matches
    the genesis-seeded DB row and the live SEVERITY_POLICY mapping.
    """
    stub_provider = make_stub_llm_provider()
    lifespan = build_lifespan(
        engine_factory=_build_engine_factory(migrated_db),
        provider_factory=lambda _persister, _model_config: stub_provider,
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
            provider_factory=lambda _persister, _model_config: stub_provider,
        )

        app = FastAPI()
        with pytest.raises(StartupError, match="has no row in severity_policies"):
            async with lifespan(app):
                pass


_VALUE_DRIFT_POLICY_JSONB = """{
    "sql_injection": "low",
    "auth_bypass": "low",
    "hardcoded_secret": "low",
    "xss": "low",
    "path_traversal": "low",
    "missing_input_validation": "low",
    "n_plus_one_query": "low",
    "blocking_call_in_async": "low",
    "missing_error_handling": "low",
    "missing_test": "low",
    "unused_import": "low",
    "deprecated_api": "low"
}"""

# Extra-key drift: the JSONB is invalid against the policy loader's
# completeness/shape check (extra key `phantom_type` isn't a FindingType).
# `load_policy_for_version` raises `PolicyVersionShapeError` before the
# mismatch comparison runs, and the fingerprint wrapper re-raises that as
# the underlying error. This is still a fingerprint-check failure path
# (different mode); the test pins the "wrong shape" arm distinctly from
# the value-drift arm so a future refactor that collapses the two error
# paths fails this assertion. Per §0c DevEx LOW-1.
_EXTRA_KEY_POLICY_JSONB = """{
    "sql_injection": "critical",
    "auth_bypass": "critical",
    "hardcoded_secret": "high",
    "xss": "high",
    "path_traversal": "high",
    "missing_input_validation": "medium",
    "n_plus_one_query": "medium",
    "blocking_call_in_async": "medium",
    "missing_error_handling": "low",
    "missing_test": "low",
    "unused_import": "info",
    "deprecated_api": "info",
    "phantom_type": "medium"
}"""


@pytest.mark.parametrize(
    ("policy_jsonb", "expected_error_class"),
    [
        # Case (c) per §0c: values differ (live mapping is stale).
        (_VALUE_DRIFT_POLICY_JSONB, StartupError),
        # Case (a) per §0c: keys differ (DB has an extra type not in live).
        # The loader's shape check raises `PolicyVersionShapeError` BEFORE
        # the fingerprint mismatch comparison runs (the loader can't
        # round-trip an unknown FindingType key into the typed enum world).
        # Post-PR review fold: pin the precise type rather than the broad
        # `Exception` — an `Exception` assertion would silently pass on any
        # unrelated DB/connectivity failure and mask the regression class
        # this test exists to catch.
        (_EXTRA_KEY_POLICY_JSONB, PolicyVersionShapeError),
    ],
)
async def test_policy_mismatch_raises_startup_error(
    migrated_db: str,
    make_stub_llm_provider: type,
    policy_jsonb: str,
    expected_error_class: type[Exception],
) -> None:
    """Two drift shapes both raise on lifespan startup:
    value-divergence and extra-key. Setup: seed an extra row at version
    '9.9.9' carrying the drift shape. Patch `ACTIVE_POLICY_VERSION` to
    point at it. Lifespan startup must refuse on both arms.
    """
    setup_engine = create_async_engine(migrated_db)
    try:
        async with setup_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO severity_policies (version, policy) "
                    "VALUES ('9.9.9', CAST(:policy AS jsonb))"
                ),
                {"policy": policy_jsonb},
            )
    finally:
        await setup_engine.dispose()

    stub_provider = make_stub_llm_provider()
    with patch("outrider.api.lifespan.ACTIVE_POLICY_VERSION", "9.9.9"):
        lifespan = build_lifespan(
            engine_factory=_build_engine_factory(migrated_db),
            provider_factory=lambda _persister, _model_config: stub_provider,
        )

        app = FastAPI()
        with pytest.raises(expected_error_class):
            async with lifespan(app):
                pass


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
