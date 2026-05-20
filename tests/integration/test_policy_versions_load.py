"""load_policy_for_version reads the seeded v1.0.0 row into typed enums.

Backs ``severity-policy-versioned-for-replay``. The genesis migration
seeded ``severity_policies`` with version "1.0.0" carrying the canonical
spec §7.4 mapping; the loader must round-trip that JSONB into the typed
``dict[FindingType, FindingSeverity]`` shape that the application code
(and replay) consumes.

Three cases:

  1. Happy path: load v1.0.0 → returns the canonical mapping with the
     right enum types on both keys and values.
  2. Unknown version → ``UnknownPolicyVersionError``.
  3. Malformed JSONB (we INSERT a row with an unknown FindingType key)
     → ``PolicyVersionShapeError``. Verifies the type-system gate at
     the load boundary; a future migration that drifts from the
     FindingType enum fails loud at replay rather than silently
     poisoning classification.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from outrider.policy import (
    FindingSeverity,
    FindingType,
    PolicyVersionShapeError,
    UnknownPolicyVersionError,
    load_policy_for_version,
)

EXPECTED_V1_POLICY: dict[FindingType, FindingSeverity] = {
    FindingType.SQL_INJECTION: FindingSeverity.CRITICAL,
    FindingType.AUTH_BYPASS: FindingSeverity.CRITICAL,
    FindingType.HARDCODED_SECRET: FindingSeverity.HIGH,
    FindingType.XSS: FindingSeverity.HIGH,
    FindingType.PATH_TRAVERSAL: FindingSeverity.HIGH,
    FindingType.MISSING_INPUT_VALIDATION: FindingSeverity.MEDIUM,
    FindingType.N_PLUS_ONE_QUERY: FindingSeverity.MEDIUM,
    FindingType.BLOCKING_CALL_IN_ASYNC: FindingSeverity.MEDIUM,
    FindingType.MISSING_ERROR_HANDLING: FindingSeverity.LOW,
    FindingType.MISSING_TEST: FindingSeverity.LOW,
    FindingType.UNUSED_IMPORT: FindingSeverity.INFO,
    FindingType.DEPRECATED_API: FindingSeverity.INFO,
}


async def test_load_v1_0_0_returns_canonical_mapping(migrated_db: str) -> None:
    """v1.0.0 (seeded by genesis migration) loads as the canonical typed mapping."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            policy = await load_policy_for_version("1.0.0", conn)

        assert policy == EXPECTED_V1_POLICY
        # Sanity: every key is FindingType, every value is FindingSeverity.
        # The dict equality above already implies type identity, but make
        # the type-system gate explicit so a regression that returns a
        # raw str-key dict fails on this assertion specifically.
        for key, value in policy.items():
            assert isinstance(key, FindingType), (
                f"Key {key!r} is not FindingType (type={type(key).__name__})"
            )
            assert isinstance(value, FindingSeverity), (
                f"Value {value!r} is not FindingSeverity (type={type(value).__name__})"
            )
    finally:
        await engine.dispose()


async def test_load_unknown_version_raises(migrated_db: str) -> None:
    """A version not in the table raises UnknownPolicyVersionError."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            with pytest.raises(UnknownPolicyVersionError, match="9.9.9"):
                await load_policy_for_version("9.9.9", conn)
    finally:
        await engine.dispose()


async def test_load_malformed_policy_raises_shape_error(migrated_db: str) -> None:
    """JSONB with a key that isn't a valid FindingType raises PolicyVersionShapeError.

    Inserts a fake "9.9.0" version with a key that doesn't
    match any FindingType. The loader's type-system gate catches it.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO severity_policies (version, policy) "
                    "VALUES ('9.9.0', "
                    '\'{"not_a_real_finding_type": "medium"}\'::jsonb)'
                )
            )

        async with engine.connect() as conn:
            with pytest.raises(PolicyVersionShapeError, match="not a valid FindingType"):
                await load_policy_for_version("9.9.0", conn)
    finally:
        await engine.dispose()


async def test_load_malformed_severity_value_raises_shape_error(
    migrated_db: str,
) -> None:
    """JSONB with a value that isn't a valid FindingSeverity raises PolicyVersionShapeError."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO severity_policies (version, policy) "
                    "VALUES ('9.9.1', "
                    '\'{"sql_injection": "catastrophic"}\'::jsonb)'
                )
            )

        async with engine.connect() as conn:
            with pytest.raises(PolicyVersionShapeError, match="not a valid FindingSeverity"):
                await load_policy_for_version("9.9.1", conn)
    finally:
        await engine.dispose()


async def test_load_incomplete_policy_raises_shape_error(migrated_db: str) -> None:
    """A policy with valid entries but missing FindingTypes raises PolicyVersionShapeError.

    Inserts a policy that has 11 of 12 FindingType keys (every value is
    valid; just one type is missing). The loader must refuse to return a
    partial policy — silently returning it would let classification fall
    through to lookup_severity's MEDIUM fallback for the missing type at
    runtime, breaking severity-set-by-policy.

    Closes Codex high-confidence audit finding on commit 9efb008.
    """
    # Build an 11-of-12 policy: drop UNUSED_IMPORT.
    canonical = dict(EXPECTED_V1_POLICY)
    del canonical[FindingType.UNUSED_IMPORT]
    incomplete_json = (
        "{" + ",".join(f'"{k.value}": "{v.value}"' for k, v in canonical.items()) + "}"
    )

    engine = create_async_engine(migrated_db)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO severity_policies (version, policy) "
                    "VALUES (:version, CAST(:policy AS jsonb))"
                ),
                {"version": "9.9.2", "policy": incomplete_json},
            )

        async with engine.connect() as conn:
            with pytest.raises(PolicyVersionShapeError, match="missing entries for"):
                await load_policy_for_version("9.9.2", conn)
    finally:
        await engine.dispose()
