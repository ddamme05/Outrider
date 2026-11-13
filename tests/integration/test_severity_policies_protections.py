"""Check that stored policy versions require strict bare semantic versions.

Reject leading zeros and release/build suffixes; accept a new valid version.
The append-only trigger is tested separately.
See specs/2026-05-19-analyze-foundation.md §0c."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

_VALID_POLICY_JSONB = """{
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
    "deprecated_api": "info"
}"""


async def test_check_constraint_exists(migrated_db: str) -> None:
    """The CHECK constraint exists with the documented name post-migration.

    Scoped to the `severity_policies` table via `conrelid` so a same-named
    CHECK on a different table cannot make this assertion pass spuriously.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conname = 'ck_severity_policies_version_semver' "
                    "AND contype = 'c' "
                    "AND conrelid = 'severity_policies'::regclass"
                )
            )
            assert row.scalar_one() == "ck_severity_policies_version_semver"
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "malformed_version",
    [
        "v1.0.0",  # leading 'v' prefix
        "1.0",  # only two segments
        "1.0.0.0",  # four segments
        "1.0.0-rc1",  # pre-release suffix
        "",  # empty string
        "1.0.0 ",  # trailing whitespace
    ],
)
async def test_insert_with_malformed_version_raises_check(
    migrated_db: str,
    malformed_version: str,
) -> None:
    """INSERT with non-bare-semver version raises via the CHECK constraint."""
    engine = create_async_engine(migrated_db)
    try:
        with pytest.raises(
            IntegrityError,
            match="ck_severity_policies_version_semver",
        ):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO severity_policies (version, policy) "
                        "VALUES (:version, CAST(:policy AS jsonb))"
                    ),
                    {"version": malformed_version, "policy": _VALID_POLICY_JSONB},
                )
    finally:
        await engine.dispose()


async def test_insert_with_valid_new_semver_succeeds(migrated_db: str) -> None:
    """INSERT with a new bare-semver version succeeds (the canonical add-row path)."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO severity_policies (version, policy) "
                    "VALUES ('1.0.1', CAST(:policy AS jsonb))"
                ),
                {"policy": _VALID_POLICY_JSONB},
            )
            result = await conn.execute(
                text("SELECT version FROM severity_policies WHERE version = '1.0.1'")
            )
            assert result.scalar_one() == "1.0.1"
    finally:
        await engine.dispose()
