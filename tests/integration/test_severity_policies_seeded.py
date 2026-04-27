"""severity_policies has v1.0.0 with the canonical SEVERITY_POLICY mapping.

Backs ``severity-set-by-policy``: baseline severity comes from this
mapping keyed by finding type, never from model output. The v1.0.0 row
is seeded by the genesis migration (the seed is non-negotiable: without
it every findings insert FK-fails on a fresh DB), and the policy JSON
must be the spec §7.4 mapping verbatim.

Closes the loop on the audit fix that changed the seed from `{}` to
the canonical mapping. If a future migration accidentally edits v1.0.0
in place (a `severity-policy-versioned-for-replay` violation), this
test catches it before the schema reaches CI.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

EXPECTED_V1_POLICY = {
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
}


async def test_v1_0_0_seeded_with_canonical_mapping(migrated_db: str) -> None:
    """severity_policies has exactly v1.0.0, with the canonical 12-key mapping."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            versions_result = await conn.execute(
                text("SELECT version FROM severity_policies ORDER BY version")
            )
            versions = [row[0] for row in versions_result]
            assert versions == ["1.0.0"], (
                f"Expected only v1.0.0 seeded by genesis; found: {versions}"
            )

            policy_result = await conn.execute(
                text("SELECT policy FROM severity_policies WHERE version = '1.0.0'")
            )
            policy = policy_result.scalar_one()
            assert policy == EXPECTED_V1_POLICY, (
                "v1.0.0 policy diverges from spec §7.4. If this is intentional, "
                "the change should ship as a new migration that inserts v1.0.1, "
                "not as an UPDATE to v1.0.0 (severity-policy-versioned-for-replay)."
            )
    finally:
        await engine.dispose()
