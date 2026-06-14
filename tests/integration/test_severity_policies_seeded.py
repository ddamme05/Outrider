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
    """v1.0.0 exists with the canonical mapping from spec §7.4.

    The invariant under test is "policy versions are immutable and
    replayable" (severity-policy-versioned-for-replay), NOT "only one
    version exists forever." A future migration MAY insert v1.0.1 (or a
    Git-SHA version) when the policy changes — that's the supported way
    to update the policy. So the assertion is about v1.0.0's content,
    not about it being the only row in the table.

    What this test catches: in-place edits to v1.0.0 (the violation
    pattern named in the invariant). If a developer accidentally lands
    an UPDATE statement against version='1.0.0' in a future migration,
    the policy check below will fail.

    What this test does NOT catch (and shouldn't): the addition of new
    versions. Asserting versions == ["1.0.0"] would block legitimate
    forward migrations.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            versions_result = await conn.execute(
                text("SELECT version FROM severity_policies ORDER BY version")
            )
            versions = [row[0] for row in versions_result]
            assert "1.0.0" in versions, (
                f"v1.0.0 must always exist (genesis seed); found versions: {versions}"
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


# v1.1.0 extends v1.0.0 with three OBSERVED-tier security types
# (DECISIONS.md#048). The lifespan fingerprint binds the live SEVERITY_POLICY
# to this row at ACTIVE_POLICY_VERSION=1.1.0.
EXPECTED_V1_1_POLICY = {
    **EXPECTED_V1_POLICY,
    "command_injection": "critical",
    "unsafe_deserialization": "high",
    "tls_verify_disabled": "high",
}


async def test_v1_1_0_seeded_with_active_mapping(migrated_db: str) -> None:
    """v1.1.0 exists with the 15-entry active mapping (DECISIONS.md#048).

    Additive over v1.0.0: the three new keys are appended; the original
    twelve are unchanged. Mirrors the v1.0.0 content check above so an
    accidental in-place edit (or a divergence from the live mapping that
    would fail the lifespan fingerprint) is caught before CI.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            policy_result = await conn.execute(
                text("SELECT policy FROM severity_policies WHERE version = '1.1.0'")
            )
            policy = policy_result.scalar_one()
            assert policy == EXPECTED_V1_1_POLICY, (
                "v1.1.0 policy diverges from DECISIONS.md#048 / the live "
                "SEVERITY_POLICY. A change ships as a new version row, never an "
                "UPDATE to v1.1.0 (severity-policy-versioned-for-replay)."
            )
    finally:
        await engine.dispose()
