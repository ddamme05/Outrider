# Policy versioning per docs/trust-boundaries.md §2 (severity-policy-versioned-for-replay)
"""Replay-time loader for historical SEVERITY_POLICY versions.

`severity-policy-versioned-for-replay` requires that historical reviews
replay under the policy version they were classified under, not the
current policy. Every policy version is stored in the `severity_policies`
table keyed by version string; this module exposes the loader.

The loader returns a typed ``dict[FindingType, FindingSeverity]`` rather
than a raw JSON dict — the type system enforces the policy shape at the
load boundary, so a future migration that drifts from the FindingType
enum (extra keys; missing required types) fails loud at replay rather
than silently corrupting the classification chain.
"""

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from outrider.policy.severity import FindingSeverity, FindingType


class UnknownPolicyVersionError(LookupError):
    """Raised when a requested policy version doesn't exist in the DB.

    Subclasses LookupError so callers using the standard "key/version not
    found" exception family get the right semantics. Replay invokes this
    loader with a version recorded on a finding; if that version was never
    seeded (DB corruption, partial migration rollback), the failure is
    forensic — it must surface, not be papered over with a fallback.
    """


class PolicyVersionShapeError(ValueError):
    """Raised when the loaded JSONB doesn't match FindingType/FindingSeverity.

    The DB stores policies as JSONB strings; this loader translates back
    to the typed enum world. If the JSONB has a key that isn't a valid
    FindingType (e.g., a removed-since type), or a value that isn't a
    valid FindingSeverity, the loader raises rather than silently
    coercing — replay correctness depends on the type system being
    intact across versions.
    """


async def load_policy_for_version(
    version: str,
    conn: AsyncConnection,
) -> dict[FindingType, FindingSeverity]:
    """Load and decode the SEVERITY_POLICY mapping for the given version.

    Queries ``severity_policies WHERE version = :version`` and returns
    a typed ``dict[FindingType, FindingSeverity]``. Raises:

      - ``UnknownPolicyVersionError`` if the version row doesn't exist.
      - ``PolicyVersionShapeError`` if the JSONB has a key/value that
        doesn't translate to a FindingType / FindingSeverity.

    The caller manages the transaction; this function takes a
    connection and issues a single SELECT.
    """
    result = await conn.execute(
        text("SELECT policy FROM severity_policies WHERE version = :version"),
        {"version": version},
    )
    row = result.first()
    if row is None:
        raise UnknownPolicyVersionError(
            f"No policy found for version {version!r}. "
            "Versions are seeded by migrations; missing version means "
            "either a partial migration rollback or a finding that "
            "references a never-seeded version (DB corruption)."
        )

    raw_policy: Any = row[0]
    # psycopg returns JSONB as already-parsed Python; defensive parse if
    # it ever changes (e.g., a future driver that returns text).
    if isinstance(raw_policy, str):
        raw_policy = json.loads(raw_policy)

    if not isinstance(raw_policy, dict):
        raise PolicyVersionShapeError(
            f"Policy for version {version!r} is not a JSON object; got {type(raw_policy).__name__}"
        )

    typed_policy: dict[FindingType, FindingSeverity] = {}
    for raw_key, raw_value in raw_policy.items():
        try:
            finding_type = FindingType(raw_key)
        except ValueError as exc:
            raise PolicyVersionShapeError(
                f"Policy for version {version!r} has key {raw_key!r} "
                f"that is not a valid FindingType. Either the FindingType "
                f"enum is out of date relative to this policy version, "
                f"or the version was seeded with an invalid mapping."
            ) from exc
        try:
            severity = FindingSeverity(raw_value)
        except ValueError as exc:
            raise PolicyVersionShapeError(
                f"Policy for version {version!r} has value {raw_value!r} "
                f"for key {raw_key!r} that is not a valid FindingSeverity."
            ) from exc
        typed_policy[finding_type] = severity

    # Completeness check: every FindingType MUST have a severity assignment.
    # A missing entry would silently fall through to the MEDIUM fallback in
    # lookup_severity at classification time, breaking severity-set-by-policy
    # for that finding type. The boundary is "policy is the source of
    # severity"; an incomplete policy is not a source. Fail loud at load
    # time rather than allow a partial policy to poison classification.
    missing_types = set(FindingType) - set(typed_policy.keys())
    if missing_types:
        missing_values = sorted(t.value for t in missing_types)
        raise PolicyVersionShapeError(
            f"Policy for version {version!r} is missing entries for: "
            f"{missing_values}. Every FindingType must have a severity "
            "assignment for the policy to be admissible — otherwise "
            "classification silently falls through to the MEDIUM fallback "
            "for the missing types, breaking severity-set-by-policy. "
            "If the gap is intentional (e.g., a future policy version "
            "deprecates a FindingType), the right path is to deprecate "
            "the enum value first, not to ship a partial policy row."
        )

    return typed_policy
