# Policy versioning per docs/trust-boundaries.md §2 (severity-policy-versioned-for-replay)
"""Replay-time loader for historical SEVERITY_POLICY versions.

`severity-policy-versioned-for-replay` requires that historical reviews
replay under the policy version they were classified under, not the
current policy. Every policy version is stored in the `severity_policies`
table keyed by version string; this module exposes the loader.

The loader returns a typed ``dict[FindingType, FindingSeverity]`` rather
than a raw JSON dict — the type system enforces the policy shape at the
load boundary, so a row with a key/value that doesn't translate to a
FindingType/FindingSeverity fails loud rather than silently corrupting
classification. Completeness (every current FindingType present) is
enforced for the ACTIVE version only; historical versions are exempt so
replay survives later enum growth (``severity-policy-versioned-for-replay``).
See ``load_policy_for_version`` for the active-vs-historical rationale.
"""

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from outrider.policy.severity import ACTIVE_POLICY_VERSION, FindingSeverity, FindingType


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

    # Completeness check — ACTIVE version ONLY. The active policy must assign a
    # severity to every CURRENT FindingType, or live classification silently falls
    # through to lookup_severity's MEDIUM fallback (breaking severity-set-by-policy).
    #
    # Historical (non-active) versions are deliberately EXEMPT: a historical row is
    # correct as of its seeding date, and a finding classified under it can only
    # reference FindingTypes that existed then. If the enum GROWS later (a new type
    # added in a future release + seeded as a new active version), the current enum
    # would no longer match an older row's key set — and re-checking historical rows
    # against the grown enum would raise on EVERY prior version, breaking
    # `severity-policy-versioned-for-replay` (the replay path at
    # `audit/replay.py::_policy_snapshots` and the dashboard policy browser both load
    # historical versions). The active-version check still catches the real bug it
    # was added for: a FindingType added to the enum without seeding a complete new
    # active policy. (A historical key that is no longer a valid FindingType is a
    # separate concern, still caught per-key above.)
    if version == ACTIVE_POLICY_VERSION:
        missing_types = set(FindingType) - set(typed_policy.keys())
        if missing_types:
            missing_values = sorted(t.value for t in missing_types)
            raise PolicyVersionShapeError(
                f"Active policy version {version!r} is missing entries for: "
                f"{missing_values}. Every current FindingType must have a severity "
                "assignment in the ACTIVE policy, or classification silently falls "
                "through to the MEDIUM fallback for the missing types, breaking "
                "severity-set-by-policy. This usually means a FindingType was added "
                "to the enum without seeding a complete new policy version — add the "
                "missing entries via a new severity_policies version migration "
                "(never edit an existing version row; severity-policy-versioned-for-"
                "replay forbids in-place changes)."
            )

    return typed_policy
