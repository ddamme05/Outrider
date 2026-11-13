"""Return the stored severity policy for GET /api/policy/{version}.

load_policy_for_version reads that version from the database, never substitutes
the active in-code mapping. This preserves historical replay
(severity-policy-versioned-for-replay). Severity remains policy-assigned
(severity-set-by-policy); dimensions use the shared mapping.

See specs/2026-06-01-policy-version-endpoint.md, FUP-132 and DECISIONS.md#021.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.versions import (
    PolicyVersionShapeError,
    UnknownPolicyVersionError,
    load_policy_for_version,
)


class PolicyEntry(BaseModel):
    """One `FindingType` → severity row for a policy version. `dimension` always
    resolves (lockstep, #021); `severity` is the STORED value for the version.
    """

    model_config = ConfigDict(extra="forbid")

    finding_type: str
    dimension: str
    severity: str


class PolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    entries: list[PolicyEntry]


router = APIRouter(
    prefix="/api/policy",
    tags=["dashboard"],
    dependencies=[Depends(require_admin_api_key)],
)


@router.get("/{version}", response_model=PolicyResponse)
async def get_policy(request: Request, version: str) -> PolicyResponse:
    """The versioned `FindingType` → severity table for `version`.

    Reads the STORED policy via `load_policy_for_version` (replay-safe — not the
    active in-code policy). 404 when the version row is absent
    (`UnknownPolicyVersionError`); structured 500 when the stored row is corrupt —
    e.g. an undecodable finding_type key (`PolicyVersionShapeError`), loud rather
    than a partial table. Entries sorted by `finding_type` for a stable render.
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        conn = await session.connection()
        try:
            policy = await load_policy_for_version(version, conn)
        except UnknownPolicyVersionError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="policy version not found"
            ) from exc
        except PolicyVersionShapeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "policy_version_shape", "note": str(exc)},
            ) from exc

    entries = sorted(
        (
            PolicyEntry(
                finding_type=finding_type.value,
                dimension=lookup_dimension(finding_type).value,
                severity=severity.value,
            )
            for finding_type, severity in policy.items()
        ),
        key=lambda e: e.finding_type,
    )
    return PolicyResponse(version=version, entries=entries)
