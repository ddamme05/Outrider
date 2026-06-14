"""AuditPersister.emit_finding — serve-path replay idempotency (FUP-176).

Pins the persister-level guarantee the cache `serve` flip depends on: when a
cache-served finding is RE-EMITTED under the same deterministic served
`finding_id` (`compute_served_finding_id(review_id, content_hash)` — proposal_hash
EXCLUDED per FUP-177 edge 1 / DECISIONS.md#025), the persister must be idempotent
on a benign refresh and loud on real content drift.

Two cases, mirroring the cache row's refresh-in-place behavior:

- **Case A — proposal_hash drift is benign.** A cache row refreshed in place
  between the original serve and a checkpoint re-execution carries the SAME
  `content_hash` (and therefore the SAME served `finding_id`) but a NEW
  `proposal_hash` (the LLM free-text moved). Re-emitting must write ONE findings
  row and raise nothing: `proposal_hash` is in neither the served-id recipe NOR
  the Case-B verify set (`_finding_verify_values`).
- **Case B — real content drift is loud.** Same served `finding_id`/`content_hash`
  but a changed analyze-time-immutable column (here `title`) raises
  `AuditPersisterIdempotencyConflict`, naming the drifted field.

Why this is distinct from `test_findings_content_writer.py`: that file pins the
GENERIC re-emit idempotency (same object twice → one row; drifted STORED row →
raise) for analyze's own retries. This file pins the SERVE-SPECIFIC shape — a
re-minted served `finding_id` plus a drifted `proposal_hash` across a cache
refresh — which the serve flip introduces and the generic tests never exercise.
The re-execution itself is simulated at the persister (the resume driver
checkpoints AFTER analyze, so analyze never re-runs through the graph — see the
FUP-176 re-scope in specs/2026-06-14-serve-flip-hardening.md).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from outrider.audit.events import compute_finding_content_hash
from outrider.audit.persister import AuditPersisterIdempotencyConflict
from outrider.policy.canonical import compute_served_finding_id
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity, FindingType
from outrider.schemas import ReviewDimension
from outrider.schemas.review_finding import ReviewFinding

if TYPE_CHECKING:
    from uuid import UUID

    from tests.integration.conftest import PersisterTestSetup

_FILE_PATH = "src/app/cached.py"
_LINE_START = 4
_LINE_END = 6
_CONTENT_HASH = compute_finding_content_hash(
    _FILE_PATH,
    line_start=_LINE_START,
    line_end=_LINE_END,
    finding_type=FindingType.SQL_INJECTION,
)


def _make_served_finding(
    review_id: UUID,
    installation_id: int,
    *,
    proposal_hash: str,
    title: str = "SQL injection in query builder",
) -> ReviewFinding:
    """Build a served `ReviewFinding` exactly as `_serve_cache_hit` re-mints it:
    `finding_id = compute_served_finding_id(review_id, content_hash)`, so two
    findings sharing `content_hash` share the served `finding_id` regardless of
    their `proposal_hash`/`title`."""
    return ReviewFinding(
        review_id=review_id,
        installation_id=installation_id,
        finding_id=compute_served_finding_id(review_id=review_id, content_hash=_CONTENT_HASH),
        policy_version="1.0.0",
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        evidence_tier=EvidenceTier.JUDGED,
        file_path=_FILE_PATH,
        line_start=_LINE_START,
        line_end=_LINE_END,
        title=title,
        description="User input flows into a raw SQL string.",
        evidence="cursor.execute(f'SELECT * FROM t WHERE id={user_id}')",
        content_hash=_CONTENT_HASH,
        proposal_hash=proposal_hash,
    )


async def _count_findings(setup: PersisterTestSetup, finding_id: UUID) -> int:
    async with setup.engine.connect() as conn:
        result = await conn.execute(
            text("SELECT COUNT(*) FROM findings WHERE finding_id = :fid"),
            {"fid": finding_id},
        )
        return result.scalar_one()


async def test_reemit_with_drifted_proposal_hash_is_idempotent(
    persister_setup: PersisterTestSetup,
) -> None:
    """Case A: a served finding re-emitted under the SAME served `finding_id` but
    a DRIFTED `proposal_hash` (cache refresh-in-place) writes ONE findings row and
    raises nothing — `proposal_hash` is excluded from the served-id recipe and the
    Case-B verify set, so the refresh is benign."""
    setup = persister_setup
    first = _make_served_finding(
        setup.review_id, setup.installation_id, proposal_hash=hashlib.sha256(b"v1").hexdigest()
    )
    refreshed = _make_served_finding(
        setup.review_id, setup.installation_id, proposal_hash=hashlib.sha256(b"v2").hexdigest()
    )
    # Same stable served finding_id, different proposal_hash.
    assert first.finding_id == refreshed.finding_id
    assert first.proposal_hash != refreshed.proposal_hash

    await setup.persister.emit_finding(first, is_eval=False)
    await setup.persister.emit_finding(refreshed, is_eval=False)  # must NOT raise

    assert await _count_findings(setup, first.finding_id) == 1


async def test_reemit_with_drifted_content_raises(
    persister_setup: PersisterTestSetup,
) -> None:
    """Case B: same served `finding_id`/`content_hash` but a changed
    analyze-time-immutable column (`title`) raises `AuditPersisterIdempotencyConflict`
    naming the drifted field — real content drift under a stable served id is loud,
    never a silent overwrite."""
    setup = persister_setup
    original = _make_served_finding(
        setup.review_id,
        setup.installation_id,
        proposal_hash=hashlib.sha256(b"v1").hexdigest(),
        title="Original title",
    )
    drifted = _make_served_finding(
        setup.review_id,
        setup.installation_id,
        proposal_hash=hashlib.sha256(b"v1").hexdigest(),
        title="Drifted title",
    )
    assert original.finding_id == drifted.finding_id  # content_hash unchanged → same served id

    await setup.persister.emit_finding(original, is_eval=False)

    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await setup.persister.emit_finding(drifted, is_eval=False)

    assert "title" in exc_info.value.mismatched_fields
    # Exactly one findings row survived — the conflict aborted the second write.
    assert await _count_findings(setup, original.finding_id) == 1
