"""AuditPersister.emit_finding — findings-content co-insert edge cases.

Pins the load-bearing branch logic of the findings-content writer
(`specs/2026-05-30-findings-content-writer.md`): the no-resurrection guard,
the analyze-time-immutable verify set (including the deliberate exclusion of
later-node columns), the installation-scope cross-check, and is_eval threading.

The happy FULL-mode path is proven in `test_audit_replay.py
::test_full_mode_through_production_persister` and end-to-end through the graph
in `test_e2e_smoke.py`; this file pins the conflict / purge / mismatch edges.

**Case A vs Case B.** `emit_finding(finding, *, is_eval)` mints a fresh
`FindingEvent.event_id` (uuid4) on every call, so two calls for the same
`ReviewFinding` always carry DIFFERENT event_ids. The audit INSERT therefore
never conflicts on `event_id` → the writer always takes the Case-B branch
(re-emit detected via a `finding_id` lookup), never Case A
(`audit_row_already_existed`). Case A + its audit-payload-equality check are a
defensive mirror of the `persist()` / `llm_call_content` path (where the event
is built by the caller and its event_id is stable) and are unreachable through
`emit_finding`'s public surface. The equivalent finding-side integrity
guarantee is the Case-B verify set, exercised below by
`test_reemit_with_drifted_stored_content_raises`.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from outrider.audit.events import compute_finding_content_hash
from outrider.audit.persister import (
    AuditPersisterFindingInstallationIdMismatchError,
    AuditPersisterIdempotencyConflict,
)
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity, FindingType
from outrider.schemas import ReviewDimension
from outrider.schemas.review_finding import ReviewFinding

if TYPE_CHECKING:
    from uuid import UUID

    from tests.integration.conftest import PersisterTestSetup

_FILE_PATH = "src/app/models.py"
_LINE_START = 10
_LINE_END = 20


def _make_finding(
    review_id: UUID,
    installation_id: int,
    *,
    title: str = "SQL injection in query builder",
) -> ReviewFinding:
    """Build an admitted `ReviewFinding` matching the seeded review/installation.

    `content_hash` is computed via the canonical recipe so the lifted
    `FindingEvent`'s `_verify_content_hash` validator passes.
    """
    return ReviewFinding(
        review_id=review_id,
        installation_id=installation_id,
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
        content_hash=compute_finding_content_hash(
            _FILE_PATH,
            line_start=_LINE_START,
            line_end=_LINE_END,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash=hashlib.sha256(b"proposal").hexdigest(),
    )


async def _count_findings(setup: PersisterTestSetup, finding_id: UUID) -> int:
    async with setup.engine.connect() as conn:
        result = await conn.execute(
            text("SELECT COUNT(*) FROM findings WHERE finding_id = :fid"),
            {"fid": finding_id},
        )
        return result.scalar_one()


async def _count_finding_events(setup: PersisterTestSetup, finding_id: UUID) -> int:
    async with setup.engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT COUNT(*) FROM audit_events WHERE event_type = 'finding' "
                "AND payload->>'finding_id' = :fid"
            ),
            {"fid": str(finding_id)},
        )
        return result.scalar_one()


# ---------------------------------------------------------------------------
# No-resurrection guard (Case B — the reachable shape).
# ---------------------------------------------------------------------------


async def test_reemit_after_findings_purge_does_not_resurrect(
    persister_setup: PersisterTestSetup,
) -> None:
    """emit_finding lands both rows; the retention sweep purges the findings
    content row (its append-only FindingEvent audit row survives). Re-emitting
    the SAME ReviewFinding (fresh event_id → Case B; a prior FindingEvent with
    this finding_id exists → re-emit) must NOT re-insert the purged content.

    This is the load-bearing regression: without the finding_id-keyed re-emit
    detection, the fresh-event_id re-emit would take the first-emit branch and
    resurrect content the retention sweep deliberately removed.
    """
    setup = persister_setup
    finding = _make_finding(setup.review_id, setup.installation_id)

    await setup.persister.emit_finding(finding, is_eval=False)
    assert await _count_findings(setup, finding.finding_id) == 1

    # Simulate the retention sweep purging the content row.
    async with setup.engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM findings WHERE finding_id = :fid"),
            {"fid": finding.finding_id},
        )
    assert await _count_findings(setup, finding.finding_id) == 0

    # Re-emit the same finding (fresh event_id). No-resurrection guard fires.
    await setup.persister.emit_finding(finding, is_eval=False)

    # Content row STILL absent; the prior FindingEvent audit row survives and a
    # second FindingEvent (fresh event_id) was appended by the re-emit.
    assert await _count_findings(setup, finding.finding_id) == 0
    assert await _count_finding_events(setup, finding.finding_id) == 2


# ---------------------------------------------------------------------------
# Idempotent re-emit: one findings row per finding_id, N audit events.
# ---------------------------------------------------------------------------


async def test_reemit_writes_one_findings_row_two_audit_events(
    persister_setup: PersisterTestSetup,
) -> None:
    """Re-emitting the same ReviewFinding (content present, not purged) writes
    exactly ONE findings row (keyed on finding_id PK) while appending a second
    FindingEvent audit row (fresh event_id). Pins the supersession of the old
    content-hash-dedup wording: uniqueness is finding_id-keyed, and one finding
    can have multiple append-only audit rows."""
    setup = persister_setup
    finding = _make_finding(setup.review_id, setup.installation_id)

    await setup.persister.emit_finding(finding, is_eval=False)
    await setup.persister.emit_finding(finding, is_eval=False)

    assert await _count_findings(setup, finding.finding_id) == 1
    assert await _count_finding_events(setup, finding.finding_id) == 2


# ---------------------------------------------------------------------------
# Verify set: catches real content drift; ignores later-node columns.
# ---------------------------------------------------------------------------


async def test_reemit_with_drifted_stored_content_raises(
    persister_setup: PersisterTestSetup,
) -> None:
    """If the stored findings row disagrees with the re-emitted finding on an
    analyze-time-immutable column (here `title`), the Case-B verify set raises
    AuditPersisterIdempotencyConflict. Proves the verify set is not vacuous —
    `content_hash` alone (file/line/type) would NOT catch a title drift."""
    setup = persister_setup
    finding = _make_finding(setup.review_id, setup.installation_id, title="Original title")
    await setup.persister.emit_finding(finding, is_eval=False)

    # Simulate a corrupted / buggy-writer drift on a content column.
    async with setup.engine.begin() as conn:
        await conn.execute(
            text("UPDATE findings SET title = :t WHERE finding_id = :fid"),
            {"t": "DRIFTED title", "fid": finding.finding_id},
        )

    with pytest.raises(AuditPersisterIdempotencyConflict):
        await setup.persister.emit_finding(finding, is_eval=False)


async def test_reemit_does_not_false_raise_on_later_node_columns(
    persister_setup: PersisterTestSetup,
) -> None:
    """`publish_destination` + the override quartet are written by LATER nodes
    (publish / HITL), so the analyze-time re-emit carries them NULL. A re-emit
    must NOT false-raise against a row a later node has since populated — the
    verify set deliberately excludes those columns.

    Regression for the converse of the drift test: include too many columns and
    legitimate post-analyze writes trip a spurious conflict."""
    setup = persister_setup
    finding = _make_finding(setup.review_id, setup.installation_id)
    await setup.persister.emit_finding(finding, is_eval=False)

    # Simulate the publish node populating publish_destination + the HITL node
    # populating the override quartet AFTER analyze wrote the row.
    async with setup.engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE findings SET publish_destination = :pd, "
                "original_severity = :os, override_reason = :orr "
                "WHERE finding_id = :fid"
            ),
            {
                "pd": "inline_comment",
                "os": "high",
                "orr": "reviewer downgraded severity",
                "fid": finding.finding_id,
            },
        )

    # Re-emit the analyze-time finding (those columns NULL). No false-raise.
    await setup.persister.emit_finding(finding, is_eval=False)
    assert await _count_findings(setup, finding.finding_id) == 1


# ---------------------------------------------------------------------------
# Installation-scope cross-check.
# ---------------------------------------------------------------------------


async def test_installation_id_mismatch_fails_loud(
    persister_setup: PersisterTestSetup,
) -> None:
    """A ReviewFinding whose installation_id disagrees with the reviews row's
    installation_id is refused with the typed
    AuditPersisterFindingInstallationIdMismatchError — the reviews row is the
    FK-scope source of truth; the persister must not write a content row under
    a fabricated installation scope."""
    setup = persister_setup
    finding = _make_finding(setup.review_id, setup.installation_id + 9999)

    with pytest.raises(AuditPersisterFindingInstallationIdMismatchError):
        await setup.persister.emit_finding(finding, is_eval=False)

    # Neither row landed (the guard fires before any INSERT).
    assert await _count_findings(setup, finding.finding_id) == 0
    assert await _count_finding_events(setup, finding.finding_id) == 0


# ---------------------------------------------------------------------------
# is_eval threading to both rows.
# ---------------------------------------------------------------------------


async def test_is_eval_threads_to_both_rows(
    persister_setup: PersisterTestSetup,
) -> None:
    """The is_eval kwarg threads to BOTH the FindingEvent audit row AND the
    findings content row. Eval isolation depends on every row a review touches
    carrying the same flag (docs/testing.md)."""
    setup = persister_setup
    finding = _make_finding(setup.review_id, setup.installation_id)

    await setup.persister.emit_finding(finding, is_eval=True)

    async with setup.engine.connect() as conn:
        findings_is_eval = (
            await conn.execute(
                text("SELECT is_eval FROM findings WHERE finding_id = :fid"),
                {"fid": finding.finding_id},
            )
        ).scalar_one()
        audit_is_eval = (
            await conn.execute(
                text(
                    "SELECT is_eval FROM audit_events WHERE event_type = 'finding' "
                    "AND payload->>'finding_id' = :fid"
                ),
                {"fid": str(finding.finding_id)},
            )
        ).scalar_one()

    assert findings_is_eval is True
    assert audit_is_eval is True
