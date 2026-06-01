"""Integration proof for `audit/replay.py` — reconstruct + assert from Postgres.

The first end-to-end proof of replay equivalence: seed real `audit_events`
+ content-table rows, reconstruct via `AuditReplayer`, and assert. The seed
is **replay-local and non-eval** — it reuses the `migrated_db` fixture and
raw-SQL inserts (omitting `is_eval` so the server default `false` applies),
deliberately NOT the eval factories (which force `is_eval=True` +
`installation_id=-1` and would smuggle eval semantics into a
production-semantic proof). The hitl_resume end-to-end scenario (FUP-107)
is the eventual proof once the graph driver lands (FUP-105/106/108).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from outrider.audit.events import (
    RESERVED_HISTORICAL_PROPOSAL_HASH,
    AuditEventBase,
    FindingEvent,
    LLMCallEvent,
    ReviewPhaseEvent,
    compute_finding_content_hash,
)
from outrider.audit.replay import (
    AuditReplayer,
    ReplayEquivalenceError,
    ReplayMode,
    ReplayReviewNotFoundError,
)
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import (
    SEVERITY_POLICY,
    FindingSeverity,
    FindingType,
)
from outrider.schemas import ReviewDimension
from outrider.schemas.review_finding import ReviewFinding

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from tests.integration.conftest import (
        LLMCallEventFactory,
        LLMRequestFactory,
        LLMResponseFactory,
        PersisterTestSetup,
    )

_INSTALLATION_ID = 12345

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def engine(migrated_db: str) -> AsyncGenerator[AsyncEngine]:
    # NullPool: no lingering idle connections to race the fresh_db teardown's
    # terminate-then-DROP (pg_terminate_backend only signals; DROP can still
    # see a pooled backend mid-shutdown).
    eng = create_async_engine(migrated_db, poolclass=NullPool)
    try:
        yield eng
    finally:
        await eng.dispose()


# ---------------------------------------------------------------------------
# Event builders (production-semantic; is_eval defaults False)
# ---------------------------------------------------------------------------


def _finding_event(
    review_id: UUID,
    *,
    evidence_tier: EvidenceTier = EvidenceTier.JUDGED,
    query_match_id: str | None = None,
    finding_type: FindingType = FindingType.SQL_INJECTION,
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    policy_version: str = "1.0.0",
    file_path: str = "src/app/models.py",
    line_start: int = 10,
    line_end: int = 20,
) -> FindingEvent:
    return FindingEvent(
        review_id=review_id,
        finding_id=uuid4(),
        finding_type=finding_type,
        severity=severity,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        dimension=ReviewDimension.SECURITY,
        finding_content_hash=compute_finding_content_hash(
            file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        evidence_tier=evidence_tier,
        query_match_id=query_match_id,
        trace_path=None,
        policy_version=policy_version,
        proposal_hash=hashlib.sha256(b"proposal").hexdigest(),
    )


def _llm_call_event(review_id: UUID) -> LLMCallEvent:
    return LLMCallEvent(
        review_id=review_id,
        model="claude-sonnet-4-5",
        node_id="analyze",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        cost_usd=0.01,
        pricing_version="v1",
        latency_ms=1200,
        prompt_hash=hashlib.sha256(b"prompt").hexdigest(),
        cache_hit=False,
        context_summary=(),
        prompt_template_version="analyze.v1",
        system_prompt_hash=hashlib.sha256(b"sys").hexdigest(),
        degraded_mode=False,
    )


def _phase_event(
    review_id: UUID,
    *,
    node_id: Literal["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"],
    marker: Literal["start", "end"],
) -> ReviewPhaseEvent:
    return ReviewPhaseEvent(
        review_id=review_id,
        phase_id=f"{node_id}:0",
        node_id=node_id,
        marker=marker,
        phase_key=None,
    )


def _phase_pair(
    review_id: UUID,
    node_id: Literal["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"],
    *work: AuditEventBase,
) -> list[AuditEventBase]:
    """Wrap work events in a `node_id` phase start/end pair.

    Phase markers survive retention (they are audit rows), so a faithful
    metadata-only stream still carries them — and `phase-events-bound-work`
    requires every work event to be phase-bounded.
    """
    return [
        _phase_event(review_id, node_id=node_id, marker="start"),
        *work,
        _phase_event(review_id, node_id=node_id, marker="end"),
    ]


# ---------------------------------------------------------------------------
# Seed helpers (raw SQL; FK-ordered; is_eval omitted ⇒ server default false)
# ---------------------------------------------------------------------------


async def _seed_installation(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                "account_login, account_type, permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": _INSTALLATION_ID},
        )


async def _seed_review(engine: AsyncEngine, review_id: UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, files_examined, files_traced_beyond_diff, llm_calls_made, "
                "total_input_tokens, total_output_tokens, total_cost_usd, wall_clock_seconds, "
                "retention_expires_at) VALUES (:id, :iid, 100, 1, 'sha1', 'completed', "
                "1, 0, 1, 100, 50, 0.01, 1.5, NOW() + INTERVAL '180 days')"
            ),
            {"id": review_id, "iid": _INSTALLATION_ID},
        )


async def _insert_event(engine: AsyncEngine, event: AuditEventBase) -> None:
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    phase_key = event.phase_key if isinstance(event, ReviewPhaseEvent) else None
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, payload) VALUES (:event_id, :review_id, :event_type, :phase_key, "
                ":timestamp, CAST(:payload AS jsonb))"
            ),
            {
                "event_id": event.event_id,
                "review_id": event.review_id,
                "event_type": event.event_type,
                "phase_key": phase_key,
                "timestamp": event.timestamp,
                "payload": json.dumps(payload),
            },
        )


async def _insert_event_dropping(
    engine: AsyncEngine, event: AuditEventBase, *, drop: set[str]
) -> None:
    """Insert an event row with `drop` fields removed from the stored payload —
    simulates a persisted historical row written before those fields were
    required (DECISIONS.md#032 / FUP-136). Raw insert because the model itself
    can't be constructed without the now-required field.
    """
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    for field in drop:
        payload.pop(field, None)
    phase_key = event.phase_key if isinstance(event, ReviewPhaseEvent) else None
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, payload) VALUES (:event_id, :review_id, :event_type, :phase_key, "
                ":timestamp, CAST(:payload AS jsonb))"
            ),
            {
                "event_id": event.event_id,
                "review_id": event.review_id,
                "event_type": event.event_type,
                "phase_key": phase_key,
                "timestamp": event.timestamp,
                "payload": json.dumps(payload),
            },
        )


async def _seed_finding_row(engine: AsyncEngine, event: FindingEvent) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO findings (finding_id, review_id, installation_id, policy_version, "
                "finding_type, dimension, severity, evidence_tier, file_path, line_start, "
                "line_end, title, description, evidence, content_hash, retention_expires_at) "
                "VALUES (:fid, :rid, :iid, :pv, :ft, :dim, :sev, :tier, :fp, :ls, :le, "
                "'t', 'd', 'e', :hash, NOW() + INTERVAL '180 days')"
            ),
            {
                "fid": event.finding_id,
                "rid": event.review_id,
                "iid": _INSTALLATION_ID,
                "pv": event.policy_version,
                "ft": event.finding_type.value,
                "dim": event.dimension.value,
                "sev": event.severity.value,
                "tier": event.evidence_tier.value,
                "fp": event.file_path,
                "ls": event.line_start,
                "le": event.line_end,
                "hash": event.finding_content_hash,
            },
        )


async def _seed_llm_content(engine: AsyncEngine, event: LLMCallEvent) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO llm_call_content (event_id, installation_id, prompt, completion, "
                "retention_expires_at) VALUES (:eid, :iid, 'the prompt', 'the completion', "
                "NOW() + INTERVAL '90 days')"
            ),
            {"eid": event.event_id, "iid": _INSTALLATION_ID},
        )


async def _seed_severity_policy(engine: AsyncEngine, version: str) -> None:
    mapping = {ft.value: sev.value for ft, sev in SEVERITY_POLICY.items()}
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO severity_policies (version, policy) VALUES (:v, CAST(:p AS jsonb))"),
            {"v": version, "p": json.dumps(mapping)},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_full_mode_reconstruct_and_assert(engine: AsyncEngine) -> None:
    review_id = uuid4()
    finding = _finding_event(review_id)
    llm_call = _llm_call_event(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id)
    for event in (
        _phase_event(review_id, node_id="intake", marker="start"),
        _phase_event(review_id, node_id="intake", marker="end"),
        _phase_event(review_id, node_id="analyze", marker="start"),
        llm_call,
        finding,
        _phase_event(review_id, node_id="analyze", marker="end"),
    ):
        await _insert_event(engine, event)
    await _seed_finding_row(engine, finding)
    await _seed_llm_content(engine, llm_call)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    review = await replayer.reconstruct(review_id)

    assert review.mode == ReplayMode.FULL
    assert review.review is not None
    assert review.review.status == "completed"
    assert len(review.findings) == 1
    assert review.findings[0].content is not None
    assert len(review.llm_exchanges) == 1
    assert review.llm_exchanges[0].prompt == "the prompt"
    assert review.llm_exchanges[0].completion == "the completion"
    assert [p.phase_id for p in review.phases] == ["intake:0", "analyze:0"]
    analyze_phase = review.phases[1]
    assert {type(e).__name__ for e in analyze_phase.events} == {"LLMCallEvent", "FindingEvent"}

    await replayer.assert_replay_equivalent(review_id)  # no raise


async def test_metadata_only_mode_reconstruct_and_assert(engine: AsyncEngine) -> None:
    # No review / findings / content rows — only the append-only audit stream.
    review_id = uuid4()
    finding = _finding_event(review_id)
    llm_call = _llm_call_event(review_id)
    for event in _phase_pair(review_id, "analyze", llm_call, finding):
        await _insert_event(engine, event)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    review = await replayer.reconstruct(review_id)

    assert review.mode == ReplayMode.METADATA_ONLY
    assert review.review is None
    assert review.findings[0].content is None  # a stub
    assert review.llm_exchanges[0].prompt is None

    await replayer.assert_replay_equivalent(review_id)  # no raise — no content-equality claim


async def test_historical_finding_missing_proposal_hash_reconstructs(engine: AsyncEngine) -> None:
    """Regression for FUP-136: a persisted finding row written before
    `proposal_hash` was required no longer 500s `reconstruct()`. The read-side
    normalizer defaults it to the reserved sentinel (DECISIONS.md#032); the
    whole verify path tolerates it.
    """
    review_id = uuid4()
    finding = _finding_event(review_id)
    await _insert_event(engine, _phase_event(review_id, node_id="analyze", marker="start"))
    await _insert_event_dropping(engine, finding, drop={"proposal_hash"})  # historical row
    await _insert_event(engine, _phase_event(review_id, node_id="analyze", marker="end"))

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    review = await replayer.reconstruct(review_id)  # must NOT raise

    assert len(review.findings) == 1
    rebuilt = next(
        e for phase in review.phases for e in phase.events if isinstance(e, FindingEvent)
    )
    assert rebuilt.proposal_hash == RESERVED_HISTORICAL_PROPOSAL_HASH
    await replayer.assert_replay_equivalent(review_id)  # verify path tolerates the historical row


async def test_mixed_mode_when_llm_content_purged(engine: AsyncEngine) -> None:
    # MIXED window: review + findings present, llm_call_content purged. Under the
    # retention ordering (llm_content <= findings <= review), the shorter-or-equal
    # LLM content can purge while finding content remains.
    review_id = uuid4()
    finding = _finding_event(review_id)
    llm_call = _llm_call_event(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id)
    for event in _phase_pair(review_id, "analyze", llm_call, finding):
        await _insert_event(engine, event)
    await _seed_finding_row(engine, finding)
    # deliberately NO _seed_llm_content — content row purged

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    review = await replayer.reconstruct(review_id)

    assert review.mode == ReplayMode.MIXED
    assert review.findings[0].content is not None  # finding content survives
    assert review.llm_exchanges[0].prompt is None  # llm content purged

    await replayer.assert_replay_equivalent(review_id)  # no raise — per-item labeled


async def test_unknown_review_raises_not_found(engine: AsyncEngine) -> None:
    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayReviewNotFoundError):
        await replayer.reconstruct(uuid4())


async def test_historical_policy_severity_reconstructs(engine: AsyncEngine) -> None:
    review_id = uuid4()
    await _seed_severity_policy(engine, "0.9.0")
    finding = _finding_event(
        review_id,
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.CRITICAL,  # matches 0.9.0's mapping
        policy_version="0.9.0",
    )
    for event in _phase_pair(review_id, "analyze", finding):
        await _insert_event(engine, event)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    await replayer.assert_replay_equivalent(review_id)  # loads 0.9.0, severity matches


async def test_historical_policy_severity_mismatch_raises(engine: AsyncEngine) -> None:
    review_id = uuid4()
    await _seed_severity_policy(engine, "0.9.0")
    # 0.9.0 maps sql_injection→CRITICAL; this row claims LOW. The schema
    # validator skips the live check for non-ACTIVE versions, so the row
    # constructs — replay is what catches the drift.
    finding = _finding_event(
        review_id,
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.LOW,
        policy_version="0.9.0",
    )
    for event in _phase_pair(review_id, "analyze", finding):
        await _insert_event(engine, event)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayEquivalenceError, match="does not match"):
        await replayer.assert_replay_equivalent(review_id)


async def test_orphan_stored_finding_raises(engine: AsyncEngine) -> None:
    # A findings-table row whose finding_id has no FindingEvent in the audit
    # stream is an append-only violation — replay must reject it, not ignore it.
    review_id = uuid4()
    finding = _finding_event(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id)
    for event in _phase_pair(review_id, "analyze", finding):
        await _insert_event(engine, event)
    await _seed_finding_row(engine, finding)
    # A second stored finding with NO corresponding FindingEvent (never inserted
    # into audit_events) — the orphan.
    orphan = _finding_event(review_id)
    await _seed_finding_row(engine, orphan)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayEquivalenceError, match="no FindingEvent in the audit stream"):
        await replayer.assert_replay_equivalent(review_id)


async def test_completed_review_unterminated_phase_raises(engine: AsyncEngine) -> None:
    # A completed review must close every phase (phase-events-bound-work:
    # missing phase-end on success is a violation). The analyze phase opens
    # with a finding but never ends.
    review_id = uuid4()
    finding = _finding_event(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id)  # status='completed'
    await _insert_event(engine, _phase_event(review_id, node_id="analyze", marker="start"))
    await _insert_event(engine, finding)
    await _seed_finding_row(engine, finding)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayEquivalenceError, match="unterminated phase"):
        await replayer.assert_replay_equivalent(review_id)


async def test_row_base_field_drift_raises(engine: AsyncEngine) -> None:
    # The audit row's is_eval column drifts from its payload's is_eval; replay
    # reconstructs from the payload and must catch the column/payload divergence.
    review_id = uuid4()
    finding = _finding_event(review_id)  # is_eval defaults False in payload
    payload = finding.model_dump(mode="json", exclude={"sequence_number"})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, is_eval, payload) VALUES (:event_id, :review_id, :event_type, "
                "NULL, :timestamp, TRUE, CAST(:payload AS jsonb))"  # column TRUE vs payload false
            ),
            {
                "event_id": finding.event_id,
                "review_id": review_id,
                "event_type": finding.event_type,
                "timestamp": finding.timestamp,
                "payload": json.dumps(payload),
            },
        )

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayEquivalenceError, match="disagree with the payload"):
        await replayer.reconstruct(review_id)


async def test_review_absent_with_surviving_content_raises(engine: AsyncEngine) -> None:
    # Impossible under the retention ordering (llm_content <= findings <= review):
    # a purged review row with a surviving llm_call_content row is corruption, not a
    # legitimate mixed window. No reviews row is seeded; the audit LLMCallEvent
    # + its content row are. reconstruct() classifies the mode and must reject.
    review_id = uuid4()
    llm_call = _llm_call_event(review_id)
    await _seed_installation(engine)
    # NO _seed_review — the review row is absent (purged).
    await _insert_event(engine, llm_call)
    await _seed_llm_content(engine, llm_call)  # content survives the (absent) review

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayEquivalenceError, match="surviving content with no review row"):
        await replayer.reconstruct(review_id)


async def test_review_present_llm_survives_finding_purged_raises(engine: AsyncEngine) -> None:
    # Sibling of the review-absent corruption case: review present, LLM content
    # surviving, but the finding's content row purged. Impossible under the
    # retention ordering (llm_content <= findings) -- LLM content cannot outlive
    # finding content -- so reconstruct() must reject it, not classify MIXED.
    review_id = uuid4()
    finding = _finding_event(review_id)
    llm_call = _llm_call_event(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id)
    for event in _phase_pair(review_id, "analyze", llm_call, finding):
        await _insert_event(engine, event)
    # LLM content survives; finding content row deliberately NOT seeded (purged).
    await _seed_llm_content(engine, llm_call)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayEquivalenceError, match="LLM content survives while finding content"):
        await replayer.reconstruct(review_id)


async def test_review_row_is_eval_drift_raises(engine: AsyncEngine) -> None:
    # Eval-isolation drift: the reviews row carries is_eval=TRUE while the audit
    # stream is is_eval=False (the _insert_event default). reconstruct() must
    # reject the table-vs-stream divergence rather than mis-bucket the review.
    review_id = uuid4()
    finding = _finding_event(review_id)  # is_eval=False in payload + column
    await _seed_installation(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, is_eval, files_examined, files_traced_beyond_diff, llm_calls_made, "
                "total_input_tokens, total_output_tokens, total_cost_usd, wall_clock_seconds, "
                "retention_expires_at) VALUES (:id, :iid, 100, 1, 'sha1', 'completed', "
                "TRUE, 1, 0, 1, 100, 50, 0.01, 1.5, NOW() + INTERVAL '180 days')"
            ),
            {"id": review_id, "iid": _INSTALLATION_ID},
        )
    for event in _phase_pair(review_id, "analyze", finding):
        await _insert_event(engine, event)
    await _seed_finding_row(engine, finding)

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ReplayEquivalenceError, match="reviews row is_eval"):
        await replayer.reconstruct(review_id)


async def test_full_mode_through_production_persister(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """FULL-mode proof driven through the real ``AuditPersister`` write path.

    The other integration tests raw-SQL-seed ``audit_events`` to construct the
    corruption / purge states the writer cannot produce (a writer whose job is
    preventing corruption can't emit a corrupt row). This test is the
    production-faithful complement: it drives the happy path through
    ``AuditPersister.emit_phase`` / ``persist`` / ``emit_finding`` — the same
    methods the graph nodes call — so replay is proven against real persister
    output (payload normalization, atomic LLMCallEvent + llm_call_content
    co-insert, the persister's hash/field cross-checks), not hand-shaped rows.

    The ``findings`` *content* row is now driven through the production writer
    too: ``emit_finding`` co-inserts the ``FindingEvent`` audit row and the
    ``findings`` content row in one transaction (lifting the event from the
    ``ReviewFinding`` internally), so FULL-mode finding content is produced
    end-to-end through the real persister — no raw-SQL ``findings`` seed.
    Closes FUP-111.
    """
    persister = persister_setup.persister
    engine = persister_setup.engine
    review_id = persister_setup.review_id

    # Drive the production write path for the audit stream, graph-faithfully:
    # the LLM call carries node_id="triage" (the conftest factory's default),
    # so it belongs inside a triage phase — replay's node-containment check
    # rejects a triage call inside an analyze phase. The finding (no node_id)
    # is emitted in its own analyze phase, mirroring the real node sequence.
    await persister.emit_phase(_phase_event(review_id, node_id="triage", marker="start"))
    llm_event = llm_call_event_factory(review_id)  # node_id="triage"
    await persister.persist(
        llm_event,
        llm_request_factory(review_id),
        llm_response_factory(),
    )
    await persister.emit_phase(_phase_event(review_id, node_id="triage", marker="end"))

    await persister.emit_phase(_phase_event(review_id, node_id="analyze", marker="start"))
    # Drive the findings content writer: emit_finding co-inserts the FindingEvent
    # audit row AND the findings content row in one transaction (lifting the event
    # from the ReviewFinding internally). installation_id must match the seeded
    # reviews row (the writer cross-checks); policy_version "1.0.0" is the
    # migration-seeded active policy (severity_policies FK).
    finding = ReviewFinding(
        review_id=review_id,
        installation_id=persister_setup.installation_id,
        policy_version="1.0.0",
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        evidence_tier=EvidenceTier.JUDGED,
        file_path="src/app/models.py",
        line_start=10,
        line_end=20,
        title="SQL injection in query builder",
        description="User input flows into a raw SQL string.",
        evidence="cursor.execute(f'SELECT * FROM t WHERE id={user_id}')",
        content_hash=compute_finding_content_hash(
            "src/app/models.py",
            line_start=10,
            line_end=20,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash=hashlib.sha256(b"proposal").hexdigest(),
    )
    await persister.emit_finding(finding, is_eval=False)
    await persister.emit_phase(_phase_event(review_id, node_id="analyze", marker="end"))

    # The seed review is status='running'; a FULL-mode replay asserts the
    # completed-phase-termination invariant, so flip it to completed. The
    # findings content row was already written by emit_finding above — no raw-SQL seed.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE reviews SET status = 'completed' WHERE id = :rid"),
            {"rid": review_id},
        )

    replayer = AuditReplayer(session_factory=async_sessionmaker(engine, expire_on_commit=False))
    review = await replayer.reconstruct(review_id)

    assert review.mode == ReplayMode.FULL
    assert review.review is not None
    assert review.review.status == "completed"
    assert len(review.findings) == 1
    assert review.findings[0].content is not None
    assert len(review.llm_exchanges) == 1
    # Content came through the real persister (llm_response_factory's default text).
    assert review.llm_exchanges[0].prompt is not None
    assert review.llm_exchanges[0].completion is not None
    # Graph-faithful phase structure: triage phase (its LLM call) then analyze.
    assert [p.node_id for p in review.phases] == ["triage", "analyze"]

    await replayer.assert_replay_equivalent(review_id)  # no raise
