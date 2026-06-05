"""Integration tests for the background replay-verdict projector
(`sweep/replay_verdict.py`).

Seeds a valid completed-review audit stream (phase pairs — the minimal stream that
passes `assert_equivalent`; no findings/LLM rows needed), runs the projector, and
asserts: an equivalent verdict with the right envelope is appended; re-projection is
idempotent (one verdict); eval reviews are excluded (the sweep contract); a
tampered stream (an unterminated phase on a completed review) yields an inequivalent
verdict carrying a reason; a corrupt audit row (reconstruct RAISES) yields an
inequivalent verdict with an ABSENT envelope + a sanitized, content-free reason (the
DECISIONS#014/#016 leak guard); and one bad review (no stream) is counted `failed`
without aborting the tick. Replay-local + non-eval; reuses the `migrated_db` fixture.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from outrider.audit.config import RetentionSettings
from outrider.audit.events import ReplayVerdictEvent, ReviewPhaseEvent
from outrider.audit.persister import AuditPersister
from outrider.sweep.replay_verdict import project_replay_verdicts

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine

_INSTALLATION_ID = 778899


@pytest_asyncio.fixture
async def engine(migrated_db: str) -> AsyncGenerator[AsyncEngine]:
    eng = create_async_engine(migrated_db, poolclass=NullPool)
    try:
        yield eng
    finally:
        await eng.dispose()


def _persister(engine: AsyncEngine) -> AuditPersister:
    return AuditPersister(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        retention_settings=RetentionSettings(),
    )


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


async def _seed_review(
    engine: AsyncEngine, review_id: UUID, *, status: str = "completed", is_eval: bool = False
) -> None:
    # Derive head_sha from review_id so every seeded review has a distinct
    # (repo_id, pr_number, head_sha) natural key (uq_review_natural_key).
    head_sha = review_id.hex[:40]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, is_eval, completed_at, retention_expires_at) VALUES (:id, :iid, 100, 1, "
                ":sha, :status, :is_eval, NOW(), NOW() + INTERVAL '180 days')"
            ),
            {
                "id": review_id,
                "iid": _INSTALLATION_ID,
                "sha": head_sha,
                "status": status,
                "is_eval": is_eval,
            },
        )


async def _insert_phase(
    engine: AsyncEngine,
    review_id: UUID,
    *,
    node_id: Literal["intake", "analyze"],
    marker: Literal["start", "end"],
    is_eval: bool = False,
) -> None:
    event = ReviewPhaseEvent(
        review_id=review_id, phase_id=f"{node_id}:0", node_id=node_id, marker=marker, phase_key=None
    )
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, is_eval, payload) VALUES (:eid, :rid, :etype, NULL, :ts, :is_eval, "
                "CAST(:payload AS jsonb))"
            ),
            {
                "eid": event.event_id,
                "rid": review_id,
                "etype": event.event_type,
                "ts": event.timestamp,
                "is_eval": is_eval,
                "payload": json.dumps(payload),
            },
        )


async def _seed_valid_stream(
    engine: AsyncEngine, review_id: UUID, *, is_eval: bool = False
) -> None:
    """A minimal stream that passes assert_equivalent: two terminated phase pairs."""
    await _seed_review(engine, review_id, is_eval=is_eval)
    for node_id, marker in (
        ("intake", "start"),
        ("intake", "end"),
        ("analyze", "start"),
        ("analyze", "end"),
    ):
        await _insert_phase(engine, review_id, node_id=node_id, marker=marker, is_eval=is_eval)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_projects_equivalent_verdict_for_valid_completed_review(engine: AsyncEngine) -> None:
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_valid_stream(engine, review_id)

    persister = _persister(engine)
    result = await project_replay_verdicts(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        audit_persister=persister,
    )
    assert result == {"projected": 1, "failed": 0}

    verdict = await persister.query_replay_verdict_event(review_id=review_id)
    assert verdict is not None
    assert verdict.replay_equivalent is True
    assert verdict.mode == "full"  # review row present, no content → FULL (vacuous)
    assert verdict.event_count == 4  # the four phase events
    assert verdict.finding_count == 0
    assert verdict.orphan_finding_count == 0
    assert verdict.target_max_sequence_number == 4  # fresh-DB IDENTITY 1..4
    assert verdict.reason is None
    assert verdict.is_eval is False


@pytest.mark.asyncio
async def test_emit_replay_verdict_idempotent_on_conflict(engine: AsyncEngine) -> None:
    # Directly exercise the load-bearing on_conflict_do_nothing path (the idempotency
    # that justifies NO advisory lock under concurrent ticks) — NOT the candidate
    # anti-join. Two emits for the same review (distinct event_ids, bypassing the
    # projector's NOT EXISTS filter): the first inserts (True), the second is a no-op
    # (False), and exactly one verdict row survives.
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id)  # FK target for the verdict rows
    persister = _persister(engine)

    def _verdict() -> ReplayVerdictEvent:
        return ReplayVerdictEvent(
            review_id=review_id,
            replay_equivalent=True,
            mode="full",
            event_count=4,
            finding_count=0,
            orphan_finding_count=0,
            target_max_sequence_number=4,
        )

    assert await persister.emit_replay_verdict(_verdict()) is True  # newly inserted
    assert await persister.emit_replay_verdict(_verdict()) is False  # conflict no-op

    async with engine.begin() as conn:
        count = await conn.scalar(
            text(
                "SELECT count(*) FROM audit_events "
                "WHERE review_id = :rid AND event_type = 'replay_verdict'"
            ),
            {"rid": review_id},
        )
    assert count == 1


@pytest.mark.asyncio
async def test_reprojection_is_idempotent(engine: AsyncEngine) -> None:
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_valid_stream(engine, review_id)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    persister = _persister(engine)

    first = await project_replay_verdicts(
        session_factory=session_factory, audit_persister=persister
    )
    assert first == {"projected": 1, "failed": 0}
    # Second tick: the candidate anti-join excludes the now-verdicted review.
    second = await project_replay_verdicts(
        session_factory=session_factory, audit_persister=persister
    )
    assert second == {"projected": 0, "failed": 0}

    # Exactly one verdict row exists.
    async with engine.begin() as conn:
        count = await conn.scalar(
            text(
                "SELECT count(*) FROM audit_events "
                "WHERE review_id = :rid AND event_type = 'replay_verdict'"
            ),
            {"rid": review_id},
        )
    assert count == 1


@pytest.mark.asyncio
async def test_only_production_review_verdicted_eval_excluded(engine: AsyncEngine) -> None:
    # One eval + one production review, identical valid streams. The projector must
    # verdict EXACTLY the production one — proving the exclusion is is_eval-specific,
    # not a global no-candidate bug (which {projected:0} alone would also satisfy).
    prod_id = uuid4()
    eval_id = uuid4()
    await _seed_installation(engine)
    await _seed_valid_stream(engine, prod_id, is_eval=False)
    await _seed_valid_stream(engine, eval_id, is_eval=True)

    persister = _persister(engine)
    result = await project_replay_verdicts(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        audit_persister=persister,
    )
    assert result == {"projected": 1, "failed": 0}  # only the production review
    assert await persister.query_replay_verdict_event(review_id=prod_id) is not None
    assert await persister.query_replay_verdict_event(review_id=eval_id) is None


@pytest.mark.asyncio
async def test_inequivalent_verdict_for_unterminated_phase(engine: AsyncEngine) -> None:
    # A completed review with an unterminated phase: reconstruct SUCCEEDS, but
    # assert_equivalent's require_all_terminated check fails → inequivalent verdict
    # with the full envelope + a reason.
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id)
    for node_id, marker in (("intake", "start"), ("intake", "end"), ("analyze", "start")):
        await _insert_phase(engine, review_id, node_id=node_id, marker=marker)  # type: ignore[arg-type]

    persister = _persister(engine)
    result = await project_replay_verdicts(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        audit_persister=persister,
    )
    assert result == {"projected": 1, "failed": 0}

    verdict = await persister.query_replay_verdict_event(review_id=review_id)
    assert verdict is not None
    assert verdict.replay_equivalent is False
    assert verdict.reason is not None
    assert "unterminated" in verdict.reason
    assert verdict.mode == "full"  # reconstruction succeeded → full envelope present
    assert verdict.event_count == 3


@pytest.mark.asyncio
async def test_corrupt_row_yields_sanitized_inequivalent_verdict(engine: AsyncEngine) -> None:
    # A corrupt audit_events payload (an invalid Literal `marker`) makes reconstruct
    # RAISE a pydantic ValidationError → the projector emits an inequivalent verdict
    # with an ABSENT envelope and a SANITIZED reason. The reason must NOT echo the
    # offending payload VALUE — the DECISIONS#014/#016 content-leak guard that
    # `_reconstruct_failure_reason` exists to enforce. Revert-the-fold: drop the
    # `except (ReplayError, ValidationError)` block or regress the sanitizer to
    # `str(exc)`, and this test is the one that fails.
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id)
    leak_canary = "LEAKED_PAYLOAD_VALUE_should_not_surface"
    corrupt_payload = {
        "event_id": str(uuid4()),
        "review_id": str(review_id),
        "event_type": "review_phase",
        "phase_id": "intake:0",
        "node_id": "intake",
        "marker": leak_canary,  # invalid: ReviewPhaseEvent.marker is Literal["start", "end"]
        "phase_key": None,
        "timestamp": "2026-06-04T00:00:00+00:00",
        "is_eval": False,
    }
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, is_eval, payload) VALUES (:eid, :rid, 'review_phase', NULL, "
                ":ts, false, CAST(:payload AS jsonb))"
            ),
            {
                "eid": corrupt_payload["event_id"],
                "rid": review_id,
                "ts": corrupt_payload["timestamp"],
                "payload": json.dumps(corrupt_payload),
            },
        )

    persister = _persister(engine)
    result = await project_replay_verdicts(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        audit_persister=persister,
    )
    assert result == {"projected": 1, "failed": 0}  # a verdict IS produced (inequivalent)

    verdict = await persister.query_replay_verdict_event(review_id=review_id)
    assert verdict is not None
    assert verdict.replay_equivalent is False
    assert verdict.mode is None  # absent envelope — reconstruction failed
    assert verdict.event_count is None
    assert verdict.finding_count is None
    assert verdict.orphan_finding_count is None
    assert verdict.reason is not None
    assert "ValidationError" in verdict.reason
    assert leak_canary not in verdict.reason  # the content-leak guard: no raw payload value


@pytest.mark.asyncio
async def test_one_bad_review_does_not_abort_the_tick(engine: AsyncEngine) -> None:
    # A completed review with ZERO audit events makes `_max_non_verdict_sequence`
    # raise (a data anomaly) → counted as `failed`, NOT a fabricated verdict. A valid
    # review in the SAME tick still gets projected — the per-row try/except is the
    # load-bearing "one bad review never aborts the tick" resilience. Revert-the-fold:
    # remove the per-row except and the good verdict is lost when the bad one raises.
    good_id = uuid4()
    bad_id = uuid4()
    await _seed_installation(engine)
    await _seed_valid_stream(engine, good_id)
    await _seed_review(engine, bad_id)  # completed, but no audit events seeded

    persister = _persister(engine)
    result = await project_replay_verdicts(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        audit_persister=persister,
    )
    assert result == {"projected": 1, "failed": 1}
    assert await persister.query_replay_verdict_event(review_id=good_id) is not None
    assert await persister.query_replay_verdict_event(review_id=bad_id) is None  # no fabrication
