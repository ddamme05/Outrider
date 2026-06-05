"""Integration tests for `GET /api/reviews/{id}/replay-timeline` (ROADMAP feature 6).

Proves the grouped+verified timeline contract against real seeded audit streams:

- an equivalent review → grouped `.phases` + per-phase event rows + the inter-phase bucket;
- the projected `ReplayVerdictEvent` is surfaced via the verdict, never in events/phases/bucket;
- the FUP-125 gate: a malformed (nested) phase stream → `.phases` SUPPRESSED (the lossy
  `_group_phases` grouping never reaches the response) — the proof FUP-125 is closed;
- the `end is None` status-split (Codex): a NON-completed review with an open phase → equivalent +
  phases exposed; a `completed` review with a dangling phase (publish-crash) → non-equivalent +
  phases suppressed (NOT a 500);
- `reconstruct`-raised (is_eval drift) → verdict only, phases null, NOT a 500;
- 404 + auth.

PR 2 (content expansion): `findings`/`llm_exchanges` carry the expandable content from the same
verified snapshot — FULL populates it, METADATA_ONLY/MIXED redact per-item (`content_redacted`),
override provenance comes from the `HITLDecisionEvent` stream (DECISIONS.md#034, not the V1-null
table columns), the redaction date comes from `purge_audit`, and a non-equivalent verdict suppresses
content the same way it suppresses `phases`. Content seeding mirrors
`tests/integration/test_audit_replay.py`; HITL + purge_audit seeding mirrors
`tests/integration/test_dashboard_findings_api.py`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.api.dashboard import reviews_router
from outrider.audit.events import (
    AgentTransitionEvent,
    AuditEventBase,
    FindingEvent,
    HITLDecisionEvent,
    LLMCallEvent,
    ReplayVerdictEvent,
    ReviewPhaseEvent,
    compute_finding_content_hash,
)
from outrider.policy.canonical import compute_hitl_decision_content_hash
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity, FindingType
from outrider.schemas import ReviewDimension
from outrider.schemas.hitl import PerFindingDecision, PerFindingOutcome

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_INSTALLATION_ID = 661122


# ---- event factories (is_eval threads through to the event so the column matches the payload) ----
def _phase(
    review_id: UUID,
    node_id: Literal["intake", "analyze", "hitl"],
    marker: Literal["start", "end"],
    *,
    is_eval: bool = False,
) -> ReviewPhaseEvent:
    return ReviewPhaseEvent(
        review_id=review_id,
        phase_id=f"{node_id}:0",
        node_id=node_id,
        marker=marker,
        phase_key=None,
        is_eval=is_eval,
    )


def _transition(review_id: UUID, *, is_eval: bool = False) -> AgentTransitionEvent:
    return AgentTransitionEvent(
        review_id=review_id, from_node="webhook", to_node="intake", latency_ms=3, is_eval=is_eval
    )


def _llm_call(review_id: UUID, *, is_eval: bool = False) -> LLMCallEvent:
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
        is_eval=is_eval,
    )


def _finding(review_id: UUID, *, is_eval: bool = False) -> FindingEvent:
    return FindingEvent(
        review_id=review_id,
        finding_id=uuid4(),
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.CRITICAL,
        file_path="src/app/models.py",
        line_start=10,
        line_end=20,
        dimension=ReviewDimension.SECURITY,
        finding_content_hash=compute_finding_content_hash(
            "src/app/models.py", line_start=10, line_end=20, finding_type=FindingType.SQL_INJECTION
        ),
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=None,
        policy_version="1.0.0",
        proposal_hash=hashlib.sha256(b"proposal").hexdigest(),
        is_eval=is_eval,
    )


async def _insert_event(engine: AsyncEngine, event: AuditEventBase) -> None:
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    phase_key = event.phase_key if isinstance(event, ReviewPhaseEvent) else None
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, timestamp, "
                "is_eval, payload) VALUES (:eid, :rid, :etype, :pk, :ts, :is_eval, "
                "CAST(:payload AS jsonb))"
            ),
            {
                "eid": event.event_id,
                "rid": event.review_id,
                "etype": event.event_type,
                "pk": phase_key,
                "ts": event.timestamp,
                "is_eval": event.is_eval,
                "payload": json.dumps(payload),
            },
        )


async def _seed_installation(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                "account_login, account_type, permissions_at_install) VALUES (:id, 'test-app', "
                "1, 'octocat', 'User', '{}'::jsonb) ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": _INSTALLATION_ID},
        )


async def _seed_review(
    engine: AsyncEngine, review_id: UUID, *, status: str, is_eval: bool = False, repo_id: int = 100
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, status, "
                "is_eval, created_at, retention_expires_at) VALUES (:id, :iid, :repo, 1, :sha, "
                ":status, :is_eval, NOW(), NOW() + INTERVAL '90 days')"
            ),
            {
                "id": review_id,
                "iid": _INSTALLATION_ID,
                "repo": repo_id,
                "sha": review_id.hex[:40],
                "status": status,
                "is_eval": is_eval,
            },
        )


def _mount(engine: AsyncEngine) -> TestClient:
    app = FastAPI()
    app.include_router(reviews_router)
    app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    return TestClient(app)


@pytest_asyncio.fixture
async def engine(migrated_db: str) -> AsyncGenerator[AsyncEngine]:
    eng = create_async_engine(migrated_db, hide_parameters=True)
    try:
        yield eng
    finally:
        await eng.dispose()


def _get(client: TestClient, review_id: UUID) -> dict[str, Any]:
    resp = client.get(f"/api/reviews/{review_id}/replay-timeline", headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


@pytest.mark.asyncio
async def test_equivalent_timeline_exposes_phases_and_inter_phase_bucket(
    engine: AsyncEngine,
) -> None:
    # transition (outside phases) → inter_phase bucket; analyze phase wraps the llm + finding.
    review_id = uuid4()
    await _insert_event(engine, _transition(review_id))
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, _llm_call(review_id))
    await _insert_event(engine, _finding(review_id))
    await _insert_event(engine, _phase(review_id, "analyze", "end"))

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is True
    assert body["mode"] == "metadata_only"  # no review/content rows
    assert body["phases"] is not None
    assert [p["node_id"] for p in body["phases"]] == ["analyze"]
    # The analyze phase's per-operation rows are the llm_call + finding (NOT the markers).
    assert {e["event_type"] for e in body["phases"][0]["events"]} == {"llm_call", "finding"}
    # The webhook→intake transition is outside any phase → the inter-phase bucket, NOT in a phase.
    assert [e["event_type"] for e in body["inter_phase_events"]] == ["agent_transition"]


@pytest.mark.asyncio
async def test_projected_verdict_is_excluded_from_stream_phases_and_bucket(
    engine: AsyncEngine,
) -> None:
    # A review that already has a projected ReplayVerdictEvent appended (post-completion metadata).
    review_id = uuid4()
    await _insert_event(engine, _transition(review_id))
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, _llm_call(review_id))
    await _insert_event(engine, _phase(review_id, "analyze", "end"))
    await _insert_event(
        engine,
        ReplayVerdictEvent(
            review_id=review_id,
            replay_equivalent=True,
            mode="metadata_only",
            event_count=4,
            finding_count=0,
            orphan_finding_count=0,
            target_max_sequence_number=4,
        ),
    )
    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is True
    # The verdict event is surfaced via the verdict header, never as an operation row anywhere.
    all_types = (
        {e["event_type"] for e in body["events"]}
        | {e["event_type"] for e in body["inter_phase_events"]}
        | {e["event_type"] for p in body["phases"] for e in p["events"]}
    )
    assert "replay_verdict" not in all_types


@pytest.mark.asyncio
async def test_in_flight_open_phase_on_running_review_is_equivalent(engine: AsyncEngine) -> None:
    # A NON-completed (running) review with an open trailing phase → require_all_terminated is
    # False, the open phase is tolerated → equivalent, phases exposed with the last phase end=None.
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="running")
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, _llm_call(review_id))  # open analyze phase, never closed

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is True  # in-flight, NOT a non-equivalent verdict
    assert body["status"] == "running"
    assert body["phases"] is not None
    assert body["phases"][0]["end"] is None  # the open/in-flight phase


@pytest.mark.asyncio
async def test_completed_dangling_phase_publish_crash_suppresses_phases(
    engine: AsyncEngine,
) -> None:
    # publish writes status='completed' BEFORE the publish phase-end (the deliberate "interrupted"
    # signal). A crash in that window leaves a COMPLETED review with a dangling phase →
    # require_all_terminated fires → assert raises → NON-equivalent → phases SUPPRESSED. NOT a 500.
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="completed")
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(
        engine, _llm_call(review_id)
    )  # dangling: no analyze-end on a completed review

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is False
    assert body["phases"] is None  # the FUP-125-unsafe grouping is suppressed
    assert body["reason"] is not None and "unterminated" in body["reason"]
    assert body["mode"] is not None  # reconstruct succeeded → mode present
    assert body["events"], "the flat ordered stream is still returned for the fallback"


@pytest.mark.asyncio
async def test_malformed_nested_phase_suppresses_phases(engine: AsyncEngine) -> None:
    # FUP-125 proof: a NESTED phase stream — `_group_phases` silently tolerates it (lossy), but
    # `_verify_phase_wellformed` rejects the non-nesting → assert raises → phases SUPPRESSED. The
    # lossy grouping NEVER reaches the response.
    review_id = uuid4()
    await _insert_event(engine, _phase(review_id, "intake", "start"))
    await _insert_event(
        engine, _phase(review_id, "analyze", "start")
    )  # nested: analyze opens while intake open
    await _insert_event(engine, _phase(review_id, "analyze", "end"))
    await _insert_event(engine, _phase(review_id, "intake", "end"))

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is False
    assert body["phases"] is None  # the lossy nested grouping is never exposed
    assert body["reason"] is not None and "non-nested" in body["reason"]


@pytest.mark.asyncio
async def test_reconstruct_raised_is_eval_drift_returns_verdict_not_500(
    engine: AsyncEngine,
) -> None:
    # A production review (is_eval=False) whose events are is_eval=True → reconstruct's
    # _verify_is_eval_consistent raises → verdict only (mode/status/phases null), 200 not 500.
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="completed", is_eval=False)
    await _insert_event(engine, _phase(review_id, "analyze", "start", is_eval=True))  # DRIFT
    await _insert_event(engine, _phase(review_id, "analyze", "end", is_eval=True))

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is False
    assert body["phases"] is None
    assert body["mode"] is None  # reconstruct itself raised
    assert body["events"] == []
    assert body["reason"] is not None


def test_unknown_review_404(engine: AsyncEngine) -> None:
    resp = _mount(engine).get(f"/api/reviews/{uuid4()}/replay-timeline", headers=_AUTH)
    assert resp.status_code == 404


def test_auth_required(engine: AsyncEngine) -> None:
    resp = _mount(engine).get(f"/api/reviews/{uuid4()}/replay-timeline")
    assert resp.status_code == 401


# ---- PR 2 content expansion: content seeding (mirrors test_audit_replay.py) ----
async def _seed_finding_row(engine: AsyncEngine, event: FindingEvent) -> None:
    """A surviving `findings` content row matching the FindingEvent (FULL mode).

    Override columns (`original_severity`/`override_reason`/`overrider_id`) are
    deliberately left NULL — the V1 read-model state (DECISIONS.md#034): provenance
    lives in the stream, never the table.
    """
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


def _hitl_override(review_id: UUID, finding_id: UUID) -> HITLDecisionEvent:
    """A valid `HITLDecisionEvent` overriding one finding CRITICAL → HIGH.

    Constructed via the model (not a raw insert) because the timeline goes through
    `reconstruct()`, which validates every row against the discriminated union — a
    minimal payload would fail the `event_type`/content-hash checks.
    """
    decision = PerFindingDecision(
        finding_id=finding_id,
        outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
        reason="downgraded: test-only path",
        original_severity=FindingSeverity.CRITICAL,
        override_severity=FindingSeverity.HIGH,
    )
    return HITLDecisionEvent(
        review_id=review_id,
        reviewer_id="admin",
        decisions=(decision,),
        annotation=None,
        decided_at=datetime(2026, 6, 1, tzinfo=UTC),
        decision_latency_seconds=12.0,
        decisions_content_hash=compute_hitl_decision_content_hash(
            decisions=(decision,), annotation=None
        ),
    )


async def _insert_purge_audit(
    engine: AsyncEngine, *, target_table: str, timestamp_iso: str, installation_id: int = 0
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO purge_audit "
                "(installation_id, target_table, rows_affected, purge_role, timestamp) "
                "VALUES (:iid, :tt, 1, 'sweep', :ts)"
            ),
            {"iid": installation_id, "tt": target_table, "ts": timestamp_iso},
        )


@pytest.mark.asyncio
async def test_full_mode_serializes_finding_and_llm_content(engine: AsyncEngine) -> None:
    # FULL: review + finding row + llm content all present → content populated, not redacted.
    review_id = uuid4()
    finding = _finding(review_id)
    llm = _llm_call(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="completed")
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, llm)
    await _insert_event(engine, finding)
    await _insert_event(engine, _phase(review_id, "analyze", "end"))
    await _seed_finding_row(engine, finding)
    await _seed_llm_content(engine, llm)

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is True
    assert body["mode"] == "full"
    (fv,) = body["findings"]
    assert fv["finding_id"] == str(finding.finding_id)
    assert fv["content_redacted"] is False
    assert (fv["title"], fv["description"], fv["evidence"]) == ("t", "d", "e")
    assert fv["hitl_decision"] is None  # no override
    assert fv["redaction_sweep_at"] is None
    (lv,) = body["llm_exchanges"]
    assert lv["event_id"] == str(llm.event_id)
    assert lv["content_redacted"] is False
    assert (lv["prompt"], lv["completion"]) == ("the prompt", "the completion")


@pytest.mark.asyncio
async def test_metadata_only_redacts_content(engine: AsyncEngine) -> None:
    # No review/content rows → METADATA_ONLY: content views are redacted stubs, but the
    # finding/llm metadata still rides the events/phases rows.
    review_id = uuid4()
    finding = _finding(review_id)
    llm = _llm_call(review_id)
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, llm)
    await _insert_event(engine, finding)
    await _insert_event(engine, _phase(review_id, "analyze", "end"))

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is True
    assert body["mode"] == "metadata_only"
    (fv,) = body["findings"]
    assert fv["content_redacted"] is True
    assert (fv["title"], fv["description"], fv["evidence"], fv["suggested_fix"]) == (
        None,
        None,
        None,
        None,
    )
    (lv,) = body["llm_exchanges"]
    assert lv["content_redacted"] is True
    assert (lv["prompt"], lv["completion"]) == (None, None)


@pytest.mark.asyncio
async def test_mixed_mode_per_item_redaction(engine: AsyncEngine) -> None:
    # MIXED: finding content survives, shorter-TTL llm_call_content purged → per-item flags,
    # never silently hybridized.
    review_id = uuid4()
    finding = _finding(review_id)
    llm = _llm_call(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="completed")
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, llm)
    await _insert_event(engine, finding)
    await _insert_event(engine, _phase(review_id, "analyze", "end"))
    await _seed_finding_row(engine, finding)
    # deliberately NO _seed_llm_content — the LLM content row is purged.

    body = _get(_mount(engine), review_id)
    assert body["mode"] == "mixed"
    (fv,) = body["findings"]
    assert fv["content_redacted"] is False and fv["title"] == "t"
    (lv,) = body["llm_exchanges"]
    assert lv["content_redacted"] is True and lv["prompt"] is None


@pytest.mark.asyncio
async def test_hitl_override_provenance_from_stream_not_table(engine: AsyncEngine) -> None:
    # DECISIONS#034: the override shows from the HITLDecisionEvent stream even though the
    # `findings` override columns are NULL (the seeded row leaves them unset).
    review_id = uuid4()
    finding = _finding(review_id)  # severity=CRITICAL
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="completed")
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, finding)
    await _insert_event(engine, _phase(review_id, "analyze", "end"))
    await _seed_finding_row(engine, finding)  # override columns NULL
    # The decision is phase-bound work → it lives inside a hitl phase, not loose in the stream.
    await _insert_event(engine, _phase(review_id, "hitl", "start"))
    await _insert_event(engine, _hitl_override(review_id, finding.finding_id))
    await _insert_event(engine, _phase(review_id, "hitl", "end"))

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is True
    (fv,) = body["findings"]
    hitl = fv["hitl_decision"]
    assert hitl is not None
    assert hitl["outcome"] == "severity_override"
    assert hitl["reviewer_id"] == "admin"
    assert (hitl["original_severity"], hitl["override_severity"]) == ("critical", "high")


@pytest.mark.asyncio
async def test_non_equivalent_verdict_suppresses_content(engine: AsyncEngine) -> None:
    # A completed review with a dangling phase → non-equivalent → content suppressed (empty),
    # parallel to the `phases` gate, even though the finding content row exists.
    review_id = uuid4()
    finding = _finding(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="completed")
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, finding)  # dangling: no analyze-end on a completed review
    await _seed_finding_row(engine, finding)

    body = _get(_mount(engine), review_id)
    assert body["replay_equivalent"] is False
    assert body["phases"] is None
    assert body["findings"] == []
    assert body["llm_exchanges"] == []


@pytest.mark.asyncio
async def test_redaction_sweep_date_from_purge_audit(engine: AsyncEngine) -> None:
    # FUP-129: the redacted LLM stub's date comes from the latest purge_audit row scoped to
    # target_table='llm_call_content' (NOT 'findings'). No review/content rows → both redacted.
    review_id = uuid4()
    finding = _finding(review_id)
    llm = _llm_call(review_id)
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, llm)
    await _insert_event(engine, finding)
    await _insert_event(engine, _phase(review_id, "analyze", "end"))
    await _insert_purge_audit(
        engine, target_table="findings", timestamp_iso="2026-05-01T00:00:00+00:00"
    )
    await _insert_purge_audit(
        engine, target_table="llm_call_content", timestamp_iso="2026-05-20T00:00:00+00:00"
    )

    body = _get(_mount(engine), review_id)
    (fv,) = body["findings"]
    (lv,) = body["llm_exchanges"]
    # Each content type reads its OWN target_table sweep row, not the other's.
    assert fv["redaction_sweep_at"].startswith("2026-05-01")
    assert lv["redaction_sweep_at"].startswith("2026-05-20")


@pytest.mark.asyncio
async def test_redaction_sweep_date_scoped_to_review_installation(engine: AsyncEngine) -> None:
    # MIXED: the review survives (installation_id present) while the shorter-TTL llm_call_content is
    # purged. The LLM stub's sweep date must resolve from a purge_audit row scoped to the REVIEW's
    # installation — exercising the (installation_id, GLOBAL) branch via the INSTALLATION match, not
    # just the global sentinel covered by test_redaction_sweep_date_from_purge_audit.
    review_id = uuid4()
    finding = _finding(review_id)
    llm = _llm_call(review_id)
    await _seed_installation(engine)
    await _seed_review(engine, review_id, status="completed")
    await _insert_event(engine, _phase(review_id, "analyze", "start"))
    await _insert_event(engine, llm)
    await _insert_event(engine, finding)
    await _insert_event(engine, _phase(review_id, "analyze", "end"))
    await _seed_finding_row(engine, finding)  # finding content survives
    # llm content purged -> MIXED; the sweep row carries the review's installation, not global 0.
    await _insert_purge_audit(
        engine,
        target_table="llm_call_content",
        timestamp_iso="2026-05-15T00:00:00+00:00",
        installation_id=_INSTALLATION_ID,
    )

    body = _get(_mount(engine), review_id)
    assert body["mode"] == "mixed"
    (lv,) = body["llm_exchanges"]
    assert lv["content_redacted"] is True
    assert lv["redaction_sweep_at"].startswith("2026-05-15")
    # The surviving finding content carries no redaction date.
    (fv,) = body["findings"]
    assert fv["content_redacted"] is False
    assert fv["redaction_sweep_at"] is None
