"""Unit-level pin of the concurrent-decide endpoint race per the
HITL spec line 313.

Both endpoint calls observe `preflight.hitl_decision is None`
(JSONB cache hasn't been written yet because no graph body has
completed). Both pass auth + state + mismatch gates. Both return
202 Accepted. The natural-key partial unique index on
`audit_events(review_id) WHERE event_type='hitl_decision'` is the
serialization point: only one HITLDecisionEvent lands. The
second background task hits `AuditPersisterHITLDecisionNaturalKeyConflict`,
which the resume wrapper absorbs at WARNING level (per the
post-fold-3 contract — INFO `hitl_resume_duplicate_submission` is
the pre-fold spec wording).

Per the spec's Group 8 framing + the F2 audit-fold note, real
serialization of the GRAPH invocation comes from langgraph's
per-thread checkpointer. The endpoint-level race (both submissions
admitted) is bounded at the audit-row natural-key check. This test
pins the endpoint behavior in isolation; the integration-level
serialization assertion (only one publish landing) belongs in an
integration test against a real graph + checkpointer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from outrider.api.dashboard import hitl_router
from outrider.db.sinks import ReviewDecidePreflight
from outrider.policy import FindingSeverity
from outrider.schemas.hitl import HITLRequest

_FINDING_A = UUID("11111111-1111-1111-1111-111111111111")
_FINDING_B = UUID("22222222-2222-2222-2222-222222222222")


def _make_preflight() -> ReviewDecidePreflight:
    now = datetime.now(UTC)
    return ReviewDecidePreflight(
        status="awaiting_approval",
        hitl_request=HITLRequest(
            findings_requiring_approval=(_FINDING_A, _FINDING_B),
            auto_post_findings=(),
            created_at=now,
            expires_at=now + timedelta(minutes=30),
        ),
        hitl_decision=None,
        gated_finding_severities={
            _FINDING_A: FindingSeverity.HIGH,
            _FINDING_B: FindingSeverity.CRITICAL,
        },
    )


def _build_app(preflight: ReviewDecidePreflight) -> tuple[TestClient, MagicMock]:
    app = FastAPI()
    app.include_router(hitl_router)
    app.state.admin_api_key = SecretStr("test-key")

    reader = MagicMock()
    reader.fetch_for_decide = AsyncMock(return_value=preflight)
    app.state.review_status_reader = reader

    compiled_graph = MagicMock()
    compiled_graph.ainvoke = AsyncMock(return_value=None)
    app.state.compiled_graph = compiled_graph

    return TestClient(app), compiled_graph


def _decision_payload(*, reason_suffix: str) -> dict[str, Any]:
    return {
        "decisions": [
            {
                "finding_id": str(_FINDING_A),
                "outcome": "approve",
                "reason": f"approved-{reason_suffix}",
            },
            {
                "finding_id": str(_FINDING_B),
                "outcome": "approve",
                "reason": f"approved-{reason_suffix}",
            },
        ],
        "annotation": f"annotation-{reason_suffix}",
    }


_AUTH = {"Authorization": "Bearer test-key"}


def test_both_concurrent_calls_pass_preflight_and_return_202() -> None:
    """Two operators with the same admin key submit concurrently.
    Both preflights see hitl_decision=None (JSONB cache not yet
    written), both pass mismatch (same finding set), both return
    202 Accepted. The natural-key check happens INSIDE the background
    task at audit-emit time; the endpoint itself never blocks."""
    pf = _make_preflight()
    client, _ = _build_app(pf)

    review_id = "33333333-3333-3333-3333-333333333333"
    resp_a = client.post(
        f"/reviews/{review_id}/decide",
        json=_decision_payload(reason_suffix="A"),
        headers=_AUTH,
    )
    resp_b = client.post(
        f"/reviews/{review_id}/decide",
        json=_decision_payload(reason_suffix="B"),
        headers=_AUTH,
    )

    assert resp_a.status_code == 202
    assert resp_b.status_code == 202
    assert resp_a.json()["status"] == "resuming"
    assert resp_b.json()["status"] == "resuming"


def test_concurrent_calls_each_enqueue_their_own_resume() -> None:
    """Both endpoint calls trigger their OWN background-task resume.
    The endpoint isn't responsible for serialization — that's the
    background-task wrapper + the audit-row natural-key check."""
    pf = _make_preflight()
    client, compiled_graph = _build_app(pf)

    review_id = "44444444-4444-4444-4444-444444444444"
    client.post(
        f"/reviews/{review_id}/decide",
        json=_decision_payload(reason_suffix="A"),
        headers=_AUTH,
    )
    client.post(
        f"/reviews/{review_id}/decide",
        json=_decision_payload(reason_suffix="B"),
        headers=_AUTH,
    )

    # Both background tasks called the graph. Each carries the
    # caller's distinct annotation in the resume payload, so the
    # audit-layer natural-key check can correctly detect divergent
    # content (different decisions_content_hash).
    assert compiled_graph.ainvoke.await_count == 2
    awaited_args = list(compiled_graph.ainvoke.await_args_list)
    # Defensive call-shape assertions: if the wrapper changes how
    # `ainvoke` is called (kwarg vs positional, dict vs Command
    # instance), the downstream `c.args[0].resume` would raise
    # AttributeError or IndexError with no diagnostic context. Assert
    # the load-bearing shape first so a wrapper-signature change
    # surfaces a clear failure here, not a generic AttributeError.
    assert all(len(c.args) > 0 for c in awaited_args), (
        "Expected positional arg in ainvoke calls — wrapper signature changed?"
    )
    assert all(hasattr(c.args[0], "resume") for c in awaited_args), (
        "Expected first positional arg to carry `.resume` attribute "
        "(Command instance from langgraph) — wrapper signature changed?"
    )
    resume_payloads = [c.args[0].resume for c in awaited_args]
    annotations = sorted(p["annotation"] for p in resume_payloads)
    assert annotations == ["annotation-A", "annotation-B"]


@pytest.mark.asyncio
async def test_failure_wrapper_absorbs_loser_of_natural_key_race() -> None:
    """The actual race resolution happens inside
    `_run_resume_under_failure_wrapper`. The audit-layer natural-key
    check raises `AuditPersisterHITLDecisionNaturalKeyConflict` on
    the loser; the wrapper catches at WARNING level and returns
    gracefully. The winner's graph completes through publish.

    This pins the wrapper behavior in isolation; the
    asyncio.gather concurrency case follows below."""
    from outrider.api.dashboard.hitl import _run_resume_under_failure_wrapper
    from outrider.audit.persister import (
        AuditPersisterHITLDecisionNaturalKeyConflict,
    )
    from outrider.schemas.hitl import HITLDecision, PerFindingDecision, PerFindingOutcome

    review_id = UUID("55555555-5555-5555-5555-555555555555")
    decision = HITLDecision(
        reviewer_id="admin",
        decisions=(
            PerFindingDecision(
                finding_id=_FINDING_A, outcome=PerFindingOutcome.APPROVE, reason="loser"
            ),
        ),
        decided_at=datetime.now(UTC),
    )

    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        side_effect=AuditPersisterHITLDecisionNaturalKeyConflict(
            existing_event_id=UUID("66666666-6666-6666-6666-666666666666"),
            incoming_event_id=UUID("77777777-7777-7777-7777-777777777777"),
            review_id=review_id,
            mismatched_fields=(("decisions_content_hash", "winner-hash", "loser-hash"),),
            natural_key=(("review_id", str(review_id)),),
        )
    )

    # Wrapper absorbs cleanly (no raise). The winner's task — which
    # we don't simulate here — completes the publish independently.
    await _run_resume_under_failure_wrapper(
        review_id=review_id,
        hitl_decision=decision,
        graph=graph,
    )


@pytest.mark.asyncio
async def test_concurrent_resume_wrappers_via_gather_loser_absorbed() -> None:
    """Two `_run_resume_under_failure_wrapper` calls execute concurrently
    via `asyncio.gather`. One graph succeeds (winner); the other raises
    `AuditPersisterHITLDecisionNaturalKeyConflict` (loser the natural-
    key partial unique index rejected). Both wrapper coroutines must
    complete without re-raising — the loser's WARNING-level absorb is
    exercised here under genuine concurrency, not the sync
    `TestClient.post` serialization the earlier tests in this file use.
    """
    from outrider.api.dashboard.hitl import _run_resume_under_failure_wrapper
    from outrider.audit.persister import (
        AuditPersisterHITLDecisionNaturalKeyConflict,
    )
    from outrider.schemas.hitl import HITLDecision, PerFindingDecision, PerFindingOutcome

    review_id = UUID("88888888-8888-8888-8888-888888888888")

    winner_graph = MagicMock()
    winner_graph.ainvoke = AsyncMock(return_value=None)

    loser_graph = MagicMock()
    loser_graph.ainvoke = AsyncMock(
        side_effect=AuditPersisterHITLDecisionNaturalKeyConflict(
            existing_event_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            incoming_event_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            review_id=review_id,
            mismatched_fields=(("decisions_content_hash", "winner-hash", "loser-hash"),),
            natural_key=(("review_id", str(review_id)),),
        )
    )

    def _decision(reason: str) -> HITLDecision:
        return HITLDecision(
            reviewer_id="admin",
            decisions=(
                PerFindingDecision(
                    finding_id=_FINDING_A,
                    outcome=PerFindingOutcome.APPROVE,
                    reason=reason,
                ),
            ),
            decided_at=datetime.now(UTC),
        )

    results = await asyncio.gather(
        _run_resume_under_failure_wrapper(
            review_id=review_id,
            hitl_decision=_decision("alpha"),
            graph=winner_graph,
        ),
        _run_resume_under_failure_wrapper(
            review_id=review_id,
            hitl_decision=_decision("beta"),
            graph=loser_graph,
        ),
    )

    # Both wrapper calls returned None (no re-raise). The loser's
    # NaturalKeyConflict was absorbed by the wrapper's catch arm;
    # the winner ran through ainvoke cleanly.
    assert len(results) == 2
    assert all(r is None for r in results)
    assert winner_graph.ainvoke.await_count == 1
    assert loser_graph.ainvoke.await_count == 1
