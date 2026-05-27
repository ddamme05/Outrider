"""Unit tests for the POST /reviews/{review_id}/decide endpoint.

Covers the M12 step-order contract: auth -> state -> mismatch. Also
pins the body-schema rejection rules (no `reviewer_id`, no
`original_severity` in the payload — both are server-derived) and the
failure-wrapper exit semantics.

Integration-style: spins up a FastAPI TestClient with stub
`review_status_reader`, stub `compiled_graph`, and a recorded
`background_tasks.add_task` callable so we can assert the wrapper is
enqueued with the right shape.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003  (runtime: helper signature)
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

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


def _make_preflight(
    *,
    status: str = "awaiting_approval",
    hitl_request: HITLRequest | None = None,
    hitl_decision: object = None,
    gated_severities: Mapping[UUID, FindingSeverity] | None = None,
) -> ReviewDecidePreflight:
    now = datetime.now(UTC)
    default_request = HITLRequest(
        findings_requiring_approval=(_FINDING_A, _FINDING_B),
        auto_post_findings=(),
        created_at=now,
        expires_at=now + timedelta(minutes=30),
    )
    # Explicit None-check so callers passing an empty dict get an empty
    # map (testing the missing-severity branch). `gated_severities or
    # {...}` would coerce the empty dict to the default — silently
    # masking tests that rely on the empty-map behavior.
    return ReviewDecidePreflight(
        status=status,
        hitl_request=hitl_request if hitl_request is not None else default_request,
        hitl_decision=hitl_decision,  # type: ignore[arg-type]
        gated_finding_severities=(
            gated_severities
            if gated_severities is not None
            else {_FINDING_A: FindingSeverity.HIGH, _FINDING_B: FindingSeverity.CRITICAL}
        ),
    )


def _build_app(
    *,
    preflight: ReviewDecidePreflight | None,
    api_key: str = "test-key",
) -> tuple[TestClient, MagicMock, MagicMock]:
    """Build a FastAPI test app + stub reader + stub compiled_graph."""
    app = FastAPI()
    app.include_router(hitl_router)
    app.state.admin_api_key = SecretStr(api_key)

    reader = MagicMock()
    reader.fetch_for_decide = AsyncMock(return_value=preflight)
    app.state.review_status_reader = reader

    compiled_graph = MagicMock()
    compiled_graph.ainvoke = AsyncMock(return_value=None)
    app.state.compiled_graph = compiled_graph

    return TestClient(app), reader, compiled_graph


def _decision_payload(
    *, finding_ids: tuple[UUID, ...] = (_FINDING_A, _FINDING_B)
) -> dict[str, Any]:
    return {
        "decisions": [
            {"finding_id": str(fid), "outcome": "approve", "reason": "ok"} for fid in finding_ids
        ],
        "annotation": "looks good",
    }


def _auth_headers(api_key: str = "test-key") -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Auth gate (M12 step-order: FIRST)
# ---------------------------------------------------------------------------


def test_no_auth_returns_401_even_with_invalid_payload() -> None:
    """M12: auth fires BEFORE state + mismatch. Even an obviously-invalid
    payload (no decisions) gets 401 if auth is missing."""
    client, _, _ = _build_app(preflight=None)
    resp = client.post(f"/reviews/{uuid4()}/decide", json={"decisions": []})
    assert resp.status_code == 401


def test_no_auth_returns_401_even_against_unknown_review() -> None:
    """Calling /decide on a non-existent review without auth -> 401, not 409.
    Prevents an attacker from enumerating review_ids without a token."""
    client, reader, _ = _build_app(preflight=None)
    resp = client.post(f"/reviews/{uuid4()}/decide", json=_decision_payload())
    assert resp.status_code == 401
    # Reader was NOT called — auth short-circuited.
    reader.fetch_for_decide.assert_not_called()


# ---------------------------------------------------------------------------
# State gate (M12 step-order: SECOND, after auth)
# ---------------------------------------------------------------------------


def test_unknown_review_returns_409() -> None:
    """preflight is None -> 409 Conflict."""
    client, _, _ = _build_app(preflight=None)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=_decision_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 409


def test_no_hitl_request_returns_409() -> None:
    """preflight.hitl_request is None -> 409."""
    now = datetime.now(UTC)
    pf = ReviewDecidePreflight(
        status="running",
        hitl_request=None,
        hitl_decision=None,
        gated_finding_severities={},
    )
    _ = now
    client, _, _ = _build_app(preflight=pf)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=_decision_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 409


def test_status_not_in_hitl_set_returns_409() -> None:
    """preflight.status outside {awaiting_approval, awaiting_approval_expired}
    -> 409 even when hitl_request is populated (forged state)."""
    pf = _make_preflight(status="running")
    client, _, _ = _build_app(preflight=pf)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=_decision_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 409


def test_decision_already_landed_returns_409() -> None:
    """preflight.hitl_decision is not None -> 409 (single-shot)."""
    from outrider.schemas.hitl import HITLDecision, PerFindingDecision, PerFindingOutcome

    existing = HITLDecision(
        reviewer_id="admin",
        decisions=(
            PerFindingDecision(
                finding_id=_FINDING_A, outcome=PerFindingOutcome.APPROVE, reason="prior"
            ),
            PerFindingDecision(
                finding_id=_FINDING_B, outcome=PerFindingOutcome.APPROVE, reason="prior"
            ),
        ),
        decided_at=datetime.now(UTC),
    )
    pf = _make_preflight(hitl_decision=existing)
    client, _, _ = _build_app(preflight=pf)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=_decision_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 409


def test_awaiting_approval_expired_is_admitted() -> None:
    """Remediation path: a late decision against an expired review
    progresses past the state gate."""
    pf = _make_preflight(status="awaiting_approval_expired")
    client, _, _ = _build_app(preflight=pf)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=_decision_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Mismatch gate (M12 step-order: THIRD)
# ---------------------------------------------------------------------------


def test_missing_finding_returns_422() -> None:
    """Payload missing a gated finding -> 422 with missing list."""
    pf = _make_preflight()
    client, _, _ = _build_app(preflight=pf)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=_decision_payload(finding_ids=(_FINDING_A,)),
        headers=_auth_headers(),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert str(_FINDING_B) in body["detail"]["missing"]


def test_extra_finding_returns_422() -> None:
    """Payload with a non-gated finding_id -> 422 with extras list."""
    extra = uuid4()
    pf = _make_preflight()
    client, _, _ = _build_app(preflight=pf)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=_decision_payload(finding_ids=(_FINDING_A, _FINDING_B, extra)),
        headers=_auth_headers(),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert str(extra) in body["detail"]["extras"]


def test_duplicate_finding_id_returns_422_not_500() -> None:
    """F3 regression: payload with duplicate finding_id satisfies
    set-equality against the gate, but the endpoint MUST reject it
    BEFORE downstream HITLDecision construction (which would raise
    mid-handler and surface as 500).

    Pre-fix: `[{fid_a}, {fid_a}, {fid_b}]` against gate
    `{fid_a, fid_b}` passed `submitted_ids != expected_ids` (both
    are `{fid_a, fid_b}`), then HITLDecision construction blew up.
    """
    pf = _make_preflight()
    client, _, compiled_graph = _build_app(preflight=pf)
    body = {
        "decisions": [
            {"finding_id": str(_FINDING_A), "outcome": "approve", "reason": "ok-1"},
            {"finding_id": str(_FINDING_A), "outcome": "approve", "reason": "ok-2"},
            {"finding_id": str(_FINDING_B), "outcome": "approve", "reason": "ok-3"},
        ],
        "annotation": None,
    }
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=body,
        headers=_auth_headers(),
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "duplicate_finding_ids" in detail
    assert str(_FINDING_A) in detail["duplicate_finding_ids"]
    # The background task was NOT enqueued — endpoint short-circuited
    # before reaching the resume dispatch.
    compiled_graph.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Body-schema rejection (no reviewer_id, no original_severity)
# ---------------------------------------------------------------------------


def test_reviewer_id_in_payload_is_rejected() -> None:
    """`HITLDecisionPayload.model_config['extra']='forbid'` rejects
    reviewer_id (set server-side from auth context)."""
    pf = _make_preflight()
    client, _, _ = _build_app(preflight=pf)
    body = _decision_payload()
    body["reviewer_id"] = "attacker"
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=body,
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


def test_original_severity_in_payload_is_rejected() -> None:
    """`PerFindingDecisionPayload` forbids `original_severity` — server-
    derives from the preflight map. A reviewer-submitted value would be
    a re-litigation surface for the policy baseline."""
    pf = _make_preflight()
    client, _, _ = _build_app(preflight=pf)
    body = {
        "decisions": [
            {
                "finding_id": str(_FINDING_A),
                "outcome": "approve",
                "reason": "ok",
                "original_severity": "low",  # forbidden
            },
            {"finding_id": str(_FINDING_B), "outcome": "approve", "reason": "ok"},
        ],
        "annotation": None,
    }
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=body,
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Caps (M8)
# ---------------------------------------------------------------------------


def test_annotation_over_2000_chars_returns_422() -> None:
    pf = _make_preflight()
    client, _, _ = _build_app(preflight=pf)
    body = _decision_payload()
    body["annotation"] = "x" * 2001
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=body,
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


def test_empty_decisions_list_returns_422() -> None:
    """min_length=1 on the decisions tuple."""
    pf = _make_preflight()
    client, _, _ = _build_app(preflight=pf)
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json={"decisions": [], "annotation": None},
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Happy path + 202 enqueue
# ---------------------------------------------------------------------------


def test_happy_path_returns_202_and_enqueues_resume() -> None:
    pf = _make_preflight()
    client, _, compiled_graph = _build_app(preflight=pf)
    review_id = uuid4()
    resp = client.post(
        f"/reviews/{review_id}/decide",
        json=_decision_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["review_id"] == str(review_id)
    assert body["status"] == "resuming"
    # BackgroundTasks fired the wrapper, which called graph.ainvoke.
    compiled_graph.ainvoke.assert_awaited_once()


def test_severity_override_uses_preflight_map() -> None:
    """A SEVERITY_OVERRIDE outcome triggers a lookup against
    `preflight.gated_finding_severities[finding_id]` for server-side
    `original_severity`. Tests the lookup happens and produces a
    domain HITLDecision (asserted indirectly via 202)."""
    pf = _make_preflight(
        gated_severities={
            _FINDING_A: FindingSeverity.CRITICAL,
            _FINDING_B: FindingSeverity.CRITICAL,
        }
    )
    client, _, _ = _build_app(preflight=pf)
    body = {
        "decisions": [
            {
                "finding_id": str(_FINDING_A),
                "outcome": "severity_override",
                "reason": "downgrade per context",
                "override_severity": "low",
            },
            {
                "finding_id": str(_FINDING_B),
                "outcome": "approve",
                "reason": "ok",
            },
        ],
        "annotation": None,
    }
    resp = client.post(
        f"/reviews/{uuid4()}/decide",
        json=body,
        headers=_auth_headers(),
    )
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Failure wrapper exit semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_wrapper_absorbs_natural_key_conflict_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Divergent-content NaturalKeyConflict: wrapper logs at WARNING
    level (not INFO) with the explicit diagnostic note, and returns
    gracefully (no re-raise). Status is NOT flipped to failed.

    Per the F2 audit-fold rationale: the wrapper cannot distinguish a
    concurrent-loser case from a window-(f) crash-retry case without
    checkpointer state. Logging at WARNING surfaces the case for
    operator alerting; the sweep's reclaim_stuck_hitl_states is the
    canonical recovery path."""
    import logging

    from outrider.api.dashboard.hitl import _run_resume_under_failure_wrapper
    from outrider.audit.persister import (
        AuditPersisterHITLDecisionNaturalKeyConflict,
    )
    from outrider.schemas.hitl import HITLDecision, PerFindingDecision, PerFindingOutcome

    review_id = uuid4()
    decision = HITLDecision(
        reviewer_id="admin",
        decisions=(
            PerFindingDecision(
                finding_id=_FINDING_A, outcome=PerFindingOutcome.APPROVE, reason="ok"
            ),
        ),
        decided_at=datetime.now(UTC),
    )

    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        side_effect=AuditPersisterHITLDecisionNaturalKeyConflict(
            existing_event_id=uuid4(),
            incoming_event_id=uuid4(),
            review_id=review_id,
            mismatched_fields=(("decisions_content_hash", "abc", "xyz"),),
            natural_key=(("review_id", str(review_id)),),
        )
    )

    with caplog.at_level(logging.WARNING, logger="outrider.api.dashboard.hitl"):
        # No raise — wrapper absorbs cleanly.
        await _run_resume_under_failure_wrapper(
            review_id=review_id,
            hitl_decision=decision,
            graph=graph,
        )

    # Log at WARNING level, not INFO. Operators with WARNING-or-higher
    # alerts see the case.
    matches = [r for r in caplog.records if r.message == "hitl_resume_natural_key_conflict"]
    assert len(matches) == 1, f"expected one WARNING log, got {[r.message for r in caplog.records]}"
    assert matches[0].levelno == logging.WARNING


@pytest.mark.asyncio
async def test_failure_wrapper_re_raises_generic_exception() -> None:
    """A non-NaturalKeyConflict exception logs + re-raises (no status flip)."""
    from outrider.api.dashboard.hitl import _run_resume_under_failure_wrapper
    from outrider.schemas.hitl import HITLDecision, PerFindingDecision, PerFindingOutcome

    review_id = uuid4()
    decision = HITLDecision(
        reviewer_id="admin",
        decisions=(
            PerFindingDecision(
                finding_id=_FINDING_A, outcome=PerFindingOutcome.APPROVE, reason="ok"
            ),
        ),
        decided_at=datetime.now(UTC),
    )

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("transient db error"))

    with pytest.raises(RuntimeError, match="transient"):
        await _run_resume_under_failure_wrapper(
            review_id=review_id,
            hitl_decision=decision,
            graph=graph,
        )
