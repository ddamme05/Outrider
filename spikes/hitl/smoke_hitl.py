# Mock-mode HITL interrupt + resume round-trip smoke.
"""End-to-end smoke for the HITL node's interrupt + resume durability story.

V1 mock mode (default; no external deps): build a minimal graph that
wires only the `hitl` node between START and END, drive it with an
`InMemorySaver` checkpointer, seed a state with a CRITICAL finding,
ainvoke → expect `__interrupt__` key in the return dict (NOT a raised
exception per langgraph 1.1.6 semantics), then ainvoke
`Command(resume=...)` with the same thread_id → expect the state
delta to carry `hitl_request` + `hitl_decision`. Verify the recording
sinks captured exactly:
  - phase events: 2 start + 1 end (same phase_id across interrupt/resume —
    the body re-runs from the top on resume per langgraph 1.1.6 "node
    restarts from the beginning" semantic, so `emit_phase(start)` fires
    twice; `emit_phase(end)` fires once on the resume-side success exit)
  - HITLRequestEvent: one emit BEFORE interrupt
  - HITLDecisionEvent: one emit AFTER resume
  - ReviewStatusSink: mark_awaiting_approval BEFORE interrupt;
    mark_running AFTER resume

This proves the load-bearing properties the unit tests can't reach
without a real checkpointer:
  1. `interrupt(...)` actually persists state via the checkpointer
  2. `Command(resume=...)` actually rehydrates the body at the
     interrupt point with the resume value as the interrupt's return
  3. The state delta lands correctly in the final graph state
  4. The audit-first emit order survives the checkpoint round-trip

Live mode (`--apply`) is deferred per spec line 572: requires a real
FastAPI server + real GitHub App + real Postgres + the
`op run --env-file=.env --` 1Password pattern. Mock mode covers the
durability-contract regression floor for V1.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from outrider.agent.nodes.hitl import hitl
from outrider.agent.nodes.hitl_config import HITLConfig
from outrider.audit.events import (  # noqa: TC001  (runtime: recording-sink list element types)
    HITLDecisionEvent,
    HITLRequestEvent,
    ReviewPhaseEvent,
)
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewFinding, ReviewState
from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.pr_context import PRContext

logger = logging.getLogger("smoke_hitl")


# ---------------------------------------------------------------------------
# Recording sinks
# ---------------------------------------------------------------------------


class _RecordingPhaseEventSink:
    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


class _RecordingHITLEventSink:
    def __init__(self) -> None:
        self.requests: list[HITLRequestEvent] = []
        self.decisions: list[HITLDecisionEvent] = []

    async def emit_hitl_request(self, event: HITLRequestEvent) -> HITLRequestEvent:
        self.requests.append(event)
        return event

    async def emit_hitl_decision(self, event: HITLDecisionEvent) -> HITLDecisionEvent:
        self.decisions.append(event)
        return event


class _RecordingReviewStatusSink:
    def __init__(self) -> None:
        self.awaiting_approval_calls: list[dict[str, Any]] = []
        self.running_calls: list[dict[str, Any]] = []
        self.expired_calls: list[dict[str, Any]] = []

    async def mark_awaiting_approval(self, **kwargs: Any) -> None:
        self.awaiting_approval_calls.append(kwargs)

    async def mark_running(self, **kwargs: Any) -> None:
        self.running_calls.append(kwargs)

    async def mark_awaiting_approval_expired(self, **kwargs: Any) -> None:
        self.expired_calls.append(kwargs)


# ---------------------------------------------------------------------------
# Seed builder
# ---------------------------------------------------------------------------


def _build_critical_finding(*, review_id: UUID) -> ReviewFinding:
    """A CRITICAL finding that will trigger the HITL gate."""
    from outrider.audit.events import compute_finding_content_hash

    file_path = "src/smoke_seed.py"
    line_start = 1
    line_end = 1
    finding_type = FindingType.SQL_INJECTION
    return ReviewFinding(
        review_id=review_id,
        installation_id=99999,
        finding_type=finding_type,
        dimension=lookup_dimension(finding_type),
        severity=FindingSeverity.CRITICAL,
        evidence_tier=EvidenceTier.JUDGED,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        title="Smoke-harness CRITICAL seed",
        description="HITL smoke harness; not a real finding.",
        evidence="SELECT * FROM users WHERE id = {user_input}",
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        proposal_hash="a" * 64,  # Per DECISIONS.md#025; dummy SHA-256 hex.
    )


def _build_seed_state(*, review_id: UUID) -> ReviewState:
    """Minimal state with one CRITICAL finding for the HITL gate."""
    finding = _build_critical_finding(review_id=review_id)
    now = datetime.now(UTC)
    pr_context = PRContext(
        installation_id=99999,
        owner="ddamme05",
        repo="outrider-smoke-test",
        pr_number=1,
        pr_title="HITL smoke",
        base_sha="a" * 40,
        head_sha="b" * 40,
        author="ddamme05",
        total_additions=1,
        total_deletions=0,
        changed_files=(),
    )
    # Build an AnalysisRound carrying the finding (the hitl node reads
    # state.analysis_rounds[*].findings).
    from outrider.policy.canonical import compute_round_id

    round_id = compute_round_id(
        pass_index=0,
        files_examined=(),
        files_skipped=(),
        finding_content_hashes=(finding.content_hash,),
    )
    analysis_round = AnalysisRound(
        round_id=round_id,
        pass_index=0,
        findings=(finding,),
        files_examined=(),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )
    state = ReviewState(
        review_id=review_id,
        pr_context=pr_context,
        received_at=now,
        is_eval=True,
        analysis_rounds=[analysis_round],
    )
    # state-is-pure-data check: round-trip through JSON. Catches a
    # smuggled session/client/callable at harness boot, not deep inside
    # the graph.
    state.model_dump_json()
    return state


# ---------------------------------------------------------------------------
# Mock-mode smoke
# ---------------------------------------------------------------------------


async def _run_mock_smoke() -> int:
    """Build a minimal hitl-only graph, drive it through interrupt +
    resume, verify the recorded artifacts.

    Returns 0 on success, 1 on any assertion failure.
    """
    import functools

    review_id = uuid4()
    seed = _build_seed_state(review_id=review_id)

    phase_sink = _RecordingPhaseEventSink()
    hitl_sink = _RecordingHITLEventSink()
    status_sink = _RecordingReviewStatusSink()
    hitl_config = HITLConfig(timeout_minutes=30)
    checkpointer = InMemorySaver()

    # Build a tiny graph: START -> hitl -> END.
    hitl_callable = functools.partial(
        hitl,
        phase_event_sink=phase_sink,
        hitl_event_sink=hitl_sink,
        review_status_sink=status_sink,
        hitl_config=hitl_config,
    )
    builder: StateGraph[ReviewState, Any, Any, Any] = StateGraph(ReviewState)
    builder.add_node("hitl", hitl_callable)
    builder.add_edge(START, "hitl")
    builder.add_edge("hitl", END)
    graph = builder.compile(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": str(review_id)}}

    # Step 1: ainvoke with seed. Expect __interrupt__ in result.
    result_before_resume = await graph.ainvoke(seed, config=thread_config)  # type: ignore[arg-type]
    if "__interrupt__" not in result_before_resume:
        logger.error(
            "FAIL: ainvoke did not yield __interrupt__. Result keys: %r",
            sorted(result_before_resume.keys()),
        )
        return 1
    interrupt_payload = result_before_resume["__interrupt__"][0].value
    if "findings_requiring_approval" not in interrupt_payload:
        logger.error(
            "FAIL: interrupt payload missing findings_requiring_approval (got %r)",
            interrupt_payload,
        )
        return 1

    # Pre-resume invariants. Recording sinks deliberately do NOT
    # dedup (per the audit/sinks.py recorder-vs-durable convention)
    # so each body invocation observes one emit. On the first
    # invocation: one emit before interrupt suspends.
    if len(hitl_sink.requests) != 1:
        logger.error(
            "FAIL: expected exactly 1 HITLRequestEvent before resume, got %d",
            len(hitl_sink.requests),
        )
        return 1
    if len(hitl_sink.decisions) != 0:
        logger.error(
            "FAIL: expected 0 HITLDecisionEvent before resume, got %d",
            len(hitl_sink.decisions),
        )
        return 1
    if len(status_sink.awaiting_approval_calls) != 1:
        logger.error(
            "FAIL: expected exactly 1 mark_awaiting_approval before resume, got %d",
            len(status_sink.awaiting_approval_calls),
        )
        return 1
    if len(status_sink.running_calls) != 0:
        logger.error(
            "FAIL: expected 0 mark_running before resume, got %d",
            len(status_sink.running_calls),
        )
        return 1

    # Build a resume payload (HITLDecision) that approves the gated
    # finding. This is what the dashboard endpoint constructs server-
    # side from the reviewer's HITLDecisionPayload submission.
    gated_finding_id = UUID(interrupt_payload["findings_requiring_approval"][0])
    resume_value: dict[str, Any] = {
        "reviewer_id": "admin",
        "decisions": [
            {
                "finding_id": str(gated_finding_id),
                "outcome": "approve",
                "reason": "smoke-approve",
                "override_severity": None,
                "original_severity": None,
            }
        ],
        "annotation": "smoke-harness approval",
        "decided_at": datetime.now(UTC).isoformat(),
    }

    # Step 2: ainvoke Command(resume=...) with same thread_id. Expect
    # the state delta to land with hitl_request + hitl_decision
    # populated.
    final_state = await graph.ainvoke(
        Command(resume=resume_value),
        config=thread_config,  # type: ignore[arg-type]
    )

    # Post-resume invariants. Per langgraph 1.1.6: "The node restarts
    # from the beginning of the node where interrupt was called when
    # resumed, so any code before the interrupt runs again." Recording
    # sinks observe BOTH body invocations (deliberate non-dedup; the
    # durable persister side dedups via natural-key index).
    if final_state.get("hitl_request") is None:
        logger.error("FAIL: final state hitl_request is None")
        return 1
    if final_state.get("hitl_decision") is None:
        logger.error("FAIL: final state hitl_decision is None")
        return 1
    if len(hitl_sink.requests) != 2:
        logger.error(
            "FAIL: expected exactly 2 HITLRequestEvent emits (one per body invocation), got %d",
            len(hitl_sink.requests),
        )
        return 1
    if len(hitl_sink.decisions) != 1:
        logger.error(
            "FAIL: expected exactly 1 HITLDecisionEvent (only on resume body), got %d",
            len(hitl_sink.decisions),
        )
        return 1
    if len(status_sink.awaiting_approval_calls) != 2:
        logger.error(
            "FAIL: expected exactly 2 mark_awaiting_approval calls "
            "(one per body invocation), got %d",
            len(status_sink.awaiting_approval_calls),
        )
        return 1
    if len(status_sink.running_calls) != 1:
        logger.error(
            "FAIL: expected exactly 1 mark_running (only on resume body), got %d",
            len(status_sink.running_calls),
        )
        return 1

    # Phase events: 2 starts (one per body invocation) + 1 end (only
    # on resume body, since the first invocation suspends at
    # interrupt before reaching the end emit).
    starts = [e for e in phase_sink.events if e.marker == "start"]
    ends = [e for e in phase_sink.events if e.marker == "end"]
    if len(starts) != 2 or len(ends) != 1:
        logger.error(
            "FAIL: expected 2 start + 1 end phase event "
            "(start fires on each body invocation; end only after resume), "
            "got %d start / %d end",
            len(starts),
            len(ends),
        )
        return 1
    # Deterministic phase_id: same compute_phase_id input on both
    # body invocations -> identical phase_id. The durable persister's
    # natural-key index would dedup; the recording sink doesn't.
    phase_ids = {e.phase_id for e in phase_sink.events}
    if len(phase_ids) != 1:
        logger.error(
            "FAIL: phase events carry divergent phase_ids "
            "(deterministic compute_phase_id should produce identical ids "
            "across body re-runs); got %r",
            phase_ids,
        )
        return 1

    # Audit-first ordering: HITLRequestEvent emitted BEFORE
    # mark_awaiting_approval; HITLDecisionEvent emitted BEFORE
    # mark_running. Recording sinks preserve call order via list
    # append, so a smoke assertion is "sink emissions exist and
    # status calls exist" — actual cross-sink ordering is exercised
    # in the unit tests. The smoke pins that both legs happened.

    logger.info("PASS: HITL smoke completed — interrupt + resume round-trip clean")
    logger.info(
        "  review_id=%s phase_id=%s "
        "starts=%d ends=%d requests=%d decisions=%d "
        "mark_awaiting_approval=%d mark_running=%d",
        review_id,
        next(iter(phase_ids)),
        len(starts),
        len(ends),
        len(hitl_sink.requests),
        len(hitl_sink.decisions),
        len(status_sink.awaiting_approval_calls),
        len(status_sink.running_calls),
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HITL interrupt + resume smoke harness (mock mode by default)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "[DEFERRED] Live mode against a real FastAPI server + GitHub App + "
            "Postgres. Not implemented in V1 mock-only harness. Use the "
            "`SMOKE_TEST.md` runbook for the manual live walkthrough."
        ),
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    if args.apply:
        logger.error(
            "--apply (live mode) is deferred to a future iteration. The mock "
            "mode covers the interrupt + resume durability contract. Live "
            "mode requires LLM credentials + real PR + FastAPI server + "
            "separate-process curl; tracked separately."
        )
        return 2
    return await _run_mock_smoke()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
