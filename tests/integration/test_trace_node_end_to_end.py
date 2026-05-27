"""Trace node end-to-end integration — controlled fakes + real persister.

Per `specs/2026-05-23-trace-node.md` M7 + M8: the unit tests
(`tests/unit/test_trace_node.py`) pin individual contracts (join
integrity, bucket build, probe-path construction). The integration
tests in `test_audit_persister_natural_key.py` pin the persister
side. Neither exercises the FULL trace node body through Phase 1
probes + audit-first emission + Phase 2 fetch + state delta
assembly. This file closes that gap with controlled GitHub fakes +
mock LLM provider + the real AuditPersister against a real Postgres.

Coverage:

  - **Resolved path (Phase 1 + Phase 2):** one candidate, one probe
    returns content, Phase 2 fetches the resolved target, the state
    delta carries the TraceDecision + TraceFetchedFile, the audit
    row was actually written.
  - **Unresolved path (Phase 1 only):** one candidate, both probes
    return None, state delta carries the TraceDecision with
    `resolution_status="unresolved"`, NO TraceFetchedFile lands
    (M8 invariant: probes do NOT populate trace_fetched_files).
  - **Target-in-PR-files skip (M8):** Phase 1 resolves the target,
    but `target_file` IS in `pr_context.changed_files` → decision
    emitted, Phase 2 skipped, NO TraceFetchedFile.
  - **Audit-first lockstep contract (M7 b):** node-side composition
    is exercised here (trace constructs `TraceDecision` from the
    persister's returned event), while the persister's no-op-on-
    retry behavior (existing-row return on natural-key conflict) is
    pinned in `tests/integration/test_audit_persister_natural_key.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest  # noqa: TC002 — used at runtime as parameter type

from outrider.agent.nodes import trace as trace_module
from outrider.agent.nodes.trace import trace
from outrider.audit.events import compute_finding_content_hash
from outrider.llm.base import LLMResponse
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import (
    compute_candidate_id,
    compute_round_id,
)
from outrider.schemas import (
    AnalysisRound,
    ReviewDimension,
    ReviewFinding,
    ReviewState,
    TraceCandidate,
    TraceFetchedFile,
)
from outrider.schemas.pr_context import ChangedFile, PRContext

if TYPE_CHECKING:
    from tests.integration.conftest import PersisterTestSetup

    from outrider.llm.base import LLMRequest


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


class _MockLLMProvider:
    """Provider that returns a canned ranking for the trace node call.

    `node_id="trace"` is the only allowed route — unknown node_ids
    raise so a future test invoking the provider for a different node
    surfaces the mismatch loud, not silent.

    `ranked_candidate_ids` is constructed at fixture-build time so the
    response matches whatever candidates the test passed in state.
    """

    def __init__(self, ranked_candidate_ids: tuple[str, ...]) -> None:
        self._ranked_candidate_ids = ranked_candidate_ids
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if request.node_id != "trace":
            msg = f"_MockLLMProvider: unexpected node_id {request.node_id!r}"
            raise AssertionError(msg)
        text = json.dumps({"ranked_candidate_ids": list(self._ranked_candidate_ids)})
        return LLMResponse(
            text=text,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=42,
        )


def _build_finding(
    *,
    review_id: object,
    proposal_hash: str,
    file_path: str = "src/app.py",
) -> ReviewFinding:
    """Build a ReviewFinding fixture matching the test scenario."""
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=review_id,  # type: ignore[arg-type]
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path=file_path,
        line_start=10,
        line_end=12,
        title="SQL injection",
        description="raw concat",
        evidence=f"concat at {file_path}:11",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version="1.0.0",
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash=proposal_hash,
    )


def _build_candidate(
    *,
    source_proposal_hash: str,
    import_string: str,
) -> TraceCandidate:
    reason = "ranked candidate"
    return TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=source_proposal_hash,
            import_string=import_string,
            reason=reason,
        ),
        source_proposal_hash=source_proposal_hash,
        reason=reason,
        import_string=import_string,
    )


def _build_state(
    *,
    review_id: object,
    finding: ReviewFinding,
    candidate: TraceCandidate,
    pr_changed_files: tuple[ChangedFile, ...] = (),
) -> ReviewState:
    """Assemble a ReviewState with one finding + one candidate."""
    now = datetime.now(UTC)
    round_id = compute_round_id(
        pass_index=0,
        files_examined=(finding.file_path,),
        files_skipped=(),
        finding_content_hashes=(finding.content_hash,),
    )
    analysis_round = AnalysisRound(
        round_id=round_id,
        pass_index=0,
        findings=(finding,),
        files_examined=(finding.file_path,),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )
    return ReviewState(
        review_id=review_id,  # type: ignore[arg-type]
        pr_context=PRContext(
            installation_id=12345,
            owner="o",
            repo="r",
            pr_number=1,
            pr_title="x",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="dev",
            total_additions=5,
            total_deletions=2,
            changed_files=pr_changed_files,
        ),
        received_at=now,
        analysis_rounds=[analysis_round],
        trace_candidates=[candidate],
    )


def _stub_github_factory(_installation_id: int) -> object:
    """Returns a sentinel — trace.py passes this opaquely to
    fetch_file_content_at, which we monkeypatch per-test."""
    return object()


# ---------------------------------------------------------------------------
# Resolved path: Phase 1 + Phase 2 both succeed.
# ---------------------------------------------------------------------------


async def test_resolved_candidate_writes_decision_and_fetched_file(
    persister_setup: PersisterTestSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end happy path: one candidate, Phase 1 resolves to one
    target file (out of the two probed module + package paths), Phase 2
    fetches the resolved file. Verify the audit row was written AND
    the state delta carries both a TraceDecision and a TraceFetchedFile.
    """
    review_id = persister_setup.review_id
    proposal_hash = "a" * 64
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="middleware.auth",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)

    # Mock fetch_file_content_at: module form resolves, package form
    # doesn't (404-equivalent → None). Phase 2 then re-fetches the
    # resolved target.
    resolved_path = "middleware/auth.py"
    unresolved_path = "middleware/auth/__init__.py"
    file_content = b"def authenticate(token: str) -> bool:\n    return True\n"
    call_log: list[str] = []

    async def fake_fetch(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        call_log.append(path)
        if path == resolved_path:
            return file_content
        if path == unresolved_path:
            return None
        msg = f"unexpected fetch path: {path!r}"
        raise AssertionError(msg)

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)

    provider = _MockLLMProvider(ranked_candidate_ids=(candidate.candidate_id,))

    state_delta = await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    # Contract: Phase 1 probed BOTH candidate paths (order is
    # implementation choice — V1.5 may parallelize probes), then
    # Phase 2 re-fetched ONLY the resolved one. Assert as set
    # semantics on Phase 1 + ordered single-element Phase 2 so the
    # test survives a parallel-probes refactor.
    assert set(call_log[:2]) == {resolved_path, unresolved_path}
    assert call_log[2:] == [resolved_path]

    # State delta carries one decision + one fetched file.
    decisions = state_delta["trace_decisions"]
    fetched_files = state_delta["trace_fetched_files"]
    assert len(decisions) == 1  # type: ignore[arg-type]
    assert len(fetched_files) == 1  # type: ignore[arg-type]
    # Per-invocation scalar: `_trace_router` reads this (NOT
    # cumulative `len(trace_fetched_files)`). Pin it here so a future
    # regression in trace's delta-construction can't silently break
    # router routing.
    assert state_delta["last_trace_pass_fetched_count"] == 1

    decision = decisions[0]  # type: ignore[index]
    assert decision.source_finding_id == finding.finding_id
    assert decision.resolution_status == "resolved"
    assert decision.target_file == resolved_path
    assert decision.resolved_candidate_paths == (resolved_path,)

    fetched = fetched_files[0]  # type: ignore[index]
    assert fetched.path == resolved_path
    assert fetched.content_head == file_content.decode("utf-8")
    assert fetched.source_finding_id == finding.finding_id

    # Audit-completeness link per M7 (data-integrity G1 fold):
    # the state delta contains a TraceDecision iff the audit_events
    # row was actually written. A future refactor where the sink
    # accepted-and-discarded would pass every state-delta assertion
    # above; this SELECT closes that gap by reading the audit table
    # directly.
    import sqlalchemy as sa

    async with persister_setup.engine.connect() as conn:
        result = await conn.execute(
            sa.text(
                "SELECT count(*)::int FROM audit_events "
                "WHERE review_id = :rid "
                "  AND event_type = 'trace_decision' "
                "  AND payload->>'source_finding_id' = :sfid"
            ),
            {"rid": str(review_id), "sfid": str(finding.finding_id)},
        )
        audit_row_count = result.scalar_one()
    assert audit_row_count == 1, (
        f"expected exactly one trace_decision audit row for finding "
        f"{finding.finding_id}; got {audit_row_count}"
    )


# ---------------------------------------------------------------------------
# Unresolved path: Phase 1 only; M8 invariant — no TraceFetchedFile.
# ---------------------------------------------------------------------------


async def test_unresolved_candidate_emits_decision_without_fetched_file(
    persister_setup: PersisterTestSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M8 invariant: when Phase 1 probes return None for every candidate
    path, the decision is `resolution_status="unresolved"` AND no
    Phase 2 fetch fires AND no TraceFetchedFile lands in state."""
    review_id = persister_setup.review_id
    proposal_hash = "b" * 64
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="missing.module",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)

    call_log: list[str] = []

    async def fake_fetch(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        call_log.append(path)
        return None  # Every probe returns "does not exist"

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)

    provider = _MockLLMProvider(ranked_candidate_ids=(candidate.candidate_id,))

    state_delta = await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    # Phase 1 probed both candidate paths; Phase 2 did NOT fire
    # (unresolved → no target_file to fetch). Set semantics on the
    # probe pair survives a parallel-probes refactor.
    assert set(call_log) == {"missing/module.py", "missing/module/__init__.py"}
    assert len(call_log) == 2  # Phase 2 didn't fire

    decisions = state_delta["trace_decisions"]
    fetched_files = state_delta["trace_fetched_files"]
    assert len(decisions) == 1  # type: ignore[arg-type]
    assert len(fetched_files) == 0  # type: ignore[arg-type]  # M8: no probe → no fetched file
    # Router-signal scalar: zero new fetches this pass → router routes
    # to publish on the next graph step.
    assert state_delta["last_trace_pass_fetched_count"] == 0

    decision = decisions[0]  # type: ignore[index]
    assert decision.resolution_status == "unresolved"
    assert decision.target_file is None
    assert decision.resolved_candidate_paths == ()


# ---------------------------------------------------------------------------
# Probe-side HTTP error handling: 404 admitted as "did not resolve",
# other errors propagate per M8 transient semantics.
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stub mimicking httpx-style response shape."""

    def __init__(self, *, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeRequestFailed(Exception):  # noqa: N818  (mirrors githubkit's name shape)
    """Stand-in for githubkit's RequestFailed exception shape — same
    pattern as `tests/unit/test_github_publisher.py:_FakeRequestFailed`."""

    def __init__(self, *, status_code: int, text: str = "") -> None:
        super().__init__(f"HTTP {status_code}: {text[:50]}")
        self.response = _FakeResponse(status_code=status_code, text=text)


async def test_probe_404_admits_as_unresolved(
    persister_setup: PersisterTestSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the M8 contract per `_resolve_via_probes` docstring: a 404
    from the GitHub fetch during Phase 1 is the COMMON probe outcome
    (LLM proposed a path that doesn't exist) and must be admitted as
    "candidate did not resolve" — NOT propagated as an exception that
    aborts the trace pass."""
    review_id = persister_setup.review_id
    proposal_hash = "f0" * 32
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="ghost.module",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)

    async def fetch_raises_404(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        del path
        raise _FakeRequestFailed(status_code=404, text="Not Found")

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fetch_raises_404)

    provider = _MockLLMProvider(ranked_candidate_ids=(candidate.candidate_id,))

    state_delta = await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    decisions = state_delta["trace_decisions"]
    assert len(decisions) == 1  # type: ignore[arg-type]
    decision = decisions[0]  # type: ignore[index]
    assert decision.resolution_status == "unresolved"
    assert decision.target_file is None
    assert state_delta["trace_fetched_files"] == []


async def test_probe_500_propagates(
    persister_setup: PersisterTestSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Companion contract: 5xx errors from GitHub during Phase 1 are
    NOT routine probe outcomes — they signal upstream issues. Per the
    `_resolve_via_probes` docstring "Probe failures (non-404 errors)
    propagate per M8 transient semantics," they must abort the trace
    pass rather than silently dropping the candidate."""
    review_id = persister_setup.review_id
    proposal_hash = "f1" * 32
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="some.module",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)

    async def fetch_raises_500(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        del path
        raise _FakeRequestFailed(status_code=500, text="Internal Server Error")

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fetch_raises_500)

    provider = _MockLLMProvider(ranked_candidate_ids=(candidate.candidate_id,))

    with pytest.raises(_FakeRequestFailed):
        await trace(
            state,
            provider=provider,  # type: ignore[arg-type]
            trace_model="claude-haiku-test",
            phase_event_sink=persister_setup.persister,
            trace_sink=persister_setup.persister,
            github_factory=_stub_github_factory,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Target-in-PR-files: Phase 1 resolves, but Phase 2 skipped per M8.
# ---------------------------------------------------------------------------


async def test_resolved_but_target_in_pr_files_skips_phase_two_fetch(
    persister_setup: PersisterTestSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per Q3 + M8: when Phase 1 resolves to a target that's already in
    `pr_context.changed_files`, the decision is emitted (status=resolved)
    but Phase 2 skips the content fetch — analyze already sees the file
    via PR diff. No TraceFetchedFile lands."""
    review_id = persister_setup.review_id
    proposal_hash = "c" * 64
    target_path = "middleware/auth.py"
    finding = _build_finding(
        review_id=review_id, proposal_hash=proposal_hash, file_path="src/other.py"
    )
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="middleware.auth",
    )
    pr_file = ChangedFile(
        path=target_path,
        status="modified",
        additions=1,
        deletions=0,
        previous_path=None,
        patch="@@ -1 +1 @@\n-old\n+new\n",
        content_base="old\n",
        content_head="new\n",
    )
    state = _build_state(
        review_id=review_id,
        finding=finding,
        candidate=candidate,
        pr_changed_files=(pr_file,),
    )

    call_log: list[str] = []

    async def fake_fetch(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        call_log.append(path)
        # Module form resolves; package form doesn't.
        if path == target_path:
            return b"def authenticate(): ..."
        return None

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)

    provider = _MockLLMProvider(ranked_candidate_ids=(candidate.candidate_id,))

    state_delta = await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    # Phase 1 probed both paths; Phase 2 did NOT fire (target in PR
    # files — already analyzed at pass 0, no re-fetch needed). Set
    # semantics on the probe pair survives a parallel-probes refactor.
    assert set(call_log) == {target_path, "middleware/auth/__init__.py"}
    assert len(call_log) == 2  # Phase 2 didn't fire

    decisions = state_delta["trace_decisions"]
    fetched_files = state_delta["trace_fetched_files"]
    assert len(decisions) == 1  # type: ignore[arg-type]
    assert len(fetched_files) == 0  # type: ignore[arg-type]  # Q3 + M8 skip
    # Router-signal scalar: target is in pr_context.changed_files, so
    # Phase 2 didn't fire → zero new fetched files this pass → router
    # routes to publish (no analyze re-entry on already-in-diff files).
    assert state_delta["last_trace_pass_fetched_count"] == 0

    decision = decisions[0]  # type: ignore[index]
    assert decision.resolution_status == "resolved"
    assert decision.target_file == target_path


# ---------------------------------------------------------------------------
# Audit-first lockstep on retry (M7 b).
# ---------------------------------------------------------------------------


# M7 (b) lockstep-recovery contract — verified at the persister tier
# (`test_audit_persister_natural_key.py::test_no_op_path_returns_existing_
# event_when_identity_subset_matches`) which exercises the
# persister-side no-op path with diverging incoming per-emission fields
# (different event_id, timestamp, reason, proposed_import_strings) and
# asserts the returned event carries the FIRST call's fields. The
# trace node itself is a thin composition over that pinned contract:
# trace.py:299-308 builds the state-layer TraceDecision from
# `persisted_event.*`, never from `decision_event.*`. The node-tier
# end-to-end test that previously sat here was vacuous (deterministic
# mock LLM made first-vs-second incoming events identical, so the
# lockstep assertion was satisfied trivially regardless of contract
# correctness). Removed 2026-05-24 per multi-lens audit convergence
# (sharp-edges H-A + data-integrity G2). The structural M7(b)
# composition over the persister contract is documented in trace.py's
# docstring + the `# Build state-layer TraceDecision from the RETURNED
# event` comment at the call site.


# ---------------------------------------------------------------------------
# Graph routing assertions (per Codex review #8): trace_router decides
# next-node based on trace_fetched_files + round count.
# ---------------------------------------------------------------------------


def test_trace_router_routes_to_analyze_when_fetched_files_non_empty() -> None:
    """`_trace_router` returns 'analyze' when trace produced at least one
    NEW fetched file IN THE MOST RECENT trace() CALL AND we're below the
    depth-2 round limit. This is the inbound side of the adaptive
    analyze ⇄ trace loop — the routing decision the trace node's state
    delta drives.

    Reads `state.last_trace_pass_fetched_count` (the per-invocation
    scalar trace() writes per CodeRabbit R1), NOT the cumulative
    `len(state.trace_fetched_files)`. Test sets the scalar to 1 to
    simulate "trace just produced one new fetch this pass."
    """
    from outrider.agent.graph import _trace_router

    review_id = uuid4()
    proposal_hash = "a" * 64
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="some.module",
    )
    # State after a successful trace pass: analysis_rounds=[round_1],
    # trace_fetched_files=[one file], last_trace_pass_fetched_count=1.
    # Router should send back to analyze.
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)
    fetched = TraceFetchedFile(
        path="some/module.py",
        content_head="x = 1\n",
        source_finding_id=finding.finding_id,
    )
    state_with_fetch = state.model_copy(
        update={
            "trace_fetched_files": [fetched],
            "last_trace_pass_fetched_count": 1,
        }
    )

    assert _trace_router(state_with_fetch) == "analyze"


def test_trace_router_routes_to_hitl_when_no_new_fetches_this_pass() -> None:
    """Trace pass produced no NEW fetches this invocation; router sends
    to `hitl` (next non-trace destination in the 7-node topology;
    `hitl` then either pass-through to publish OR interrupts on a
    CRITICAL/HIGH finding). Per the router contract: the per-invocation
    scalar `last_trace_pass_fetched_count` is the signal, NOT the
    cumulative `state.trace_fetched_files`.

    Seeds a STALE TraceFetchedFile from a prior pass while keeping
    `last_trace_pass_fetched_count=0` so the test fails if a future
    regression makes `_trace_router` consult cumulative state again.
    Without the stale seed, both signals would be zero — the test
    would pass even under the wrong (cumulative) implementation."""
    from outrider.agent.graph import _trace_router

    review_id = uuid4()
    proposal_hash = "b" * 64
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="missing.module",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)
    stale_fetch = TraceFetchedFile(
        path="already/fetched.py",
        content_head="x = 1\n",
        source_finding_id=finding.finding_id,
    )
    state_with_stale_fetch = state.model_copy(
        update={
            "trace_fetched_files": [stale_fetch],
            "last_trace_pass_fetched_count": 0,
        }
    )
    # Cumulative list non-empty AND per-invocation scalar zero —
    # router must read the scalar and route to `hitl` (the next non-
    # trace destination per the HITL spec Group 5 topology).
    assert len(state_with_stale_fetch.trace_fetched_files) == 1
    assert state_with_stale_fetch.last_trace_pass_fetched_count == 0

    # Routes through `hitl` (always next non-trace destination from
    # trace per Group 5 of the HITL spec); pass-through if no gated
    # findings, then publish runs.
    assert _trace_router(state_with_stale_fetch) == "hitl"


def test_trace_router_routes_to_hitl_at_max_rounds() -> None:
    """Depth-2 ceiling: even when the most recent trace() call yielded
    new fetches, if analysis_rounds has already reached
    MAX_ANALYSIS_ROUNDS, route to `hitl` to bound the loop's total
    wall-clock cost (`hitl` then routes to publish for non-gated
    findings; the depth gate fires at the trace→next-node boundary).
    Sets `last_trace_pass_fetched_count=1` to ensure the depth gate
    (not the scalar gate) is what blocks routing to analyze — without
    that, the test would pass for the wrong reason."""
    from outrider.agent.graph import _trace_router
    from outrider.agent.nodes.trace import MAX_ANALYSIS_ROUNDS
    from outrider.schemas import TraceFetchedFile

    review_id = uuid4()
    proposal_hash = "c" * 64
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="some.module",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)
    # Build extra rounds up to the ceiling.
    extra_round = state.analysis_rounds[0].model_copy(
        update={"round_id": "d" * 64, "pass_index": 1}
    )
    fetched = TraceFetchedFile(
        path="some/module.py",
        content_head="x = 1\n",
        source_finding_id=finding.finding_id,
    )
    state_at_ceiling = state.model_copy(
        update={
            "analysis_rounds": [state.analysis_rounds[0], extra_round],
            "trace_fetched_files": [fetched],
            "last_trace_pass_fetched_count": 1,
        }
    )
    assert len(state_at_ceiling.analysis_rounds) == MAX_ANALYSIS_ROUNDS
    assert state_at_ceiling.last_trace_pass_fetched_count > 0  # scalar gate would pass
    assert _trace_router(state_at_ceiling) == "hitl"  # depth gate fires
