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
from outrider.audit.events import TraceDecisionEvent, compute_finding_content_hash
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
    TraceDecision,
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

    async def aclose(self) -> None:
        return None

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


async def _stub_github_factory(_installation_id: int) -> object:
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

    # Phase 1 probed both tiers — the module-form pair missed, so the
    # symbol-form parent fallback probed too (FUP-209) — and Phase 2 did
    # NOT fire (unresolved → no target_file to fetch). Set semantics on
    # the probe paths survives a parallel-probes refactor.
    assert set(call_log) == {
        "missing/module.py",
        "missing/module/__init__.py",
        "missing.py",
        "missing/__init__.py",
    }
    assert len(call_log) == 4  # Phase 2 didn't fire

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

    # Phase 1 served the module-form path from the probe memo (seeded
    # with the PR file's head content — zero GitHub round-trips for
    # in-PR paths per FUP-209 review) and fetch-probed only the
    # package-form sibling; Phase 2 did NOT fire (target in PR files —
    # already analyzed at pass 0, no re-fetch needed).
    assert call_log == ["middleware/auth/__init__.py"]

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
# Shared pass fetch budget: Phase 2 draws from the same ceiling as
# Phase 1 probes (FUP-209 review F2 — a probes-only budget left content
# fetches uncapped and the stated rate-limit ceiling wasn't real).
# ---------------------------------------------------------------------------


async def test_phase_two_fetch_skipped_when_pass_budget_exhausted(
    persister_setup: PersisterTestSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the pass budget sized to exactly the Phase 1 probes, the
    resolved decision still lands (audit unchanged) but the Phase 2
    content fetch is skipped: no TraceFetchedFile, zero router signal —
    the same graceful unfunded degradation Phase 1 uses."""
    review_id = persister_setup.review_id
    proposal_hash = "e" * 64
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="middleware.auth",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)

    call_log: list[str] = []

    async def fake_fetch(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        call_log.append(path)
        if path == "middleware/auth.py":
            return b"def authenticate(): ..."
        return None

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)
    # Exactly the two level-0 probe fetches — nothing left for Phase 2.
    monkeypatch.setattr(trace_module, "MAX_PROBE_FETCHES_PER_PASS", 2)

    provider = _MockLLMProvider(ranked_candidate_ids=(candidate.candidate_id,))

    state_delta = await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    # Both budget units went to Phase 1; Phase 2 never fetched.
    assert call_log == ["middleware/auth.py", "middleware/auth/__init__.py"]

    decisions = state_delta["trace_decisions"]
    assert len(decisions) == 1  # type: ignore[arg-type]
    decision = decisions[0]  # type: ignore[index]
    assert decision.resolution_status == "resolved"
    assert decision.target_file == "middleware/auth.py"

    fetched_files = state_delta["trace_fetched_files"]
    assert len(fetched_files) == 0  # type: ignore[arg-type]
    assert state_delta["last_trace_pass_fetched_count"] == 0


# ---------------------------------------------------------------------------
# Crash-resume recovery: audit-ahead-of-state adoption (FUP-209 review).
# ---------------------------------------------------------------------------


async def test_persisted_decision_adopted_on_resume_without_reprobe(
    persister_setup: PersisterTestSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decision row persisted by a run that crashed after emit but
    before its state delta merged is ADOPTED on the re-run: no Phase 1
    re-probe (whose outcome could differ under newer resolver semantics
    and trip the natural-key identity guard, aborting the resumed
    review), the adopted decision lands in the state delta (lockstep
    repair), and Phase 2 still fetches a resolved target so the lost
    fetched-file state repairs too."""
    review_id = persister_setup.review_id
    proposal_hash = "d" * 64
    finding = _build_finding(review_id=review_id, proposal_hash=proposal_hash)
    candidate = _build_candidate(
        source_proposal_hash=proposal_hash,
        import_string="middleware.auth",
    )
    state = _build_state(review_id=review_id, finding=finding, candidate=candidate)

    # The pre-crash run persisted this decision; state never learned it
    # (state.trace_decisions is empty, so the step-5 gate can't skip).
    prior_event = TraceDecisionEvent(
        review_id=review_id,
        is_eval=False,
        source_finding_id=finding.finding_id,
        target_file="middleware/auth.py",
        reason="pre-crash reason",
        resolution_status="resolved",
        proposed_import_strings=("middleware.auth",),
        resolved_candidate_paths=("middleware/auth.py",),
    )
    await persister_setup.persister.emit_trace_decision(prior_event)

    call_log: list[str] = []

    async def fake_fetch(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        call_log.append(path)
        if path == "middleware/auth.py":
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

    # No Phase 1 probes fired — the persisted decision was adopted. The
    # single fetch is Phase 2's content fetch of the resolved target.
    assert call_log == ["middleware/auth.py"]

    decisions = state_delta["trace_decisions"]
    assert len(decisions) == 1  # type: ignore[arg-type]
    decision = decisions[0]  # type: ignore[index]
    assert decision.resolution_status == "resolved"
    assert decision.target_file == "middleware/auth.py"
    # Adopted verbatim from the persisted row, not re-derived.
    assert decision.reason == "pre-crash reason"

    fetched_files = state_delta["trace_fetched_files"]
    assert len(fetched_files) == 1  # type: ignore[arg-type]
    assert state_delta["last_trace_pass_fetched_count"] == 1


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


def test_trace_router_routes_to_synthesize_when_no_new_fetches_this_pass() -> None:
    """Trace pass produced no NEW fetches this invocation; router sends
    to `synthesize` (next non-trace destination in the 7-node topology;
    synthesize aggregates findings into ReviewReport, then HITL gates,
    then publish runs). Per the router contract: the per-invocation
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
    # router must read the scalar and route to `synthesize` (the next
    # non-trace destination per the 7-node topology; synthesize folds
    # findings, HITL gates, then publish).
    assert len(state_with_stale_fetch.trace_fetched_files) == 1
    assert state_with_stale_fetch.last_trace_pass_fetched_count == 0

    assert _trace_router(state_with_stale_fetch) == "synthesize"


def test_trace_router_routes_to_synthesize_at_max_rounds() -> None:
    """Depth-2 ceiling: even when the most recent trace() call yielded
    new fetches, if analysis_rounds has already reached
    MAX_ANALYSIS_ROUNDS, route to `synthesize` to bound the loop's
    total wall-clock cost (synthesize folds findings, then HITL gates,
    then publish; the depth gate fires at the trace→next-node
    boundary). Sets `last_trace_pass_fetched_count=1` to ensure the
    depth gate (not the scalar gate) is what blocks routing to
    analyze — without that, the test would pass for the wrong reason."""
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
    assert _trace_router(state_at_ceiling) == "synthesize"  # depth gate fires


# ---------------------------------------------------------------------------
# Candidate selection through the real trace(): excluding findings already
# decided, bounding unique imports, and ordering candidates for ranking
# ---------------------------------------------------------------------------


def _build_multi_state(
    *,
    review_id: object,
    findings: tuple[ReviewFinding, ...],
    candidates: tuple[TraceCandidate, ...],
    prior_decisions: tuple[TraceDecision, ...] = (),
) -> ReviewState:
    """A state carrying several findings and their candidates.

    `_build_state` above holds exactly one finding and one candidate; selection
    behavior that depends on more than one of either needs this builder.
    """
    now = datetime.now(UTC)
    analysis_round = AnalysisRound(
        round_id=compute_round_id(
            pass_index=0,
            files_examined=tuple(f.file_path for f in findings),
            files_skipped=(),
            finding_content_hashes=tuple(f.content_hash for f in findings),
        ),
        pass_index=0,
        findings=findings,
        files_examined=tuple(f.file_path for f in findings),
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
            changed_files=(),
        ),
        received_at=now,
        analysis_rounds=[analysis_round],
        trace_candidates=list(candidates),
        trace_decisions=list(prior_decisions),
    )


@pytest.mark.asyncio
async def test_already_traced_finding_is_excluded_from_selection(
    persister_setup: PersisterTestSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finding whose decision is already in state is not decided again.

    Within-graph re-entry idempotency (`DECISIONS.md#025` point 5): a later pass
    must leave a finding the earlier pass already decided alone, while still
    deciding the ones it has not seen.
    """
    review_id = persister_setup.review_id
    traced = _build_finding(review_id=review_id, proposal_hash="a" * 64, file_path="src/a.py")
    fresh = _build_finding(review_id=review_id, proposal_hash="b" * 64, file_path="src/b.py")
    cand_traced = _build_candidate(source_proposal_hash="a" * 64, import_string="pkg.already")
    cand_fresh = _build_candidate(source_proposal_hash="b" * 64, import_string="pkg.fresh")
    prior = TraceDecision(
        source_finding_id=traced.finding_id,
        target_file="src/already.py",
        reason="decided on the previous pass",
        resolution_status="resolved",
        proposed_import_strings=("pkg.already",),
        resolved_candidate_paths=("src/already.py",),
    )
    state = _build_multi_state(
        review_id=review_id,
        findings=(traced, fresh),
        candidates=(cand_traced, cand_fresh),
        prior_decisions=(prior,),
    )

    async def fake_fetch(*_args: object, **_kwargs: object) -> bytes | None:
        return None  # every probe misses; resolution is not what this pins

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)
    provider = _MockLLMProvider(ranked_candidate_ids=(cand_fresh.candidate_id,))

    state_delta = await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    decided = {d.source_finding_id for d in state_delta.get("trace_decisions", [])}
    assert fresh.finding_id in decided, "the un-traced finding must still be decided"
    assert traced.finding_id not in decided, (
        "a finding already carrying a TraceDecision must not be re-decided"
    )


@pytest.mark.asyncio
async def test_ranking_request_carries_unique_imports_not_raw_entries(
    persister_setup: PersisterTestSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dedupe runs BEFORE the per-finding cap, so the cap bounds UNIQUE imports.

    A bucket of 7 candidates over 3 distinct import strings must reach ranking
    as those 3, not as the first `MAX_CANDIDATES_PER_FINDING` raw entries — a
    loquacious analyze-LLM repeating one import with different rationales would
    otherwise crowd every slot and starve the unique ones.
    """
    from outrider.agent.nodes.trace import MAX_CANDIDATES_PER_FINDING

    review_id = persister_setup.review_id
    noisy = _build_finding(review_id=review_id, proposal_hash="c" * 64, file_path="src/c.py")
    other = _build_finding(review_id=review_id, proposal_hash="d" * 64, file_path="src/d.py")
    # 7 raw entries, 3 unique imports, with the repeat filling the whole cap:
    # capping FIRST would keep only `pkg.one` and starve the other two. A fixture
    # whose uniques all fall inside the first MAX_CANDIDATES_PER_FINDING entries
    # cannot tell the two orders apart — both yield the same set.
    noisy_cands = tuple(
        TraceCandidate(
            candidate_id=compute_candidate_id(
                source_proposal_hash="c" * 64, import_string=imp, reason=f"rationale {i}"
            ),
            source_proposal_hash="c" * 64,
            reason=f"rationale {i}",
            import_string=imp,
        )
        for i, imp in enumerate(
            ["pkg.one", "pkg.one", "pkg.one", "pkg.one", "pkg.one", "pkg.two", "pkg.three"]
        )
    )
    # A second bucket with two candidates, so the singleton bypass cannot fire.
    other_cands = (
        _build_candidate(source_proposal_hash="d" * 64, import_string="pkg.alpha"),
        _build_candidate(source_proposal_hash="d" * 64, import_string="pkg.beta"),
    )
    state = _build_multi_state(
        review_id=review_id,
        findings=(noisy, other),
        candidates=(*noisy_cands, *other_cands),
    )

    async def fake_fetch(*_args: object, **_kwargs: object) -> bytes | None:
        return None

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)
    provider = _MockLLMProvider(ranked_candidate_ids=())

    await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    assert len(provider.calls) == 1, "two multi-candidate buckets must fire exactly one ranking"
    prompt = provider.calls[0].user_prompt
    assert prompt.count("pkg.one") == 1, (
        "the repeated import must reach ranking once, not once per rationale"
    )
    for unique in ("pkg.one", "pkg.two", "pkg.three"):
        assert unique in prompt, f"{unique} was crowded out of the capped set"
    assert len(noisy_cands) > MAX_CANDIDATES_PER_FINDING, (
        "fixture must exceed the cap or it proves nothing about dedupe-before-cap"
    )


@pytest.mark.asyncio
async def test_ranking_request_orders_buckets_by_finding_id(
    persister_setup: PersisterTestSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-bucket order is `sorted(source_finding_id)`; intra-bucket is insertion order.

    Intra-bucket is deliberately NOT sorted by `candidate_id` — that is a hash of
    LLM-controlled fields, so sorting on it would let a hostile analyze-LLM grind
    ids to win the top of the ranking prompt.
    """
    review_id = persister_setup.review_id
    first = _build_finding(review_id=review_id, proposal_hash="e" * 64, file_path="src/e.py")
    second = _build_finding(review_id=review_id, proposal_hash="f" * 64, file_path="src/f.py")
    lo, hi = sorted((first, second), key=lambda f: f.finding_id)
    lo_hash = "e" * 64 if lo is first else "f" * 64
    hi_hash = "e" * 64 if hi is first else "f" * 64
    lo_cands = (
        _build_candidate(source_proposal_hash=lo_hash, import_string="lo.second_emitted"),
        _build_candidate(source_proposal_hash=lo_hash, import_string="lo.first_emitted"),
    )
    hi_cands = (
        _build_candidate(source_proposal_hash=hi_hash, import_string="hi.second_emitted"),
        _build_candidate(source_proposal_hash=hi_hash, import_string="hi.first_emitted"),
    )
    state = _build_multi_state(
        review_id=review_id,
        findings=(first, second),
        # Emitted hi-bucket first, so a correct flatten must still put lo first.
        candidates=(*hi_cands, *lo_cands),
    )

    async def fake_fetch(*_args: object, **_kwargs: object) -> bytes | None:
        return None

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)
    # Rank the four INTERLEAVED across buckets and reversed within each, so the
    # provider's order differs from both the request order and insertion order.
    provider = _MockLLMProvider(
        ranked_candidate_ids=(
            hi_cands[1].candidate_id,
            lo_cands[1].candidate_id,
            hi_cands[0].candidate_id,
            lo_cands[0].candidate_id,
        )
    )

    state_delta = await trace(
        state,
        provider=provider,  # type: ignore[arg-type]
        trace_model="claude-haiku-test",
        phase_event_sink=persister_setup.persister,
        trace_sink=persister_setup.persister,
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    assert len(provider.calls) == 1
    prompt = provider.calls[0].user_prompt
    positions = [
        prompt.index("lo.second_emitted"),
        prompt.index("lo.first_emitted"),
        prompt.index("hi.second_emitted"),
        prompt.index("hi.first_emitted"),
    ]
    assert positions == sorted(positions), (
        "buckets must flatten in sorted(source_finding_id) order with insertion "
        f"order preserved inside each; got offsets {positions}"
    )

    decisions = list(state_delta.get("trace_decisions", []))
    assert [d.source_finding_id for d in decisions] == sorted(
        d.source_finding_id for d in decisions
    ), "decisions must come back sorted by finding id"

    # Re-grouping puts each candidate back under its own finding in the order the
    # provider ranked them. `proposed_import_strings` cannot show this — both
    # schemas canonicalize it via a sorting field_validator — but the aggregated
    # `reason` concatenates per candidate in bucket order and does survive.
    reason_by_finding = {d.source_finding_id: d.reason for d in decisions}
    for finding, cands in ((lo, lo_cands), (hi, hi_cands)):
        reason = reason_by_finding[finding.finding_id]
        ranked_first, ranked_second = cands[1].import_string, cands[0].import_string
        assert reason.index(ranked_first) < reason.index(ranked_second), (
            f"aggregated reason must follow the provider's ranking "
            f"({ranked_first} before {ranked_second}); got {reason!r}"
        )
