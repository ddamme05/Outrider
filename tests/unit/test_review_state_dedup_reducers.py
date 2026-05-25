# See specs/2026-05-19-analyze-foundation.md §3.
"""`ReviewState.analysis_rounds` + `ReviewState.trace_candidates` reducer tests.

The §3 commit adds two slots, each consuming `append_with_dedup_by(key_fn)`
from `agent/reducers.py`. The reducer itself is exercised in
`tests/unit/test_agent_reducers.py`; this file pins the integration —
that the reducer is wired into `ReviewState` with the right merge key
and survives checkpoint-replay re-application idempotently.

Per `docs/conventions.md` "LangGraph specifics" + §3 of the foundation
spec: replay re-application of the same delta must not double-count.
The `round_id` and `candidate_id` fields are content-derived SHA-256
hex digests; re-emission of the same logical item produces the same
key and is dropped on merge.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, get_args, get_type_hints
from uuid import UUID, uuid4

from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import (
    compute_candidate_id,
    compute_identity_hash,
    compute_round_id,
)
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import (
    AnalysisRound,
    ReviewDimension,
    ReviewFinding,
    ReviewState,
    TraceCandidate,
    TraceDecision,
    TraceFetchedFile,
)


def _finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path="src/foo.py",
        line_start=10,
        line_end=12,
        title="SQL injection",
        description="raw concat",
        evidence="concat at src/foo.py:11",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path="src/foo.py",
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash="a" * 64,  # Per DECISIONS.md#025.
    )


def _round(pass_index: int = 0, *, distinct: str = "") -> AnalysisRound:
    """Construct an AnalysisRound with a canonical round_id.

    `distinct` parameterizes the file path so different rounds get
    different content-derived ids (canonical recipe is content-keyed,
    so we vary the content to vary the id).
    """
    now = datetime.now(UTC)
    file_path = f"src/foo_{distinct}.py" if distinct else "src/foo.py"
    finding = _finding()
    return AnalysisRound(
        round_id=compute_round_id(
            pass_index=pass_index,
            files_examined=(file_path,),
            files_skipped=(),
            finding_content_hashes=(finding.content_hash,),
        ),
        pass_index=pass_index,
        findings=(finding,),
        files_examined=(file_path,),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )


def _candidate(seed: str = "a") -> TraceCandidate:
    """Construct a TraceCandidate with a canonical candidate_id derived
    from its own payload (required by `_enforce_candidate_id_matches_payload`).
    Per DECISIONS.md#024, trace candidates are dotted Python import
    strings — the seed is sanitized (hyphens → underscores) and folded
    into a single-part identifier so test-author-friendly seeds like
    'auth-mw' don't trip the identifier-validity check."""
    source_proposal_hash = compute_identity_hash({"prop": seed})
    import_string = f"pkg_{seed.replace('-', '_')}"
    reason = "x"
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


def _get_reducer(field_name: str) -> Any:
    """Extract the reducer callable from the Annotated field metadata.

    `ReviewState` declares `analysis_rounds: Annotated[list[...], reducer]`;
    Pydantic preserves the Annotated metadata so the test can pull the
    reducer out by type-hint introspection. This is the same pattern
    LangGraph itself uses to discover reducers.
    """
    hints = get_type_hints(ReviewState, include_extras=True)
    annotated = hints[field_name]
    metadata = get_args(annotated)[1:]
    reducers = [m for m in metadata if callable(m)]
    assert reducers, f"no callable metadata on field {field_name!r}"
    return reducers[0]


def test_analysis_rounds_field_carries_reducer() -> None:
    """The Annotated metadata must include a callable reducer; without
    it LangGraph falls back to plain list-concat which double-accumulates
    on replay."""
    reducer = _get_reducer("analysis_rounds")
    # Smoke: the reducer admits two args + returns a list.
    merged = reducer([], [_round()])
    assert isinstance(merged, list)
    assert len(merged) == 1


def test_trace_candidates_field_carries_reducer() -> None:
    reducer = _get_reducer("trace_candidates")
    merged = reducer([], [_candidate()])
    assert isinstance(merged, list)
    assert len(merged) == 1


def test_analysis_rounds_replay_idempotent() -> None:
    """The §3 replay-equivalence invariant: re-emitting the same round
    on top of an existing-state list is a no-op."""
    reducer = _get_reducer("analysis_rounds")
    r = _round(pass_index=0, distinct="seed")
    merged_once = reducer([], [r])
    merged_twice = reducer(merged_once, [r])
    assert len(merged_twice) == 1
    assert merged_twice[0] is r


def test_trace_candidates_replay_idempotent() -> None:
    reducer = _get_reducer("trace_candidates")
    c = _candidate(seed="auth-mw")
    merged_once = reducer([], [c])
    merged_twice = reducer(merged_once, [c])
    assert len(merged_twice) == 1
    assert merged_twice[0] is c


def test_analysis_rounds_distinct_round_ids_both_admitted() -> None:
    """Different content-derived round_ids represent distinct rounds —
    both land. The reducer is dedup-by-key, not dedup-by-equality, so
    different keys are always preserved."""
    reducer = _get_reducer("analysis_rounds")
    r1 = _round(pass_index=0, distinct="seed-a")
    r2 = _round(pass_index=1, distinct="seed-b")
    merged = reducer([r1], [r2])
    assert len(merged) == 2
    assert {x.round_id for x in merged} == {r1.round_id, r2.round_id}


def test_review_state_default_lists_are_empty() -> None:
    """New state slots default to empty lists per `Field(default_factory=list)`.
    Without this, the seed `ReviewState` constructed by the webhook
    receiver would fail to validate (the slots are required-but-empty
    at intake time)."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from outrider.schemas.pr_context import PRContext

    state = ReviewState(
        review_id=uuid4(),
        pr_context=PRContext(
            installation_id=1,
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
        received_at=datetime.now(UTC),
    )
    assert state.analysis_rounds == []
    assert state.trace_candidates == []
    # Trace-node spec slots also default to empty lists.
    assert state.trace_decisions == []
    assert state.trace_fetched_files == []


# ---------------------------------------------------------------------------
# trace_decisions reducer (per DECISIONS.md#017 × #024 amendment)
# ---------------------------------------------------------------------------


def _trace_decision(source_finding_id: UUID | None = None) -> TraceDecision:
    """Build a TraceDecision fixture for reducer tests. Per #024 amendment:
    parallel proposed_import_strings + resolved_candidate_paths tuples;
    `resolved` case has exactly one resolved_candidate_paths entry
    matching target_file."""
    return TraceDecision(
        source_finding_id=source_finding_id if source_finding_id is not None else uuid4(),
        target_file="src/bar.py",
        reason="x",
        resolution_status="resolved",
        proposed_import_strings=("bar",),
        resolved_candidate_paths=("src/bar.py",),
    )


def test_trace_decisions_field_carries_reducer() -> None:
    """`ReviewState.trace_decisions` carries an `append_with_dedup_by` reducer
    keyed on `source_finding_id` per DECISIONS.md#017 amended point 1."""
    reducer = _get_reducer("trace_decisions")
    assert reducer is not None


def test_trace_decisions_replay_idempotent() -> None:
    """Same TraceDecision applied twice via the reducer collapses to one row.
    Replay idempotency per #017's `source_finding_id`-alone dedup-key
    contract."""
    reducer = _get_reducer("trace_decisions")
    d = _trace_decision()
    merged_once = reducer([], [d])
    merged_twice = reducer(merged_once, [d])
    assert len(merged_once) == 1
    assert len(merged_twice) == 1  # duplicate dropped
    assert merged_once == merged_twice


def test_trace_decisions_distinct_source_finding_ids_both_admitted() -> None:
    """Different source_finding_ids = distinct decisions; reducer keeps both."""
    reducer = _get_reducer("trace_decisions")
    d1 = _trace_decision()
    d2 = _trace_decision()
    assert d1.source_finding_id != d2.source_finding_id
    merged = reducer([d1], [d2])
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# trace_fetched_files reducer (per spec Q3 + M2)
# ---------------------------------------------------------------------------


def _trace_fetched_file(path: str = "src/middleware/auth.py") -> TraceFetchedFile:
    """Build a TraceFetchedFile fixture for reducer tests."""
    return TraceFetchedFile(
        path=path,
        content_head="def authenticate(): pass\n",
        source_finding_id=uuid4(),
    )


def test_trace_fetched_files_field_carries_reducer() -> None:
    """`ReviewState.trace_fetched_files` carries an `append_with_dedup_by`
    reducer keyed on `path` per spec Q3."""
    reducer = _get_reducer("trace_fetched_files")
    assert reducer is not None


def test_trace_fetched_files_first_write_wins_on_path_collision() -> None:
    """Per M2 audit-fold: when two findings resolve to the same target
    path, the reducer's first-write-wins semantics keep only the first
    emission. Multi-cause provenance recovers via cross-reference to
    `state.trace_decisions` by `target_file`."""
    reducer = _get_reducer("trace_fetched_files")
    f1 = _trace_fetched_file(path="src/foo.py")
    f2 = _trace_fetched_file(path="src/foo.py")  # same path, different source_finding_id
    assert f1.source_finding_id != f2.source_finding_id
    merged = reducer([f1], [f2])
    assert len(merged) == 1  # f2 dropped
    assert merged[0].source_finding_id == f1.source_finding_id  # first-write-wins


def test_trace_fetched_files_distinct_paths_both_admitted() -> None:
    """Different paths = distinct fetches; reducer keeps both."""
    reducer = _get_reducer("trace_fetched_files")
    f1 = _trace_fetched_file(path="src/foo.py")
    f2 = _trace_fetched_file(path="src/bar.py")
    merged = reducer([f1], [f2])
    assert len(merged) == 2
