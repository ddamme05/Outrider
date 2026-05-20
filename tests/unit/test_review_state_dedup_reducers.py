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
from uuid import uuid4

from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import compute_identity_hash, compute_round_id
from outrider.schemas import (
    AnalysisRound,
    ReviewDimension,
    ReviewFinding,
    ReviewState,
    TraceCandidate,
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
        policy_version="1.0.0",
        content_hash=compute_identity_hash(
            {
                "file_path": "src/foo.py",
                "line_start": 10,
                "line_end": 12,
                "finding_type": "sql_injection",
            }
        ),
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
    return TraceCandidate(
        candidate_id=compute_identity_hash({"seed": seed}),
        source_proposal_hash=compute_identity_hash({"prop": seed}),
        reason="x",
        candidate_path=f"src/{seed}.py",
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
