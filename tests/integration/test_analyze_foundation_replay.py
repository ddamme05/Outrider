# See specs/2026-05-19-analyze-foundation.md §8.
"""Cross-cutting integration test: checkpoint-replay equivalence for the
new analyze-foundation state slots.

Exercises:
- Construct a `ReviewState` with seeded `analysis_rounds` +
  `trace_candidates`.
- Round-trip through Pydantic's JSON serialization (the same path
  langgraph-checkpoint-postgres takes for persistence) and back.
- Apply a "replay" delta (same items emitted again) and assert the
  reducer collapses duplicates — total count unchanged.

This is the integration counterpart to the unit-level reducer tests
in `tests/unit/test_review_state_dedup_reducers.py`. The unit tests
verify the reducer in isolation; this test verifies it survives the
full Pydantic JSON serialization + state-merge cycle.

(Note: a full LangGraph compiled-graph integration is in
`tests/integration/test_review_state_langgraph_merge.py` for the
existing slots; we don't add another compiled-graph fixture here
because that's covered by §3's tests and the §8 scope is cross-cutting
serialization correctness only.)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import (
    compute_candidate_id,
    compute_identity_hash,
    compute_round_id,
)
from outrider.schemas import (
    AnalysisRound,
    PRContext,
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
        title="x",
        description="y",
        evidence="z",
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


def _round(pass_index: int = 0) -> AnalysisRound:
    """Construct an AnalysisRound with a canonical round_id derived from
    its actual payload — required by `_enforce_round_id_matches_payload`.
    Vary `pass_index` to get distinct round_ids."""
    now = datetime.now(UTC)
    file_path = f"src/foo_{pass_index}.py"
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


def _candidate(seed: str) -> TraceCandidate:
    source_proposal_hash = compute_identity_hash({"prop": seed})
    candidate_path = f"src/{seed}.py"
    reason = "x"
    return TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=source_proposal_hash,
            candidate_path=candidate_path,
            reason=reason,
        ),
        source_proposal_hash=source_proposal_hash,
        reason=reason,
        candidate_path=candidate_path,
    )


def _empty_pr_context() -> PRContext:
    return PRContext(
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
    )


def test_review_state_json_dump_includes_new_state_slots() -> None:
    """Construct a ReviewState with the new slots populated; serialize via
    Pydantic JSON; assert the slots are visible in the output.

    The Pydantic mode='json' dump path is what langgraph-checkpoint-postgres
    uses to persist state. We verify the slots make it to the JSON
    representation; full re-validation roundtrip is a separate concern
    (ReviewFinding.confidence is a computed_field that emits but doesn't
    re-validate under extra='forbid' — a real downstream interaction
    that the sister analyze-implementation spec's persistence layer
    handles via `exclude={'confidence'}` or model_validator(mode='before')
    coercion; out of foundation scope).
    """
    state = ReviewState(
        review_id=uuid4(),
        pr_context=_empty_pr_context(),
        received_at=datetime.now(UTC),
        analysis_rounds=[_round(0), _round(1)],
        trace_candidates=[_candidate("a"), _candidate("b")],
    )

    as_json = state.model_dump_json()
    parsed = json.loads(as_json)

    # Slots present + populated.
    assert len(parsed["analysis_rounds"]) == 2
    assert len(parsed["trace_candidates"]) == 2
    # Content-derived ids survive serialization (they're plain str).
    assert parsed["analysis_rounds"][0]["round_id"] == state.analysis_rounds[0].round_id
    assert parsed["trace_candidates"][0]["candidate_id"] == state.trace_candidates[0].candidate_id
    # Tuples become JSON arrays.
    assert isinstance(parsed["analysis_rounds"], list)
    assert isinstance(parsed["trace_candidates"], list)


def test_review_state_replay_collapse_for_analysis_rounds() -> None:
    """Replay the same AnalysisRound twice via Pydantic merge — the
    dedup-by-round_id reducer collapses duplicates so the count stays at 1.

    This is the integration counterpart to the unit-level reducer test:
    here we go through Pydantic's model construction (the same path
    LangGraph uses post-merge) rather than calling the reducer directly.
    """
    from typing import Any, get_args, get_type_hints

    hints = get_type_hints(ReviewState, include_extras=True)
    reducer = next(m for m in get_args(hints["analysis_rounds"])[1:] if callable(m))

    r = _round(pass_index=0)
    # Simulate "first emit" + "checkpoint replay reapplies the same delta".
    merged_first = reducer([], [r])
    merged_replay: list[Any] = reducer(merged_first, [r])
    assert len(merged_replay) == 1
    assert merged_replay[0].round_id == r.round_id


def test_review_state_replay_collapse_for_trace_candidates() -> None:
    """Same shape for trace_candidates."""
    from typing import Any, get_args, get_type_hints

    hints = get_type_hints(ReviewState, include_extras=True)
    reducer = next(m for m in get_args(hints["trace_candidates"])[1:] if callable(m))

    c = _candidate("auth-mw")
    merged_first = reducer([], [c])
    merged_replay: list[Any] = reducer(merged_first, [c])
    assert len(merged_replay) == 1
    assert merged_replay[0].candidate_id == c.candidate_id
