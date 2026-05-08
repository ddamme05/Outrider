"""ReviewState: V1 skeletal slice — webhook-seeded + triage-populated slots only.

Per spec §7.1 + DECISIONS.md#020, ReviewState is the LangGraph state
envelope. This V1 slice carries only review_id / pr_context / received_at
(webhook-seeded by the receiver per #020) and triage_result (triage node
output). Intake enriches `pr_context` in place by fetching the file list +
per-file content and returning a fresh PRContext via {"pr_context":
new_pr_context}. Slots populated by analyze, trace, synthesize, hitl,
publish are deferred to their respective node specs (see
schemas/review_state.py module docstring).

ReviewState is NOT frozen — LangGraph nodes return partial-update dicts that
reducers merge. A frozen state would break the reducer contract per
docs/conventions.md "LangGraph specifics".

received_at is AwareDatetime — naive datetimes round-trip as subtly wrong
times through Postgres timestamptz per docs/conventions.md "Code style".
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.schemas import (
    ChangedFile,
    PRContext,
    ReviewDimension,
    ReviewState,
    ReviewTier,
    RiskLevel,
    TriageResult,
)


def _minimal_pr_context() -> PRContext:
    return PRContext(
        installation_id=12345,
        owner="acme",
        repo="widget",
        pr_number=42,
        pr_title="Add the thing",
        pr_body="Adds a thing.",
        base_sha="a" * 40,
        head_sha="b" * 40,
        author="alice",
        changed_files=[
            ChangedFile(
                path="src/foo.py",
                status="modified",
                additions=3,
                deletions=1,
                patch="@@ -1 +1,3 @@\n a\n+b\n+c",
            )
        ],
        total_additions=3,
        total_deletions=1,
    )


def _minimal_review_state(**overrides: object) -> ReviewState:
    base = dict(
        review_id=uuid4(),
        pr_context=_minimal_pr_context(),
        received_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC),
    )
    base.update(overrides)
    return ReviewState(**base)  # type: ignore[arg-type]


def test_review_state_minimal_construction_succeeds() -> None:
    state = _minimal_review_state()
    assert state.pr_context.owner == "acme"
    assert state.triage_result is None  # default


def test_review_state_triage_result_defaults_to_none() -> None:
    """V1 skeleton: triage_result is None until the triage node populates it."""
    state = _minimal_review_state()
    assert state.triage_result is None


def test_review_state_accepts_triage_result_instance() -> None:
    triage = TriageResult(
        file_tiers={"src/foo.py": ReviewTier.STANDARD},
        overall_risk=RiskLevel.LOW,
        relevant_dimensions=[ReviewDimension.CODE_QUALITY],
        reasoning="standard application change.",
    )
    state = _minimal_review_state(triage_result=triage)
    assert state.triage_result is not None
    assert state.triage_result.overall_risk == RiskLevel.LOW


def test_review_state_received_at_rejects_naive_datetime() -> None:
    """AwareDatetime per docs/conventions.md 'datetimes are AwareDatetime, never naive'."""
    with pytest.raises(ValidationError):
        _minimal_review_state(received_at=datetime(2026, 5, 8, 12, 0, 0))  # naive


def test_review_state_extra_forbid() -> None:
    """Unknown fields raise — guards against silently growing the V1 skeleton."""
    with pytest.raises(ValidationError, match="extra"):
        ReviewState(  # type: ignore[call-arg]
            review_id=uuid4(),
            pr_context=_minimal_pr_context(),
            received_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC),
            analysis_rounds=[],  # canonical-deferred slot, not in V1 skeleton
        )


def test_review_state_is_not_frozen() -> None:
    """ReviewState is NOT frozen per docs/conventions.md 'LangGraph specifics'.

    LangGraph nodes return partial-update dicts that reducers merge; a frozen
    state would break the reducer contract. This test guards against an
    accidental ConfigDict(frozen=True) flip — but well-typed assignments
    must still pass validate_assignment=True. The companion tests below pin
    that bad-typed assignments raise, so this test pins exactly the
    intended escape hatch (typed mutation works; misuse-resistance fires).
    """
    state = _minimal_review_state()
    new_id = uuid4()
    state.review_id = new_id  # well-typed; must succeed
    assert state.review_id == new_id


def test_review_state_assigning_naive_datetime_raises() -> None:
    """validate_assignment=True must fire on every attribute assignment, not
    only at construction. A naive datetime assigned post-construction must
    raise (else the AwareDatetime gate is bypassable; same hole the
    ReviewFinding module docstring documents)."""
    state = _minimal_review_state()
    with pytest.raises(ValidationError):
        state.received_at = datetime(2026, 5, 8, 12, 0, 0)  # naive — type: ignore[assignment]


def test_review_state_assigning_wrong_type_pr_context_raises() -> None:
    """Cross-model assignment guard: nested-model fields must revalidate.
    A bare dict that doesn't match the PRContext shape must raise; a string
    must raise; an unrelated object must raise."""
    state = _minimal_review_state()
    with pytest.raises(ValidationError):
        state.pr_context = "not a PRContext"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        state.pr_context = {"owner": "acme"}  # incomplete dict — type: ignore[assignment]


def test_review_state_assigning_wrong_type_triage_result_raises() -> None:
    """Optional-typed-field guard: `triage_result: TriageResult | None` must
    revalidate on assignment. Passing a non-None non-TriageResult value
    (e.g., a string or an unrelated dict) must raise."""
    state = _minimal_review_state()
    with pytest.raises(ValidationError):
        state.triage_result = "ok"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        state.triage_result = {"random_field": "bogus"}  # type: ignore[assignment]


def test_review_state_well_typed_pr_context_assignment_succeeds() -> None:
    """validate_assignment must accept correctly-typed values (the escape
    hatch the previous test relies on). Reconstructing pr_context with a
    fresh PRContext instance is the supported pattern."""
    state = _minimal_review_state()
    new_ctx = _minimal_pr_context()  # different instance
    state.pr_context = new_ctx
    assert state.pr_context is new_ctx


def test_review_state_round_trip_without_triage_result() -> None:
    """Pre-triage state checkpoints must round-trip through Postgres JSON."""
    state = _minimal_review_state()
    rehydrated = ReviewState.model_validate_json(state.model_dump_json())
    assert rehydrated == state
    assert rehydrated.triage_result is None


def test_review_state_round_trip_with_triage_result() -> None:
    """Post-triage state checkpoints must round-trip; nested TriageResult
    must rehydrate as a TriageResult instance, not a dict."""
    triage = TriageResult(
        file_tiers={"src/foo.py": ReviewTier.DEEP},
        overall_risk=RiskLevel.HIGH,
        relevant_dimensions=[ReviewDimension.SECURITY],
        reasoning="auth-related changes warrant deep review.",
    )
    state = _minimal_review_state(triage_result=triage)
    rehydrated = ReviewState.model_validate_json(state.model_dump_json())
    assert rehydrated == state
    assert isinstance(rehydrated.triage_result, TriageResult)
    assert rehydrated.triage_result.overall_risk == RiskLevel.HIGH


def test_review_state_required_fields_raise_when_omitted() -> None:
    """No defaults on the three webhook-seeded slots (per DECISIONS.md#020):
    review_id, pr_context, received_at."""
    with pytest.raises(ValidationError):
        ReviewState()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ReviewState(  # type: ignore[call-arg]
            review_id=uuid4(),
            pr_context=_minimal_pr_context(),
            # received_at omitted
        )


def test_review_state_dict_round_trip() -> None:
    """LangGraph reducer merges receive partial-update dicts; model_dump() →
    model_validate() must preserve all nested structure exactly. This is the
    in-process state-carrier round trip; the JSON round trips above cover
    the Postgres-checkpoint serialization path."""
    triage = TriageResult(
        file_tiers={"src/foo.py": ReviewTier.STANDARD},
        overall_risk=RiskLevel.LOW,
        relevant_dimensions=(ReviewDimension.CODE_QUALITY,),
        reasoning="standard application change.",
    )
    state = _minimal_review_state(triage_result=triage)
    rehydrated = ReviewState.model_validate(state.model_dump())
    assert rehydrated == state
    assert isinstance(rehydrated.triage_result, TriageResult)
    assert isinstance(rehydrated.pr_context.changed_files, tuple)
