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

from datetime import UTC, datetime, timedelta, timezone
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
                content_base="a\n",
                content_head="a\nb\nc\n",
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


def test_review_state_is_eval_defaults_to_false() -> None:
    """Production reviews default to is_eval=False. Eval-harness factories
    construct seeds with is_eval=True; nodes thread state.is_eval into
    LLMRequest so audit rows produced during eval runs are correctly
    tagged per `docs/testing.md` "Eval isolation"."""
    state = _minimal_review_state()
    assert state.is_eval is False


def test_review_state_is_eval_accepts_true() -> None:
    """Eval-harness factories MUST be able to set is_eval=True on the seed
    so the entire downstream audit trail (LLMCallEvent rows, future phase
    events, future findings) inherits the flag."""
    state = _minimal_review_state(is_eval=True)
    assert state.is_eval is True


def test_review_state_is_eval_rejects_non_bool_garbage() -> None:
    """Pydantic field validation: is_eval rejects values it can't coerce
    to bool. Pydantic 2 DOES loose-coerce "yes"/"no"/"true"/"false"/1/0
    per its standard boolean-validation rules; the test uses an
    obviously-garbage value to confirm the type IS validated (not just
    silently accepted as object). Pin so a future refactor that drops
    the bool annotation doesn't silently admit lists/dicts/etc."""
    with pytest.raises(ValidationError):
        _minimal_review_state(is_eval=["not", "a", "bool"])  # type: ignore[arg-type]


def test_review_state_validate_assignment_catches_is_eval_post_construction() -> None:
    """validate_assignment=True on ReviewState means mid-graph mutation
    of state.is_eval to garbage raises. Without this gate, a node could
    silently flip the eval-isolation flag to a non-bool."""
    state = _minimal_review_state(is_eval=False)
    with pytest.raises(ValidationError):
        state.is_eval = {"not": "bool"}  # type: ignore[assignment]


def test_review_state_is_eval_json_round_trip() -> None:
    """is_eval must round-trip through model_dump_json + model_validate_json
    — critical for LangGraph checkpoint persistence which serializes
    state to JSON on every interrupt. A regression that drops is_eval
    from the JSON shape would silently lose the eval flag on every
    HITL resume / checkpoint reload."""
    for flag in (True, False):
        state = _minimal_review_state(is_eval=flag)
        rehydrated = type(state).model_validate_json(state.model_dump_json())
        assert rehydrated.is_eval is flag, (
            f"is_eval={flag} dropped through JSON round-trip; got {rehydrated.is_eval}"
        )


def test_review_state_received_at_rejects_naive_datetime() -> None:
    """AwareDatetime per docs/conventions.md 'datetimes are AwareDatetime, never naive'."""
    with pytest.raises(ValidationError):
        _minimal_review_state(received_at=datetime(2026, 5, 8, 12, 0, 0))  # naive


def test_review_state_extra_forbid() -> None:
    """Unknown fields raise — guards against silently growing the V1 skeleton.

    Uses a clearly-fictional field name so the assertion stays meaningful
    against the canonical state shape (review_report / hitl_request /
    hitl_decision all landed; this test pins the extra="forbid" config,
    not a specific deferred-slot name).
    """
    with pytest.raises(ValidationError, match="extra"):
        ReviewState(  # type: ignore[call-arg]
            review_id=uuid4(),
            pr_context=_minimal_pr_context(),
            received_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC),
            future_v2_field=None,  # arbitrary unknown field; extra="forbid" gates
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


def test_review_state_assigning_wrong_type_review_id_raises() -> None:
    """validate_assignment coverage: review_id is a UUID; reassignment with
    a non-UUID value must raise. The module docstring's 'primary structural
    defense across the post-first-input lifetime' claim is load-bearing on
    EVERY field being type-enforced post-construction. Without this test,
    a future refactor that loosens review_id's Pydantic type (e.g., bare
    str) would silently slip past validate_assignment."""
    state = _minimal_review_state()
    with pytest.raises(ValidationError):
        state.review_id = "not-a-uuid"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        state.review_id = 12345  # type: ignore[assignment]


def test_review_state_received_at_round_trip_preserves_microseconds_and_offset() -> None:
    """JSON round-trip on AwareDatetime must preserve microsecond precision
    AND non-UTC tzinfo offset. The schema's existing round-trip test uses
    UTC + zero microseconds — a regression that drops microseconds or
    normalizes the offset to Z would still pass equality and never be
    caught. This test pins both signals against the canonical contract."""
    plus_two_hours = timezone(timedelta(hours=2))
    precise = datetime(2026, 5, 8, 12, 34, 56, 789012, tzinfo=plus_two_hours)
    state = _minimal_review_state(received_at=precise)
    rehydrated = ReviewState.model_validate_json(state.model_dump_json())
    assert rehydrated.received_at == state.received_at
    assert rehydrated.received_at.microsecond == 789012, (
        f"microsecond precision dropped: got {rehydrated.received_at.microsecond}"
    )
    assert rehydrated.received_at.utcoffset() == precise.utcoffset(), (
        f"timezone offset normalized: got {rehydrated.received_at.utcoffset()}"
    )


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


def test_last_trace_pass_fetched_count_rejects_negative() -> None:
    """`Field(default=0, ge=0)` per the router-contract: the field models
    a per-invocation count of NEW trace-fetched files; negative values are
    meaningless AND silently violate `_trace_router`'s `> 0` predicate.
    Fail fast at the schema boundary rather than at the router."""
    with pytest.raises(ValidationError):
        _minimal_review_state(last_trace_pass_fetched_count=-1)


def test_last_trace_pass_fetched_count_admits_zero_and_positive() -> None:
    """Default is 0 (no trace pass yet); positive values are valid
    per-invocation deltas. Pin both ends of the valid range."""
    zero_state = _minimal_review_state(last_trace_pass_fetched_count=0)
    assert zero_state.last_trace_pass_fetched_count == 0
    positive_state = _minimal_review_state(last_trace_pass_fetched_count=5)
    assert positive_state.last_trace_pass_fetched_count == 5
