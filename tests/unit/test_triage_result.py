"""TriageResult: construction-time validation, enum gates, frozen guard.

Per spec §7.2: TriageResult is the typed output contract of the triage node.
file_tiers maps changed-file paths to ReviewTier; overall_risk is a RiskLevel;
relevant_dimensions enumerates which review dimensions apply (pure CSS doesn't
get a security review per §4.1.2 cost-control rationale); reasoning is the
LLM's brief justification, capped at 500 chars.

Round 24: file_tiers is wrapped in MappingProxyType post-construction so
`triage.file_tiers["x"] = ...` raises TypeError. Closes FUP-018.
"""

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from outrider.schemas import ReviewDimension, ReviewTier, RiskLevel, TriageResult


def _minimal_triage_result(**overrides: object) -> TriageResult:
    base = dict(
        file_tiers={"src/auth.py": ReviewTier.DEEP, "src/css/main.css": ReviewTier.SKIM},
        overall_risk=RiskLevel.MEDIUM,
        relevant_dimensions=[ReviewDimension.SECURITY, ReviewDimension.CODE_QUALITY],
        reasoning="auth changes warrant deep security review; CSS-only changes skim.",
    )
    base.update(overrides)
    return TriageResult(**base)  # type: ignore[arg-type]


def test_triage_result_minimal_construction_succeeds() -> None:
    result = _minimal_triage_result()
    assert result.file_tiers["src/auth.py"] == ReviewTier.DEEP
    assert result.overall_risk == RiskLevel.MEDIUM


def test_triage_result_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="extra"):
        TriageResult(  # type: ignore[call-arg]
            file_tiers={},
            overall_risk=RiskLevel.LOW,
            relevant_dimensions=[],
            reasoning="ok",
            unknown_field="oops",
        )


def test_triage_result_is_frozen() -> None:
    result = _minimal_triage_result()
    with pytest.raises(ValidationError):
        result.reasoning = "different"  # type: ignore[misc]


def test_triage_result_rejects_unknown_review_tier_string() -> None:
    """file_tiers values are typed ReviewTier; strings outside the enum reject."""
    with pytest.raises(ValidationError):
        TriageResult(  # type: ignore[arg-type]
            file_tiers={"src/foo.py": "ultra-deep"},
            overall_risk=RiskLevel.MEDIUM,
            relevant_dimensions=[],
            reasoning="ok",
        )


def test_triage_result_rejects_unknown_risk_level_string() -> None:
    with pytest.raises(ValidationError):
        TriageResult(  # type: ignore[arg-type]
            file_tiers={},
            overall_risk="catastrophic",
            relevant_dimensions=[],
            reasoning="ok",
        )


def test_triage_result_rejects_unknown_dimension_string() -> None:
    with pytest.raises(ValidationError):
        TriageResult(  # type: ignore[arg-type]
            file_tiers={},
            overall_risk=RiskLevel.LOW,
            relevant_dimensions=["accessibility"],
            reasoning="ok",
        )


def test_triage_result_reasoning_max_length_500_admits() -> None:
    """500-char reasoning is the cap (inclusive)."""
    result = _minimal_triage_result(reasoning="x" * 500)
    assert len(result.reasoning) == 500


def test_triage_result_reasoning_over_500_rejects() -> None:
    with pytest.raises(ValidationError):
        _minimal_triage_result(reasoning="x" * 501)


def test_triage_result_accepts_each_tier_in_file_tiers() -> None:
    """Pin: file_tiers values can be any of the four ReviewTier members."""
    result = TriageResult(
        file_tiers={
            "a.py": ReviewTier.DEEP,
            "b.py": ReviewTier.STANDARD,
            "c.py": ReviewTier.SKIM,
            "d.py": ReviewTier.SKIP,
        },
        overall_risk=RiskLevel.HIGH,
        relevant_dimensions=[],
        reasoning="ok",
    )
    assert result.file_tiers["d.py"] == ReviewTier.SKIP


def test_triage_result_round_trip_json() -> None:
    """LangGraph state serialization round-trip: nested enums must rehydrate."""
    result = _minimal_triage_result()
    rehydrated = TriageResult.model_validate_json(result.model_dump_json())
    assert rehydrated == result
    assert rehydrated.file_tiers["src/auth.py"] == ReviewTier.DEEP
    assert rehydrated.overall_risk == RiskLevel.MEDIUM
    assert ReviewDimension.SECURITY in rehydrated.relevant_dimensions


def test_triage_result_empty_file_tiers_admits() -> None:
    """Empty file_tiers is valid at the schema level. In practice, an empty
    file_tiers reaches a TriageResult only if the §6.10 size-cap policy
    classified all files SKIP upstream of the triage LLM call (per spec
    §6.10: 'Hard caps trigger before any LLM call') — the LLM itself does
    not produce SKIP entries (per the module docstring on triage_result.py:
    'DEEP / STANDARD / SKIM are the LLM-produced classifications; SKIP is
    populated by the §6.10 size-cap policy gate'). The test pins the schema
    admittance, not the publishing pathway."""
    result = _minimal_triage_result(file_tiers={})
    assert result.file_tiers == {}


def test_triage_result_empty_relevant_dimensions_admits() -> None:
    """Empty relevant_dimensions is valid (e.g., a pure-formatting PR that
    none of the five review dimensions meaningfully apply to)."""
    result = _minimal_triage_result(relevant_dimensions=[])
    assert result.relevant_dimensions == ()


def test_triage_result_required_fields_all_required() -> None:
    """No defaults — all four fields must be provided at construction."""
    with pytest.raises(ValidationError):
        TriageResult()  # type: ignore[call-arg]


def test_triage_result_relevant_dimensions_is_tuple_not_list() -> None:
    """frozen=True is faux-immutable over .append() on a list field; spec §7.2
    was amended 2026-05-08 to use tuple[ReviewDimension, ...] for true
    immutability. Same precedent as PRContext.changed_files / HITLDecision.decisions."""
    result = _minimal_triage_result()
    assert isinstance(result.relevant_dimensions, tuple)


def test_triage_result_relevant_dimensions_rejects_in_place_append() -> None:
    """Tuple has no .append(); a downstream node attempting to mutate the
    triage's dimension list now raises AttributeError instead of silently
    succeeding."""
    result = _minimal_triage_result()
    with pytest.raises(AttributeError):
        result.relevant_dimensions.append(  # type: ignore[attr-defined]
            ReviewDimension.PERFORMANCE
        )


def test_triage_result_dict_round_trip() -> None:
    """LangGraph reducer merges receive partial-update dicts; model_dump() →
    model_validate() must preserve all nested structure exactly. Round 24:
    file_tiers' runtime type is now MappingProxyType (not dict) per FUP-018
    closure, so the isinstance check widens to Mapping (its abstract supertype)."""
    result = _minimal_triage_result()
    rehydrated = TriageResult.model_validate(result.model_dump())
    assert rehydrated == result
    assert isinstance(rehydrated.relevant_dimensions, tuple)
    assert isinstance(rehydrated.file_tiers, Mapping)


def test_triage_result_file_tiers_rejects_in_place_mutation() -> None:
    """Round 24 closes FUP-018: TriageResult.file_tiers is wrapped in
    MappingProxyType post-construction so any attempt to mutate the mapping
    in-place raises TypeError. This pins the closure of the dict-mutation
    gap that was deferred from Round 14 / Round 18 (the validator's name was
    half-truthing this gap; now the gap itself is closed)."""
    result = _minimal_triage_result()
    with pytest.raises(TypeError):
        result.file_tiers["src/sneaky.py"] = ReviewTier.DEEP  # type: ignore[index]
    with pytest.raises(TypeError):
        del result.file_tiers["src/auth.py"]  # type: ignore[attr-defined]


def test_triage_result_file_tiers_json_round_trip_preserves_mapping_semantics() -> None:
    """Round 24 regression guard: MappingProxyType-wrapped file_tiers must
    round-trip through model_dump_json() → model_validate_json() correctly.
    The field_serializer dumps as a regular dict (StrEnum values become
    strings); the field_validator on rehydration re-wraps in MappingProxyType.
    Without the serializer, Pydantic might fail to JSON-encode MappingProxyType
    or might emit a non-dict shape that breaks downstream consumers."""
    result = _minimal_triage_result()
    rehydrated = TriageResult.model_validate_json(result.model_dump_json())
    assert rehydrated == result
    # Rehydrated value is also MappingProxyType — same immutability gate fires.
    assert isinstance(rehydrated.file_tiers, Mapping)
    with pytest.raises(TypeError):
        rehydrated.file_tiers["x"] = ReviewTier.SKIM  # type: ignore[index]


def test_triage_result_file_tiers_input_dict_is_copied_not_aliased() -> None:
    """Round 24 invariant: the input dict passed to TriageResult must be
    COPIED before being wrapped in MappingProxyType. Otherwise, mutating
    the original dict post-construction would leak through the proxy and
    silently change file_tiers — defeating the immutability gate."""
    input_dict = {"src/auth.py": ReviewTier.DEEP}
    result = _minimal_triage_result(file_tiers=input_dict)
    # Mutate the original dict.
    input_dict["src/sneaky.py"] = ReviewTier.SKIM
    # The proxy must NOT see the mutation.
    assert "src/sneaky.py" not in result.file_tiers
