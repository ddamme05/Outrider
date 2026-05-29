# See DECISIONS.md#028-per-review-policy-version-snapshot-anchor-on-triageresult
# — `test_triage_result_policy_version_admits_any_valid_semver` is #028's
# shape-vs-value pin (BARE_SEMVER_PATTERN is the schema floor; the
# triage Rule (d) gate is what closes the value-injection path).
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


def test_triage_result_policy_version_defaults_to_active() -> None:
    """The `policy_version` field captures `ACTIVE_POLICY_VERSION` via
    default_factory at construction time. This snapshot rides on state
    downstream and is the trusted anchor that synthesize uses to detect
    policy_version smuggle (see specs/2026-05-28-synthesize-node.md
    Severity policy section, triage-anchored snapshot resolution)."""
    from outrider.policy.severity import ACTIVE_POLICY_VERSION

    result = _minimal_triage_result()
    assert result.policy_version == ACTIVE_POLICY_VERSION


def test_triage_result_default_factory_fires_on_model_validate_json_absent_key() -> None:
    """`DECISIONS.md#028` depends on Pydantic firing
    `default_factory=lambda: ACTIVE_POLICY_VERSION` when the JSON input
    omits the `policy_version` key. Per Pass-1 multi-lens audit
    MCP-grounded lens recommendation: Pydantic docs document
    `default_factory` as a field-init mechanism but DON'T explicitly
    assert it fires on `model_validate_json` for an absent key. This
    test converts the implicit-by-docs claim into an asserted contract
    — protects DECISIONS#028's trust-root analysis against a future
    Pydantic upgrade silently changing the JSON-input default path."""
    from outrider.policy.severity import ACTIVE_POLICY_VERSION

    # JSON input deliberately OMITS policy_version. All four other
    # required fields present.
    json_input = (
        '{"file_tiers": {"src/auth.py": "deep"}, '
        '"overall_risk": "medium", '
        '"relevant_dimensions": ["security"], '
        '"reasoning": "snapshot-anchor default_factory pin"}'
    )

    result = TriageResult.model_validate_json(json_input)

    # If Pydantic did NOT fire the default_factory on the absent key,
    # this would either error during validation (required-field
    # missing) OR ship a sentinel value. Asserting equality with
    # ACTIVE pins the documented behavior.
    assert result.policy_version == ACTIVE_POLICY_VERSION, (
        f"default_factory did not fire on absent JSON key — "
        f"DECISIONS#028 trust-root depends on this. Got "
        f"{result.policy_version!r}, expected {ACTIVE_POLICY_VERSION!r}."
    )


def test_triage_result_policy_version_explicit_override_admits() -> None:
    """Replay path: a historical TriageResult rehydrated from audit can
    carry a historical policy_version string. The field accepts any
    bare-semver string (no mid-flight clamping to ACTIVE)."""
    result = _minimal_triage_result(policy_version="0.0.1")
    assert result.policy_version == "0.0.1"


def test_triage_result_policy_version_rejects_non_semver() -> None:
    """The field's regex constrains the SHAPE to bare semver
    (X.Y.Z). Attacker-controlled strings or accidental garbage with
    invalid shape reject at construction time — the schema floor."""
    with pytest.raises(ValidationError):
        _minimal_triage_result(policy_version="evil-snapshot")
    with pytest.raises(ValidationError):
        _minimal_triage_result(policy_version="1.2")  # missing patch
    with pytest.raises(ValidationError):
        _minimal_triage_result(policy_version="1.2.3-rc1")  # prerelease suffix


def test_triage_result_policy_version_admits_any_valid_semver() -> None:
    """The schema-level `pattern=BARE_SEMVER_PATTERN` is shape-only —
    `"0.0.0"`, `"99.99.99"`, `"5.0.0"` ALL pass. Demonstrates that the
    schema floor alone does NOT prevent an LLM-emitted explicit value
    from landing on the field. The trusted-snapshot property depends
    on the triage-node gate (`_enforce_triage_policy` Rule (d), see
    `tests/unit/test_triage_node.py::test_enforce_policy_rejects_llm_injected_policy_version`)
    which compares the post-validation value to live ACTIVE.

    This test exists as the deliberate negative pin: any change that
    tightens the schema-level pattern to a single-value match would
    make Rule (d) redundant, and vice versa — the two gates serve
    different purposes (shape vs value). Removing either is a
    regression."""
    for valid_semver in ("0.0.0", "99.99.99", "5.0.0", "10.20.30"):
        # MUST NOT raise — the schema admits any valid-shape semver.
        result = _minimal_triage_result(policy_version=valid_semver)
        assert result.policy_version == valid_semver
