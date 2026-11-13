"""Check source grounding, scope boundaries and honest context gaps in v2 scores."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from outrider.audit.events import ContextManifestEntry, compute_finding_content_hash
from outrider.llm.base import INCLUDE_TEXT_OPT_IN, LLMRequest
from outrider.policy import EvidenceTier, FindingType, lookup_severity
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.review_finding import ReviewFinding

from .grading import ExpectedFinding
from .model_comparison import run_analyze_instrumented, state_from_eval_fixture
from .review_quality import IssueExpectation, VisibleSource, assess_case, assess_issue
from .test_model_comparison import _GROUND_TRUTH_BY_FIXTURE, _ScriptedProvider

_SOURCE = (
    "def shipping_cost(weight):\n"
    "    if weight <= 2:\n"
    "        return 4.99\n"
    "    return weight * 1.25\n"
    "\n"
    "def other_cost(weight):\n"
    "    if weight <= 2:\n"
    "        return 4.99\n"
    "    return weight * 2.50\n"
)


def finding(
    *,
    start: int = 2,
    end: int = 3,
    kind: FindingType = FindingType.MISSING_TEST,
    evidence: str = "    if weight <= 2:\n        return 4.99",
    description: str = "shipping_cost adds untested weight boundaries and pricing arithmetic.",
) -> ReviewFinding:
    return ReviewFinding(
        review_id=UUID(int=1),
        installation_id=1,
        policy_version=ACTIVE_POLICY_VERSION,
        finding_type=kind,
        dimension=lookup_dimension(kind),
        severity=lookup_severity(kind),
        evidence_tier=EvidenceTier.JUDGED,
        file_path="pricing.py",
        line_start=start,
        line_end=end,
        title="Untested pricing logic",
        description=description,
        evidence=evidence,
        proposal_hash="0" * 64,
        content_hash=compute_finding_content_hash(
            file_path="pricing.py", line_start=start, line_end=end, finding_type=kind
        ),
    )


def expectation(kind: FindingType = FindingType.MISSING_TEST) -> IssueExpectation:
    expected = ExpectedFinding(
        file_path="pricing.py",
        line_start=1,
        line_end=1,
        finding_type=kind,
        severity=lookup_severity(kind),
    )
    return IssueExpectation(
        expected, "shipping_cost", 1, 4, "supported", "Supported context supplied for this test."
    )


def test_visible_head_grading_refuses_a_base_finding_with_the_same_quote_and_lines():
    head = finding(start=3, end=4)
    base = ReviewFinding.model_validate(
        {
            **head.model_dump(),
            "source_side": "base",
            "content_hash": compute_finding_content_hash(
                head.file_path,
                line_start=head.line_start,
                line_end=head.line_end,
                finding_type=head.finding_type,
                source_side="base",
            ),
        }
    )
    with pytest.raises(ValueError, match="HEAD-only.*BASE"):
        assess_issue([base], expectation(), VisibleSource("pricing.py", _SOURCE, ((1, 9),)))


@pytest.mark.parametrize("base_context", [True, False])
def test_empty_case_refuses_base_context_before_reading_a_fixture(base_context):
    request = LLMRequest(
        system_prompt="Review source.",
        user_prompt="Removed source.",
        model="test-model",
        max_tokens=100,
        temperature=0,
        review_id=UUID(int=1),
        node_id="analyze",
        is_eval=True,
        prompt_template_version="test",
        degraded_mode=not base_context,
        degradation_reason=None if base_context else "file_removal",
        context_summary=(
            ContextManifestEntry(
                file_path="pricing.py",
                source_side="base",
                scope_unit_name="removed file",
                line_start=1,
                line_end=9,
                inclusion_reason="changed_region",
            ),
        )
        if base_context
        else (),
    )
    with pytest.raises(ValueError, match="HEAD-only.*BASE"):
        assess_case([], [], fixture_path="nonexistent-fixture.json", request=request)


def test_unit_wide_issue_keeps_quote_location_and_declared_offset_separate() -> None:
    result = assess_issue(
        [finding(start=3, end=4)], expectation(), VisibleSource("pricing.py", _SOURCE, ((1, 9),))
    )
    assert result.status == "matched"
    assert result.matching_indices == (0,)
    assessment = result.findings[0]
    assert assessment.quote_spans == ((2, 3), (7, 8))
    assert assessment.declared_span == (3, 4)
    assert assessment.location_quality == "quote-offset"
    assert assessment.quote_offsets == ((1, 1),)


@pytest.mark.parametrize("control", ["wrong-function", "wrong-type", "wrong-file", "wrong-quote"])
def test_scope_type_file_and_exact_quote_are_independent_gates(control: str) -> None:
    actual = finding()
    if control == "wrong-function":
        actual = finding(start=7, end=8)
    elif control == "wrong-type":
        actual = finding(kind=FindingType.SQL_INJECTION)
    elif control == "wrong-file":
        actual = actual.model_copy(update={"file_path": "other.py"})
    else:
        actual = finding(evidence="    return weight * 99")
    result = assess_issue([actual], expectation(), VisibleSource("pricing.py", _SOURCE, ((1, 9),)))
    assert result.status == "missed"
    assert result.grounded_expected_issue_match is False
    assert not result.matching_indices


@pytest.mark.parametrize(
    "description",
    ["This code may need more tests. Use best practices.", "shipping_cost needs tests."],
)
def test_generic_warning_with_a_real_quote_requires_adjudication(description: str) -> None:
    actual = finding(description=description)
    result = assess_issue([actual], expectation(), VisibleSource("pricing.py", _SOURCE, ((1, 9),)))
    assert result.status == "needs-adjudication"
    assert result.grounded_expected_issue_match is None
    assert result.findings[0].source_grounding == "exact-visible-quote"


@pytest.mark.parametrize("context", ["unknown", "unsupported", "not-visible"])
def test_unsupported_context_never_turns_empty_output_into_a_miss_or_pass(context: str) -> None:
    expected = expectation()
    source = VisibleSource("pricing.py", _SOURCE, ((1, 9),))
    if context == "not-visible":
        source = replace(source, visible_ranges=((6, 9),))
    else:
        expected = replace(expected, context_status=context)
    result = assess_issue([], expected, source)
    assert result.status == "unscored"
    assert result.grounded_expected_issue_match is None
    assert not result.matching_indices


def test_local_defect_cannot_claim_unrelated_code_elsewhere_in_the_function() -> None:
    source = "def work():\n" + "    fetch_profile()\n" * 10
    expected = replace(expectation(FindingType.SQL_INJECTION), scope_name="work", scope_end=11)
    actual = finding(
        start=10,
        end=10,
        kind=FindingType.SQL_INJECTION,
        evidence="    fetch_profile()",
        description="fetch_profile can execute unsafe input.",
    )
    # The same quote also occurs near the expected site. The declared anchor must
    # not let an unrelated far-away operation claim that occurrence.
    result = assess_issue([actual], expected, VisibleSource("pricing.py", source, ((1, 11),)))
    assert result.status == "missed"


def test_tied_repeated_quote_cannot_choose_the_expected_occurrence_for_credit() -> None:
    source = "def work():\n    cursor.execute(query)\n" + "    pass\n" * 5
    source += "    cursor.execute(query)\n"
    base = expectation(FindingType.SQL_INJECTION)
    expected = replace(
        base,
        expected=base.expected.model_copy(update={"line_start": 2, "line_end": 2}),
        scope_name="work",
        scope_end=8,
    )
    actual = finding(
        start=5,
        end=5,
        kind=FindingType.SQL_INJECTION,
        evidence="    cursor.execute(query)",
        description="cursor executes an unsafe query.",
    )
    result = assess_issue([actual], expected, VisibleSource("pricing.py", source, ((1, 8),)))
    assert result.status == "needs-adjudication"
    assert result.grounded_expected_issue_match is None
    assert result.findings[0].quote_spans == ((2, 2), (8, 8))
    assert result.findings[0].quote_offsets == ((3, 3), (-3, -3))
    assert result.findings[0].location_quality == "unverified"


def test_quote_offsets_use_source_lines_with_multibyte_text() -> None:
    source = "# café\ndef shipping_cost(weight):\n    return weight * 1.25\n"
    base = expectation()
    expected = replace(
        base,
        expected=base.expected.model_copy(update={"line_start": 2, "line_end": 2}),
        scope_start=2,
        scope_end=3,
    )
    actual = finding(start=3, end=3, evidence="    return weight * 1.25")
    result = assess_issue([actual], expected, VisibleSource("pricing.py", source, ((2, 3),)))
    assert result.status == "matched"
    assert result.findings[0].quote_spans == ((3, 3),)
    assert result.findings[0].location_quality == "quote-aligned"


@pytest.mark.parametrize("name", ["missing_test_shipping_rates", "missing_error_handling"])
def test_case_context_preserves_unknown_absence_and_error_ownership(name: str) -> None:
    path = Path(__file__).parent / "fixtures/mock_github" / f"{name}.json"
    fixture = json.loads(path.read_text())
    file = fixture["files"][0]
    content = file["content_head"]
    start = 5 if name.startswith("missing_test") else 4
    end = len(content.splitlines())
    scope_name = "shipping_cost" if name.startswith("missing_test") else "fetch_profile"
    body = "\n".join(content.splitlines()[start - 1 : end])
    request = LLMRequest(
        system_prompt="Review the supplied source.",
        user_prompt=(
            f"File: {file['path']}\n### function `{scope_name}` (lines {start}-{end})\n"
            f"```python\n{body}\n```"
        ),
        model="model-name-does-not-affect-scoring",
        max_tokens=100,
        temperature=0,
        review_id=UUID(int=1),
        node_id="analyze",
        is_eval=True,
        prompt_template_version="test",
        degraded_mode=False,
        context_summary=(
            ContextManifestEntry(
                file_path=file["path"],
                scope_unit_name=scope_name,
                line_start=start,
                line_end=end,
                inclusion_reason="changed_scope",
            ),
        ),
    )
    kind = (
        FindingType.MISSING_TEST
        if name.startswith("missing_test")
        else FindingType.MISSING_ERROR_HANDLING
    )
    expected = ExpectedFinding(
        file_path=file["path"],
        line_start=start,
        line_end=start,
        finding_type=kind,
        severity=lookup_severity(kind),
    )
    result = assess_case([], [expected], fixture_path=path, request=request)
    assert result[0].status == "unscored"
    assert result[0].context_status == "unknown"
    assert "not fully verified" not in " ".join(result[0].reasons)


@pytest.mark.parametrize("context", ["intact", "changed-body", "wrong-manifest", "changed-fixture"])
async def test_case_assessment_verifies_real_served_request(context: str, tmp_path: Path) -> None:
    path = "tests/eval/fixtures/mock_github/pygoat_sql_injection.json"
    fixture = json.loads(Path(path).read_text())
    response = json.loads(fixture["llm_responses"]["analyze"][0])
    response["findings"][0].update(evidence_tier="judged", query_match_id=None)
    provider = _ScriptedProvider(json.dumps(response))
    run = await run_analyze_instrumented(
        state_from_eval_fixture(path), provider=provider, model="claude-sonnet-5"
    )
    assert len(provider.calls) == 1
    request = provider.calls[0]
    if context == "changed-body":
        request = request.model_copy(
            update={"user_prompt": request.user_prompt.replace("query =", "other =")}
        )
    elif context == "wrong-manifest":
        request = request.model_copy(
            update={
                "context_summary": tuple(
                    entry.model_copy(update={"scope_unit_name": "unrelated"})
                    for entry in request.context_summary
                )
            }
        )
    fixture_path = Path(path)
    if context == "changed-fixture":
        fixture_path = tmp_path / path
        fixture_path.parent.mkdir(parents=True)
        fixture_path.write_bytes(Path(path).read_bytes() + b" ")
    result = assess_case(
        run.findings, _GROUND_TRUTH_BY_FIXTURE[path], fixture_path=fixture_path, request=request
    )
    assert result[0].status == ("matched" if context == "intact" else "unscored")
    if context == "changed-fixture":
        assert "Fixture bytes changed" in result[0].reasons[0]


def test_every_positive_case_has_an_explicit_context_review() -> None:
    from .model_refresh import scenarios

    ledger_path = Path(__file__).parent / "fixtures/review_quality_context.json"
    reviewed = json.loads(ledger_path.read_text())["cases"]
    positive_cases = {row.path for row in scenarios() if row.expected}
    assert set(reviewed) == positive_cases
    assert all(
        row["status"] in {"supported", "unknown", "unsupported"} and row["reason"]
        for row in reviewed.values()
    )


@pytest.mark.parametrize(
    "case_id,scope_name,scope_kind",
    [
        ("dbt-common-182", "safe_extract._is_within_directory", "function"),
        ("starlette-985", "StaticFiles.lookup_path", "method"),
    ],
)
async def test_real_pr_context_requires_the_reviewed_source_premises(
    case_id: str, scope_name: str, scope_kind: str
) -> None:
    from . import model_refresh

    scenario = next(
        row for row in model_refresh.scenarios("real-pr") if Path(row.path).parent.name == case_id
    )
    provider = _ScriptedProvider('{"findings":[]}')
    await run_analyze_instrumented(
        model_refresh.state_for_scenario(scenario, suite="real-pr"),
        provider=provider,
        model="claude-sonnet-5",
    )
    assert len(provider.calls) == 1
    full_request = provider.calls[0]
    full = assess_case([], scenario.expected, fixture_path=scenario.path, request=full_request)[0]
    assert (full.context_status, full.status) == ("supported", "missed")

    # Keep the exact inner source and its manifest entry while hiding the required caller.
    entry = next(
        item for item in full_request.context_summary if item.scope_unit_name == scope_name
    )
    fixture = json.loads(Path(scenario.path).read_text())
    content = next(
        file["content_head"] for file in fixture["files"] if file["path"] == entry.file_path
    )
    body = "\n".join(content.splitlines()[entry.line_start - 1 : entry.line_end])
    partial_request = LLMRequest.model_validate(
        {
            **full_request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
            "user_prompt": (
                f"File: {entry.file_path}\n"
                f"### {scope_kind} `{scope_name}` (lines {entry.line_start}-{entry.line_end})\n"
                f"```python\n{body}\n```"
            ),
            "context_summary": [entry.model_dump(mode="json")],
        }
    )
    partial = assess_case(
        [], scenario.expected, fixture_path=scenario.path, request=partial_request
    )[0]
    assert (partial.context_status, partial.status) == ("unknown", "unscored")
    assert partial.grounded_expected_issue_match is None
    assert "omits source required" in " ".join(partial.reasons)
