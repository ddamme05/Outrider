"""Assess grounded issue matches separately from quote and location accuracy.

These checks establish type, scope and source support, not semantic correctness.
Context-dependent expectations remain unscored when the served input cannot
establish them. Frozen v1 grades are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from keyword import iskeyword
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from outrider.audit.events import DegradationReason
from outrider.coordinates import query_span_to_source_lines
from outrider.policy import FindingType

from .grading import DEFAULT_LINE_WINDOW, ExpectedFinding

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.llm.base import LLMRequest
    from outrider.schemas.review_finding import ReviewFinding

ContextStatus = Literal["supported", "unknown", "unsupported"]
MatchStatus = Literal["matched", "missed", "needs-adjudication", "unscored"]
MEASUREMENT_CONTRACT_V2 = "refresh-grounded-visible-context-v2"


@dataclass(frozen=True, slots=True)
class IssueExpectation:
    expected: ExpectedFinding
    scope_name: str
    scope_start: int
    scope_end: int
    context_status: ContextStatus = "unknown"
    context_reason: str = "The visible context has not established this expectation."

    def __post_init__(self) -> None:
        if not 1 <= self.scope_start <= self.scope_end:
            raise ValueError("expected scope must be a nonempty source-line range")
        if (
            not self.scope_start
            <= self.expected.line_start
            <= self.expected.line_end
            <= self.scope_end
        ):
            raise ValueError("expected defect site must be inside its scope")
        if self.context_status not in {"supported", "unknown", "unsupported"}:
            raise ValueError("unknown context status")


@dataclass(frozen=True, slots=True)
class VisibleSource:
    file_path: str
    content: str | None
    visible_ranges: tuple[tuple[int, int], ...]

    def contains(self, start: int, end: int) -> bool:
        if self.content is None or not 1 <= start <= end <= len(self.content.splitlines()):
            return False
        return all(
            any(lo <= line <= hi for lo, hi in self.visible_ranges)
            for line in range(start, end + 1)
        )


@dataclass(frozen=True, slots=True)
class FindingAssessment:
    finding_index: int
    grounded_expected_issue_match: bool
    source_grounding: Literal["exact-visible-quote", "not-grounded"]
    quote_spans: tuple[tuple[int, int], ...]
    declared_span: tuple[int, int]
    location_quality: Literal["quote-aligned", "quote-offset", "outside-scope", "unverified"]
    quote_offsets: tuple[tuple[int, int], ...]
    needs_adjudication: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IssueAssessment:
    file_path: str
    finding_type: str
    status: MatchStatus
    grounded_expected_issue_match: bool | None
    matching_indices: tuple[int, ...]
    findings: tuple[FindingAssessment, ...]
    context_status: ContextStatus
    reasons: tuple[str, ...]


def _quote_spans(evidence: str, source: str) -> tuple[tuple[int, int], ...]:
    if not evidence.strip():
        return ()
    source_bytes, quote = source.encode("utf-8"), evidence.encode("utf-8")
    spans, offset = [], 0
    while (start := source_bytes.find(quote, offset)) >= 0:
        spans.append(
            query_span_to_source_lines(
                byte_start=start, byte_end=start + len(quote), head_content=source
            )
        )
        offset = start + 1
    return tuple(spans)


def _has_code_specific_claim(finding: ReviewFinding, scope_name: str) -> bool:
    """Reject generic prose; a concrete code noun is not semantic adjudication."""
    prose = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", finding.title + " " + finding.description))
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", finding.evidence))
    identifiers -= set(scope_name.split("."))
    nouns = {part for identifier in identifiers for part in identifier.split("_")}
    return any(len(noun) > 2 and not iskeyword(noun) and noun in prose for noun in nouns)


def assess_issue(
    findings: Sequence[ReviewFinding], expectation: IssueExpectation, source: VisibleSource
) -> IssueAssessment:
    """Assess one expectation, retaining uncertain omissions and placement errors.

    Local defects require quoted evidence at the expected site. A missing-test
    claim can instead quote the relevant behavior within its named scope.
    Inputs are admitted findings; this scorer does not rerun production admission.
    """
    findings = tuple(findings)
    if any(finding.source_side != "head" for finding in findings):
        raise ValueError("HEAD-only visible-context grading does not support BASE findings")
    expected = expectation.expected
    assessments = []
    for index, finding in enumerate(findings):
        reasons = []
        identity_matches = (
            finding.file_path == expected.file_path
            and finding.finding_type == expected.finding_type
        )
        if not identity_matches:
            reasons.append("finding type or file differs from the expectation")
        declared = (finding.line_start, finding.line_end)
        in_scope = expectation.scope_start <= declared[0] <= declared[1] <= expectation.scope_end
        if not in_scope:
            reasons.append("declared location is outside the expected scope")
        spans = (
            _quote_spans(finding.evidence, source.content)
            if source.content is not None and finding.file_path == source.file_path
            else ()
        )
        visible = tuple(span for span in spans if source.contains(*span))
        in_scope_quotes = tuple(
            span
            for span in visible
            if expectation.scope_start <= span[0] <= span[1] <= expectation.scope_end
        )
        # Repeated snippets belong to their nearest declared occurrence. A quote
        # at a distant unrelated call must not claim a duplicate near the target.
        nearest = min(
            (abs(declared[0] - lo) + abs(declared[1] - hi) for lo, hi in in_scope_quotes),
            default=None,
        )
        nearest_quotes = tuple(
            span
            for span in in_scope_quotes
            if abs(declared[0] - span[0]) + abs(declared[1] - span[1]) == nearest
        )
        ambiguous_quote = len(nearest_quotes) > 1
        relevant = tuple(
            span
            for span in nearest_quotes
            if (
                expected.finding_type is FindingType.MISSING_TEST
                or span[0] <= expected.line_end + DEFAULT_LINE_WINDOW
                and expected.line_start - DEFAULT_LINE_WINDOW <= span[1]
            )
        )
        if not visible:
            reasons.append("evidence is not an exact quote from visible source")
        elif not relevant:
            reasons.append("quoted code does not cover the expected defect site or relevant scope")
        if ambiguous_quote:
            reasons.append("repeated quote has tied nearest occurrences; needs adjudication")
        specific = _has_code_specific_claim(finding, expectation.scope_name)
        if not specific:
            reasons.append("generic claim needs semantic adjudication")
        offsets = tuple((declared[0] - lo, declared[1] - hi) for lo, hi in nearest_quotes)
        location: Literal["quote-aligned", "quote-offset", "outside-scope", "unverified"] = (
            "outside-scope" if not in_scope else "unverified"
        )
        if in_scope and relevant and not ambiguous_quote:
            location = "quote-aligned" if (0, 0) in offsets else "quote-offset"
        grounded = identity_matches and in_scope and bool(relevant)
        assessments.append(
            FindingAssessment(
                finding_index=index,
                grounded_expected_issue_match=grounded and specific and not ambiguous_quote,
                source_grounding="exact-visible-quote" if visible else "not-grounded",
                quote_spans=spans,
                declared_span=declared,
                location_quality=location,
                quote_offsets=offsets,
                needs_adjudication=grounded and (not specific or ambiguous_quote),
                reasons=tuple(reasons),
            )
        )

    context_status = expectation.context_status
    reasons = [expectation.context_reason]
    if (
        source.file_path != expected.file_path
        or source.content is None
        or not source.contains(expectation.scope_start, expectation.scope_end)
    ):
        context_status = "unknown"
        reasons.append("the expected scope is not fully verified in the served source")
    matching = tuple(a.finding_index for a in assessments if a.grounded_expected_issue_match)
    status: MatchStatus
    if context_status != "supported":
        status, matched = "unscored", None
        matching = ()
    elif matching:
        status, matched = "matched", True
    elif any(a.needs_adjudication for a in assessments):
        status, matched = "needs-adjudication", None
    else:
        status, matched = "missed", False
    return IssueAssessment(
        file_path=expected.file_path,
        finding_type=expected.finding_type.value,
        status=status,
        grounded_expected_issue_match=matched,
        matching_indices=matching,
        findings=tuple(assessments),
        context_status=context_status,
        reasons=tuple(reasons),
    )


def _context_support(
    fixture_path: Path,
    fixture_bytes: bytes,
    expected: ExpectedFinding,
    verified_ranges: tuple[tuple[int, int], ...],
) -> tuple[ContextStatus, str]:
    from .grounded_cases import context_for_case
    from .real_pr_cases import context_for_case as real_pr_context

    real_pr = real_pr_context(fixture_path, expected, verified_ranges=verified_ranges)
    if real_pr is not None:
        return real_pr
    grounded = context_for_case(fixture_path, expected)
    if grounded is not None:
        return grounded
    ledger = json.loads(
        Path(__file__).with_name("fixtures").joinpath("review_quality_context.json").read_text()
    )
    if ledger.get("schema") != "review-quality-context-v1":
        raise ValueError("unsupported visible-context ledger")
    path = fixture_path.as_posix()
    entries = [
        entry
        for name, entry in ledger["cases"].items()
        if path == name or path.endswith("/" + name)
    ]
    if len(entries) != 1:
        return "unknown", "This fixture has no reviewed visible-context premise."
    entry = entries[0]
    if hashlib.sha256(fixture_bytes).hexdigest() != entry["fixture_sha256"]:
        return "unknown", "Fixture bytes changed after visible-context review."
    if any(
        entry[key] != getattr(expected, key)
        for key in ("finding_type", "file_path", "line_start", "line_end")
    ):
        return "unknown", "The expectation changed after visible-context review."
    if entry["status"] not in {"supported", "unknown", "unsupported"}:
        raise ValueError("unsupported visible-context status")
    return entry["status"], entry["reason"]


def assess_case(
    findings: Sequence[ReviewFinding],
    expected: Sequence[ExpectedFinding],
    *,
    fixture_path: str | Path,
    request: LLMRequest,
) -> tuple[IssueAssessment, ...]:
    """Use only source scopes verified in this exact rendered review request.

    A reviewed, byte-bound context ledger records each expectation's premises.
    Unreviewed inputs and missing premises stay unscored, even with empty outputs.
    """
    findings = tuple(findings)
    if (
        request.degradation_reason == DegradationReason.FILE_REMOVAL
        or any(finding.source_side != "head" for finding in findings)
        or any(entry.source_side != "head" for entry in request.context_summary)
    ):
        raise ValueError("HEAD-only visible-context grading does not support BASE inputs")
    fixture_path = Path(fixture_path)
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    files = {f["path"]: f.get("content_head") for f in fixture["files"]}
    prompt = request.user_prompt
    scopes = []
    pattern = re.compile(
        r"^### (?:function|method|class) `([^`]+)` \(lines (\d+)-(\d+)\)\n"
        r"(`{3,})[^\n]*\n(.*?)\n\4$",
        re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(prompt):
        name, lo, hi, _, body = match.groups()
        scopes.append((name, int(lo), int(hi), body))
    assessments = []
    for item in expected:
        content = files.get(item.file_path)
        lines = content.splitlines() if content is not None else []
        verified = [
            (name, lo, hi)
            for name, lo, hi, body in scopes
            if f"File: {item.file_path}\n" in prompt
            and any(
                entry.file_path == item.file_path
                and entry.scope_unit_name == name
                and entry.line_start == lo
                and entry.line_end == hi
                for entry in request.context_summary
            )
            and 1 <= lo <= hi <= len(lines)
            and "\n".join(lines[lo - 1 : hi]) == body.rstrip("\n")
        ]
        containing = [
            scope for scope in verified if scope[1] <= item.line_start <= item.line_end <= scope[2]
        ]
        scope = (
            min(containing, key=lambda s: s[2] - s[1])
            if containing
            else ("", item.line_start, item.line_end)
        )
        status, reason = _context_support(
            fixture_path, fixture_bytes, item, tuple((lo, hi) for _, lo, hi in verified)
        )
        expectation = IssueExpectation(
            expected=item,
            scope_name=scope[0],
            scope_start=scope[1],
            scope_end=scope[2],
            context_status=status,
            context_reason=reason,
        )
        source = VisibleSource(item.file_path, content, tuple((lo, hi) for _, lo, hi in verified))
        assessments.append(assess_issue(findings, expectation, source))
    return tuple(assessments)
