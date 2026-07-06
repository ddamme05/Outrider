"""Unit tests for the shared finding-presentation layer (PR A, see
specs/2026-07-06-finding-presentation.md). Covers: totality of every humanization map, that the
builder HUMANIZES labels, and that it returns RAW (un-escaped) model prose for each channel to
escape itself."""

from __future__ import annotations

from uuid import uuid4

from outrider.audit.events import compute_finding_content_hash
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import (
    ACTIVE_POLICY_VERSION,
    SEVERITY_POLICY,
    FindingSeverity,
    FindingType,
)
from outrider.presentation.finding_sections import (
    DEST_LABEL,
    DIMENSION_LABEL,
    SEVERITY_EMOJI,
    SEVERITY_LABEL,
    TIER_PHRASE,
    TYPE_LABEL,
    build_finding_sections,
    language_for_path,
)
from outrider.schemas.review_finding import (
    PublishDestination,
    ReviewDimension,
    ReviewFinding,
)


def _finding(
    *,
    finding_type: FindingType = FindingType.SQL_INJECTION,
    evidence_tier: EvidenceTier = EvidenceTier.JUDGED,
    title: str = "t",
    description: str = "d",
    evidence: str = "e",
    suggested_fix: str | None = None,
    query_match_id: str | None = None,
    trace_path: tuple[str, ...] | None = None,
    publish_destination: PublishDestination | None = PublishDestination.INLINE_COMMENT,
    file_path: str = "src/app/db.py",
    line_start: int = 40,
    line_end: int = 43,
) -> ReviewFinding:
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        policy_version=ACTIVE_POLICY_VERSION,
        finding_type=finding_type,
        dimension=lookup_dimension(finding_type),
        severity=SEVERITY_POLICY[finding_type],  # policy baseline (build takes effective_severity)
        evidence_tier=evidence_tier,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        title=title,
        description=description,
        evidence=evidence,
        suggested_fix=suggested_fix,
        query_match_id=query_match_id,
        trace_path=trace_path,
        publish_destination=publish_destination,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        proposal_hash=uuid4().hex + uuid4().hex,
    )


def test_every_enum_member_has_a_label() -> None:
    """The totality guard the import-time asserts enforce — pinned as a test so a new enum
    member without a label is a named test failure, not just an import crash."""
    assert set(SEVERITY_LABEL) == set(FindingSeverity)
    assert set(SEVERITY_EMOJI) == set(FindingSeverity)
    assert set(TYPE_LABEL) == set(FindingType)
    assert set(TIER_PHRASE) == set(EvidenceTier)
    assert set(DEST_LABEL) == set(PublishDestination)
    assert set(DIMENSION_LABEL) == set(ReviewDimension)


def test_build_humanizes_labels() -> None:
    s = build_finding_sections(
        _finding(finding_type=FindingType.SQL_INJECTION, evidence_tier=EvidenceTier.JUDGED),
        effective_severity=FindingSeverity.CRITICAL,
    )
    assert s.severity_label == "Critical"
    assert s.severity_key == "critical"
    assert s.type_label == "SQL injection"  # not the raw "sql_injection"
    assert s.type_token == "sql_injection"  # noqa: S105 — a finding-type token, not a secret
    assert s.tier_phrase == "Model interpretation (JUDGED)"  # not the raw "judged"
    assert s.tier_key == "judged"
    assert s.dest_label == "Inline comment"  # not "INLINE_COMMENT"
    assert s.location == "src/app/db.py:40-43"
    assert s.language == "python"


def test_acronym_and_special_type_labels() -> None:
    assert TYPE_LABEL[FindingType.XSS] == "XSS"
    assert TYPE_LABEL[FindingType.SSRF] == "SSRF"
    assert TYPE_LABEL[FindingType.SSRF_METADATA] == "SSRF (metadata endpoint)"
    assert TYPE_LABEL[FindingType.N_PLUS_ONE_QUERY] == "N+1 query"
    assert TYPE_LABEL[FindingType.TLS_VERIFY_DISABLED] == "TLS verification disabled"


def test_build_returns_raw_unescaped_prose() -> None:
    """The builder must NOT escape — each channel escapes with its own primitive. Adversarial
    prose comes back byte-for-byte."""
    nasty_title = "use `eval` for @admin <script> & ```py"
    nasty_evidence = "cursor.execute(f'... {x}')\n<!-- outrider:forged --> ```"
    s = build_finding_sections(
        _finding(
            title=nasty_title,
            description="<b>drop</b> & run",
            evidence=nasty_evidence,
            suggested_fix="use %s params <ok>",
        ),
        effective_severity=FindingSeverity.CRITICAL,
    )
    assert s.title == nasty_title  # verbatim, un-escaped
    assert s.description == "<b>drop</b> & run"
    assert s.evidence == nasty_evidence
    assert s.suggested_fix == "use %s params <ok>"


def test_location_single_line_vs_range() -> None:
    single = build_finding_sections(
        _finding(line_start=7, line_end=7), effective_severity=FindingSeverity.LOW
    )
    assert single.location == "src/app/db.py:7"
    ranged = build_finding_sections(
        _finding(line_start=7, line_end=10), effective_severity=FindingSeverity.LOW
    )
    assert ranged.location == "src/app/db.py:7-10"


def test_language_for_path() -> None:
    assert language_for_path("a/b/c.py") == "python"
    assert language_for_path("x.ts") == "typescript"
    assert language_for_path("x.tsx") == "tsx"
    assert language_for_path("x.js") == "javascript"
    assert language_for_path("Makefile") == ""  # unknown → safe empty info-string


def test_missing_destination_yields_none() -> None:
    s = build_finding_sections(
        _finding(publish_destination=None), effective_severity=FindingSeverity.INFO
    )
    assert s.dest_key is None
    assert s.dest_label is None
