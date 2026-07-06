# See specs/2026-07-06-finding-presentation.md
"""Canonical finding → structured display sections, shared by the GitHub, Slack, and
dashboard renderers.

`FindingSections` is STRUCTURED DATA ONLY — never markdown/mrkdwn/JSX and never pre-escaped.
It carries safe developer-authored display LABELS (severity_label, type_label, tier_phrase, …)
alongside the RAW model-prose fields (title, description, evidence, suggested_fix) that each
renderer escapes with its OWN primitive (`sanitize_display_string` / `render_fenced_block` for
GitHub, `_escape_mrkdwn` for Slack, React text-nodes for the dashboard). A raw display string is
NEVER shared across channels.

The humanization maps are the single source of truth for the labels Python renders (GitHub +
Slack). The dashboard mirrors them in `dashboard/src/lib/findingSections.ts`; a parity fixture
keeps the two in lockstep. Eligibility / HITL-outcome phrasing is dashboard-only and lives in
the TS mirror. An import-time totality assert makes a new enum member without a label crash
loudly (mirrors `policy.severity`'s `_V1_SEVERITY_GATE`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity, FindingType
from outrider.schemas.review_finding import PublishDestination, ReviewDimension

if TYPE_CHECKING:
    from collections.abc import Mapping
    from enum import Enum

    from outrider.schemas.review_finding import ReviewFinding

# --- Humanization maps (keyed by enum member; totality-asserted at import) ------------------

SEVERITY_LABEL: Final[dict[FindingSeverity, str]] = {
    FindingSeverity.CRITICAL: "Critical",
    FindingSeverity.HIGH: "High",
    FindingSeverity.MEDIUM: "Medium",
    FindingSeverity.LOW: "Low",
    FindingSeverity.INFO: "Info",
}

# Decorative cue only — `severity_label` is the authoritative text (never color/emoji-only).
SEVERITY_EMOJI: Final[dict[FindingSeverity, str]] = {
    FindingSeverity.CRITICAL: "🔴",
    FindingSeverity.HIGH: "🟠",
    FindingSeverity.MEDIUM: "🟡",
    FindingSeverity.LOW: "🔵",
    FindingSeverity.INFO: "⚪",
}

# Acronym-correct humanized names for all 22 FindingType members.
TYPE_LABEL: Final[dict[FindingType, str]] = {
    FindingType.SQL_INJECTION: "SQL injection",
    FindingType.XSS: "XSS",
    FindingType.HARDCODED_SECRET: "Hardcoded secret",
    FindingType.AUTH_BYPASS: "Auth bypass",
    FindingType.PATH_TRAVERSAL: "Path traversal",
    FindingType.MISSING_INPUT_VALIDATION: "Missing input validation",
    FindingType.N_PLUS_ONE_QUERY: "N+1 query",
    FindingType.BLOCKING_CALL_IN_ASYNC: "Blocking call in async",
    FindingType.UNUSED_IMPORT: "Unused import",
    FindingType.MISSING_ERROR_HANDLING: "Missing error handling",
    FindingType.MISSING_TEST: "Missing test",
    FindingType.DEPRECATED_API: "Deprecated API",
    FindingType.COMMAND_INJECTION: "Command injection",
    FindingType.UNSAFE_DESERIALIZATION: "Unsafe deserialization",
    FindingType.TLS_VERIFY_DISABLED: "TLS verification disabled",
    FindingType.WEAK_CRYPTO: "Weak cryptography",
    FindingType.WEAK_PASSWORD_HASH: "Weak password hash",
    FindingType.INSECURE_RANDOMNESS: "Insecure randomness",
    FindingType.SSRF: "SSRF",
    FindingType.SSRF_METADATA: "SSRF (metadata endpoint)",
    FindingType.OPEN_REDIRECT: "Open redirect",
    FindingType.OPEN_REDIRECT_AUTHED: "Open redirect (authenticated)",
}

# Human-readable evidence-tier phrasing — replaces the raw lowercase enum / the "PROOF · JUDGED"
# slug. Display of an already-decided tier; never model-set, never re-derived.
TIER_PHRASE: Final[dict[EvidenceTier, str]] = {
    EvidenceTier.OBSERVED: "Structural match (OBSERVED)",
    EvidenceTier.INFERRED: "Traced (INFERRED)",
    EvidenceTier.JUDGED: "Model interpretation (JUDGED)",
}

DEST_LABEL: Final[dict[PublishDestination, str]] = {
    PublishDestination.INLINE_COMMENT: "Inline comment",
    PublishDestination.REVIEW_BODY: "Review summary",
    PublishDestination.DASHBOARD_ONLY: "Dashboard only",
}

DIMENSION_LABEL: Final[dict[ReviewDimension, str]] = {
    ReviewDimension.CODE_QUALITY: "Code quality",
    ReviewDimension.SECURITY: "Security",
    ReviewDimension.PERFORMANCE: "Performance",
    ReviewDimension.TEST_COVERAGE: "Test coverage",
    ReviewDimension.BEST_PRACTICES: "Best practices",
}

# Fence info-string per file extension. "" (no language) is a safe default — the fence still
# renders; it just gets no syntax highlight. Extension → highlight token only, no path math.
_LANGUAGE_BY_EXT: Final[dict[str, str]] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
}


def language_for_path(file_path: str) -> str:
    """Highlight token for a file path's fenced snippet, or "" when unknown. Pure suffix
    lookup — no filesystem access, no coordinate math."""
    return _LANGUAGE_BY_EXT.get(PurePosixPath(file_path).suffix.lower(), "")


def _assert_total(mapping: Mapping[Any, str], enum: type[Enum], name: str) -> None:
    """Fail loud at import if any enum member lacks a display label (the fail-closed totality
    guard — a new FindingType/severity/tier/etc. cannot ship unlabeled)."""
    missing = set(enum) - set(mapping)
    if missing:
        raise RuntimeError(
            f"{name} is missing display labels for {sorted(m.value for m in missing)} — "
            f"every {enum.__name__} member needs one (presentation/finding_sections.py)."
        )


_assert_total(SEVERITY_LABEL, FindingSeverity, "SEVERITY_LABEL")
_assert_total(SEVERITY_EMOJI, FindingSeverity, "SEVERITY_EMOJI")
_assert_total(TYPE_LABEL, FindingType, "TYPE_LABEL")
_assert_total(TIER_PHRASE, EvidenceTier, "TIER_PHRASE")
_assert_total(DEST_LABEL, PublishDestination, "DEST_LABEL")
_assert_total(DIMENSION_LABEL, ReviewDimension, "DIMENSION_LABEL")


# --- Structured sections --------------------------------------------------------------------


@dataclass(frozen=True)
class FindingSections:
    """Display sections for one finding. LABEL fields are safe developer-authored strings;
    the *_RAW model-prose fields (title, description, evidence, suggested_fix, and file_path
    off the diff) are attacker-influenced and MUST be escaped by each renderer with its own
    primitive. Structured data only — no markdown/mrkdwn/JSX, no pre-escaping."""

    # header
    severity_key: str  # "critical" — for CSS/emoji lookup
    severity_label: str  # "Critical" — authoritative text
    severity_emoji: str  # decorative cue
    type_token: str  # "sql_injection" — machine/marker use
    type_label: str  # "SQL injection"
    title: str  # RAW
    # location
    file_path: str  # RAW (validated but attacker-influenced off the diff)
    line_start: int
    line_end: int
    location: str  # "file:line" | "file:start-end"
    # summary
    description: str  # RAW
    # evidence
    evidence: str  # RAW
    language: str  # fence info-string ("" when unknown)
    # proof (permanent — survives content redaction)
    tier_key: str  # "judged"
    tier_phrase: str  # "Model interpretation (JUDGED)"
    query_match_id: str | None
    trace_path: tuple[str, ...] | None
    # remediation
    suggested_fix: str | None  # RAW
    # routing + dimension (secondary meta)
    dest_key: str | None
    dest_label: str | None
    dimension_key: str
    dimension_label: str


def build_finding_sections(
    finding: ReviewFinding, *, effective_severity: FindingSeverity
) -> FindingSections:
    """Derive display sections from a canonical `ReviewFinding`.

    `effective_severity` is an INPUT (the post-HITL/policy value resolved by the caller) — this
    layer displays it, never derives it. Returns RAW prose fields un-escaped; each channel
    escapes. Pure: no I/O, no coordinate math, no SDK.
    """
    dest = finding.publish_destination
    location = (
        f"{finding.file_path}:{finding.line_start}"
        if finding.line_start == finding.line_end
        else f"{finding.file_path}:{finding.line_start}-{finding.line_end}"
    )
    return FindingSections(
        severity_key=effective_severity.value,
        severity_label=SEVERITY_LABEL[effective_severity],
        severity_emoji=SEVERITY_EMOJI[effective_severity],
        type_token=finding.finding_type.value,
        type_label=TYPE_LABEL[finding.finding_type],
        title=finding.title,
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        location=location,
        description=finding.description,
        evidence=finding.evidence,
        language=language_for_path(finding.file_path),
        tier_key=finding.evidence_tier.value,
        tier_phrase=TIER_PHRASE[finding.evidence_tier],
        query_match_id=finding.query_match_id,
        trace_path=finding.trace_path,
        suggested_fix=finding.suggested_fix,
        dest_key=dest.value if dest is not None else None,
        dest_label=DEST_LABEL[dest] if dest is not None else None,
        dimension_key=finding.dimension.value,
        dimension_label=DIMENSION_LABEL[finding.dimension],
    )


__all__ = [
    "DEST_LABEL",
    "DIMENSION_LABEL",
    "SEVERITY_EMOJI",
    "SEVERITY_LABEL",
    "TIER_PHRASE",
    "TYPE_LABEL",
    "FindingSections",
    "build_finding_sections",
    "language_for_path",
]
