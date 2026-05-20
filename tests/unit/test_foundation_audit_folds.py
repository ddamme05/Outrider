# See specs/2026-05-19-analyze-foundation.md — foundation-wide crazy-audit folds.
"""Foundation-wide crazy-audit fold tests.

Pins each fold from the 4-lens foundation-wide audit so a future
refactor that loosens the guarantee fires loud:

- I-1: `outrider.policy` does NOT re-export `dimensions` or `canonical`.
- I-3: `SkipReason.stage()` returns "parser" or "analyze" correctly.
- I-7: `verify_lockstep` is public (no underscore prefix).
- DevEx F1: `ScopeUnit.to_span()` returns a Span with matching bytes.
- DevEx F3: typed hash wrappers (proposal/response/round/candidate)
  produce stable digests + correct byte-level encoding.
- DI F1: schema-level path validators reject non-canonical paths.
- DI F2: `ReviewFinding._enforce_dimension_lockstep` rejects drift.
- DI F4 (cleanup): canonicalize_for_hash rejects BaseModel values.
- Adv M1: `policy_version` field requires bare semver.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from outrider.ast_facts.models import ScopeUnit, SkipReason, Span
from outrider.audit.events import AnalyzeCompletedEvent
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import (
    canonicalize_for_hash,
    compute_candidate_id,
    compute_identity_hash,
    compute_proposal_hash,
    compute_response_hash,
    compute_round_id,
)
from outrider.schemas import AnalysisRound, ReviewDimension, ReviewFinding, TraceCandidate

# ---------------------------------------------------------------------------
# I-1: deep-import-only discipline for canonical + dimensions.
# ---------------------------------------------------------------------------


def test_policy_does_not_reexport_canonical_or_dimensions() -> None:
    """`outrider.policy.__init__.py` deliberately omits both modules to
    prevent two-paths drift + the circular import via review_finding.

    Without this assertion the comment in `policy/__init__.py` is the
    only guard against a future contributor adding `from outrider.policy.dimensions
    import ...` to "complete the namespace" — re-introducing the cycle.
    """
    import outrider.policy as policy_mod

    must_not_export = {
        "FINDING_TYPE_TO_DIMENSION",
        "lookup_dimension",
        "SHA256_HEX_PATTERN",
        "SHA256_HEX_PATTERN_SHORT",
        "compute_identity_hash",
        "canonicalize_for_hash",
        "compute_proposal_hash",
        "compute_response_hash",
        "compute_round_id",
        "compute_candidate_id",
    }
    actually_exported = set(policy_mod.__all__)
    overlap = must_not_export & actually_exported
    assert not overlap, (
        f"outrider.policy re-exports {overlap} which must stay deep-import-only "
        f"to prevent the circular import via outrider.schemas.review_finding. "
        f"See policy/__init__.py module comment."
    )


def test_deep_import_paths_actually_work() -> None:
    """The deep-import escape hatches must remain available."""
    from outrider.policy.canonical import (  # noqa: F401
        SHA256_HEX_PATTERN,
        SHA256_HEX_PATTERN_SHORT,
        canonicalize_for_hash,
        compute_identity_hash,
    )
    from outrider.policy.dimensions import (  # noqa: F401
        FINDING_TYPE_TO_DIMENSION,
        lookup_dimension,
        verify_lockstep,
    )


# ---------------------------------------------------------------------------
# I-3: SkipReason.stage() discriminator.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected_stage"),
    [
        (SkipReason.OVERSIZED, "parser"),
        (SkipReason.VENDORED, "parser"),
        (SkipReason.GENERATED_FILENAME, "parser"),
        (SkipReason.MINIFIED, "parser"),
        (SkipReason.GENERATED_BANNER, "parser"),
        (SkipReason.COST_BUDGET_EXHAUSTED, "analyze"),
        (SkipReason.NO_REVIEWABLE_CONTEXT, "analyze"),
        (SkipReason.NO_CHANGED_SCOPE_UNITS, "analyze"),
    ],
)
def test_skip_reason_stage_returns_correct_axis(reason: SkipReason, expected_stage: str) -> None:
    """The 8 SkipReason values split into 5 parser-stage + 3 analyze-stage."""
    assert reason.stage() == expected_stage


# ---------------------------------------------------------------------------
# DevEx F1: ScopeUnit.to_span() canonical bridge.
# ---------------------------------------------------------------------------


def test_scope_unit_to_span_returns_matching_bytes() -> None:
    su = ScopeUnit(
        unit_id="a" * 64,
        kind="function",
        name="foo",
        qualified_name="m.foo",
        file_path="src/foo.py",
        line_start=10,
        line_end=20,
        byte_start=100,
        byte_end=200,
    )
    span = su.to_span()
    assert isinstance(span, Span)
    assert span.byte_start == 100
    assert span.byte_end == 200


# ---------------------------------------------------------------------------
# DevEx F3: typed hash wrappers.
# ---------------------------------------------------------------------------


def test_compute_proposal_hash_deterministic_on_same_inputs() -> None:
    """Same kwargs → same digest. Wrapper builds the canonical payload
    internally so callers can't drift on field order."""
    h1 = compute_proposal_hash(
        finding_type="sql_injection",
        evidence_tier="JUDGED",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        byte_start=100,
        byte_end=120,
    )
    h2 = compute_proposal_hash(
        finding_type="sql_injection",
        evidence_tier="JUDGED",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        byte_start=100,
        byte_end=120,
    )
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_proposal_hash_field_sensitive() -> None:
    """Different inputs produce different digests."""
    h_a = compute_proposal_hash(
        finding_type="sql_injection",
        evidence_tier="JUDGED",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        byte_start=100,
        byte_end=120,
    )
    h_b = compute_proposal_hash(
        finding_type="xss",  # different
        evidence_tier="JUDGED",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        byte_start=100,
        byte_end=120,
    )
    assert h_a != h_b


def test_compute_response_hash_full_text() -> None:
    """Response hash is sha256 of the FULL response, not a prefix."""
    short = "short response"
    long = short + "x" * 10000
    assert compute_response_hash(short) != compute_response_hash(long)


def test_compute_round_id_keys_include_finding_content_hashes() -> None:
    """Round id changes when a finding's content_hash changes."""
    h_empty = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py",),
        files_skipped=(),
        finding_content_hashes=(),
    )
    h_with = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py",),
        files_skipped=(),
        finding_content_hashes=("a" * 64,),
    )
    assert h_empty != h_with


def test_compute_candidate_id_includes_source_proposal_hash() -> None:
    """Two candidates with same path + reason but different source
    proposal produce different ids — the source proposal is identity-bearing."""
    h_a = compute_candidate_id(
        source_proposal_hash="a" * 64,
        candidate_path="src/foo.py",
        reason="r",
    )
    h_b = compute_candidate_id(
        source_proposal_hash="b" * 64,
        candidate_path="src/foo.py",
        reason="r",
    )
    assert h_a != h_b


# ---------------------------------------------------------------------------
# DI F1: schema-level path validators.
# ---------------------------------------------------------------------------


def _valid_finding() -> ReviewFinding:
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


def test_analysis_round_rejects_traversal_in_files_examined() -> None:
    """`validate_diff_path` rejects `..` traversal at the schema layer.

    Pydantic V2's `field_validator` re-raises the underlying
    `CoordinateError` directly when it derives from Exception (it
    only wraps `ValueError`/`AssertionError`), so the test pins the
    domain error type.
    """
    from outrider.coordinates import CoordinateError

    now = datetime.now(UTC)
    with pytest.raises((ValidationError, CoordinateError)):
        AnalysisRound(
            round_id=compute_identity_hash({"x": 1}),
            pass_index=0,
            findings=(_valid_finding(),),
            files_examined=("../escape.py",),
            files_skipped=(),
            started_at=now,
            ended_at=now,
        )


def test_trace_candidate_rejects_traversal_in_candidate_path() -> None:
    """Same rule on the candidate-side schema."""
    from outrider.coordinates import CoordinateError

    with pytest.raises((ValidationError, CoordinateError)):
        TraceCandidate(
            candidate_id=compute_identity_hash({"a": 1}),
            source_proposal_hash=compute_identity_hash({"b": 1}),
            reason="r",
            candidate_path="../escape.py",
        )


# ---------------------------------------------------------------------------
# DI F2: ReviewFinding.dimension lockstep validator.
# ---------------------------------------------------------------------------


def test_review_finding_rejects_drifted_dimension() -> None:
    """A finding constructed with the wrong dimension for its finding_type
    must fail at validation. Closes the stored-vs-computed gap that would
    otherwise let a stale audit-events payload survive replay reconstruction.
    """
    with pytest.raises(ValidationError, match="drifted from"):
        ReviewFinding(
            finding_id=uuid4(),
            review_id=uuid4(),
            installation_id=12345,
            finding_type=FindingType.SQL_INJECTION,  # → SECURITY
            dimension=ReviewDimension.PERFORMANCE,  # WRONG
            severity=FindingSeverity.CRITICAL,
            file_path="src/foo.py",
            line_start=10,
            line_end=12,
            title="x",
            description="y",
            evidence="z",
            evidence_tier=EvidenceTier.JUDGED,
            policy_version="1.0.0",
            content_hash="a" * 64,
        )


def test_review_finding_admits_canonical_dimension() -> None:
    """The happy path: dimension matches FINDING_TYPE_TO_DIMENSION."""
    finding = _valid_finding()
    assert finding.dimension == ReviewDimension.SECURITY


# ---------------------------------------------------------------------------
# DI F4: canonicalize_for_hash rejects BaseModel values.
# ---------------------------------------------------------------------------


def test_canonicalize_for_hash_rejects_basemodel_value() -> None:
    """A caller that passes a Pydantic model directly hits a typed error
    naming the right escape hatch (`model_dump(mode='json')`)."""

    class _Inner(BaseModel):
        x: int

    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        canonicalize_for_hash({"nested": _Inner(x=1)})


# ---------------------------------------------------------------------------
# Adv M1: policy_version semver pattern.
# ---------------------------------------------------------------------------


def _completed_kwargs_minimum() -> dict[str, object]:
    return {
        "review_id": uuid4(),
        "pass_index": 0,
        "n_files_analyzed": 0,
        "n_files_skipped": 0,
        "n_llm_calls": 0,
        "n_proposals_seen": 0,
        "n_findings_emitted": 0,
        "n_proposals_rejected": 0,
        "n_responses_rejected": 0,
        "n_trace_candidates_emitted": 0,
        "total_input_tokens": 0,
        "total_cached_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_usd": 0.0,
        "pricing_version": "v2",  # NOT bare semver — pricing has its own scheme
        "analyze_model": "claude-sonnet-4-6",
    }


def test_analyze_completed_event_rejects_non_semver_policy_version() -> None:
    """`policy_version` must be bare semver (matches DB CHECK + lifespan
    fingerprint). A bogus value lands in the append-only audit log otherwise."""
    with pytest.raises(ValidationError):
        AnalyzeCompletedEvent(
            policy_version="banana",
            **_completed_kwargs_minimum(),
        )


def test_analyze_completed_event_admits_semver_policy_version() -> None:
    """`1.0.0` matches the bare-semver pattern."""
    event = AnalyzeCompletedEvent(
        policy_version="1.0.0",
        **_completed_kwargs_minimum(),
    )
    assert event.policy_version == "1.0.0"


def test_analyze_completed_event_pricing_version_remains_free_form() -> None:
    """Pricing has its own scheme (`v2`); the semver pattern is policy-only."""
    event = AnalyzeCompletedEvent(
        policy_version="1.0.0",
        **_completed_kwargs_minimum(),
    )
    assert event.pricing_version == "v2"


# ---------------------------------------------------------------------------
# Post-foundation push/PR audit folds.
# ---------------------------------------------------------------------------


def test_compute_round_id_is_order_insensitive() -> None:
    """compute_round_id sorts inputs internally so two producers emitting
    the same logical round in different orders get the same id. Without
    this, dedup-by-round_id admits both as distinct rounds on replay."""
    from outrider.policy.canonical import compute_round_id

    h_a = compute_round_id(
        pass_index=0,
        files_examined=("src/a.py", "src/b.py"),
        files_skipped=(),
        finding_content_hashes=("a" * 64, "b" * 64),
    )
    h_b = compute_round_id(
        pass_index=0,
        files_examined=("src/b.py", "src/a.py"),
        files_skipped=(),
        finding_content_hashes=("b" * 64, "a" * 64),
    )
    assert h_a == h_b


def test_analysis_round_round_id_bound_to_payload() -> None:
    """Arbitrary 64-hex round_id rejected even though it matches the
    pattern — the validator re-derives the canonical id and asserts
    equality. Closes the bypass-replay-dedup risk."""
    from outrider.policy.canonical import compute_round_id

    now = datetime.now(UTC)
    finding = _valid_finding()
    with pytest.raises(ValidationError, match="does not match the canonical id"):
        AnalysisRound(
            round_id="0" * 64,
            pass_index=0,
            findings=(finding,),
            files_examined=("src/foo.py",),
            files_skipped=(),
            started_at=now,
            ended_at=now,
        )
    # Canonical id is admitted.
    canonical_id = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py",),
        files_skipped=(),
        finding_content_hashes=(finding.content_hash,),
    )
    AnalysisRound(
        round_id=canonical_id,
        pass_index=0,
        findings=(finding,),
        files_examined=("src/foo.py",),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )


def test_validate_diff_path_normalizes_nfc() -> None:
    """NFC normalization per spec §1: NFD and NFC forms collapse to the
    same canonical output so downstream identity hashes don't diverge
    across systems with different default normalization.

    Constructs the NFD form at runtime via unicodedata so source-file
    save-time normalization can't silently collapse the test literals.
    """
    import unicodedata

    from outrider.coordinates import validate_diff_path

    # NFC: 'é' as single code point U+00E9.
    nfc_path = unicodedata.normalize("NFC", "src/cafe\u0301/foo.py")
    # NFD: 'e' + combining acute U+0301.
    nfd_path = unicodedata.normalize("NFD", nfc_path)
    assert nfc_path != nfd_path
    assert unicodedata.normalize("NFC", nfd_path) == nfc_path

    # validate_diff_path collapses them to the same canonical output.
    assert validate_diff_path(nfc_path) == validate_diff_path(nfd_path)
    assert validate_diff_path(nfd_path) == nfc_path


def test_validate_diff_path_ascii_is_idempotent_under_nfc() -> None:
    """Pure-ASCII paths are NFC-idempotent; pin so a refactor dropping
    the step doesn't slip past."""
    from outrider.coordinates import validate_diff_path

    assert validate_diff_path("src/foo.py") == "src/foo.py"


def test_llm_call_event_rejects_degraded_mode_on_non_analyze_node() -> None:
    """LLMCallEvent mirrors LLMRequest's analyze-only scoping. Closes the
    read-boundary asymmetry where the audit-event schema admitted
    `trace + degraded_mode=True` even though the request rejected it."""
    from outrider.audit.events import ContextManifestEntry, LLMCallEvent

    ctx = (
        ContextManifestEntry(
            file_path="src/foo.py",
            scope_unit_name="Foo.bar",
            line_start=1,
            line_end=10,
            inclusion_reason="changed_scope",
        ),
    )
    with pytest.raises(ValidationError, match="only valid for node_id='analyze'"):
        LLMCallEvent(
            review_id=uuid4(),
            model="claude-haiku-4-5",
            node_id="trace",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.01,
            pricing_version="v2",
            latency_ms=250,
            prompt_hash="a" * 64,
            cache_hit=False,
            context_summary=ctx,
            prompt_template_version="trace:1",
            system_prompt_hash="b" * 64,
            degraded_mode=True,
            degradation_reason="parse_failed",
        )
