"""FindingEvent proof + severity + computed-field surfaces.

Backs `evidence-tier-schema-enforced` (validator wired at the audit-event
layer per the spec, not just at ReviewFinding), `severity-set-by-policy`,
`finding-type-enum-constrained`, and `confidence-is-computed-not-assigned`
(by absence — confidence is NOT a field on the event; replay re-derives it
from evidence_tier).
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import FindingEvent, compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension


def _build_event(**overrides: Any) -> FindingEvent:
    file_path = overrides.get("file_path", "src/foo.py")
    line_start = overrides.get("line_start", 10)
    line_end = overrides.get("line_end", 12)
    finding_type = overrides.get("finding_type", FindingType.SQL_INJECTION)

    # Compute canonical hash only when finding_type is a real FindingType.
    # Tests that pass invalid finding_type (bare string) hit the enum gate
    # at field-validation time before _verify_content_hash runs, so a
    # placeholder hash is fine here.
    if isinstance(finding_type, FindingType):
        content_hash = compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        )
    else:
        content_hash = "a" * 64

    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "finding_id": uuid4(),
        "finding_type": finding_type,
        "severity": FindingSeverity.CRITICAL,
        "file_path": file_path,
        "line_start": line_start,
        "line_end": line_end,
        "dimension": ReviewDimension.SECURITY,
        "finding_content_hash": content_hash,
        "evidence_tier": EvidenceTier.JUDGED,
        "policy_version": "1.0.0",
    }
    fields.update(overrides)
    return FindingEvent(**fields)


def test_finding_event_carries_evidence_tier() -> None:
    """evidence_tier is required; missing raises."""
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "finding_id": uuid4(),
        "finding_type": FindingType.SQL_INJECTION,
        "severity": FindingSeverity.CRITICAL,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "dimension": ReviewDimension.SECURITY,
        "finding_content_hash": compute_finding_content_hash(
            file_path="src/foo.py",
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        "policy_version": "1.0.0",
    }
    with pytest.raises(ValidationError):
        FindingEvent(**fields)


def test_finding_event_severity_is_finding_severity_enum() -> None:
    """Bare invalid string raises (severity-set-by-policy gate)."""
    with pytest.raises(ValidationError):
        _build_event(severity="catastrophic")


def test_finding_event_finding_type_is_constrained_enum() -> None:
    """Bare invalid string raises (finding-type-enum-constrained)."""
    with pytest.raises(ValidationError):
        _build_event(finding_type="not_a_real_finding_type")


def test_finding_event_has_no_confidence_field() -> None:
    """confidence is NOT a field on the event (confidence-is-computed-not-assigned).

    Replay derives confidence at read time from evidence_tier, mirroring
    the ReviewFinding rule. Storing it here would duplicate state.
    """
    event = _build_event()
    assert "confidence" not in event.model_dump()
    with pytest.raises(ValidationError):
        _build_event(confidence=0.9)


def test_finding_event_observed_admits_with_query_match_id() -> None:
    """Happy path: OBSERVED + non-empty query_match_id constructs cleanly."""
    event = _build_event(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.placeholder",
    )
    assert event.evidence_tier == EvidenceTier.OBSERVED


def test_finding_event_observed_rejects_without_query_match_id() -> None:
    """OBSERVED + None query_match_id raises ValidationError via the validator."""
    with pytest.raises(ValidationError, match="non-empty str query_match_id"):
        _build_event(
            evidence_tier=EvidenceTier.OBSERVED,
            query_match_id=None,
        )


def test_finding_event_inferred_rejects_without_trace_path() -> None:
    """INFERRED + None trace_path raises ValidationError via the validator."""
    with pytest.raises(ValidationError, match="non-empty list"):
        _build_event(
            evidence_tier=EvidenceTier.INFERRED,
            query_match_id=None,
            trace_path=None,
        )


def test_finding_event_judged_admits_without_artifacts() -> None:
    """JUDGED admits without query_match_id or trace_path (no-structural-claim path)."""
    event = _build_event(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=None,
    )
    assert event.evidence_tier == EvidenceTier.JUDGED


def test_finding_event_finding_content_hash_format() -> None:
    """finding_content_hash format gate per spec §8.5: 64 lowercase hex chars.

    Format-only failures (wrong prefix, uppercase, non-hex chars, wrong
    length) all raise via the Field pattern before the canonical-hash
    verifier even runs.
    """
    with pytest.raises(ValidationError):
        _build_event(finding_content_hash="sha256-h")
    with pytest.raises(ValidationError):
        _build_event(finding_content_hash="A" * 64)
    with pytest.raises(ValidationError):
        _build_event(finding_content_hash="g" * 64)
    with pytest.raises(ValidationError):
        _build_event(finding_content_hash="a" * 63)


def test_finding_event_finding_content_hash_must_equal_canonical() -> None:
    """Spec §8.5: hash MUST equal SHA-256 of canonical input tuple.

    Format-only gating accepts any 64-hex string for any input; this
    test verifies the canonical-equality validator catches an emitter
    that supplies a format-valid but non-canonical hash.
    """
    # A format-valid hash that doesn't match the canonical computation.
    bogus_but_format_valid = "a" * 64
    with pytest.raises(ValidationError, match="finding_content_hash mismatch"):
        _build_event(finding_content_hash=bogus_but_format_valid)


def test_compute_finding_content_hash_is_deterministic() -> None:
    """Same input tuple → same hash; different inputs → different hashes."""
    h1 = compute_finding_content_hash(
        file_path="src/foo.py",
        line_start=10,
        line_end=12,
        finding_type=FindingType.SQL_INJECTION,
    )
    h2 = compute_finding_content_hash(
        file_path="src/foo.py",
        line_start=10,
        line_end=12,
        finding_type=FindingType.SQL_INJECTION,
    )
    assert h1 == h2

    h3 = compute_finding_content_hash(
        file_path="src/foo.py",
        line_start=10,
        line_end=13,  # different line_end
        finding_type=FindingType.SQL_INJECTION,
    )
    assert h1 != h3

    h4 = compute_finding_content_hash(
        file_path="src/foo.py",
        line_start=10,
        line_end=12,
        finding_type=FindingType.AUTH_BYPASS,  # different finding_type
    )
    assert h1 != h4


def test_finding_event_line_start_ge_1() -> None:
    """line_start = 0 raises (1-indexed per coordinates/)."""
    with pytest.raises(ValidationError):
        _build_event(line_start=0, line_end=5)


def test_finding_event_line_end_ge_line_start() -> None:
    """line_end < line_start raises via the model_validator."""
    with pytest.raises(ValidationError, match="line_end"):
        _build_event(line_start=10, line_end=5)


def test_finding_event_line_end_equal_line_start_admits() -> None:
    """Single-line findings (line_start == line_end) admit."""
    event = _build_event(line_start=42, line_end=42)
    assert event.line_start == event.line_end == 42


# ---------------------------------------------------------------------------
# Codex round-5 audit fold: severity-set-by-policy gate on FindingEvent.
# ---------------------------------------------------------------------------


def test_finding_event_rejects_severity_drifted_from_policy() -> None:
    """`FindingEvent.severity` must equal SEVERITY_POLICY[finding_type]
    under live policy. Backs `severity-set-by-policy`.

    Codex round-5 audit: pre-fold a row like
    `(SQL_INJECTION, LOW, policy_version="1.0.0")` admitted even though
    SEVERITY_POLICY[SQL_INJECTION] == CRITICAL under policy_version 1.0.0,
    so a policy-invalid event could enter the append-only audit stream.
    """
    with pytest.raises(ValidationError, match="severity-set-by-policy"):
        _build_event(
            finding_type=FindingType.SQL_INJECTION,
            severity=FindingSeverity.LOW,  # policy says CRITICAL
        )


def test_finding_event_admits_severity_matching_policy() -> None:
    """The happy path: severity matches SEVERITY_POLICY[finding_type]."""
    event = _build_event(
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.CRITICAL,
    )
    assert event.severity == FindingSeverity.CRITICAL


def test_finding_event_admits_historical_policy_version() -> None:
    """Under a non-live `policy_version`, the validator skips —
    historical events under a frozen policy carry severity correct at
    write-time per `severity-policy-versioned-for-replay`. Schema layer
    cannot do versioned-replay lookup; that's the persister's job."""
    event = _build_event(
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.LOW,  # would fail under live policy
        policy_version="0.9.0",  # not ACTIVE_POLICY_VERSION
    )
    assert event.severity == FindingSeverity.LOW


# ---------------------------------------------------------------------------
# Category F sweep: FindingEvent.dimension lockstep (mirror of
# ReviewFinding._enforce_dimension_lockstep at the audit-event layer).
# Same gap class as severity-set-by-policy — would have been caught by
# Codex in a follow-up round.
# ---------------------------------------------------------------------------


def test_finding_event_rejects_drifted_dimension() -> None:
    """`(SQL_INJECTION, dimension=PERFORMANCE)` fails because
    FINDING_TYPE_TO_DIMENSION[SQL_INJECTION] == SECURITY. Same shape as
    the existing ReviewFinding test."""
    with pytest.raises(ValidationError, match="drifted from"):
        _build_event(
            finding_type=FindingType.SQL_INJECTION,
            dimension=ReviewDimension.PERFORMANCE,  # WRONG
        )


def test_finding_event_admits_canonical_dimension() -> None:
    """The happy path: dimension matches FINDING_TYPE_TO_DIMENSION."""
    event = _build_event(
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
    )
    assert event.dimension == ReviewDimension.SECURITY


# ---------------------------------------------------------------------------
# Category F sweep continuation: LLMCallEvent.cache_hit ↔ cached_tokens
# bidirectional coupling.
# ---------------------------------------------------------------------------


def test_llm_call_event_rejects_cache_hit_true_with_zero_cached_tokens() -> None:
    """cache_hit MUST equal (cached_tokens > 0). The producer computes
    cache_hit FROM cached_tokens; a divergent event means wrapper drift."""
    from datetime import UTC, datetime

    from outrider.audit.events import ContextManifestEntry, LLMCallEvent

    with pytest.raises(ValidationError, match="cache_hit"):
        LLMCallEvent(
            review_id=uuid4(),
            timestamp=datetime.now(UTC),
            model="claude-sonnet-4-6",
            node_id="analyze",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.01,
            pricing_version="v2",
            latency_ms=1000,
            prompt_hash="a" * 64,
            cache_hit=True,  # WRONG — cached_tokens is 0
            context_summary=(
                ContextManifestEntry(
                    file_path="src/foo.py",
                    scope_unit_name="foo",
                    line_start=1,
                    line_end=10,
                    inclusion_reason="changed_scope",
                ),
            ),
            prompt_template_version="v1",
            system_prompt_hash="b" * 64,
            degraded_mode=False,
        )


def test_llm_call_event_rejects_cache_hit_false_with_positive_cached_tokens() -> None:
    """The other direction: cache_hit=False with cached_tokens>0."""
    from datetime import UTC, datetime

    from outrider.audit.events import ContextManifestEntry, LLMCallEvent

    with pytest.raises(ValidationError, match="cache_hit"):
        LLMCallEvent(
            review_id=uuid4(),
            timestamp=datetime.now(UTC),
            model="claude-sonnet-4-6",
            node_id="analyze",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=42,  # > 0
            cost_usd=0.01,
            pricing_version="v2",
            latency_ms=1000,
            prompt_hash="a" * 64,
            cache_hit=False,  # WRONG — cached_tokens > 0
            context_summary=(
                ContextManifestEntry(
                    file_path="src/foo.py",
                    scope_unit_name="foo",
                    line_start=1,
                    line_end=10,
                    inclusion_reason="changed_scope",
                ),
            ),
            prompt_template_version="v1",
            system_prompt_hash="b" * 64,
            degraded_mode=False,
        )
