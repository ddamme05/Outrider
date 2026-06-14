"""ReviewFinding model configuration: extra='forbid', NOT frozen, validate_assignment.

Four rule families covered here:
  - Pydantic config: extra='forbid' rejects unknown fields; frozen is
    deliberately OFF (multi-stage lifecycle — see review_finding.py
    module docstring); validate_assignment=True so post-construction
    writes re-run model_validators + Field constraints + enum typing.
  - Enum gates: invalid string values raise (at construction AND on
    assignment). Pydantic V2 coerces VALID string values to enum members
    (that's fine; the resulting field is still an enum instance, so
    `severity-set-by-policy` and `finding-type-enum-constrained` still
    hold). The gate that matters is rejection of invalid values.
  - Line constraints: line_start ≥ 1, line_end ≥ line_start.
  - Validate-on-assignment: lifecycle setters cannot bypass the proof
    boundary, the line constraint, or the enum gates.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import PublishDestination, ReviewDimension, ReviewFinding


def _build_finding(**overrides: Any) -> ReviewFinding:
    """Construct a valid finding; overrides replace defaults.

    `content_hash` recomputed from the post-override payload so the
    `_verify_content_hash` validator doesn't fire on tests that
    override identity-tuple fields. Tests that DELIBERATELY exercise
    hash drift override `content_hash` directly.
    """
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "policy_version": ACTIVE_POLICY_VERSION,
        "finding_type": FindingType.SQL_INJECTION,
        "dimension": ReviewDimension.SECURITY,
        "severity": FindingSeverity.CRITICAL,
        "evidence_tier": EvidenceTier.JUDGED,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "title": "t",
        "description": "d",
        "evidence": "e",
        # Per DECISIONS.md#025: admitted findings carry proposal_hash.
        "proposal_hash": "a" * 64,
    }
    fields.update(overrides)
    if "content_hash" not in overrides and isinstance(fields["finding_type"], FindingType):
        # Only auto-compute when finding_type is a real enum member —
        # tests that exercise invalid `finding_type=<str>` rejection
        # need Pydantic's field validator to fire FIRST, not crash in
        # the helper. The placeholder hash below fails the new
        # `_verify_content_hash` validator, but Pydantic catches the
        # invalid finding_type at the field layer before the model
        # validator runs.
        fields["content_hash"] = compute_finding_content_hash(
            file_path=fields["file_path"],
            line_start=fields["line_start"],
            line_end=fields["line_end"],
            finding_type=fields["finding_type"],
        )
    elif "content_hash" not in overrides:
        # finding_type override is invalid — Pydantic will reject; the
        # helper just supplies any valid-shape hash to get there.
        fields["content_hash"] = "a" * 64
    return ReviewFinding(**fields)


def test_review_finding_extra_forbid() -> None:
    """Unknown fields raise ValidationError per docs/conventions.md."""
    with pytest.raises(ValidationError, match="extra"):
        _build_finding(unknown_field="oops")  # type: ignore[call-arg]


def test_review_finding_is_mutable_for_lifecycle() -> None:
    """ReviewFinding is NOT frozen: lifecycle stages set fields after construction.

    Regression guard against a future PR that adds frozen=True without
    recognizing the lifecycle implication. `coordinates/` sets
    publish_destination; HITL flow sets the override fields. Both rely
    on mutation. See review_finding.py module docstring + the
    schemas-module spec's Compliance notes for the full rationale.
    """
    finding = _build_finding()
    assert finding.publish_destination is None
    finding.publish_destination = PublishDestination.INLINE_COMMENT
    assert finding.publish_destination == PublishDestination.INLINE_COMMENT


def test_review_finding_severity_rejects_invalid_string() -> None:
    """A string value not in FindingSeverity raises."""
    with pytest.raises(ValidationError):
        _build_finding(severity="catastrophic")


def test_review_finding_severity_accepts_enum_member() -> None:
    """FindingSeverity enum member admits cleanly when it matches the
    policy baseline for the finding_type.

    Uses HARDCODED_SECRET (whose policy severity IS HIGH) so the
    severity field exercise stays meaningful while passing the
    `_enforce_severity_matches_policy` validator. An earlier shape of
    this test used `SQL_INJECTION + HIGH`, which the policy gate
    correctly rejects.
    """
    finding = _build_finding(
        finding_type=FindingType.HARDCODED_SECRET,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.HIGH,
    )
    assert finding.severity == FindingSeverity.HIGH


def test_review_finding_finding_type_rejects_invalid_string() -> None:
    """A string value not in FindingType raises (backs finding-type-enum-constrained)."""
    with pytest.raises(ValidationError):
        _build_finding(finding_type="not_a_real_finding_type")


def test_review_finding_line_start_ge_1() -> None:
    """line_start = 0 raises (1-indexed per coordinates/)."""
    with pytest.raises(ValidationError):
        _build_finding(line_start=0, line_end=5)


def test_review_finding_line_end_ge_line_start() -> None:
    """line_end < line_start raises via the model_validator."""
    with pytest.raises(ValidationError, match="line_end"):
        _build_finding(line_start=10, line_end=5)


def test_review_finding_line_end_equal_line_start_admits() -> None:
    """Single-line findings (line_start == line_end) admit."""
    finding = _build_finding(line_start=42, line_end=42)
    assert finding.line_start == 42
    assert finding.line_end == 42


def test_review_finding_validate_assignment_blocks_invalid_severity_string() -> None:
    """Post-construction, assigning an invalid string to .severity raises.

    Without validate_assignment=True, `finding.severity = "garbage"` would
    silently admit because Pydantic does not revalidate by default. With
    it, the assignment runs the same enum-coercion check as construction.
    """
    finding = _build_finding()
    with pytest.raises(ValidationError):
        finding.severity = "catastrophic"  # type: ignore[assignment]


def test_review_finding_validate_assignment_blocks_invalid_publish_destination() -> None:
    """Lifecycle assignment of publish_destination revalidates the enum."""
    finding = _build_finding()
    with pytest.raises(ValidationError):
        finding.publish_destination = "broadcast"  # type: ignore[assignment]


def test_review_finding_validate_assignment_runs_proof_boundary() -> None:
    """Stripping query_match_id from an OBSERVED finding post-construction raises.

    The proof-boundary model_validator runs on assignment, so the lifecycle
    cannot wash out the OBSERVED → query_match_id requirement after the fact.
    """
    finding = _build_finding(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.placeholder",
    )
    with pytest.raises(ValidationError, match="non-empty str query_match_id"):
        finding.query_match_id = None


def test_review_finding_validate_assignment_runs_line_constraint() -> None:
    """Setting line_end below line_start post-construction raises."""
    finding = _build_finding(line_start=10, line_end=20)
    with pytest.raises(ValidationError, match="line_end"):
        finding.line_end = 5


# ---------------------------------------------------------------------------
# PR-review round 5: convergent input-boundary tightening on ReviewFinding.
# ---------------------------------------------------------------------------


def test_review_finding_policy_version_rejects_non_semver() -> None:
    """`policy_version` must match the strict bare-semver pattern matching
    the audit-event side + the DB CHECK; without this, a finding could
    carry a `policy_version` that the audit-event for the same finding
    refuses, breaking the in-memory/audit-row coherence."""
    with pytest.raises(ValidationError, match="policy_version"):
        _build_finding(policy_version="banana")


def test_review_finding_policy_version_rejects_leading_zero() -> None:
    """No leading zeros (`01.0.0`) — sibling of the DB CHECK + Python
    `_SEMVER_RE` in `policy.severity`."""
    with pytest.raises(ValidationError, match="policy_version"):
        _build_finding(policy_version="01.0.0")


def test_review_finding_content_hash_rejects_non_sha256_hex() -> None:
    """`content_hash` must be 64 lowercase-hex chars. Without the
    pattern, the FindingEvent dedup join would silently admit a
    malformed hash."""
    with pytest.raises(ValidationError, match="content_hash"):
        _build_finding(content_hash="sha256-abc123")


def test_review_finding_content_hash_rejects_short_hex() -> None:
    """A 63-hex string fails (`{64}` exact)."""
    with pytest.raises(ValidationError, match="content_hash"):
        _build_finding(content_hash="a" * 63)


def test_review_finding_content_hash_rejects_uppercase() -> None:
    """Lowercase-hex only — same canonical-encoding rule the audit
    side enforces."""
    with pytest.raises(ValidationError, match="content_hash"):
        _build_finding(content_hash="A" * 64)


def test_review_finding_file_path_rejects_traversal() -> None:
    """`..` traversal in `file_path` raises at construction via the
    `validate_diff_path` field validator. Without this, a traversal-bearing
    finding could be persisted and only fail at publish boundary."""
    from outrider.coordinates import CoordinateError

    with pytest.raises((ValidationError, CoordinateError)):
        _build_finding(file_path="../escape.py")


def test_review_finding_file_path_rejects_absolute() -> None:
    from outrider.coordinates import CoordinateError

    with pytest.raises((ValidationError, CoordinateError)):
        _build_finding(file_path="/etc/passwd")


def test_review_finding_evidence_max_length() -> None:
    """2000-char cap on model-emitted evidence text."""
    with pytest.raises(ValidationError, match="evidence"):
        _build_finding(evidence="x" * 2001)


def test_review_finding_evidence_admits_at_max() -> None:
    """The inclusive boundary (2000 chars) admits cleanly — paired with
    the +1 rejection test above. Without this, a future refactor that
    accidentally tightens the cap to 1999 would only fail the rejection
    side (silently shifting the boundary)."""
    finding = _build_finding(evidence="x" * 2000)
    assert len(finding.evidence) == 2000


def test_review_finding_suggested_fix_max_length() -> None:
    """2000-char cap on model-emitted suggested fix."""
    with pytest.raises(ValidationError, match="suggested_fix"):
        _build_finding(suggested_fix="x" * 2001)


def test_review_finding_suggested_fix_admits_at_max() -> None:
    finding = _build_finding(suggested_fix="x" * 2000)
    assert finding.suggested_fix is not None
    assert len(finding.suggested_fix) == 2000


def test_review_finding_override_reason_max_length() -> None:
    """1000-char cap on HITL-supplied override reason — matches the
    other reviewer-prose caps on `description` + below `evidence`."""
    with pytest.raises(ValidationError, match="override_reason"):
        _build_finding(override_reason="x" * 1001)


def test_review_finding_override_reason_admits_at_max() -> None:
    """1000-char cap admits at the boundary. The HITL override triplet
    (`original_severity` + `override_reason` + `overrider_id`) must
    ALL be set together per the new triplet-coherence validator —
    a partial override gets caught by `_enforce_override_triplet_coherence`."""
    finding = _build_finding(
        override_reason="x" * 1000,
        original_severity=FindingSeverity.CRITICAL,
        severity=FindingSeverity.MEDIUM,  # reviewer's override
        overrider_id=uuid4(),
    )
    assert finding.override_reason is not None
    assert len(finding.override_reason) == 1000


def test_review_finding_query_match_id_max_length() -> None:
    """200-char cap on query-registry id (well above realistic max)."""
    with pytest.raises(ValidationError, match="query_match_id"):
        _build_finding(
            evidence_tier=EvidenceTier.OBSERVED,
            query_match_id="x" * 201,
        )


def test_review_finding_query_match_id_admits_at_max() -> None:
    finding = _build_finding(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="x" * 200,
    )
    assert finding.query_match_id is not None
    assert len(finding.query_match_id) == 200


def test_review_finding_trace_path_per_element_max_length_256() -> None:
    """Per-element cap mirrors `AnalyzeFindingProposalRaw.trace_path`
    (256 chars). 257 chars rejects."""
    with pytest.raises(ValidationError, match="at most 256 characters"):
        _build_finding(
            evidence_tier=EvidenceTier.INFERRED,
            trace_path=("y" * 257,),
        )


def test_review_finding_trace_path_max_steps_32() -> None:
    """Tuple cap is 32 elements, mirroring the raw layer. 33 rejects."""
    with pytest.raises(ValidationError, match="at most 32 items"):
        _build_finding(
            evidence_tier=EvidenceTier.INFERRED,
            trace_path=tuple(f"step_{i}" for i in range(33)),
        )


def test_review_finding_trace_path_rejects_empty_string_element() -> None:
    """Per-element min_length=1 rejects empty strings in trace_path."""
    with pytest.raises(ValidationError, match="at least 1 character"):
        _build_finding(
            evidence_tier=EvidenceTier.INFERRED,
            trace_path=("",),
        )


# ---------------------------------------------------------------------------
# Severity-policy gate + content-hash recipe gate. Backs invariants
# `severity-set-by-policy` (docs/invariants.md §237) and the in-memory
# mirror of FindingEvent._verify_content_hash.
# ---------------------------------------------------------------------------


def test_review_finding_rejects_severity_drifted_from_policy() -> None:
    """A finding constructed with severity != SEVERITY_POLICY[finding_type]
    under the LIVE policy must fail.

    Regression: a row like `(SQL_INJECTION, LOW)` at the ACTIVE policy
    version was admitted even though SEVERITY_POLICY[SQL_INJECTION] ==
    CRITICAL. The helper defaults `policy_version` to ACTIVE_POLICY_VERSION
    so the drift check fires (it is skipped for historical versions).
    """
    with pytest.raises(ValidationError, match="severity-set-by-policy"):
        _build_finding(
            finding_type=FindingType.SQL_INJECTION,
            severity=FindingSeverity.LOW,  # policy says CRITICAL
        )


def test_review_finding_admits_severity_matching_policy() -> None:
    """The happy path: severity matches SEVERITY_POLICY[finding_type]."""
    finding = _build_finding(
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.CRITICAL,
    )
    assert finding.severity == FindingSeverity.CRITICAL


def test_review_finding_admits_hitl_override_with_original_severity() -> None:
    """HITL-overridden finding: baseline goes in `original_severity`,
    reviewer's choice in `severity`. The validator checks the baseline
    (original_severity) against policy, not the override."""
    finding = _build_finding(
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.MEDIUM,  # reviewer's override
        original_severity=FindingSeverity.CRITICAL,  # policy baseline
        override_reason="reviewer disagrees with severity",
        overrider_id=uuid4(),
    )
    assert finding.severity == FindingSeverity.MEDIUM
    assert finding.original_severity == FindingSeverity.CRITICAL


def test_review_finding_rejects_hitl_override_with_wrong_original_severity() -> None:
    """If `original_severity` is set, it (the baseline) must match
    SEVERITY_POLICY[finding_type]. An override path that falsifies the
    pre-override value gets caught."""
    with pytest.raises(ValidationError, match="severity-set-by-policy"):
        _build_finding(
            finding_type=FindingType.SQL_INJECTION,
            severity=FindingSeverity.MEDIUM,
            original_severity=FindingSeverity.LOW,  # WRONG — policy is CRITICAL
            override_reason="reviewer disagrees",
            overrider_id=uuid4(),
        )


def test_review_finding_admits_historical_policy_version() -> None:
    """Replay-aware scoping. `model_validate` is the same code path
    that reconstructs historical findings; a historical row under an
    older `policy_version` MUST validate cleanly. The schema's
    SEVERITY_POLICY match check only fires when `policy_version ==
    ACTIVE_POLICY_VERSION`; older versions skip and trust the row.

    The earlier hard-block on non-ACTIVE policy_version (intended as
    a fresh-write smuggle defense) broke replay reconstruction —
    `TypeAdapter.validate_python` runs the same validators, can't
    distinguish a fresh write from a historical rehydration, and has
    no synchronous historical-policy loader. The smuggle defense
    belongs in the producer/persister layer, not the schema layer.
    """
    finding = _build_finding(
        finding_type=FindingType.SQL_INJECTION,
        # Under a historical policy "0.9.0" the severity could
        # legitimately be anything that was correct AT WRITE TIME.
        severity=FindingSeverity.LOW,
        policy_version="0.9.0",  # not ACTIVE — replay scenario
    )
    assert finding.policy_version == "0.9.0"
    assert finding.severity == FindingSeverity.LOW


def test_review_finding_rejects_drifted_content_hash() -> None:
    """`content_hash` must equal `compute_finding_content_hash(...)` for
    the identity tuple. Without the validator, a drifted hash would
    survive into `AnalysisRound.round_id` and the reducer would dedup
    under the bad key on replay.

    pre-fold ReviewFinding.content_hash was only
    shape-validated. Several fixtures seeded it with
    `compute_identity_hash(...)` instead of the canonical recipe.
    """
    bad_hash = "f" * 64  # right shape, wrong content
    with pytest.raises(ValidationError, match="content_hash"):
        _build_finding(content_hash=bad_hash)


# ---------------------------------------------------------------------------
# HITL override triplet coherence + replay-aware policy_version scoping.
# ---------------------------------------------------------------------------


def test_review_finding_rejects_partial_override_original_only() -> None:
    """`original_severity` set but `override_reason` and `overrider_id`
    None — partial override. The triplet must be all-set-or-all-None.

    without this, a caller could set
    original_severity=CRITICAL + severity=LOW + override_reason=None +
    overrider_id=None — the policy-baseline check PASSES (CRITICAL
    matches policy) but no real HITL decision backs the downgrade.
    Severity drops to LOW with no reason, no reviewer."""
    with pytest.raises(ValidationError, match="all-set-or-all-None"):
        _build_finding(
            original_severity=FindingSeverity.CRITICAL,
            severity=FindingSeverity.LOW,
            override_reason=None,
            overrider_id=None,
        )


def test_review_finding_rejects_partial_override_missing_reason() -> None:
    """Two of three set, override_reason missing."""
    with pytest.raises(ValidationError, match="all-set-or-all-None"):
        _build_finding(
            original_severity=FindingSeverity.CRITICAL,
            severity=FindingSeverity.LOW,
            override_reason=None,
            overrider_id=uuid4(),
        )


def test_review_finding_rejects_partial_override_missing_overrider() -> None:
    """Two of three set, overrider_id missing."""
    with pytest.raises(ValidationError, match="all-set-or-all-None"):
        _build_finding(
            original_severity=FindingSeverity.CRITICAL,
            severity=FindingSeverity.LOW,
            override_reason="reviewer disagrees",
            overrider_id=None,
        )


def test_review_finding_admits_complete_override_triplet() -> None:
    """The happy path: all three override fields set together."""
    overrider_id = uuid4()
    finding = _build_finding(
        original_severity=FindingSeverity.CRITICAL,
        severity=FindingSeverity.MEDIUM,
        override_reason="reviewer disagrees with severity",
        overrider_id=overrider_id,
    )
    assert finding.original_severity == FindingSeverity.CRITICAL
    assert finding.severity == FindingSeverity.MEDIUM
    assert finding.override_reason == "reviewer disagrees with severity"
    assert finding.overrider_id == overrider_id


def test_review_finding_admits_no_override_state() -> None:
    """Baseline finding with no override — all three fields None."""
    finding = _build_finding()
    assert finding.original_severity is None
    assert finding.override_reason is None
    assert finding.overrider_id is None


def test_review_finding_rejects_no_op_override() -> None:
    """An override envelope with `severity == original_severity` is a
    producer bug — the reviewer's intent to ACK without change is
    `PerFindingDecision.APPROVE`, not `SEVERITY_OVERRIDE` with
    identical values. cheap to
    pin, catches a HITL UI submitting the override path without
    actually changing the value.
    """
    with pytest.raises(ValidationError, match="no-op overrides are not valid"):
        _build_finding(
            original_severity=FindingSeverity.CRITICAL,
            severity=FindingSeverity.CRITICAL,  # same — no-op
            override_reason="reviewer pressed override but didn't change anything",
            overrider_id=uuid4(),
        )


def test_review_finding_rejects_empty_override_reason() -> None:
    """the triplet's `is None` check admitted
    `override_reason=""`. An override with no substantive reason is
    the bug class `hitl-gates-high-severity` defends against.
    `PerFindingDecision` already rejects empty reasons; the carrier
    ReviewFinding now matches."""
    with pytest.raises(ValidationError, match="blank override_reason"):
        _build_finding(
            original_severity=FindingSeverity.CRITICAL,
            severity=FindingSeverity.MEDIUM,
            override_reason="",  # empty — blank
            overrider_id=uuid4(),
        )


def test_review_finding_rejects_whitespace_only_override_reason() -> None:
    """Whitespace-only reason is also blank — `.strip() == ""` catches
    both empty strings AND spaces-only strings."""
    with pytest.raises(ValidationError, match="blank override_reason"):
        _build_finding(
            original_severity=FindingSeverity.CRITICAL,
            severity=FindingSeverity.MEDIUM,
            override_reason="   \t\n  ",  # whitespace-only
            overrider_id=uuid4(),
        )


# ---------------------------------------------------------------------------
# proposal_hash field — per DECISIONS.md#025
# ---------------------------------------------------------------------------


def test_review_finding_admits_proposal_hash() -> None:
    """Per DECISIONS.md#025: ReviewFinding carries proposal_hash for trace's
    join contract. Field is pattern-validated (SHA-256 hex)."""
    finding = _build_finding(proposal_hash="b" * 64)
    assert finding.proposal_hash == "b" * 64


def test_review_finding_rejects_proposal_hash_non_hex() -> None:
    """Pattern-validated against SHA256_HEX_PATTERN — non-hex rejected."""
    with pytest.raises(ValidationError):
        _build_finding(proposal_hash="not-a-sha256-hash")


def test_review_finding_rejects_proposal_hash_wrong_length() -> None:
    """64 hex chars exactly — too short / too long rejected."""
    with pytest.raises(ValidationError):
        _build_finding(proposal_hash="a" * 63)  # 1 too short
    with pytest.raises(ValidationError):
        _build_finding(proposal_hash="a" * 65)  # 1 too long


def test_review_finding_rejects_missing_proposal_hash() -> None:
    """Per DECISIONS.md#025 point 1: no default. A construction that
    omits proposal_hash raises (not a silent default-to-empty)."""
    from typing import Any

    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "policy_version": ACTIVE_POLICY_VERSION,
        "finding_type": FindingType.SQL_INJECTION,
        "dimension": ReviewDimension.SECURITY,
        "severity": FindingSeverity.CRITICAL,
        "evidence_tier": EvidenceTier.JUDGED,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "content_hash": compute_finding_content_hash(
            file_path="src/foo.py",
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        # proposal_hash deliberately omitted
    }
    with pytest.raises(ValidationError):
        ReviewFinding(**fields)


def test_review_finding_proposal_hash_not_in_content_hash_recipe() -> None:
    """Per DECISIONS.md#025 point 3: proposal_hash is PROVENANCE, not
    part of finding_content_hash. Two findings with identical content
    fields but DIFFERENT proposal_hash values produce IDENTICAL
    content_hash. This is the load-bearing distinction between content
    identity (#022 recipe; stable across LLM phrasing) and provenance
    (#025; varies per LLM call)."""
    f1 = _build_finding(proposal_hash="a" * 64)
    f2 = _build_finding(proposal_hash="b" * 64)
    # Different proposal_hash (provenance differs)
    assert f1.proposal_hash != f2.proposal_hash
    # Same content_hash (identity unchanged)
    assert f1.content_hash == f2.content_hash
