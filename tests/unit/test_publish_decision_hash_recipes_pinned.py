"""Golden-digest pins for the three publish decision-hash helpers.

Backs the append-only enum-value + hash-recipe contract declared in
`DECISIONS.md` #023 "Append-only enum-value + hash-recipe contract"
Consequence bullet. Per that contract:

- The four publish StrEnums (`PublishRoutingReason`, `PublishEligibility`,
  `PublishEligibilityReason`, `PublishAttemptOutcome`) have append-only
  `.value` strings; existing values MUST NOT be renamed or removed.
- The three canonical decision-hash helpers
  (`compute_publish_routing_decision_hash`,
  `compute_publish_eligibility_decision_hash`,
  `compute_publish_attempt_content_hash`) have append-only recipes;
  JSON encoding order, separators, and input-tuple shape MUST NOT change.

Every `PublishRoutingEvent` / `PublishEligibilityEvent` / `PublishAttemptEvent`
model-validator recomputes the hash at construction and rejects on mismatch.
A silent rename of any `.value` string OR a re-ordering of the JSON encoding
would cause EVERY historical row's `model_validate_json(...)` to fail at
replay, with no version-stamp escape hatch. This file is the structural
floor: if a contract violation lands, these tests fail at commit time,
not at the next replay attempt against historical audit data.

The fixtures cover at least one tuple per enum member's hash-input path
(both branches of the optional-kind on routing; both branches of the
optional-reason on eligibility; success and failed on attempt; the
sorted-vs-empty branches of `sorted_finding_ids`). They are NOT a
substitute for unit coverage of validator behavior; they are the
recipe-pinning artifact specifically.

If you are reading this because a test failed: the recipe changed.
Either revert the recipe change OR — if the change is deliberate — add a
`hash_recipe_version` field to the events (mirroring
`PublishEligibilityEvent.policy_version`'s `ACTIVE_POLICY_VERSION`
skip-for-historical guard) and bump it BEFORE updating these goldens.
"""

from uuid import UUID

from outrider.audit.events import (
    PublishAttemptOutcome,
    PublishEligibility,
    PublishEligibilityReason,
    PublishRoutingReason,
    compute_publish_attempt_content_hash,
    compute_publish_eligibility_decision_hash,
    compute_publish_routing_decision_hash,
)
from outrider.coordinates.errors import CoordinateErrorKind
from outrider.schemas import PublishDestination

# Fixed UUIDs so attempt-hash goldens are reproducible. Do NOT use uuid4().
_FIXED_REVIEW_ID = UUID("00000000-0000-0000-0000-000000000001")
_FIXED_FINDING_A = UUID("11111111-1111-1111-1111-111111111111")
_FIXED_FINDING_B = UUID("22222222-2222-2222-2222-222222222222")


# ---------------------------------------------------------------------------
# compute_publish_routing_decision_hash — both kind branches pinned.
# ---------------------------------------------------------------------------


def test_routing_hash_recipe_pinned_inline_reviewable_no_kind() -> None:
    """Happy path: INLINE_COMMENT + REVIEWABLE_DIFF_LINE + None kind."""
    expected = "d5a19225b424f42f3c6a71740e4cc6d0de568bc473bcd44d25332b04a4e214f7"
    actual = compute_publish_routing_decision_hash(
        destination=PublishDestination.INLINE_COMMENT,
        reason=PublishRoutingReason.REVIEWABLE_DIFF_LINE,
        coordinate_error_kind=None,
    )
    assert actual == expected, (
        f"compute_publish_routing_decision_hash recipe drift detected.\n"
        f"  Inputs: INLINE_COMMENT + REVIEWABLE_DIFF_LINE + None.\n"
        f"  Expected: {expected}\n  Actual:   {actual}\n"
        f"  See this file's module docstring before updating the golden."
    )


def test_routing_hash_recipe_pinned_dashboard_path_validation_umbrella() -> None:
    """Umbrella path: DASHBOARD_ONLY + COORDINATE_ERROR + PATH_VALIDATION_FAILED kind."""
    expected = "23758c8fd8e279b84b6e46e2157832e9192d71ea189b5c80eb970816ab142fe8"
    actual = compute_publish_routing_decision_hash(
        destination=PublishDestination.DASHBOARD_ONLY,
        reason=PublishRoutingReason.COORDINATE_ERROR,
        coordinate_error_kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
    )
    assert actual == expected, (
        f"compute_publish_routing_decision_hash recipe drift detected.\n"
        f"  Inputs: DASHBOARD_ONLY + COORDINATE_ERROR + PATH_VALIDATION_FAILED.\n"
        f"  Expected: {expected}\n  Actual:   {actual}\n"
        f"  See this file's module docstring before updating the golden."
    )


# ---------------------------------------------------------------------------
# compute_publish_eligibility_decision_hash — both reason branches pinned.
# ---------------------------------------------------------------------------


def test_eligibility_hash_recipe_pinned_eligible_no_reason() -> None:
    """Eligible path: reason MUST be None."""
    expected = "9abefcdd2e7f69c48d34cc7b7add0b2df807d60f470b81d7ed00ed119f9c282b"
    actual = compute_publish_eligibility_decision_hash(
        eligibility=PublishEligibility.ELIGIBLE,
        reason=None,
    )
    assert actual == expected, (
        f"compute_publish_eligibility_decision_hash recipe drift detected.\n"
        f"  Inputs: ELIGIBLE + None reason.\n"
        f"  Expected: {expected}\n  Actual:   {actual}\n"
        f"  See this file's module docstring before updating the golden."
    )


def test_eligibility_hash_recipe_pinned_withheld_hitl_absent() -> None:
    """Withheld path: V1 HITL-absent withholding (most common V1 reason)."""
    expected = "de3910e7a6ab7eee5deb8f7b61e6a8022fb1d8e777dd033275749d1f7c1f97dc"
    actual = compute_publish_eligibility_decision_hash(
        eligibility=PublishEligibility.WITHHELD,
        reason=PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT,
    )
    assert actual == expected, (
        f"compute_publish_eligibility_decision_hash recipe drift detected.\n"
        f"  Inputs: WITHHELD + HITL_REQUIRED_NODE_ABSENT.\n"
        f"  Expected: {expected}\n  Actual:   {actual}\n"
        f"  See this file's module docstring before updating the golden."
    )


# ---------------------------------------------------------------------------
# compute_publish_attempt_content_hash — outcome inclusion + sorted-tuple
# determinism pinned (both empty and populated sorted_finding_ids).
# ---------------------------------------------------------------------------


def test_attempt_hash_recipe_pinned_success_empty_findings() -> None:
    """SUCCESS / zero-findings boundary case. Pins the seven-field
    recipe (review_id, attempt_index, sorted_finding_ids, outcome,
    status_code, failure_class, comments_attempted). Re-pinned when
    the recipe expanded to include the three attempt-distinguishing
    fields so two FAILED attempts with different status_code can't
    collapse on read-time dedup."""
    expected = "e89e6f4ed4ff68ce19c489c2ecc33b5ac1c729a94f4178583a45957408dfa29a"
    actual = compute_publish_attempt_content_hash(
        review_id=_FIXED_REVIEW_ID,
        attempt_index=1,
        sorted_finding_ids=(),
        outcome=PublishAttemptOutcome.SUCCESS,
        status_code=200,
        failure_class=None,
        comments_attempted=0,
    )
    assert actual == expected, (
        f"compute_publish_attempt_content_hash recipe drift detected.\n"
        f"  Inputs: SUCCESS + attempt_index=1 + empty sorted_finding_ids "
        f"+ status_code=200 + no failure_class + comments_attempted=0.\n"
        f"  Expected: {expected}\n  Actual:   {actual}\n"
        f"  See this file's module docstring before updating the golden."
    )


def test_attempt_hash_recipe_pinned_failed_two_findings_sorted() -> None:
    """FAILED outcome with two sorted finding IDs + failure context.
    Pins outcome-in-recipe AND sorted-tuple positional encoding AND
    failure-context inclusion (status_code + failure_class +
    comments_attempted). Renaming `failed` → `failed_v2`, dropping any
    field from the recipe, or re-encoding the UUID list to a different
    string form would each break this golden."""
    expected = "dc651ca70fbb6251411b302cb2d320c1b7566795612105ec33fffea382730220"
    actual = compute_publish_attempt_content_hash(
        review_id=_FIXED_REVIEW_ID,
        attempt_index=2,
        sorted_finding_ids=(_FIXED_FINDING_A, _FIXED_FINDING_B),
        outcome=PublishAttemptOutcome.FAILED,
        status_code=422,
        failure_class="GitHubReviewValidationError",
        comments_attempted=2,
    )
    assert actual == expected, (
        f"compute_publish_attempt_content_hash recipe drift detected.\n"
        f"  Inputs: FAILED + attempt_index=2 + 2 sorted finding IDs "
        f"+ status_code=422 + GitHubReviewValidationError + "
        f"comments_attempted=2.\n"
        f"  Expected: {expected}\n  Actual:   {actual}\n"
        f"  See this file's module docstring before updating the golden."
    )


# ---------------------------------------------------------------------------
# Enum-value pins — append-only contract guard. Renaming `reviewable_diff_line`
# → `inline_eligible` or removing `idempotently_skipped_external_record` would
# silently break every historical audit row's deserialization; these tests
# fail at commit time instead.
# ---------------------------------------------------------------------------


def test_publish_routing_reason_value_strings_pinned() -> None:
    """Append-only contract pin per DECISIONS.md #023."""
    assert PublishRoutingReason.REVIEWABLE_DIFF_LINE.value == "reviewable_diff_line"
    assert PublishRoutingReason.UNCHANGED_REGION.value == "unchanged_region"
    assert PublishRoutingReason.NON_DIFFED_FILE.value == "non_diffed_file"
    assert PublishRoutingReason.COORDINATE_ERROR.value == "coordinate_error"
    assert len(PublishRoutingReason) == 4


def test_publish_eligibility_value_strings_pinned() -> None:
    """Append-only contract pin per DECISIONS.md #023."""
    assert PublishEligibility.ELIGIBLE.value == "eligible"
    assert PublishEligibility.WITHHELD.value == "withheld"
    assert len(PublishEligibility) == 2


def test_publish_eligibility_reason_value_strings_pinned() -> None:
    """Append-only contract pin per DECISIONS.md #023.

    Pre-HITL: 3 reasons. Post-HITL (specs/2026-05-26-hitl-node.md Group 6):
    +3 (HITL_DECISION_MISSING / HITL_REJECTED / HITL_SUPPRESSED) = 6 total.
    """
    assert PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT.value == "hitl_required_node_absent"
    assert (
        PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT.value
        == "unexpected_override_fields_present"
    )
    assert PublishEligibilityReason.ROUTING_EMISSION_FAILED.value == "routing_emission_failed"
    assert PublishEligibilityReason.HITL_DECISION_MISSING.value == "hitl_decision_missing"
    assert PublishEligibilityReason.HITL_REJECTED.value == "hitl_rejected"
    assert PublishEligibilityReason.HITL_SUPPRESSED.value == "hitl_suppressed"
    assert len(PublishEligibilityReason) == 6


def test_publish_attempt_outcome_value_strings_pinned() -> None:
    """Append-only contract pin per DECISIONS.md #023."""
    assert PublishAttemptOutcome.SUCCESS.value == "success"
    assert PublishAttemptOutcome.FAILED.value == "failed"
    assert PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED.value == "idempotently_skipped"
    assert (
        PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD.value
        == "idempotently_skipped_external_record"
    )
    assert PublishAttemptOutcome.NO_OP_EMPTY.value == "no_op_empty"
    assert len(PublishAttemptOutcome) == 5


def test_coordinate_error_kind_value_strings_pinned() -> None:
    """Append-only contract pin — kinds ride on PublishRoutingEvent.coordinate_error_kind
    payload, so a rename silently breaks routing-row replay."""
    assert CoordinateErrorKind.UNCHANGED_REGION.value == "unchanged_region"
    assert CoordinateErrorKind.BYTE_OFFSET_INVALID.value == "byte_offset_invalid"
    assert CoordinateErrorKind.MALFORMED_PATCH.value == "malformed_patch"
    assert CoordinateErrorKind.DUPLICATE_FILE_ENTRY.value == "duplicate_file_entry"
    assert CoordinateErrorKind.FILE_NOT_IN_PATCH.value == "file_not_in_patch"
    assert CoordinateErrorKind.INVALID_DIFF_LINE.value == "invalid_diff_line"
    assert CoordinateErrorKind.PATH_VALIDATION_FAILED.value == "path_validation_failed"
    assert CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED.value == "argument_validation_failed"
    assert CoordinateErrorKind.HEAD_CONTENT_UNAVAILABLE.value == "head_content_unavailable"
    assert len(CoordinateErrorKind) == 9
