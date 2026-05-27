"""Validator coverage for `PublishRoutingEvent` + `PublishEligibilityEvent`.

Backs the schema-layer defenses introduced for the V1 publish-node spec:

- `_enforce_coordinate_error_kind_membership` — field-level membership check
  against `CoordinateErrorKind` so JSON-replay events carrying invented
  kind values can't admit.
- `_enforce_coordinate_error_kind_required_iff_coordinate_error` — total
  cover over the (reason × kind) product, not just the two ends.
- `_verify_finding_content_hash` — recompute via `compute_finding_content_hash`,
  honest hash binding rather than pattern-only field-level check.
- `_enforce_severity_matches_policy` (eligibility-side) — mirror of
  FindingEvent's validator so the eligibility audit shadow is at least
  as strict as the source.
- `_enforce_v1_no_overrides` — schema-layer rejection of `original_severity`
  on an eligibility event before HITL ships; defends the audit row against
  replay-injected pre-approved downgrades.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import (
    PublishAttemptEvent,
    PublishAttemptOutcome,
    PublishEligibility,
    PublishEligibilityEvent,
    PublishEligibilityReason,
    PublishRoutingEvent,
    PublishRoutingReason,
    compute_finding_content_hash,
    compute_publish_attempt_content_hash,
    compute_publish_eligibility_decision_hash,
    compute_publish_routing_decision_hash,
)
from outrider.coordinates.errors import CoordinateErrorKind
from outrider.policy import FindingSeverity, FindingType
from outrider.policy.severity import ACTIVE_POLICY_VERSION, SEVERITY_POLICY
from outrider.schemas import PublishDestination

# ----------------------------------------------------------------------------
# PublishRoutingEvent — happy path + every validator
# ----------------------------------------------------------------------------


def _routing_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build a valid routing-event payload; overrides take precedence."""
    file_path = overrides.pop("file_path", "src/app.py")
    line_start = overrides.pop("line_start", 10)
    line_end = overrides.pop("line_end", 12)
    finding_type = overrides.pop("finding_type", FindingType.MISSING_INPUT_VALIDATION)
    destination = overrides.pop("destination", PublishDestination.INLINE_COMMENT)
    reason = overrides.pop("reason", PublishRoutingReason.REVIEWABLE_DIFF_LINE)
    coordinate_error_kind = overrides.pop("coordinate_error_kind", None)
    base = {
        "review_id": uuid4(),
        "finding_id": uuid4(),
        "destination": destination,
        "reason": reason,
        "coordinate_error_kind": coordinate_error_kind,
        "file_path": file_path,
        "line_start": line_start,
        "line_end": line_end,
        "finding_type": finding_type,
        "finding_content_hash": compute_finding_content_hash(
            file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        "decision_content_hash": compute_publish_routing_decision_hash(
            destination=destination,
            reason=reason,
            coordinate_error_kind=coordinate_error_kind,
        ),
    }
    base.update(overrides)
    return base


def test_routing_happy_path_reviewable_diff_line() -> None:
    """Success path: reviewable_diff_line → INLINE_COMMENT, kind=None."""
    PublishRoutingEvent(**_routing_kwargs())


def test_routing_unchanged_region_requires_unchanged_region_kind() -> None:
    """unchanged_region reason MUST carry kind=UNCHANGED_REGION; the kind is
    part of the routing identity."""
    # Happy path with the right kind.
    PublishRoutingEvent(
        **_routing_kwargs(
            destination=PublishDestination.REVIEW_BODY,
            reason=PublishRoutingReason.UNCHANGED_REGION,
            coordinate_error_kind=CoordinateErrorKind.UNCHANGED_REGION.value,
        )
    )
    # Missing kind.
    with pytest.raises(ValidationError, match="requires coordinate_error_kind"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.REVIEW_BODY,
                reason=PublishRoutingReason.UNCHANGED_REGION,
                coordinate_error_kind=None,
            )
        )
    # Wrong kind.
    with pytest.raises(ValidationError, match="requires coordinate_error_kind"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.REVIEW_BODY,
                reason=PublishRoutingReason.UNCHANGED_REGION,
                coordinate_error_kind=CoordinateErrorKind.MALFORMED_PATCH.value,
            )
        )


def test_routing_non_diffed_file_accepts_registry_miss_or_file_not_in_patch() -> None:
    """non_diffed_file reason accepts kind=None (registry miss) OR
    kind=FILE_NOT_IN_PATCH (registry/patch disagreement); anything else rejects."""
    # Registry-miss case: kind=None.
    PublishRoutingEvent(
        **_routing_kwargs(
            destination=PublishDestination.DASHBOARD_ONLY,
            reason=PublishRoutingReason.NON_DIFFED_FILE,
            coordinate_error_kind=None,
        )
    )
    # File-not-in-patch case: kind=FILE_NOT_IN_PATCH.
    PublishRoutingEvent(
        **_routing_kwargs(
            destination=PublishDestination.DASHBOARD_ONLY,
            reason=PublishRoutingReason.NON_DIFFED_FILE,
            coordinate_error_kind=CoordinateErrorKind.FILE_NOT_IN_PATCH.value,
        )
    )
    # Disallowed kind (e.g., MALFORMED_PATCH) for non_diffed_file reason.
    with pytest.raises(ValidationError, match="accepts only"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.DASHBOARD_ONLY,
                reason=PublishRoutingReason.NON_DIFFED_FILE,
                coordinate_error_kind=CoordinateErrorKind.MALFORMED_PATCH.value,
            )
        )


def test_routing_coordinate_error_requires_kind_and_rejects_dedicated_kinds() -> None:
    """coordinate_error reason requires kind AND rejects UNCHANGED_REGION /
    FILE_NOT_IN_PATCH (those route via their dedicated reasons)."""
    # Happy path: coordinate_error with a generic kind.
    PublishRoutingEvent(
        **_routing_kwargs(
            destination=PublishDestination.DASHBOARD_ONLY,
            reason=PublishRoutingReason.COORDINATE_ERROR,
            coordinate_error_kind=CoordinateErrorKind.MALFORMED_PATCH.value,
        )
    )
    # Missing kind.
    with pytest.raises(ValidationError, match="requires coordinate_error_kind"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.DASHBOARD_ONLY,
                reason=PublishRoutingReason.COORDINATE_ERROR,
                coordinate_error_kind=None,
            )
        )
    # Forbidden kind (UNCHANGED_REGION belongs on its own reason).
    with pytest.raises(ValidationError, match="must use the dedicated reason"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.DASHBOARD_ONLY,
                reason=PublishRoutingReason.COORDINATE_ERROR,
                coordinate_error_kind=CoordinateErrorKind.UNCHANGED_REGION.value,
            )
        )
    # Forbidden kind (FILE_NOT_IN_PATCH belongs on its own reason).
    with pytest.raises(ValidationError, match="must use the dedicated reason"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.DASHBOARD_ONLY,
                reason=PublishRoutingReason.COORDINATE_ERROR,
                coordinate_error_kind=CoordinateErrorKind.FILE_NOT_IN_PATCH.value,
            )
        )


def test_routing_reviewable_diff_line_rejects_any_kind() -> None:
    """Success path forbids any coordinate_error_kind."""
    with pytest.raises(ValidationError, match="coordinate_error_kind must be None"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.INLINE_COMMENT,
                reason=PublishRoutingReason.REVIEWABLE_DIFF_LINE,
                coordinate_error_kind=CoordinateErrorKind.UNCHANGED_REGION.value,
            )
        )


def test_routing_rejects_invented_coordinate_error_kind_string() -> None:
    """A coordinate_error_kind value not in CoordinateErrorKind admits at
    the field-pattern layer (it's just a string), but the membership
    field-validator rejects it BEFORE the model-validators run."""
    with pytest.raises(ValidationError, match="is not a CoordinateErrorKind member"):
        PublishRoutingEvent(
            **_routing_kwargs(
                destination=PublishDestination.DASHBOARD_ONLY,
                reason=PublishRoutingReason.COORDINATE_ERROR,
                coordinate_error_kind="totally_made_up_kind",
            )
        )


def test_routing_rejects_mismatched_finding_content_hash() -> None:
    """`_verify_finding_content_hash` re-derives via `compute_finding_content_hash`
    and rejects any caller-supplied hash that doesn't match."""
    kwargs = _routing_kwargs()
    # Replace the hash with a syntactically valid but wrong value.
    kwargs["finding_content_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="does not match compute_finding_content_hash"):
        PublishRoutingEvent(**kwargs)


def test_routing_rejects_mismatched_decision_content_hash() -> None:
    """`_verify_decision_content_hash` re-derives and rejects mismatch."""
    kwargs = _routing_kwargs()
    kwargs["decision_content_hash"] = "f" * 64
    with pytest.raises(
        ValidationError, match="does not match compute_publish_routing_decision_hash"
    ):
        PublishRoutingEvent(**kwargs)


def test_routing_round_trips_through_json() -> None:
    """Replay reconstruction (model_validate_json) re-runs all validators —
    a tampered payload (e.g., wrong kind) rejects on replay, not just on
    fresh construction."""
    event = PublishRoutingEvent(**_routing_kwargs())
    blob = event.model_dump_json()
    PublishRoutingEvent.model_validate_json(blob)


# ----------------------------------------------------------------------------
# PublishEligibilityEvent — happy path + every validator
# ----------------------------------------------------------------------------


def _eligibility_kwargs(**overrides: Any) -> dict[str, Any]:
    file_path = overrides.pop("file_path", "src/app.py")
    line_start = overrides.pop("line_start", 10)
    line_end = overrides.pop("line_end", 12)
    finding_type = overrides.pop("finding_type", FindingType.MISSING_INPUT_VALIDATION)
    # Severity must match SEVERITY_POLICY under live policy version.
    severity = overrides.pop("severity", SEVERITY_POLICY[finding_type])
    eligibility = overrides.pop("eligibility", PublishEligibility.ELIGIBLE)
    reason = overrides.pop("reason", None)
    policy_version = overrides.pop("policy_version", ACTIVE_POLICY_VERSION)
    base = {
        "review_id": uuid4(),
        "finding_id": uuid4(),
        "file_path": file_path,
        "line_start": line_start,
        "line_end": line_end,
        "finding_type": finding_type,
        "severity": severity,
        "original_severity": None,
        "finding_content_hash": compute_finding_content_hash(
            file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        "decision_content_hash": compute_publish_eligibility_decision_hash(
            eligibility=eligibility,
            reason=reason,
        ),
        "eligibility": eligibility,
        "reason": reason,
        "policy_version": policy_version,
    }
    base.update(overrides)
    return base


def test_eligibility_happy_path_eligible() -> None:
    """Eligible finding with reason=None constructs cleanly."""
    PublishEligibilityEvent(**_eligibility_kwargs())


def test_eligibility_happy_path_withheld() -> None:
    """Withheld finding with a reason constructs cleanly."""
    eligibility = PublishEligibility.WITHHELD
    reason = PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT
    PublishEligibilityEvent(
        **_eligibility_kwargs(
            eligibility=eligibility,
            reason=reason,
        )
    )


def test_eligibility_rejects_severity_not_matching_policy_under_live_version() -> None:
    """`_enforce_severity_matches_policy` rejects a (finding_type, severity)
    tuple that doesn't match SEVERITY_POLICY under the live policy version.
    Mirror of FindingEvent's validator."""
    # Find a finding_type whose policy-assigned severity is NOT LOW so we
    # have a concrete mismatch.
    finding_type = FindingType.SQL_INJECTION
    expected = SEVERITY_POLICY[finding_type]
    mismatched = FindingSeverity.LOW if expected != FindingSeverity.LOW else FindingSeverity.HIGH
    with pytest.raises(ValidationError, match="does not match SEVERITY_POLICY"):
        PublishEligibilityEvent(
            **_eligibility_kwargs(finding_type=finding_type, severity=mismatched)
        )


def test_eligibility_historical_policy_version_skips_severity_check() -> None:
    """Historical events under an older policy version MUST validate cleanly
    even if the severity doesn't match current SEVERITY_POLICY — the row was
    correct under its frozen policy and there's no synchronous loader at
    this layer."""
    finding_type = FindingType.SQL_INJECTION
    # Deliberately pass a severity that does NOT match the CURRENT policy
    # but stash it under an older policy_version — should validate cleanly.
    mismatched = FindingSeverity.LOW
    PublishEligibilityEvent(
        **_eligibility_kwargs(
            finding_type=finding_type,
            severity=mismatched,
            policy_version="0.9.0",
        )
    )


def test_eligibility_admits_severity_override_with_baseline_in_original_severity() -> None:
    """Post-HITL convention (mirror of ReviewFinding): when override is
    in effect, `severity` carries the OVERRIDE value and
    `original_severity` carries the POLICY BASELINE. The baseline must
    equal SEVERITY_POLICY[finding_type] under the active policy version.

    Construction succeeds; replay can reconstruct "what severity did the
    published comment show" from this event alone.
    """
    finding_type = FindingType.SQL_INJECTION  # baseline=CRITICAL
    baseline = SEVERITY_POLICY[finding_type]
    override = FindingSeverity.LOW
    event = PublishEligibilityEvent(
        **_eligibility_kwargs(
            finding_type=finding_type,
            severity=override,
            original_severity=baseline,
        )
    )
    assert event.severity == override
    assert event.original_severity == baseline


def test_eligibility_rejects_override_whose_baseline_diverges_from_policy() -> None:
    """When `original_severity` is non-None, the baseline check uses it
    (mirror of ReviewFinding). A forged baseline that doesn't match
    SEVERITY_POLICY[finding_type] is rejected — defense against a
    replay-injected event claiming an illegitimate baseline."""
    finding_type = FindingType.SQL_INJECTION  # baseline=CRITICAL
    # original_severity claims baseline=LOW; severity is override=INFO.
    # Neither matches SEVERITY_POLICY[SQL_INJECTION]=CRITICAL.
    with pytest.raises(ValidationError, match="does not match SEVERITY_POLICY"):
        PublishEligibilityEvent(
            **_eligibility_kwargs(
                finding_type=finding_type,
                severity=FindingSeverity.INFO,
                original_severity=FindingSeverity.LOW,
            )
        )


def test_eligibility_rejects_withheld_without_reason() -> None:
    """Withheld eligibility requires a reason."""
    with pytest.raises(ValidationError, match="withheld requires a reason"):
        PublishEligibilityEvent(
            **_eligibility_kwargs(
                eligibility=PublishEligibility.WITHHELD,
                reason=None,
            )
        )


def test_eligibility_rejects_eligible_with_reason() -> None:
    """Eligible MUST have reason=None."""
    with pytest.raises(ValidationError, match="eligible must have reason=None"):
        PublishEligibilityEvent(
            **_eligibility_kwargs(
                eligibility=PublishEligibility.ELIGIBLE,
                reason=PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT,
            )
        )


def test_eligibility_rejects_mismatched_content_hash() -> None:
    """`_verify_content_hash_binding` rejects mismatched finding_content_hash."""
    kwargs = _eligibility_kwargs()
    kwargs["finding_content_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="does not match compute_finding_content_hash"):
        PublishEligibilityEvent(**kwargs)


def test_eligibility_round_trips_through_json() -> None:
    """Replay reconstruction re-runs all validators."""
    event = PublishEligibilityEvent(**_eligibility_kwargs())
    blob = event.model_dump_json()
    PublishEligibilityEvent.model_validate_json(blob)


# ----------------------------------------------------------------------------
# PublishAttemptEvent — happy path + a couple of validator branches
# ----------------------------------------------------------------------------


def _attempt_kwargs(**overrides: Any) -> dict[str, Any]:
    review_id = overrides.pop("review_id", uuid4())
    attempt_index = overrides.pop("attempt_index", 1)
    sorted_finding_ids: tuple[Any, ...] = overrides.pop("sorted_finding_ids", ())
    outcome = overrides.pop("outcome", PublishAttemptOutcome.NO_OP_EMPTY)
    status_code = overrides.pop("status_code", None)
    failure_class = overrides.pop("failure_class", None)
    comments_attempted = overrides.pop("comments_attempted", 0)
    recovered_github_review_id = overrides.pop("recovered_github_review_id", None)
    base = {
        "review_id": review_id,
        "attempt_index": attempt_index,
        "outcome": outcome,
        "status_code": status_code,
        "failure_class": failure_class,
        "comments_attempted": comments_attempted,
        "sorted_finding_ids": sorted_finding_ids,
        "recovered_github_review_id": recovered_github_review_id,
        "attempt_content_hash": compute_publish_attempt_content_hash(
            review_id=review_id,
            attempt_index=attempt_index,
            sorted_finding_ids=sorted_finding_ids,
            outcome=outcome,
            status_code=status_code,
            failure_class=failure_class,
            comments_attempted=comments_attempted,
            recovered_github_review_id=recovered_github_review_id,
        ),
    }
    base.update(overrides)
    return base


def test_attempt_happy_path_no_op_empty() -> None:
    """no_op_empty outcome with zero comments admits."""
    PublishAttemptEvent(**_attempt_kwargs())


def test_attempt_failed_requires_failure_class() -> None:
    """outcome=failed requires failure_class."""
    kwargs = _attempt_kwargs(outcome=PublishAttemptOutcome.FAILED, failure_class=None)
    with pytest.raises(ValidationError, match="outcome=failed requires failure_class"):
        PublishAttemptEvent(**kwargs)


def test_attempt_non_failed_rejects_failure_class() -> None:
    """outcome != failed must have failure_class=None."""
    kwargs = _attempt_kwargs(outcome=PublishAttemptOutcome.SUCCESS, failure_class="SomeError")
    with pytest.raises(ValidationError, match="must have failure_class=None"):
        PublishAttemptEvent(**kwargs)


def test_attempt_rejects_mismatched_content_hash() -> None:
    """`_verify_attempt_content_hash` rejects mismatch."""
    kwargs = _attempt_kwargs()
    kwargs["attempt_content_hash"] = "0" * 64
    with pytest.raises(
        ValidationError, match="does not match compute_publish_attempt_content_hash"
    ):
        PublishAttemptEvent(**kwargs)


def test_attempt_rejects_unsorted_finding_ids() -> None:
    """`_enforce_sorted_finding_ids` rejects unsorted at construction.

    Sort order is load-bearing for `compute_publish_attempt_content_hash`'s
    positional encoding; silently coercing via `sorted(...)` would mask
    a producer bug rather than surface it.
    """
    a, b = sorted((uuid4(), uuid4()))
    unsorted = (b, a)
    kwargs = _attempt_kwargs(sorted_finding_ids=unsorted)
    with pytest.raises(ValidationError, match="sorted_finding_ids must be sorted"):
        PublishAttemptEvent(**kwargs)


def test_attempt_rejects_oversized_failure_class() -> None:
    """`failure_class` is bounded — defense against attacker-influenced
    422 error strings being interpolated into the append-only audit row."""
    kwargs = _attempt_kwargs(
        outcome=PublishAttemptOutcome.FAILED,
        failure_class="X" * 129,
    )
    with pytest.raises(ValidationError, match="at most 128"):
        PublishAttemptEvent(**kwargs)


def test_attempt_external_record_skip_requires_recovered_github_review_id() -> None:
    """`outcome=idempotently_skipped_external_record` MUST carry a
    non-None `recovered_github_review_id`. Audit-only replay needs the
    binding to reconstruct the recovery — no paired `PublishEvent`
    lands on this path."""
    kwargs = _attempt_kwargs(
        outcome=PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD,
        recovered_github_review_id=None,
    )
    with pytest.raises(
        ValidationError,
        match=("outcome=idempotently_skipped_external_record requires recovered_github_review_id"),
    ):
        PublishAttemptEvent(**kwargs)


def test_attempt_external_record_skip_admits_with_recovered_id() -> None:
    """Happy path: external-record skip with a positive
    `recovered_github_review_id` constructs cleanly."""
    kwargs = _attempt_kwargs(
        outcome=PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD,
        recovered_github_review_id=12345,
    )
    event = PublishAttemptEvent(**kwargs)
    assert event.recovered_github_review_id == 12345


def test_attempt_non_external_record_skip_rejects_recovered_github_review_id() -> None:
    """Every outcome OTHER than `idempotently_skipped_external_record`
    MUST have `recovered_github_review_id=None`. The field is exclusive
    to the external-record skip path; admitting it elsewhere would
    suggest a github review id binding in audit replay where none
    actually applies."""
    kwargs = _attempt_kwargs(
        outcome=PublishAttemptOutcome.SUCCESS,
        recovered_github_review_id=42,
    )
    with pytest.raises(ValidationError, match="must have recovered_github_review_id=None"):
        PublishAttemptEvent(**kwargs)


@pytest.mark.parametrize("bad_id", [0, -1])
def test_attempt_rejects_zero_or_negative_recovered_github_review_id(bad_id: int) -> None:
    """`recovered_github_review_id` must be a positive int per the
    `Field(ge=1)` constraint — GitHub review ids are positive
    integers; zero AND negative both surface a producer bug.
    Parametrized over `[0, -1]` so the test name's "zero or negative"
    claim is honored across both boundary cases (CodeRabbit 2026-05-27
    pin)."""
    kwargs = _attempt_kwargs(
        outcome=PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD,
        recovered_github_review_id=bad_id,
    )
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        PublishAttemptEvent(**kwargs)
