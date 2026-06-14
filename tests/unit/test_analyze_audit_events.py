# See specs/2026-05-19-analyze-foundation.md §5 and
# specs/2026-06-14-observed-query-library-v1.md (ObservedSkipShadowEvent).
"""Analyze-event subclasses: shape + validator + discriminator tests.

Pins:
- `AnalyzeCompletedEvent`: counter cross-field validators
  (`_enforce_proposal_accounting`, `_enforce_response_accounting`).
- `FindingProposalRejectedEvent`: bidirectional
  `claimed_evidence_tier`/`rejection_reason` coupling; every rejection
  reason accepted; pattern guards on hash fields.
- `AnalyzeResponseRejectedEvent`: `response_hash` pattern; Literal
  `rejection_reason`.
- `ObservedSkipShadowEvent` (Cost Lever 3): outcome/blocker consistency,
  sub-model line-order, file_path canonicalization, node_id pinning.
- All four: frozen + extra="forbid", discriminator routing through
  `AuditEventAdapter`.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    AuditEventAdapter,
    FindingProposalRejectedEvent,
    ObservedSkipChangedRegion,
    ObservedSkipCoveringMatch,
    ObservedSkipShadowEvent,
)
from outrider.policy import EvidenceTier
from outrider.policy.canonical import compute_identity_hash, compute_response_hash


def _completed_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
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
        "total_cache_read_tokens": 0,
        "total_cache_write_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_usd": 0.0,
        "pricing_version": "v1",
        "policy_version": "1.0.0",
        "analyze_model": "claude-sonnet-4-6",
    }
    base.update(overrides)
    return base


def _rejected_proposal_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "review_id": uuid4(),
        "file_path": "src/foo.py",
        "proposal_hash": compute_identity_hash({"x": 1}),
        "claimed_evidence_tier": EvidenceTier.JUDGED,
        "claimed_finding_type_hash": "abcdef0123456789",
        "claimed_finding_type_len": 12,
        "rejection_reason": "span_outside_scope_unit",
        "rejection_detail": "(100,200)",
    }
    base.update(overrides)
    return base


def _rejected_response_kwargs(**overrides: Any) -> dict[str, Any]:
    """Fixture for AnalyzeResponseRejectedEvent. `response_hash` is
    derived via `compute_response_hash` (the canonical text-bytes
    recipe), NOT via `compute_identity_hash` (the structured-dict
    recipe). Post-PR review fixture-drift fix.
    """
    base: dict[str, Any] = {
        "review_id": uuid4(),
        "file_path": "src/foo.py",
        "response_hash": compute_response_hash("unparseable response text"),
        "rejection_reason": "raw_response_unparseable",
        "rejection_detail": "findings[0].finding_type x1",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# AnalyzeCompletedEvent
# ---------------------------------------------------------------------------


def test_analyze_completed_admits_zero_counters() -> None:
    """All-zero edge case: a no-op pass is valid."""
    event = AnalyzeCompletedEvent(**_completed_kwargs())
    assert event.event_type == "analyze_completed"
    assert event.pass_index == 0


def test_analyze_completed_admits_consistent_proposal_accounting() -> None:
    """`n_proposals_seen == n_findings_emitted + n_proposals_rejected` holds."""
    event = AnalyzeCompletedEvent(
        **_completed_kwargs(
            n_proposals_seen=5,
            n_findings_emitted=3,
            n_proposals_rejected=2,
            n_llm_calls=1,
        )
    )
    assert event.n_proposals_seen == 5


def test_analyze_completed_rejects_proposal_accounting_mismatch() -> None:
    """Sum off by one — 3+1 != 5."""
    with pytest.raises(ValidationError, match="Proposal accounting mismatch"):
        AnalyzeCompletedEvent(
            **_completed_kwargs(
                n_proposals_seen=5,
                n_findings_emitted=3,
                n_proposals_rejected=1,
            )
        )


def test_analyze_completed_rejects_findings_without_proposals() -> None:
    """0 != (1-0)+0: a NON-served finding emitted without a counted proposal is
    incoherent (n_findings_served defaults 0, so it does not subtract)."""
    with pytest.raises(ValidationError, match="Proposal accounting mismatch"):
        AnalyzeCompletedEvent(
            **_completed_kwargs(
                n_proposals_seen=0,
                n_findings_emitted=1,
                n_proposals_rejected=0,
            )
        )


def test_analyze_completed_admits_served_findings() -> None:
    """Stage B serve flip: cache-served findings ride n_findings_emitted but are
    subtracted from the proposal lifecycle via n_findings_served. A served-only
    pass (2 served findings, 0 proposals, 0 LLM calls) is coherent:
    0 == (2 - 2) + 0."""
    event = AnalyzeCompletedEvent(
        **_completed_kwargs(
            n_proposals_seen=0,
            n_findings_emitted=2,
            n_findings_served=2,
            n_proposals_rejected=0,
            n_llm_calls=0,
        )
    )
    assert event.n_findings_served == 2


def test_analyze_completed_admits_mixed_served_and_proposed() -> None:
    """A pass mixing model-proposed and cache-served findings: 3 proposals → 2
    findings + 1 rejected, plus 2 served → n_findings_emitted=4. The equation
    excludes the served pair: 3 == (4 - 2) + 1."""
    event = AnalyzeCompletedEvent(
        **_completed_kwargs(
            n_proposals_seen=3,
            n_findings_emitted=4,
            n_findings_served=2,
            n_proposals_rejected=1,
            n_llm_calls=1,
        )
    )
    assert event.n_findings_emitted == 4
    assert event.n_findings_served == 2


def test_analyze_completed_served_findings_revert_the_fold() -> None:
    """Revert-the-fold proof: the SAME served-only counters that pass WITH the
    n_findings_served subtraction would FAIL the pre-amendment equation
    (n_proposals_seen == n_findings_emitted + n_proposals_rejected). Guards a
    future revert from silently re-admitting the FUP-flagged incoherence."""
    AnalyzeCompletedEvent(  # passes with the subtraction
        **_completed_kwargs(n_proposals_seen=0, n_findings_emitted=2, n_findings_served=2)
    )
    with pytest.raises(ValidationError, match="Proposal accounting mismatch"):
        # The pre-amendment shape (no served subtraction): 0 != 2 + 0.
        AnalyzeCompletedEvent(
            **_completed_kwargs(n_proposals_seen=0, n_findings_emitted=2, n_findings_served=0)
        )


def test_analyze_completed_admits_observed_findings_subtract() -> None:
    """Cost Lever 3: deterministic OBSERVED findings ride n_findings_emitted but
    are subtracted via n_findings_observed. An OBSERVED-only pass (2 OBSERVED
    findings, 0 model proposals, but the LLM ran) is coherent: 0 == (2 - 0 - 2) + 0."""
    event = AnalyzeCompletedEvent(
        **_completed_kwargs(
            n_proposals_seen=0,
            n_findings_emitted=2,
            n_findings_observed=2,
            n_proposals_rejected=0,
            n_llm_calls=1,
        )
    )
    assert event.n_findings_observed == 2


def test_analyze_completed_admits_mixed_proposed_and_observed() -> None:
    """A pass mixing model-proposed and deterministic OBSERVED findings: 3
    proposals → 2 findings + 1 rejected, plus 2 OBSERVED → n_findings_emitted=4.
    The equation excludes the OBSERVED pair: 3 == (4 - 0 - 2) + 1."""
    event = AnalyzeCompletedEvent(
        **_completed_kwargs(
            n_proposals_seen=3,
            n_findings_emitted=4,
            n_findings_observed=2,
            n_proposals_rejected=1,
            n_llm_calls=1,
        )
    )
    assert event.n_findings_emitted == 4
    assert event.n_findings_observed == 2


def test_analyze_completed_observed_findings_revert_the_fold() -> None:
    """Revert-the-fold proof: the SAME OBSERVED-only counters that pass WITH the
    n_findings_observed subtraction would FAIL without it (0 != 2 + 0). Guards a
    future revert from silently re-admitting the accounting incoherence."""
    AnalyzeCompletedEvent(  # passes with the subtraction
        **_completed_kwargs(n_proposals_seen=0, n_findings_emitted=2, n_findings_observed=2)
    )
    with pytest.raises(ValidationError, match="Proposal accounting mismatch"):
        AnalyzeCompletedEvent(
            **_completed_kwargs(n_proposals_seen=0, n_findings_emitted=2, n_findings_observed=0)
        )


def test_analyze_completed_admits_response_accounting_subset() -> None:
    """`n_responses_rejected <= n_llm_calls`."""
    event = AnalyzeCompletedEvent(**_completed_kwargs(n_responses_rejected=2, n_llm_calls=3))
    assert event.n_responses_rejected == 2


def test_analyze_completed_rejects_response_accounting_exceeds() -> None:
    """Rejected responses can't exceed LLM calls."""
    with pytest.raises(ValidationError, match="cannot exceed"):
        AnalyzeCompletedEvent(**_completed_kwargs(n_responses_rejected=3, n_llm_calls=2))


def test_analyze_completed_admits_zero_calls_zero_rejected() -> None:
    """Trivial edge case: no calls, no rejections."""
    event = AnalyzeCompletedEvent(**_completed_kwargs(n_responses_rejected=0, n_llm_calls=0))
    assert event.n_llm_calls == 0


def test_analyze_completed_rejects_negative_counter() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AnalyzeCompletedEvent(**_completed_kwargs(n_files_analyzed=-1))


def test_analyze_completed_frozen() -> None:
    event = AnalyzeCompletedEvent(**_completed_kwargs())
    with pytest.raises(ValidationError, match="Instance is frozen"):
        event.pass_index = 99  # type: ignore[misc]


def test_analyze_completed_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AnalyzeCompletedEvent(**_completed_kwargs(unexpected_field="bad"))


# ---------------------------------------------------------------------------
# FindingProposalRejectedEvent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        "query_match_id_not_in_registry",
        "trace_path_not_admissible",
        "finding_type_not_in_enum",
        "span_outside_scope_unit",
        "span_outside_file",
        # FUP-162: the parameterized-call veto's deterministic rejection.
        "sql_injection_on_parameterized_call",
        "schema_construction_failed",
    ],
)
def test_finding_proposal_rejected_admits_all_non_tier_reasons(reason: str) -> None:
    """Non-tier rejection reasons accept a non-None claimed_evidence_tier."""
    event = FindingProposalRejectedEvent(**_rejected_proposal_kwargs(rejection_reason=reason))
    assert event.rejection_reason == reason
    assert event.claimed_evidence_tier == EvidenceTier.JUDGED


def test_finding_proposal_rejected_evidence_tier_failure_requires_none() -> None:
    """`evidence_tier_not_in_enum` requires claimed_evidence_tier=None."""
    event = FindingProposalRejectedEvent(
        **_rejected_proposal_kwargs(
            rejection_reason="evidence_tier_not_in_enum",
            claimed_evidence_tier=None,
        )
    )
    assert event.claimed_evidence_tier is None


def test_finding_proposal_rejected_evidence_tier_failure_with_tier_raises() -> None:
    """Tier failure + non-None tier is incoherent."""
    with pytest.raises(ValidationError, match="evidence_tier_not_in_enum.*requires"):
        FindingProposalRejectedEvent(
            **_rejected_proposal_kwargs(
                rejection_reason="evidence_tier_not_in_enum",
                claimed_evidence_tier=EvidenceTier.JUDGED,
            )
        )


def test_finding_proposal_rejected_non_tier_failure_with_none_tier_raises() -> None:
    """Non-tier-failure + None tier is incoherent (the model's tier parsed)."""
    with pytest.raises(ValidationError, match="requires a non-None"):
        FindingProposalRejectedEvent(
            **_rejected_proposal_kwargs(
                rejection_reason="span_outside_scope_unit",
                claimed_evidence_tier=None,
            )
        )


def test_finding_proposal_rejected_proposal_hash_pattern() -> None:
    """`proposal_hash` must match SHA-256 64-hex pattern."""
    with pytest.raises(ValidationError, match="(?s)proposal_hash.*String should match pattern"):
        FindingProposalRejectedEvent(**_rejected_proposal_kwargs(proposal_hash="not-a-hash"))


def test_finding_proposal_rejected_finding_type_hash_short_pattern() -> None:
    """`claimed_finding_type_hash` must match the 16-hex SHORT pattern."""
    with pytest.raises(
        ValidationError, match="(?s)claimed_finding_type_hash.*String should match pattern"
    ):
        FindingProposalRejectedEvent(
            **_rejected_proposal_kwargs(claimed_finding_type_hash="a" * 64)
        )


def test_finding_proposal_rejected_finding_type_len_bounded() -> None:
    """`claimed_finding_type_len <= 128` per raw layer cap."""
    with pytest.raises(ValidationError, match="less than or equal to 128"):
        FindingProposalRejectedEvent(**_rejected_proposal_kwargs(claimed_finding_type_len=129))


def test_finding_proposal_rejected_detail_length_bounded() -> None:
    """`rejection_detail` capped at 500 chars."""
    with pytest.raises(ValidationError, match="at most 500 characters"):
        FindingProposalRejectedEvent(**_rejected_proposal_kwargs(rejection_detail="x" * 501))


def test_finding_proposal_rejected_frozen() -> None:
    event = FindingProposalRejectedEvent(**_rejected_proposal_kwargs())
    with pytest.raises(ValidationError, match="Instance is frozen"):
        event.file_path = "other.py"  # type: ignore[misc]


def test_finding_proposal_rejected_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FindingProposalRejectedEvent(**_rejected_proposal_kwargs(unexpected="bad"))


# ---------------------------------------------------------------------------
# AnalyzeResponseRejectedEvent
# ---------------------------------------------------------------------------


def test_analyze_response_rejected_admits_well_formed() -> None:
    event = AnalyzeResponseRejectedEvent(**_rejected_response_kwargs())
    assert event.event_type == "analyze_response_rejected"
    assert event.rejection_reason == "raw_response_unparseable"


def test_analyze_response_rejected_response_hash_pattern() -> None:
    with pytest.raises(ValidationError, match="(?s)response_hash.*String should match pattern"):
        AnalyzeResponseRejectedEvent(**_rejected_response_kwargs(response_hash="bad"))


def test_analyze_response_rejected_rejects_other_reasons() -> None:
    """Literal accepts only the one value."""
    with pytest.raises(ValidationError, match="Input should be 'raw_response_unparseable'"):
        AnalyzeResponseRejectedEvent(
            **_rejected_response_kwargs(rejection_reason="some_other_reason")
        )


def test_analyze_response_rejected_detail_length_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 500 characters"):
        AnalyzeResponseRejectedEvent(**_rejected_response_kwargs(rejection_detail="x" * 501))


def test_analyze_response_rejected_frozen() -> None:
    event = AnalyzeResponseRejectedEvent(**_rejected_response_kwargs())
    with pytest.raises(ValidationError, match="Instance is frozen"):
        event.file_path = "other.py"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Discriminator routing — the AuditEvent tagged union picks each subtype
# correctly via TypeAdapter.
# ---------------------------------------------------------------------------


def test_audit_event_adapter_routes_analyze_completed() -> None:
    kwargs = _completed_kwargs()
    payload = {**kwargs, "event_type": "analyze_completed"}
    payload["review_id"] = str(payload["review_id"])
    event = AuditEventAdapter.validate_python(payload)
    assert isinstance(event, AnalyzeCompletedEvent)


def test_audit_event_adapter_routes_finding_proposal_rejected() -> None:
    kwargs = _rejected_proposal_kwargs()
    payload = {**kwargs, "event_type": "finding_proposal_rejected"}
    payload["review_id"] = str(payload["review_id"])
    payload["claimed_evidence_tier"] = payload["claimed_evidence_tier"].value
    event = AuditEventAdapter.validate_python(payload)
    assert isinstance(event, FindingProposalRejectedEvent)


def test_audit_event_adapter_routes_analyze_response_rejected() -> None:
    kwargs = _rejected_response_kwargs()
    payload = {**kwargs, "event_type": "analyze_response_rejected"}
    payload["review_id"] = str(payload["review_id"])
    event = AuditEventAdapter.validate_python(payload)
    assert isinstance(event, AnalyzeResponseRejectedEvent)


# ---------------------------------------------------------------------------
# ObservedSkipShadowEvent (Cost Lever 3,
# specs/2026-06-14-observed-query-library-v1.md). Shadow telemetry: per-file
# OBSERVED-tier skip-routing decision, never raw model output (spans + ids
# only, metadata-only audit contract DECISIONS.md#014).
# ---------------------------------------------------------------------------


def _skip_shadow_kwargs(**overrides: Any) -> dict[str, Any]:
    """Fixture for ObservedSkipShadowEvent. Default is the V1-realistic
    `not_eligible` shape: default-deny seeds zero `skip_safe` queries, so every
    changed region is a blocker. One head-side changed region, no covering
    matches, the same region echoed as the single blocker — satisfies the
    outcome/blocker-consistency validator."""
    region = ObservedSkipChangedRegion(side="head", line_start=10, line_end=14)
    base: dict[str, Any] = {
        "review_id": uuid4(),
        "file_path": "src/foo.py",
        "outcome": "not_eligible",
        "changed_regions": (region,),
        "covering_matches": (),
        "blockers": (region,),
    }
    base.update(overrides)
    return base


def test_observed_skip_shadow_admits_not_eligible() -> None:
    """Default-deny V1 case: one uncovered changed region → not_eligible."""
    event = ObservedSkipShadowEvent(**_skip_shadow_kwargs())
    assert event.event_type == "observed_skip_shadow"
    assert event.node_id == "analyze"
    assert event.outcome == "not_eligible"
    assert len(event.blockers) == 1


def test_observed_skip_shadow_admits_would_skip_fully_covered() -> None:
    """Post-promotion case: a `skip_safe` match covers the changed region, so
    there are no blockers → would_skip."""
    match = ObservedSkipCoveringMatch(
        query_match_id="py-observed-eval-call-1", side="head", line_start=10, line_end=14
    )
    event = ObservedSkipShadowEvent(
        **_skip_shadow_kwargs(outcome="would_skip", covering_matches=(match,), blockers=())
    )
    assert event.outcome == "would_skip"
    assert event.blockers == ()
    assert event.covering_matches[0].query_match_id == "py-observed-eval-call-1"


def test_observed_skip_shadow_rejects_would_skip_with_blockers() -> None:
    """outcome/blocker consistency: would_skip with a non-empty blocker set is
    incoherent (the outcome is a deterministic function of coverage)."""
    with pytest.raises(ValidationError, match="would_skip' requires empty"):
        ObservedSkipShadowEvent(**_skip_shadow_kwargs(outcome="would_skip"))


def test_observed_skip_shadow_rejects_not_eligible_without_blockers() -> None:
    """The inverse: not_eligible with no blocker has no recorded reason."""
    with pytest.raises(ValidationError, match="not_eligible' requires at least"):
        ObservedSkipShadowEvent(**_skip_shadow_kwargs(blockers=()))


def test_observed_skip_changed_region_rejects_inverted_lines() -> None:
    with pytest.raises(ValidationError, match="must be >= line_start"):
        ObservedSkipChangedRegion(side="base", line_start=14, line_end=10)


def test_observed_skip_covering_match_rejects_inverted_lines() -> None:
    with pytest.raises(ValidationError, match="must be >= line_start"):
        ObservedSkipCoveringMatch(query_match_id="q-1", side="head", line_start=14, line_end=10)


def test_observed_skip_covering_match_rejects_empty_query_match_id() -> None:
    """`query_match_id` is the registry pointer — an empty id can't reference a
    real query (min_length=1)."""
    with pytest.raises(ValidationError, match="query_match_id"):
        ObservedSkipCoveringMatch(query_match_id="", side="head", line_start=10, line_end=14)


def test_observed_skip_covering_match_rejects_non_head_side() -> None:
    """`side` is pinned to Literal["head"] — OBSERVED queries run on head content,
    so a base-side covering match is impossible in V1 and rejected at construction.
    A future base-side structural query widens the Literal deliberately."""
    with pytest.raises(ValidationError):
        ObservedSkipCoveringMatch(query_match_id="q-1", side="base", line_start=10, line_end=14)


def test_observed_skip_shadow_canonicalizes_file_path() -> None:
    """file_path rides the same `validate_diff_path` canonicalization as every
    sibling path-bearing event — `./src/foo.py` normalizes to `src/foo.py`."""
    event = ObservedSkipShadowEvent(**_skip_shadow_kwargs(file_path="./src/foo.py"))
    assert event.file_path == "src/foo.py"


def test_observed_skip_shadow_rejects_traversal_file_path() -> None:
    """`..` traversal is rejected at construction (CoordinateError is not a
    ValueError subclass, so it propagates unwrapped — same as sibling events)."""
    from outrider.coordinates import CoordinateError

    with pytest.raises((CoordinateError, ValueError)):
        ObservedSkipShadowEvent(**_skip_shadow_kwargs(file_path="../secrets.txt"))


def test_observed_skip_shadow_rejects_non_analyze_node_id() -> None:
    """node_id is pinned to the analyze phase (Literal['analyze'])."""
    with pytest.raises(ValidationError):
        ObservedSkipShadowEvent(**_skip_shadow_kwargs(node_id="triage"))


def test_observed_skip_shadow_frozen() -> None:
    event = ObservedSkipShadowEvent(**_skip_shadow_kwargs())
    with pytest.raises(ValidationError, match="Instance is frozen"):
        event.file_path = "other.py"  # type: ignore[misc]


def test_observed_skip_shadow_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservedSkipShadowEvent(**_skip_shadow_kwargs(unexpected="bad"))


def test_observed_skip_shadow_covering_and_blockers_default_empty() -> None:
    """Both tuple fields default to () — a would_skip with neither kwarg is the
    minimal admissible shape (no blockers, so consistent)."""
    region = ObservedSkipChangedRegion(side="head", line_start=10, line_end=14)
    event = ObservedSkipShadowEvent(
        review_id=uuid4(), file_path="src/foo.py", outcome="would_skip", changed_regions=(region,)
    )
    assert event.covering_matches == ()
    assert event.blockers == ()


def test_observed_skip_changed_region_frozen_and_extra_forbid() -> None:
    """The sub-model's frozen-ness does not inherit from the outer event — it is
    declared on the sub-model itself, so pin both here."""
    region = ObservedSkipChangedRegion(side="head", line_start=10, line_end=14)
    with pytest.raises(ValidationError, match="Instance is frozen"):
        region.line_start = 1  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservedSkipChangedRegion(side="head", line_start=10, line_end=14, extra="bad")


def test_observed_skip_covering_match_frozen_and_extra_forbid() -> None:
    match = ObservedSkipCoveringMatch(query_match_id="q-1", side="head", line_start=10, line_end=14)
    with pytest.raises(ValidationError, match="Instance is frozen"):
        match.line_start = 1  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservedSkipCoveringMatch(
            query_match_id="q-1", side="head", line_start=10, line_end=14, extra="bad"
        )


def test_audit_event_adapter_routes_observed_skip_shadow() -> None:
    """The discriminated union routes the tag to the right subtype and the
    round-trip preserves every field (sub-model tuples included)."""
    event = ObservedSkipShadowEvent(**_skip_shadow_kwargs())
    routed = AuditEventAdapter.validate_python(event.model_dump(mode="json"))
    assert isinstance(routed, ObservedSkipShadowEvent)
    assert routed == event
