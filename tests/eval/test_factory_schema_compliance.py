"""Eval-harness factory schema compliance.

Each factory's `.create()` produces a schema-valid instance of its target
canonical type. Catches drift between factory-generated shapes and the
underlying canonical types when those types evolve.

Backs the harness's "factories own setting `is_eval=True`" discipline:
audit-event factories construct frozen+extra=forbid Pydantic models with
`is_eval=True` set on every instance.

The `PRContext`-shape factory + the webhook-input-schema validation
arrives in a later spec (when `api/webhooks/schemas.py` exists), per the
Input boundary held item in the eval-harness spec.
"""

from typing import Any
from uuid import UUID

import pytest

from outrider.audit.events import (
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
)
from outrider.policy import EvidenceTier, FindingType, lookup_severity
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import PerFindingDecision, ReviewFinding

from .fixtures import (
    FindingEventFactory,
    FindingFactory,
    HITLDecisionEventFactory,
    HITLRequestEventFactory,
    ReviewFactory,
    TraceDecisionEventFactory,
)


def test_review_factory_produces_dict_with_is_eval_true() -> None:
    """ReviewFactory returns a dict shaped for `Review(**dict)` insertion."""
    row = ReviewFactory.create()
    assert isinstance(row, dict)
    assert row["is_eval"] is True
    assert isinstance(row["id"], UUID)
    assert isinstance(row["installation_id"], int)
    assert row["status"] == "completed"


def test_review_factory_overrides_replace_defaults() -> None:
    """Overrides win over factory defaults."""
    row = ReviewFactory.create(installation_id=99999, is_eval=True, status="failed")
    assert row["installation_id"] == 99999
    assert row["status"] == "failed"


def test_finding_factory_produces_review_finding() -> None:
    """FindingFactory returns a ReviewFinding with the canonical content_hash.

    `ReviewFinding` has no construction-time validator on `content_hash` —
    unlike `FindingEvent`, whose validator enforces hash-equality on
    construction (see `test_finding_event_factory_validator_runs_on_construction`).
    That makes the factory's hash logic the ONLY place the canonical-SHA-256
    contract per spec §8.5 is enforced for `ReviewFinding`. Asserting equality
    against a recomputed hash (not just length) guards against a factory
    hash-logic bug that would otherwise slip through.
    """
    finding = FindingFactory.create()
    assert isinstance(finding, ReviewFinding)
    assert finding.finding_type == FindingType.SQL_INJECTION
    assert finding.evidence_tier == EvidenceTier.JUDGED
    expected_hash = compute_finding_content_hash(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=finding.finding_type,
    )
    assert finding.content_hash == expected_hash
    assert len(finding.content_hash) == 64  # belt + suspenders for spec §8.5 length


def test_finding_factory_recomputes_hash_on_field_overrides() -> None:
    """Overrides to file_path / line_start / line_end / finding_type recompute hash."""
    a = FindingFactory.create(file_path="src/a.py", line_start=1, line_end=5)
    b = FindingFactory.create(file_path="src/b.py", line_start=1, line_end=5)
    assert a.content_hash != b.content_hash


def test_finding_event_factory_produces_finding_event_with_is_eval_true() -> None:
    """FindingEventFactory.create() returns a FindingEvent with is_eval=True.

    Severity comes from `SEVERITY_POLICY[finding_type]` via `lookup_severity`,
    NOT a hard-coded constant — encoding the policy rule rather than today's
    policy value, so the test catches policy drift.

    Note on hash assertion: FindingEvent has a construction-time validator
    that enforces `finding_content_hash` equality with the canonical helper,
    so a wrong-hash factory would fail at `FindingEventFactory.create()`
    before reaching the test body. Asserting equality here anyway for
    self-documentation parity with the FindingFactory test (round 7) and
    so a future change that loosens the validator doesn't silently weaken
    this test's guarantees.
    """
    event = FindingEventFactory.create()
    assert isinstance(event, FindingEvent)
    assert event.is_eval is True
    expected_hash = compute_finding_content_hash(
        file_path=event.file_path,
        line_start=event.line_start,
        line_end=event.line_end,
        finding_type=event.finding_type,
    )
    assert event.finding_content_hash == expected_hash
    assert len(event.finding_content_hash) == 64
    assert event.severity == lookup_severity(event.finding_type)


def test_finding_event_factory_validator_runs_on_construction() -> None:
    """The FindingEvent proof-boundary + canonical-hash validators fire via the factory."""
    # JUDGED is the default tier; query_match_id and trace_path can be None
    event = FindingEventFactory.create()
    assert event.evidence_tier == EvidenceTier.JUDGED
    assert event.query_match_id is None
    assert event.trace_path is None


def test_trace_decision_event_factory_satisfies_validator_rules() -> None:
    """Default factory output passes the resolved-cross-field rules per
    #017 × #024 amendment: target_file == resolved_candidate_paths[0]."""
    event = TraceDecisionEventFactory.create()
    assert isinstance(event, TraceDecisionEvent)
    assert event.is_eval is True
    assert event.resolution_status == "resolved"
    assert event.target_file is not None
    # Per #024 amendment to #017: resolved target_file must equal the
    # single resolved_candidate_paths entry.
    assert len(event.resolved_candidate_paths) == 1
    assert event.target_file == event.resolved_candidate_paths[0]


def test_trace_decision_event_factory_unresolved_override() -> None:
    """Overrides for unresolved + target_file=None + empty
    resolved_candidate_paths construct cleanly."""
    event = TraceDecisionEventFactory.create(
        resolution_status="unresolved",
        target_file=None,
        proposed_import_strings=("foo", "bar"),
        resolved_candidate_paths=(),
    )
    assert event.resolution_status == "unresolved"
    assert event.target_file is None


def test_hitl_request_event_factory_produces_hitl_request_with_is_eval() -> None:
    """HITLRequestEventFactory returns HITLRequestEvent with is_eval=True."""
    event = HITLRequestEventFactory.create()
    assert isinstance(event, HITLRequestEvent)
    assert event.is_eval is True
    assert isinstance(event.findings_requiring_approval, tuple)
    assert isinstance(event.auto_post_findings, tuple)


def test_hitl_decision_event_factory_produces_decision_with_default_approve() -> None:
    """HITLDecisionEventFactory's default decisions tuple has one APPROVE."""
    event = HITLDecisionEventFactory.create()
    assert isinstance(event, HITLDecisionEvent)
    assert event.is_eval is True
    assert isinstance(event.decisions, tuple)
    assert len(event.decisions) == 1
    assert isinstance(event.decisions[0], PerFindingDecision)


def test_hitl_decision_event_factory_admits_decision_overrides() -> None:
    """Caller-supplied decisions tuple replaces the default."""
    from uuid import uuid4

    from outrider.schemas import PerFindingOutcome

    custom = (
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.REJECT,
            reason="duplicate of prior finding",
        ),
    )
    event = HITLDecisionEventFactory.create(decisions=custom)
    assert event.decisions == custom


# Regression coverage for the discipline guards added during the eval-harness
# Copilot audit chain (rounds 6 + 7 → spec Actual Outcome Items 7 + 11).
# Surfaced by Codex audit: the helpers were correct, but the exact bug paths
# they close had no direct tests, so a future regression would only surface
# the next time someone hit the bug manually. Tests below exercise:
#   - `_normalize_finding_type`: valid str coerces; valid enum stays; invalid
#     str raises at the factory call site (not at a confusing downstream
#     Pydantic ValidationError); the original bug (string input silently
#     skipped severity + content_hash derivation) stays closed.
#   - `_reject_is_eval_false`: any non-True is_eval override is rejected.
#     Strict `is not True` (not `is False`) — Pydantic V2 lenient mode
#     coerces falsy values like 0, "", "false" → False on bool fields, so
#     the narrower `is False` check would let those slip past the gate.


_FINDING_TYPE_FACTORIES = [
    pytest.param(FindingFactory, id="FindingFactory"),
    pytest.param(FindingEventFactory, id="FindingEventFactory"),
]


@pytest.mark.parametrize("factory", _FINDING_TYPE_FACTORIES)
@pytest.mark.parametrize(
    "input_value,expected_enum",
    [
        ("sql_injection", FindingType.SQL_INJECTION),
        (FindingType.SQL_INJECTION, FindingType.SQL_INJECTION),
    ],
)
def test_factory_normalizes_finding_type_str_or_enum(
    factory: Any, input_value: str | FindingType, expected_enum: FindingType
) -> None:
    """Both finding_type-carrying factories accept str-enum value or enum.

    Cross-product over (factory × input form) — guards both
    `_normalize_finding_type()` call sites symmetrically. If a future change
    drops the helper from either factory, this test catches it.
    """
    result = factory.create(finding_type=input_value)
    assert result.finding_type == expected_enum


@pytest.mark.parametrize("factory", _FINDING_TYPE_FACTORIES)
def test_factory_rejects_invalid_finding_type_string(factory: Any) -> None:
    """Both factories raise ValueError at the call site for an invalid string.

    Loud-failure pattern: error names the bad value, not a confusing
    downstream Pydantic ValidationError or a placeholder content_hash.
    """
    with pytest.raises(ValueError, match="not a valid FindingType"):
        factory.create(finding_type="not_a_real_type")


def test_finding_factory_string_input_triggers_severity_and_hash_derivations() -> None:
    """The original Item 11 bug fix: string finding_type must trigger both derivations.

    Before round 7, the isinstance gate skipped both `lookup_severity()` and
    `compute_finding_content_hash()` when `finding_type` arrived as a string.
    Verifies that severity is derived from policy AND content_hash is the
    canonical recomputed value, not the placeholder. FindingFactory-specific
    because the field name is `content_hash` (vs `finding_content_hash` on
    FindingEventFactory) and only ReviewFinding lacks a hash validator that
    would otherwise catch a placeholder hash on construction.
    """
    finding = FindingFactory.create(finding_type="sql_injection")
    assert finding.severity == lookup_severity(FindingType.SQL_INJECTION)
    expected_hash = compute_finding_content_hash(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=finding.finding_type,
    )
    assert finding.content_hash == expected_hash


def test_review_factory_head_sha_unique_per_call() -> None:
    """Item 9 regression: ReviewFactory.create() produces unique head_sha per call.

    Guards against a future regression that hard-codes head_sha (or any other
    component of the `uq_review_natural_key` UNIQUE(repo_id, pr_number,
    head_sha) triple) back to a constant default. Five calls; five distinct
    head_shas. Also asserts the SHA-1 shape (40 hex chars) per the field's
    documented contract.
    """
    rows = [ReviewFactory.create() for _ in range(5)]
    head_shas = {row["head_sha"] for row in rows}
    assert len(head_shas) == 5  # all distinct
    for sha in head_shas:
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


@pytest.mark.parametrize(
    "factory",
    [
        ReviewFactory,
        FindingEventFactory,
        TraceDecisionEventFactory,
        HITLRequestEventFactory,
        HITLDecisionEventFactory,
    ],
    ids=[
        "ReviewFactory",
        "FindingEventFactory",
        "TraceDecisionEventFactory",
        "HITLRequestEventFactory",
        "HITLDecisionEventFactory",
    ],
)
@pytest.mark.parametrize("bad_value", [False, 0, "false", "False", None, ""])
def test_every_is_eval_carrying_factory_rejects_non_true_override(
    factory: Any, bad_value: Any
) -> None:
    """All 5 `is_eval`-carrying factories reject any non-True override.

    Cross-product test (5 factories × 6 bad values = 30 cases) — guards
    against a future regression that drops the `_reject_is_eval_false()`
    call from any one of the five factory `create()` methods. `FindingFactory`
    is not in this list because `ReviewFinding` (cross-boundary type) has no
    `is_eval` field; the eval-isolation flag lives on the corresponding
    `findings` row, not on the type.

    Strict `is not True` (round-6 tightening) covers the Pydantic V2
    lenient-coercion vector — `0`, `""`, `"false"` all coerce to `False`
    on bool fields, and would slip past a narrower `is False` check.
    """
    with pytest.raises(ValueError, match="cannot construct a record with is_eval"):
        factory.create(is_eval=bad_value)


def _is_eval_value(factory_result: Any) -> bool:
    """Read is_eval from a factory result whether it's a dict or a Pydantic model."""
    if isinstance(factory_result, dict):
        return factory_result["is_eval"]
    return factory_result.is_eval


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(ReviewFactory, id="ReviewFactory"),
        pytest.param(FindingEventFactory, id="FindingEventFactory"),
        pytest.param(TraceDecisionEventFactory, id="TraceDecisionEventFactory"),
        pytest.param(HITLRequestEventFactory, id="HITLRequestEventFactory"),
        pytest.param(HITLDecisionEventFactory, id="HITLDecisionEventFactory"),
    ],
)
def test_every_is_eval_carrying_factory_permits_explicit_true(factory: Any) -> None:
    """Every factory accepts the explicit `is_eval=True` override (no-op vs default).

    Symmetric with `test_every_is_eval_carrying_factory_rejects_non_true_override`
    — same 5 factories, positive case. Without this, a future change that
    accidentally rejected `is_eval=True` (e.g., a mistyped `is True` check)
    would only surface when a test explicitly passed True, not in the default
    happy path.
    """
    result = factory.create(is_eval=True)
    assert _is_eval_value(result) is True


# --- PR-1 (eval-harness truthful) coverage for the factory fixes ---
# #1 policy_version tracks ACTIVE_POLICY_VERSION (not a hard-coded "1.0.0").
# #2 _normalize_evidence_tier: str/enum coercion + proof-boundary loud-fail.
# #3 optional finding_id linkage on the two HITL factories.
# #8 FindingEventFactory.proposal_hash is unique-per-call (sibling parity).


@pytest.mark.parametrize("factory", _FINDING_TYPE_FACTORIES)
def test_factory_policy_version_tracks_active_policy_version(factory: Any) -> None:
    """Both finding factories stamp ACTIVE_POLICY_VERSION, not a hard-coded literal.

    Encodes the policy-version-tracking rule rather than today's value, so a
    future ACTIVE_POLICY_VERSION bump is reflected automatically and a
    regression back to a hard-coded string is caught.
    """
    assert factory.create().policy_version == ACTIVE_POLICY_VERSION


@pytest.mark.parametrize("factory", _FINDING_TYPE_FACTORIES)
@pytest.mark.parametrize(
    "input_value",
    [EvidenceTier.JUDGED, EvidenceTier.JUDGED.value],
    ids=["enum", "str"],
)
def test_factory_normalizes_evidence_tier_str_or_enum(
    factory: Any, input_value: EvidenceTier | str
) -> None:
    """Both finding factories accept the EvidenceTier enum or its str-enum value.

    Companion to the finding_type normalization tests — guards both
    `_normalize_evidence_tier()` call sites symmetrically.
    """
    assert factory.create(evidence_tier=input_value).evidence_tier == EvidenceTier.JUDGED


@pytest.mark.parametrize("factory", _FINDING_TYPE_FACTORIES)
def test_factory_rejects_invalid_evidence_tier_string(factory: Any) -> None:
    """Both factories raise ValueError at the call site for an invalid tier string."""
    with pytest.raises(ValueError, match="not a valid EvidenceTier"):
        factory.create(evidence_tier="not_a_real_tier")


@pytest.mark.parametrize("factory", _FINDING_TYPE_FACTORIES)
def test_factory_rejects_non_judged_tier_without_proof_artifact(factory: Any) -> None:
    """OBSERVED/INFERRED overrides without their proof artifact fail loud at the factory.

    The factories supply no proof artifacts by default. Without the
    `_normalize_evidence_tier` guard this would surface as a less obvious
    `enforce_proof_boundary` ValidationError during model construction.
    """
    with pytest.raises(ValueError, match="OBSERVED requires a query_match_id"):
        factory.create(evidence_tier=EvidenceTier.OBSERVED)
    with pytest.raises(ValueError, match="INFERRED requires a trace_path"):
        factory.create(evidence_tier=EvidenceTier.INFERRED)


@pytest.mark.parametrize("factory", _FINDING_TYPE_FACTORIES)
def test_factory_accepts_non_judged_tier_with_proof_artifact(factory: Any) -> None:
    """A non-JUDGED tier constructs cleanly when its proof artifact is supplied.

    Symmetric with `test_factory_rejects_non_judged_tier_without_proof_artifact`:
    both OBSERVED (query_match_id) and INFERRED (trace_path) admit when the
    matching artifact is present. `trace_path` is `tuple[str, ...]` — a bare
    string would fail the proof-boundary validator (or coerce to a tuple of
    characters), so it must be a tuple of scope-unit strings.
    """
    observed = factory.create(evidence_tier=EvidenceTier.OBSERVED, query_match_id="q-eval-1")
    assert observed.evidence_tier == EvidenceTier.OBSERVED
    assert observed.query_match_id == "q-eval-1"

    inferred = factory.create(
        evidence_tier=EvidenceTier.INFERRED,
        trace_path=("src.foo.changed_scope", "src.bar.target_scope"),
    )
    assert inferred.evidence_tier == EvidenceTier.INFERRED
    assert inferred.trace_path == ("src.foo.changed_scope", "src.bar.target_scope")


def test_finding_event_factory_proposal_hash_unique_per_call() -> None:
    """FindingEventFactory stamps a unique proposal_hash per call (sibling of FindingFactory).

    The default was previously the constant ``"a" * 64``, so two factory events
    composed into one batch shared a proposal_hash. Unique-per-call matches the
    FindingFactory default and avoids collisions on the SHA-256-shaped field.
    """
    hashes = {FindingEventFactory.create().proposal_hash for _ in range(5)}
    assert len(hashes) == 5
    for digest in hashes:
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


def test_hitl_factories_link_finding_id_into_defaults() -> None:
    """Sharing review_id + finding_id composes a coherent request+decision pair.

    A real HITL request and its decision live on the same review and reference
    the same finding; passing both ids to both factories models that pair. The
    factories default each id independently, so finding_id alone links the
    finding but leaves review_id divergent -- share both for a replay-coherent
    pair.
    """
    from uuid import uuid4

    rid, fid = uuid4(), uuid4()
    request = HITLRequestEventFactory.create(review_id=rid, finding_id=fid)
    decision = HITLDecisionEventFactory.create(review_id=rid, finding_id=fid)
    assert request.review_id == decision.review_id == rid
    assert request.findings_requiring_approval == (fid,)
    assert decision.decisions[0].finding_id == fid


def test_hitl_decision_factory_explicit_decisions_win_over_finding_id() -> None:
    """An explicit decisions tuple wins; the finding_id linkage shortcut is ignored."""
    from uuid import uuid4

    from outrider.schemas import PerFindingOutcome

    other_fid = uuid4()
    custom = (
        PerFindingDecision(
            finding_id=other_fid,
            outcome=PerFindingOutcome.APPROVE,
            reason="",
        ),
    )
    event = HITLDecisionEventFactory.create(finding_id=uuid4(), decisions=custom)
    assert event.decisions[0].finding_id == other_fid
