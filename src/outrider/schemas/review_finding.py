# Cross-boundary ReviewFinding per docs/spec.md §7.3 + docs/trust-boundaries.md §1
"""ReviewFinding + ReviewDimension + PublishDestination cross-boundary models.

ReviewFinding is the Pydantic carrier for every finding the agent produces.
It registers `enforce_proof_boundary` from `policy/findings.py` as a
model_validator so OBSERVED-without-query_match_id and INFERRED-without-
trace_path raise at construction time per `evidence-tier-schema-enforced`.
The `confidence` field is a `@computed_field` deriving deterministically
from `evidence_tier` per `confidence-is-computed-not-assigned` (OBSERVED=0.9,
INFERRED=0.75, JUDGED=0.5 per spec §7.3).

ReviewFinding is **NOT frozen** because it has a multi-stage lifecycle:
the analyze node constructs it with later-set fields at None;
`coordinates.tree_sitter_to_github` (separate spec) sets `publish_destination`
after computing the GitHub-comment-location translation; the HITL flow
may later set the override fields when a reviewer issues a SEVERITY_OVERRIDE
decision. Mutability lets each stage write its fields directly without
requiring `model_copy(update=...)` boilerplate. The audit-replay guarantee
for findings rides on `FindingEvent` (which IS frozen and append-only via
the audit_events trigger), not on `ReviewFinding`'s in-memory mutability.

PublishDestination uses uppercase Python member names with lowercase
serialized string values per spec §4.1.7, same convention as
EvidenceTier / FindingType / FindingSeverity.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Final, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from outrider.coordinates import validate_diff_path
from outrider.policy import (
    EvidenceTier,
    FindingSeverity,
    FindingType,
    enforce_proof_boundary,
)
from outrider.policy.canonical import SHA256_HEX_PATTERN
from outrider.policy.severity import BARE_SEMVER_PATTERN


class ReviewDimension(StrEnum):
    """The five review-dimension axes per spec §7.3."""

    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TEST_COVERAGE = "test_coverage"
    BEST_PRACTICES = "best_practices"


class PublishDestination(StrEnum):
    """Where a finding lands when published, per spec §4.1.7.

    Set by `coordinates.tree_sitter_to_github` after computing the
    GitHub-comment-location translation; the analyze node leaves the
    field None at construction time. Backs `publish-routes-through-coordinates`.
    """

    INLINE_COMMENT = "inline_comment"
    REVIEW_BODY = "review_body"
    DASHBOARD_ONLY = "dashboard_only"


# Module-level constant: confidence values per evidence tier. Wrapped
# in MappingProxyType so runtime mutation raises TypeError. Same
# defense-in-depth shape as `outrider.llm.pricing.RATE_TABLE` — without
# the proxy, a test fixture or buggy caller could mutate the mapping
# and silently change confidence for the rest of the pytest session.
# Inlined into the proxy call directly so there's no importable
# underscore-prefixed bare-dict back-door.
_CONFIDENCE_BY_TIER: Final[Mapping[EvidenceTier, float]] = MappingProxyType(
    {
        EvidenceTier.OBSERVED: 0.9,
        EvidenceTier.INFERRED: 0.75,
        EvidenceTier.JUDGED: 0.5,
    }
)


class ReviewFinding(BaseModel):
    """One finding produced by the agent's review of a PR.

    Construction-time validators enforce the proof boundary
    (`enforce_proof_boundary` from policy/findings) and the line
    constraint (`line_end >= line_start`). Confidence is a computed
    field, not a settable attribute.
    """

    # Not frozen: multi-stage lifecycle. See module docstring.
    # validate_assignment=True: lifecycle writes (publish_destination,
    # override fields) re-run model_validators + Field constraints + enum
    # typing on every assignment, so post-construction mutations cannot
    # bypass the proof boundary, the line constraint, or the typed-enum
    # gates. Without this, `finding.severity = "garbage"` would silently
    # admit; with it, the assignment raises.
    #
    # WARNING — DO NOT USE `model_copy(update={...})` on a ReviewFinding.
    # Pydantic v2's `model_copy` skips ALL model_validators even when
    # `validate_assignment=True`, so a copy-with-update that flips
    # severity silently bypasses `_enforce_severity_matches_policy`,
    # `_enforce_dimension_lockstep`, `_enforce_override_triplet_coherence`,
    # and `_verify_content_hash`. For multi-field lifecycle updates,
    # use `ReviewFinding.model_validate({**finding.model_dump(), **delta})`
    # which routes through the full validator chain.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    finding_id: UUID = Field(default_factory=uuid4)
    review_id: UUID
    installation_id: int
    # Bare-semver: same shape `AnalyzeCompletedEvent.policy_version`,
    # `FindingEvent.policy_version`, and the `severity_policies` DB CHECK
    # all enforce. Without the pattern here, a malformed `policy_version`
    # could be persisted on the `ReviewFinding` while the audit-event row
    # written from the SAME finding refuses construction — silent divergence
    # between the in-memory finding and its append-only audit shadow.
    policy_version: Annotated[str, Field(pattern=BARE_SEMVER_PATTERN)]
    finding_type: FindingType
    dimension: ReviewDimension
    severity: FindingSeverity
    evidence_tier: EvidenceTier
    # Repo-relative POSIX, post-`validate_diff_path` normalized — same
    # contract `AnalysisRound.files_examined` enforces at the schema
    # layer. (TraceCandidate uses `is_valid_import_string` on its
    # `import_string` field instead — different validator for dotted-
    # Python-import-string shape per DECISIONS.md#024.) Without this,
    # a path that fails `validate_diff_path` could ride on the finding
    # through replay and be rejected at the publisher boundary only —
    # too late.
    file_path: str
    line_start: int = Field(ge=1)
    line_end: int
    title: Annotated[str, Field(max_length=120)]
    description: Annotated[str, Field(max_length=1000)]
    # Model-output cap. The model is told to include a code snippet plus
    # short prose; 2000 chars is generous enough for typical findings and
    # bounded enough that a runaway response can't fill an audit row with
    # MB of fabricated evidence.
    evidence: Annotated[str, Field(max_length=2000)]
    # Model-output cap. Suggested fixes can include a code snippet so the
    # cap matches `evidence`.
    suggested_fix: Annotated[str | None, Field(max_length=2000)] = None
    # Query-registry-id cap. Today's query ids are short paths (`security/sql_injection`);
    # 200 chars is well above the realistic max and well below pathological.
    query_match_id: Annotated[str | None, Field(max_length=200)] = None
    # tuple, not list: post-construction `.append()` / `.clear()` on a list
    # would bypass validate_assignment (which only fires on attribute
    # rebinding, not in-place mutation). Tuple delivers true immutability.
    # Per-element + tuple bounds mirror `AnalyzeFindingProposalRaw.trace_path`
    # so the admitted layer enforces what the raw layer admits — without
    # this, a bypass path could land an unbounded trace_path in admitted
    # findings and the FindingEvent audit shadow.
    trace_path: (
        Annotated[
            tuple[Annotated[str, Field(max_length=256, min_length=1)], ...],
            Field(max_length=32),
        ]
        | None
    ) = None
    # Lifecycle / HITL-set fields (None at analyze-time):
    original_severity: FindingSeverity | None = None
    # Reviewer-supplied; reviewers can be human and HITL UIs can be sloppy,
    # but 1000 chars matches the model-output reason caps elsewhere and
    # caps a copy-pasted novel-length explanation.
    override_reason: Annotated[str | None, Field(max_length=1000)] = None
    overrider_id: UUID | None = None
    publish_destination: PublishDestination | None = None
    # SHA-256 hex digest, lowercase. Sibling of `AnalysisRound.round_id` and
    # `TraceCandidate.candidate_id` (both patterned). The dedup contract
    # in `FindingEvent` joins via this field, so the shape MUST match the
    # corresponding event's `finding_content_hash` (also patterned).
    content_hash: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    # SHA-256 hex digest of the raw proposal that produced this finding,
    # computed via `outrider.policy.canonical.compute_proposal_hash`. Per
    # `DECISIONS.md#025` (Accepted 2026-05-24), admitted findings carry
    # `proposal_hash` so trace can join `state.trace_candidates` (whose
    # `source_proposal_hash` field is the same digest) to admitted
    # `finding_id`s. **Provenance, NOT content identity** — explicitly
    # NOT part of `compute_finding_content_hash` recipe (#025 point 3):
    # `content_hash` is `(file_path, line, type)`-derived and stays
    # stable across LLM phrasing variation; `proposal_hash` is the
    # full-proposal digest and DIFFERS when the LLM rewrites
    # title/description/evidence (#022 PR/file-scoped semantics).
    # Required (no default) — analyze's admission path stamps this from
    # `compute_proposal_hash` output. Mirror on `FindingEvent.proposal_hash`.
    proposal_hash: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        """Re-run `validate_diff_path` so the schema layer enforces the
        repo-relative-POSIX invariant. Same shape as
        `AnalysisRound._enforce_canonical_files_examined` — propagates
        the canonical-record discipline to every cross-boundary model
        that carries a diff-side path. (TraceCandidate uses
        `_enforce_canonical_import_string` for its import-string field
        per DECISIONS.md#024; different shape, same validator-at-the-
        schema-floor discipline.)
        """
        return validate_diff_path(path)

    @model_validator(mode="before")
    @classmethod
    def _strip_computed_confidence_on_input(cls, data: Any) -> Any:
        """Drop a stray `confidence` key from input dicts before validation.

        Pydantic v2 includes `@computed_field` properties in
        `model_dump(mode='json')`, so a round-tripped payload —
        langgraph-checkpoint-postgres persists state via that path —
        carries `confidence` as a regular dict key. Without this
        stripper, `model_validate` on the round-tripped payload would
        fail under `extra="forbid"` because `confidence` isn't a
        regular field.

        The fix is "tolerate computed-field reappearance on input."
        The value gets re-derived from `evidence_tier` at attribute
        access time, so dropping it loses nothing and lets ReviewState
        checkpoint replay round-trip cleanly.

        Operates only on dict inputs (Pydantic v2 also accepts other
        model instances + raw shapes — those don't carry the computed
        field in their attribute namespace).
        """
        if isinstance(data, dict) and "confidence" in data:
            data = {k: v for k, v in data.items() if k != "confidence"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Deterministic mapping from evidence_tier per spec §7.3.

        OBSERVED=0.9, INFERRED=0.75, JUDGED=0.5. Read-only at the
        descriptor level — assigning to `.confidence` raises
        AttributeError regardless of model frozen-ness.
        """
        return _CONFIDENCE_BY_TIER[self.evidence_tier]

    @model_validator(mode="after")
    def _enforce_proof_boundary(self) -> Self:
        """Wire policy/findings.enforce_proof_boundary into Pydantic validation."""
        enforce_proof_boundary(
            evidence_tier=self.evidence_tier,
            query_match_id=self.query_match_id,
            trace_path=self.trace_path,
        )
        return self

    @model_validator(mode="after")
    def _enforce_line_constraint(self) -> Self:
        """line_end must be >= line_start (1-indexed per coordinates/)."""
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start "
                f"({self.line_start}); lines are 1-indexed per coordinates/"
            )
        return self

    @model_validator(mode="after")
    def _enforce_dimension_lockstep(self) -> Self:
        """`dimension` must equal `FINDING_TYPE_TO_DIMENSION[finding_type]`.

        dimension is a stored
        field (not computed), so a stale serialized payload could survive
        replay under an old mapping. The module-load lockstep guard in
        `outrider.policy.dimensions` fires only at import — it cannot
        detect a finding ALREADY in `audit_events.payload` carrying a
        drifted dimension. This validator closes that hole by asserting
        the stored dimension matches the current mapping at construction
        AND replay time (Pydantic re-runs `model_validator(mode='after')`
        on `model_validate`).

        Imported locally to avoid a circular import:
        `policy.dimensions` imports `ReviewDimension` from this module
        for its mapping values. The dependency goes one way at module
        load and the other way at call time, which is fine.
        """
        # Local import: `policy.dimensions` imports from this module at
        # load time, so a top-level import would cycle.
        from outrider.policy.dimensions import (  # noqa: PLC0415
            FINDING_TYPE_TO_DIMENSION,
        )

        expected = FINDING_TYPE_TO_DIMENSION[self.finding_type]
        if self.dimension != expected:
            raise ValueError(
                f"ReviewFinding.dimension={self.dimension.value!r} drifted from "
                f"FINDING_TYPE_TO_DIMENSION[{self.finding_type.value!r}]="
                f"{expected.value!r}. Per DECISIONS.md#021, FINDING_TYPE_TO_DIMENSION "
                f"is append-only for existing FindingType members: a mapping change "
                f"is a DECISIONS-level ontology rewrite, not a quiet code edit. "
                f"If this is a stored audit row hitting replay-time drift, revert "
                f"the mapping change OR land a new DECISIONS entry covering the "
                f"backfill plan. If this is fresh construction, the caller is not "
                f"going through the canonical lookup — use `lookup_dimension(finding_type)`."
            )
        return self

    @model_validator(mode="after")
    def _enforce_severity_matches_policy(self) -> Self:
        """Baseline severity must equal `SEVERITY_POLICY[finding_type]`
        under the live policy version. Backs `severity-set-by-policy`.

        Baseline = `original_severity` if a HITL override is in effect
        (then `severity` carries the reviewer's override), else
        `severity` directly. Either way, the policy-computed value is
        what we check.

        Replay-aware scoping. `model_validate` is the SAME path
        `TypeAdapter(AuditEvent).validate_python(...)` and equivalent
        ReviewFinding reconstructors use to rehydrate historical
        records. A historical finding under an older `policy_version`
        MUST validate cleanly — the severity it carries was correct
        AT WRITE TIME under its frozen policy, and there's no
        synchronous loader for the historical mapping here
        (`policy/versions.py::load_policy_for_version` is async; it's
        the persister/replay layer's job, not the schema's).

        Scope: the live-policy match check below fires ONLY when
        `policy_version == ACTIVE_POLICY_VERSION`. Older versions
        skip and trust the historical row. The "fresh-write smuggle"
        concern (a producer setting `policy_version="0.9.0"` to dodge
        the live check) is NOT defended here — it's a producer-side
        discipline enforced by the emitter and, when the replay/
        persister spec lands, by the persister's write-time check
        that incoming records carry the active version. The schema
        layer cannot distinguish fresh writes from replay
        reconstruction inside `model_validate`.

        Without the SEVERITY_POLICY match check below, a fresh row
        like (SQL_INJECTION, LOW) admits cleanly even though
        SEVERITY_POLICY[SQL_INJECTION] == CRITICAL — the schema-layer
        defense against caller-supplied severity drift.
        """
        # Local imports to avoid a circular import: `policy.severity`
        # transitively imports nothing from schemas, but `policy/__init__`
        # re-exports from review_finding for the proof-boundary helper,
        # so top-level imports route through the deep paths.
        from outrider.policy.severity import (  # noqa: PLC0415
            ACTIVE_POLICY_VERSION,
            SEVERITY_POLICY,
        )

        if self.policy_version != ACTIVE_POLICY_VERSION:
            # Historical record: trust the row. Versioned-replay
            # cross-check belongs in the persister/replay layer.
            return self

        baseline = self.original_severity if self.original_severity is not None else self.severity
        expected = SEVERITY_POLICY.get(self.finding_type)
        if expected is None or baseline != expected:
            raise ValueError(
                f"ReviewFinding.severity baseline={baseline.value!r} does not match "
                f"SEVERITY_POLICY[{self.finding_type.value!r}]="
                f"{(expected.value if expected else None)!r} under policy_version "
                f"{self.policy_version!r}. Per `severity-set-by-policy` "
                f"(docs/invariants.md), baseline severity comes from SEVERITY_POLICY "
                f"keyed by finding_type, never from caller or model output. If a HITL "
                f"override is in flight, set `original_severity` to the policy "
                f"baseline and put the override on `severity`."
            )
        return self

    @model_validator(mode="after")
    def _enforce_override_triplet_coherence(self) -> Self:
        """HITL override is a triplet: original_severity + override_reason
        + overrider_id. All three or none. Backs `hitl-gates-high-severity`
        + `severity-set-by-policy`.

        Without this gate, a caller could set
        `original_severity=CRITICAL, severity=LOW, override_reason=None,
        overrider_id=None` and bypass the HITL gate. The policy-baseline
        check at `_enforce_severity_matches_policy` PASSES (CRITICAL
        matches policy) but the override path has no real
        `PerFindingDecision.SEVERITY_OVERRIDE` backing it — no reason,
        no reviewer identity. The finding lands with a downgraded
        severity that has no audit trail.

        The rule: the three override fields are bound. Either all None
        (no override in effect) or all set (override happened, with a
        reason and a reviewer). A partial state is a producer bug.
        """
        override_fields = (
            self.original_severity,
            self.override_reason,
            self.overrider_id,
        )
        all_none = all(field is None for field in override_fields)
        all_set = all(field is not None for field in override_fields)
        if not (all_none or all_set):
            raise ValueError(
                f"ReviewFinding HITL override fields must be all-set-or-all-None: "
                f"original_severity={self.original_severity!r}, "
                f"override_reason={self.override_reason!r}, "
                f"overrider_id={self.overrider_id!r}. A real HITL override produces "
                f"a `PerFindingDecision.SEVERITY_OVERRIDE` with all three; a "
                f"partial state is a producer bug that would bypass `hitl-gates-"
                f"high-severity` by claiming an override that never happened."
            )
        # No-op override rejection: when the override envelope is set,
        # `severity` MUST differ from `original_severity` — otherwise the
        # override claims a reviewer-changed-the-severity event that did
        # nothing. A reviewer's intent to ACK-without-change is the
        # `APPROVE` outcome on `PerFindingDecision`, NOT a SEVERITY_OVERRIDE
        # with identical values. Catches the producer-bug class where a
        # HITL UI submits the override path without checking whether the
        # reviewer actually changed the value.
        if all_set and self.severity == self.original_severity:
            raise ValueError(
                f"ReviewFinding HITL override claims SEVERITY_OVERRIDE but "
                f"severity={self.severity.value!r} equals "
                f"original_severity={self.original_severity.value!r} — no-op "
                f"overrides are not valid. The reviewer's intent to ACK without "
                f"change is the `PerFindingDecision.APPROVE` outcome, not a "
                f"SEVERITY_OVERRIDE with identical values."
            )
        # Non-blank override_reason when envelope is set. The triplet's
        # `is None` check admits `override_reason=""` and
        # `override_reason="   "` because they're not None. But
        # `PerFindingDecision` (the cross-boundary HITL decision shape)
        # already rejects empty reasons via `Field(max_length=500)` +
        # the non-APPROVE-needs-reason validator. The ReviewFinding's
        # carrier-side check must align: an override without a substantive
        # reason is the bug class the `PerFindingDecision.outcome !=
        # APPROVE requires reason` rule defends against. `strip() == ""`
        # catches both empty and whitespace-only strings.
        if all_set and self.override_reason is not None and not self.override_reason.strip():
            raise ValueError(
                f"ReviewFinding HITL override has blank override_reason "
                f"({self.override_reason!r}); the override envelope requires "
                f"a non-blank justification matching the "
                f"`PerFindingDecision.SEVERITY_OVERRIDE` contract. An override "
                f"with no real reason is the bug class `hitl-gates-high-severity` "
                f"defends against."
            )
        return self

    @model_validator(mode="after")
    def _verify_content_hash(self) -> Self:
        """`content_hash` must equal the canonical
        `compute_finding_content_hash` over (`file_path`, `line_start`,
        `line_end`, `finding_type`).

        Mirror of `FindingEvent._verify_content_hash` at the in-memory
        layer. Format gating alone (the Field pattern) accepts any
        64-hex string for any input tuple, so an emitter bug producing
        a mis-computed hash would land on the finding AND survive into
        `AnalysisRound.round_id` (which folds `f.content_hash` into the
        round id at `analysis_round.py::compute_round_id`). The reducer
        would dedup under the bad key on replay.

        Pinning the hash recipe at the in-memory layer makes
        `ReviewFinding`'s dedup contract identical to `FindingEvent`'s
        append-only contract — fixture code using
        `compute_identity_hash({...})` is on the wrong recipe and
        fails here loud.
        """
        # Local import: `audit.events` imports from schemas for
        # PerFindingDecision / PublishDestination / ReviewDimension,
        # so a top-level import would cycle.
        from outrider.audit.events import (  # noqa: PLC0415
            compute_finding_content_hash,
        )

        expected = compute_finding_content_hash(
            file_path=self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            finding_type=self.finding_type,
        )
        if self.content_hash != expected:
            raise ValueError(
                f"ReviewFinding.content_hash={self.content_hash!r} does not match "
                f"compute_finding_content_hash(file_path={self.file_path!r}, "
                f"line_start={self.line_start}, line_end={self.line_end}, "
                f"finding_type={self.finding_type.value!r})={expected!r}. "
                f"Use audit.events.compute_finding_content_hash() to compute "
                f"the hash at construction; fixture code using "
                f"`compute_identity_hash({{...}})` is on the wrong recipe."
            )
        return self


__all__ = [
    "PublishDestination",
    "ReviewDimension",
    "ReviewFinding",
]
