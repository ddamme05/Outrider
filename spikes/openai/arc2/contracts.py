"""The Arc 2 generation + evaluation authorities.

Three strict, closed artifacts with a one-way transition between them:

    LockedProbeContract
        |  builds (offline, free)
        v
    DryRunManifest          <- content-addressed
        |  operator review
        v
    PaidProbeContract

Deliberately NOT one contract with an `unlocked` boolean and nullable `FROZEN_*`
constants. That earlier shape was circular: the single contract was declared to
own the per-row caps AND the source dry-run digest, while those values are
PRODUCED by the reviewed dry run it was supposed to precede. A boolean cannot
express a one-way transition between two different sets of known facts, and the
open-state shape is the same defect this arc has had to remove from the row
evidence, the manifest, and the classifier.

**Identity vs observation.** `EvaluationContract` owns identity only — what the
grader IS. It holds no counts: a count is a response-derived OBSERVATION, and an
artifact that asserts its own observations attests to itself. Observed counts are
derived by the verifier from hash-verified fixture bytes.

**No `reviewed=True` flag.** Human review is the trust anchor and cannot be
represented as a forgeable field. What IS mechanized is that
`PaidProbeContract.from_reviewed` copies immutable identity from its verified
parents rather than accepting it again, so a caller can supply only the reviewed
cap values.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spikes.openai.arc2.classifier import ROW_ORDER, ExpectedFinding, RowId

#: Alias so the verifier can name the expected-finding type without importing the
#: classifier directly for a single annotation.
ExpectedFindingLike = ExpectedFinding

__all__ = [
    "STRICT_PROBE_CONTRACT_VERSION",
    "ContractViolationError",
    "DryRunManifest",
    "EvaluationContract",
    "ExpectedFindingLike",
    "LockedProbeContract",
    "PaidProbeContract",
    "RowMeasurement",
    "digest_of",
]

#: Rotates on PROCEDURE or SCHEMA changes only — the row matrix, the predicates,
#: the contract shapes, the cap-derivation algorithm.
#:
#: It deliberately does NOT rotate for a cap VALUE change: exact caps are bound by
#: the `PaidProbeContract` digest, which already differs when they differ. Coupling
#: values to the procedure version would make every re-measurement look like a
#: procedure change and drain the version of meaning.
#:
#: v3 (2026-07-20): the whole owner model changed — three closed contracts, a
#: discriminated attempt union, and verifier-derived verdicts replacing
#: `ResourceContract` / `PAID_MODE_UNLOCKED` / stored-verdict authority.
#:
#: v4 (2026-07-20): the paid runner exists. That is a PROCEDURE change — there is
#: now a defined sequence of wire calls with a preflight, an abort rule, and a
#: shared terminality authority, where before there was a stub. Deliberately NOT
#: rotated for the cap VALUES chosen in the same change: values are bound by the
#: `PaidProbeContract` digest already, and rotating on them would make every
#: re-measurement look like a procedure change.
#:
#: v5 (2026-07-20): the defect scenario changed shape. The v4 capture graded
#: INCONCLUSIVE because the model bounded its finding to the whole `execute(...)`
#: statement (lines 3-5) while the registered identity required the concatenation
#: line exactly. The predicate was NOT loosened — the scenario was rewritten so the
#: vulnerable call occupies one line and "narrowest span containing the defect" has
#: a single reading. It moves the `acceptance_finding` request body only (the
#: clean row and the three elicitations are unchanged), but a scenario is part of
#: evaluation identity, so it rotates the
#: version and invalidates the v4 capture for grading; `capture_c6ac52a38ed27425`
#: is preserved unchanged as the v4 partial and is NOT retroactively regraded.
STRICT_PROBE_CONTRACT_VERSION: Final[ContractVersion] = "arc2-strict-schema:5"

#: The version is a LITERAL on every artifact, not a free string: an artifact
#: carrying an unrecognised version must fail to LOAD, not merely compare unequal
#: to whatever the reader expected.
ContractVersion = Literal["arc2-strict-schema:5"]

# Length is pinned alongside the pattern: pydantic-core's Rust regex has no
# `\A`/`\Z`, and `$` can admit a trailing newline, so the bound is what makes
# "exactly a sha256 hex digest" airtight.
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegInt = Annotated[int, Field(ge=0)]


class ContractViolationError(Exception):
    """A contract transition or artifact binding was invalid.

    Raised instead of proceeding, because every one of these means the artifact
    cannot support the conclusion it would otherwise be used to justify.
    """


def _canonical(payload: Any) -> str:
    """Sorted-key canonical form, for DIGESTING only.

    Deliberately different from the schema serializer: key order is generatively
    load-bearing for a schema sent to the model, but a contract is never generated
    from — only hashed and compared — so a stable sorted form is right here.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class _Strict(BaseModel):
    """Frozen, closed base. `extra="forbid"` so an unknown field is an error
    rather than silently-carried data nothing validates."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    @property
    def digest(self) -> str:
        return digest_of(self.model_dump(mode="json"))


class EvaluationContract(_Strict):
    """WHO grades, and against what — identity only, never observations.

    Everything here is frozen before the capture and cannot change as a result of
    it. Notably absent: any candidate/finding COUNT. Counts come from the response
    and are derived by the verifier; putting them here would let the artifact
    assert its own evidence.
    """

    #: sha256 of the scenario source bytes the parser will be given.
    scenario_source_digest: Sha256
    scenario_file_path: str
    scope_name: str
    scope_line_start: PositiveInt
    scope_line_end: PositiveInt
    #: The pre-registered defect identity.
    expected_finding_type: str
    expected_line_start: PositiveInt
    expected_line_end: PositiveInt
    #: HOW fired ids are derived (production builder + scope filter), plus the ids
    #: that derivation produced — identity of the derivation, not an observation
    #: of the response.
    fired_query_match_ids: tuple[str, ...]
    query_registry_digest: Sha256
    #: Grader versions.
    parser_version: str
    active_policy_version: str
    assessment_procedure_version: str
    #: Parser inputs that change admission.
    degraded_mode: bool
    pass_index: NonNegInt
    #: Narrowed to the parser's own Literal, not `str`. While the verifier merely
    #: RECORDED this field, `str` was harmless; now that it is threaded into
    #: `parse_analyze_response`, a free-form string would be a type error deferred
    #: to the paid run — the one place it must not surface.
    trace_candidate_form: Literal["module", "specifier"]

    @model_validator(mode="after")
    def _ranges_are_ordered(self) -> Self:
        if self.scope_line_end < self.scope_line_start:
            msg = "scope_line_end precedes scope_line_start"
            raise ValueError(msg)
        if self.expected_line_end < self.expected_line_start:
            msg = "expected_line_end precedes expected_line_start"
            raise ValueError(msg)
        return self


class RowMeasurement(_Strict):
    """One row's MEASURED request size and the exact bytes measured."""

    row_id: RowId
    prompt_bytes: PositiveInt
    estimated_tokens: NonNegInt
    request_body_digest: Sha256


class LockedProbeContract(_Strict):
    """Generation identity, frozen BEFORE any measurement exists.

    Holds no caps and no source-manifest digest, because at this point neither
    exists — that is the whole reason this is a separate type rather than a
    nullable field on one contract.
    """

    contract_version: ContractVersion
    model: str
    profile_contract_digest: Sha256
    strict_schema_digest: Sha256
    analyze_prompt_version: str
    evaluation_contract_digest: Sha256
    #: The frozen refusal elicitation texts. Their digests are part of identity,
    #: so a placeholder run and a real run are not interchangeable artifacts.
    refusal_prompt_digests: tuple[Sha256, ...]
    #: The request-EFFECTIVE completion limit. Positive and required: it must be
    #: the value actually sent, not a nullable constant recorded beside a
    #: different one.
    max_completion_tokens: PositiveInt
    row_order: tuple[RowId, ...]

    @model_validator(mode="after")
    def _matches_frozen_matrix(self) -> Self:
        if self.row_order != ROW_ORDER:
            msg = f"row_order {self.row_order!r} is not the frozen matrix {ROW_ORDER!r}"
            raise ValueError(msg)
        if len(self.refusal_prompt_digests) != 3:
            msg = "exactly three refusal elicitations are frozen"
            raise ValueError(msg)
        if len(set(self.refusal_prompt_digests)) != 3:
            msg = (
                "refusal elicitations are not distinct: identical prompts are ONE "
                "observation billed three times"
            )
            raise ValueError(msg)
        return self


class DryRunManifest(_Strict):
    """The reviewed, content-addressed measurement artifact.

    Binds the COMPLETE locked contract digest plus the ordered per-row
    measurements. It is the only sanctioned input to `PaidProbeContract`, and it
    carries no verdict: a run that made no network call cannot have decided
    anything.
    """

    contract_version: ContractVersion
    locked_contract_digest: Sha256
    measurements: tuple[RowMeasurement, ...]

    @model_validator(mode="after")
    def _covers_the_matrix_in_order(self) -> Self:
        rows = tuple(m.row_id for m in self.measurements)
        if rows != ROW_ORDER:
            msg = (
                f"measurements cover {rows!r} but the frozen matrix in execution "
                f"order is {ROW_ORDER!r}"
            )
            raise ValueError(msg)
        return self


class PaidProbeContract(_Strict):
    """Generation identity PLUS the reviewed caps.

    `from_reviewed` is the intended path and copies identity from the verified
    locked contract rather than accepting it again — but it is NOT a privileged
    constructor, because Python cannot make one. The guarantee rests on
    `_lineage_holds`, which re-derives the whole transition on every load, so a
    directly-constructed contract is rejected too.

    Currentness is a separate question this type cannot answer: the lineage here is
    INTERNAL, and a chain built entirely under a stale locked contract satisfies all
    of it. `verifier.verify_and_derive` reconstructs the generation plan and
    compares, which is where staleness is caught.
    """

    contract_version: ContractVersion
    model: str
    profile_contract_digest: Sha256
    strict_schema_digest: Sha256
    analyze_prompt_version: str
    evaluation_contract_digest: Sha256
    refusal_prompt_digests: tuple[Sha256, ...]
    max_completion_tokens: PositiveInt
    row_order: tuple[RowId, ...]
    #: The reviewed caps — the ONLY new data this transition accepts.
    per_row_prompt_byte_cap: dict[RowId, PositiveInt]
    #: The VERIFIED parents themselves, embedded rather than merely digested.
    #: Embedding them is what lets ordinary Pydantic loading re-check the
    #: relationship: with digests alone, a directly-constructed contract could
    #: name parents nothing ever compared it against.
    locked: LockedProbeContract
    dry_run: DryRunManifest
    #: The reviewed request identities, copied forward. Replay requires each
    #: attempt to answer the request its row was measured on, so a response to a
    #: different prompt cannot be graded under this contract.
    measurements: tuple[RowMeasurement, ...]

    @property
    def locked_contract_digest(self) -> str:
        return self.locked.digest

    @property
    def source_dry_run_digest(self) -> str:
        return self.dry_run.digest

    @model_validator(mode="after")
    def _lineage_holds(self) -> Self:
        """Re-derive the transition on EVERY load, not only in the classmethod.

        `from_reviewed` was the intended sole constructor, but `PaidProbeContract(...)`
        stayed publicly callable with arbitrary identity — so the guarantee held
        only for callers who chose to use it.
        """
        if self.dry_run.locked_contract_digest != self.locked.digest:
            msg = "embedded dry run was not produced under the embedded locked contract"
            raise ValueError(msg)
        for field in (
            "model",
            "profile_contract_digest",
            "strict_schema_digest",
            "analyze_prompt_version",
            "evaluation_contract_digest",
            "max_completion_tokens",
        ):
            if getattr(self, field) != getattr(self.locked, field):
                msg = (
                    f"{field} differs from the embedded locked contract: "
                    "identity is copied, not supplied"
                )
                raise ValueError(msg)
        if self.refusal_prompt_digests != self.locked.refusal_prompt_digests:
            msg = "refusal prompts differ from the embedded locked contract"
            raise ValueError(msg)
        if self.row_order != self.locked.row_order:
            msg = "row_order differs from the embedded locked contract"
            raise ValueError(msg)
        if self.measurements != self.dry_run.measurements:
            msg = "measurements differ from the reviewed dry run"
            raise ValueError(msg)
        measured = {m.row_id: m.prompt_bytes for m in self.measurements}
        if tuple(sorted(self.per_row_prompt_byte_cap)) != tuple(sorted(measured)):
            msg = "caps do not cover exactly the measured rows"
            raise ValueError(msg)
        for row, cap in self.per_row_prompt_byte_cap.items():
            if cap < measured[row]:
                msg = f"cap for {row.value!r} is below its reviewed measurement"
                raise ValueError(msg)
        return self

    @classmethod
    def from_reviewed(
        cls,
        *,
        locked: LockedProbeContract,
        dry: DryRunManifest,
        caps: dict[RowId, int],
    ) -> PaidProbeContract:
        """The ONLY constructor. Verifies the parents, then copies identity.

        `caps` is the sole caller-supplied input. Identity is copied from
        `locked`, never re-accepted, so a caller cannot pair reviewed caps with a
        different model, schema, or prompt set.
        """
        if dry.locked_contract_digest != locked.digest:
            msg = (
                f"dry-run manifest was produced under locked contract "
                f"{dry.locked_contract_digest!r}, not {locked.digest!r}: it does not "
                "authorize this contract"
            )
            raise ContractViolationError(msg)
        if dry.contract_version != locked.contract_version:
            msg = "dry-run manifest and locked contract disagree on procedure version"
            raise ContractViolationError(msg)

        measured = {m.row_id: m.prompt_bytes for m in dry.measurements}
        if tuple(sorted(caps)) != tuple(sorted(measured)):
            msg = (
                f"caps cover {sorted(caps)!r} but the reviewed measurements cover "
                f"{sorted(measured)!r}: every row needs exactly one cap"
            )
            raise ContractViolationError(msg)
        for row, cap in sorted(caps.items()):
            if cap < measured[row]:
                msg = (
                    f"cap for {row.value!r} is {cap} but the reviewed measurement is "
                    f"{measured[row]}: a cap below what was measured would refuse the "
                    "very request it was derived from"
                )
                raise ContractViolationError(msg)

        return cls(
            locked=locked,
            dry_run=dry,
            measurements=dry.measurements,
            contract_version=locked.contract_version,
            model=locked.model,
            profile_contract_digest=locked.profile_contract_digest,
            strict_schema_digest=locked.strict_schema_digest,
            analyze_prompt_version=locked.analyze_prompt_version,
            evaluation_contract_digest=locked.evaluation_contract_digest,
            refusal_prompt_digests=locked.refusal_prompt_digests,
            max_completion_tokens=locked.max_completion_tokens,
            row_order=locked.row_order,
            per_row_prompt_byte_cap=dict(caps),
        )
