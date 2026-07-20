"""The Arc 2 verdict classifier — pre-registered, total, and prose-blind.

Frozen before any spend so the outcome cannot be rationalized after the fact
(specs/2026-07-20-arc2-strict-schema-feasibility.md, "Frozen verdict rule").

Three properties this module is built to guarantee:

1. **Totality over BOTH arms.** Every transport outcome (any HTTP status, any
   timeout) and every capture outcome (any envelope shape, including malformed
   and ambiguous ones) reaches exactly one classification. There is no
   fall-through branch.
2. **Structure-only classification.** The verdict never reads vendor prose.
   `RawOpenAICaptureError` exposes only a validated status, a validated request
   id, and a bounded raw excerpt — an invalid-schema 400 and any other 400 are
   indistinguishable from those fields — so request rejection is decided
   POSITIONALLY (which row) and never by matching message substrings.
3. **Every fact is BOUND TO THE BYTES it came from.** A row is one immutable
   `EvaluatedRow`, not several independently-supplied booleans. An earlier
   design passed the envelope, the finding assessment, and the zero-findings
   flag as separate arguments, so a caller could pair a `{"findings":[]}` body
   with `returned_any_finding=True` — an impossible observation that still
   produced a positive verdict. Coherence is now enforced at construction, so
   that pairing raises instead of grading.

**`STOP_SHAPE` from a 400 is an observation, not a diagnosis.** It means "the
exact first-row strict request was rejected with HTTP 400" and nothing more. It
does NOT assert that the API rejected the schema, or that the design is
inexpressible in the strict subset — those readings would require interpreting
vendor prose. A narrower-encoding follow-up is a hypothesis for a later spec,
not a conclusion this classifier draws.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable  # noqa: TC003 - runtime annotation in evaluate_row
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "REFUSAL_ROWS",
    "ROW_ORDER",
    "SCHEMA_ADMISSION_ROW",
    "EvaluatedRow",
    "AdmittedFinding",
    "ExpectedFinding",
    "FindingRowAssessment",
    "ParserOutcome",
    "Observation",
    "ObservationKind",
    "RowId",
    "SessionVerdict",
    "classify_envelope",
    "validate_strict_payload",
    "classify_session",
    "classify_transport",
    "derive_assessment",
    "detect_wrong_tier_proof",
    "evaluate_row",
    "route_finding_assessment",
]


class Verdict(StrEnum):
    """The CLOSED terminal vocabulary. Exactly six; nothing else is a verdict.

    There is deliberately no bare `STOP` and no combined `INCONCLUSIVE / PARK` —
    an earlier spec draft used both, and the trace-candidate rule and the
    PARK-completeness rule drifted apart from the taxonomy they shared.
    """

    GO = "GO"
    STOP_SHAPE = "STOP-shape"
    STOP_AUTHENTICITY = "STOP-authenticity"
    STOP_FABRICATION = "STOP-fabrication"
    PARK = "PARK"
    INCONCLUSIVE = "INCONCLUSIVE"


__all__.append("Verdict")


class ObservationKind(StrEnum):
    """What a single row actually yielded. NON-terminal: these feed verdicts.

    Only `VALID_REFUSAL` satisfies the discriminator; only `VALID_NEGATIVE`
    counts toward `PARK`. `FAILED` is everything else and is never evidence of
    absence — the untrusted-arm rule the earlier refusal discovery had to learn.
    """

    VALID_REFUSAL = "valid_refusal"
    VALID_NEGATIVE = "valid_negative"
    FAILED = "failed"


class RowId(StrEnum):
    """The five rows of the bounded paid session, in execution order.

    `ACCEPTANCE_CLEAN` runs FIRST and is the schema-admission row — the only row
    whose 400 can produce `STOP_SHAPE`.
    """

    ACCEPTANCE_CLEAN = "acceptance_clean"
    ACCEPTANCE_FINDING = "acceptance_finding"
    REFUSAL_1 = "refusal_elicit_1"
    REFUSAL_2 = "refusal_elicit_2"
    REFUSAL_3 = "refusal_elicit_3"


#: Execution order is part of the contract: positional classification is only
#: sound if `ACCEPTANCE_CLEAN` is genuinely first.
ROW_ORDER: Final[tuple[RowId, ...]] = (
    RowId.ACCEPTANCE_CLEAN,
    RowId.ACCEPTANCE_FINDING,
    RowId.REFUSAL_1,
    RowId.REFUSAL_2,
    RowId.REFUSAL_3,
)

SCHEMA_ADMISSION_ROW: Final[RowId] = RowId.ACCEPTANCE_CLEAN

REFUSAL_ROWS: Final[tuple[RowId, ...]] = (RowId.REFUSAL_1, RowId.REFUSAL_2, RowId.REFUSAL_3)

#: Finish reasons that permit a body to be read as complete output. Per the
#: Structured Outputs guide's "Handling edge cases": `length` means the generation
#: ran out of context and the JSON "may be incomplete"; `content_filter` means
#: generation "was halted and may be partial". Either can still be schema-VALID by
#: luck, so validating the body is not sufficient — the finish reason must say the
#: bytes are the whole answer.
_COMPLETE_FINISH_REASON: Final[str] = "stop"


def validate_strict_payload(payload: object) -> None:
    """The CANONICAL strict-schema check. Module-owned, with no injection seam.

    Deliberately NOT a caller-supplied callable. An earlier design stored a
    `strict_validator` on the row, so `lambda _: None` turned off-schema JSON —
    including a content-channel refusal like `'"I cannot help"'` — into a valid
    negative observation, and three such rows manufactured `PARK`. That is the
    same hole the removed `UNCHECKED` sentinel opened, re-entered through a
    different door: the authority to decide "is this a conforming body" must not
    be a parameter.
    """
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    from spikes.openai.arc2.strict_schema import derive_strict_analyze_schema

    Draft202012Validator(derive_strict_analyze_schema()).validate(payload)


@dataclass(frozen=True, slots=True)
class Observation:
    """One row's classification plus the human-readable reason it got it.

    `detail` is for the operator reading the manifest; no branch reads it.
    """

    kind: ObservationKind
    detail: str


@dataclass(frozen=True, slots=True)
class FindingRowAssessment:
    """DERIVED assessment of the `acceptance_finding` row's body.

    Every flag is computed by `derive_assessment` from the raw body plus the REAL
    parser's reported facts. It is never hand-supplied on the gradeable path: an
    earlier design accepted an arbitrary callback's booleans, so a caller could
    pair `trace_candidates: []` with `nonempty_trace_candidates=True`, or a clean
    JUDGED body with `wrong_tier_proof=True`, and still drive a STOP verdict.

    `fabricated_proof_rejected_by_parser` is deliberately present and
    deliberately NOT a failure input: it is the expected negative control.
    """

    returned_any_finding: bool
    fabricated_proof_survived_parser: bool
    fabricated_proof_rejected_by_parser: bool
    nonempty_trace_candidates: bool


@dataclass(frozen=True, slots=True)
class AdmittedFinding:
    """One finding the parser ACTUALLY ADMITTED — identity plus its proof.

    Admission is the fact that matters. A schema-valid PROPOSAL is not a finding:
    the parser can reject it (`span_outside_scope_unit`, an off-enum
    `finding_type`, a registry miss) and the review then contains nothing. Reading
    the raw `findings` array instead let a body whose only proposal was rejected
    leave every STOP flag false and reach GO/PARK.
    """

    finding_type: str
    line_start: int
    line_end: int
    query_match_id: str | None
    trace_candidate_count: int


@dataclass(frozen=True, slots=True)
class ExpectedFinding:
    """The pre-registered identity of the defect the scenario plants.

    Frozen with the scenario, so "did the model find the thing we planted?" is a
    match against a declared target rather than "did anything come back?".
    """

    finding_type: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ParserOutcome:
    """The REAL parser's reported FACTS about one body — not a verdict.

    Deliberately narrow: the probe runs `parse_analyze_response` and reports what
    it observed; this module decides what that means. Keeping facts and judgment
    apart is what stops a caller from supplying the judgment directly.
    """

    #: Every proposal the parser ADMITTED, with identity and proof.
    admitted: tuple[AdmittedFinding, ...]
    #: `rejection_reason` strings for every proposal the parser REJECTED.
    rejection_reasons: tuple[str, ...]
    #: TOTAL trace candidates the parser RETAINED for this response — including
    #: those lifted from proposals it REJECTED. Production deliberately keeps
    #: valid candidates from rejected proposals (`analyze_parser.py`), so counting
    #: only the admitted finding's candidates let an admitted finding with `[]`
    #: plus a rejected candidate-bearing proposal slip past the spec's ABSOLUTE
    #: "any non-empty trace_candidates" rule.
    #:
    #: REQUIRED, with no default: a default of 0 let a caller silently omit the
    #: absolute fabrication fact, which is the same "authority one layer early"
    #: shape as every other defaulted conclusion this module has had to remove.
    retained_trace_candidate_count: int

    def __post_init__(self) -> None:
        if self.retained_trace_candidate_count < 0:
            msg = (
                f"retained_trace_candidate_count={self.retained_trace_candidate_count} "
                "is negative: a count of retained candidates cannot be below zero"
            )
            raise ValueError(msg)


def _matches_proof_subschema(value: object, subschema: dict[str, Any]) -> bool:
    """Does `value` satisfy this proof property's JSON subschema?

    Type-exact, deliberately NOT truthiness. An earlier version used
    emptiness tests and disagreed with the schema in BOTH directions:
    OBSERVED `query_match_id=""` is schema-VALID (the branch is a bare
    `{"type": "string"}`) yet was flagged, while INFERRED/JUDGED
    `query_match_id=""` VIOLATES their null-only branch yet was read as
    "absent" and missed. A wrong-tier detector that disagrees with the schema
    it is checking against is worse than none.
    """
    # The SUBSCHEMA is the authority — not a re-implementation of it. An earlier
    # version read `type` and `minItems` from the mapping but hardcoded
    # `re.search(r".", ...)` for the item pattern. That happened to match today's
    # `.+`, so the claim "the detector cannot drift from the schema" was false:
    # changing `pattern` in `TIER_PROOF_SHAPES` would have moved strict validation
    # and left this check behind, silently.
    from jsonschema import Draft202012Validator

    return bool(Draft202012Validator(subschema).is_valid(value))


def detect_wrong_tier_proof(payload: object) -> bool:
    """True iff any finding's declared tier disagrees with its proof fields.

    DERIVED from `TIER_PROOF_SHAPES` — the same mapping the strict schema is
    built from — so the semantic check and the wire encoding cannot drift apart.

    Computed from the PARSED JSON, deliberately independent of strict-schema
    validation. That independence is the whole point: the strict schema is
    supposed to make wrong-tier proof unrepresentable, so if the API honours it
    a wrong-tier body cannot arrive — and a wrong-tier body that DOES arrive is
    evidence the schema was not enforced, which is exactly the `STOP-shape`
    finding this arc exists to catch. Checking it only after validation passed
    would make that branch unreachable, and a verdict nothing can reach is not
    a gate.
    """
    from spikes.openai.arc2.strict_schema import TIER_PROOF_SHAPES

    by_tier = {tier.value: shapes for tier, shapes in TIER_PROOF_SHAPES.items()}

    if not isinstance(payload, dict):
        return False
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return False
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        tier = finding.get("evidence_tier")
        if not isinstance(tier, str) or tier not in by_tier:
            # An off-enum tier is itself a tier/proof failure: the branch enums
            # pin exactly three values, so nothing else should be representable.
            return True
        for prop, subschema in by_tier[tier].items():
            if not _matches_proof_subschema(finding.get(prop), subschema):
                return True
    return False


def derive_assessment(
    *,
    body: str,
    parser: ParserOutcome,
    fired_query_match_ids: frozenset[str],
    expected: ExpectedFinding,
) -> FindingRowAssessment:
    """Compute EVERY verdict-driving flag from the parser's ADMITTED findings.

    The project-owned assessor. Nothing is supplied as a conclusion:
    `fired_query_match_ids` is the set of query ids that actually matched for the
    file, and `expected` is the pre-registered identity of the planted defect.

    **Admission, not proposal.** `returned_any_finding` means "the expected
    finding was ADMITTED by the real parser at the expected span", not "the model
    emitted something". A schema-valid proposal the parser rejected (a span
    outside the packed scope, an off-enum type, a registry miss) leaves the
    populated-branch question unanswered, so it reads INCONCLUSIVE — never a
    silent pass. Trace-candidate and authenticity facts are likewise read off the
    ADMITTED finding, so a rejected proposal's contents cannot drive a verdict.
    """
    payload = json.loads(body)
    raw_findings = payload.get("findings", []) if isinstance(payload, dict) else []

    matching = [
        f
        for f in parser.admitted
        if f.finding_type == expected.finding_type
        and f.line_start == expected.line_start
        and f.line_end == expected.line_end
    ]
    claimed_ids = {
        f.get("query_match_id")
        for f in raw_findings
        if isinstance(f, dict) and isinstance(f.get("query_match_id"), str)
    }
    fabricated_claimed = {cid for cid in claimed_ids if cid not in fired_query_match_ids}
    # Parser-WIDE, not span-scoped: the spec says ANY fabricated proof surviving
    # into an ADMITTED finding is an authenticity escape. Restricting this to
    # findings matching the expected span meant an unrelated admitted finding
    # carrying an invented id passed unnoticed.
    admitted_fabricated = {
        f.query_match_id
        for f in parser.admitted
        if f.query_match_id is not None and f.query_match_id not in fired_query_match_ids
    }

    return FindingRowAssessment(
        returned_any_finding=bool(matching),
        # An id the file's queries never produced, on a finding the parser ADMITTED.
        fabricated_proof_survived_parser=bool(admitted_fabricated),
        # Claimed but caught — the expected negative control.
        fabricated_proof_rejected_by_parser=bool(fabricated_claimed)
        and "query_match_id_not_in_registry" in parser.rejection_reasons,
        # ANY retained candidate counts — the rule is absolute. Candidates the
        # parser lifted from a REJECTED proposal are still candidates the response
        # induced, and production retains them.
        nonempty_trace_candidates=(
            any(f.trace_candidate_count > 0 for f in matching)
            or parser.retained_trace_candidate_count > 0
        ),
    )


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")


def _is_sha256_hex(value: str | None) -> bool:
    return value is not None and _SHA256_HEX.match(value) is not None


def classify_envelope(
    *,
    content: str | None,
    refusal: str | None,
    finish_reason: str | None,
) -> Observation:
    """Classify one COMPLETED response envelope. Total over every content/refusal
    combination, including the ambiguous ones.

    `RawCapture` declares `content: str | None` and `refusal: str | None`
    INDEPENDENTLY, so both-null and both-populated envelopes are representable on
    the wire. An earlier spec draft counted "a completed response with no refusal"
    toward `PARK`, which would have let a malformed or ambiguous envelope become
    evidence of absence.

    Validation uses the module-owned `validate_strict_payload`; there is no
    validator parameter, because the authority to decide "is this a conforming
    body" must not be injectable.

    `finish_reason` gates COMPLETENESS on the CONTENT arm only: a truncated
    (`length`) or filtered (`content_filter`) generation can still be accidentally
    schema-valid, and the guide directs callers to treat both as incomplete.
    It deliberately does NOT gate the refusal arm — production treats a non-empty
    `message.refusal` as a refusal regardless of finish reason.
    """
    content_present = content is not None
    refusal_present = not _is_blank(refusal)

    # THE API-OWNED REFUSAL CHANNEL TAKES PRECEDENCE — over content and over
    # finish_reason — because that is the recorded PRODUCTION contract, not a
    # reading of ambiguous docs. `OpenAICompatibleProvider` (FUP-203, adjudicated
    # by the openai-native-host spec) normalizes "a non-empty structured
    # `message.refusal` field (which OpenAI can send with `finish_reason="stop"`)"
    # to a refusal, whatever else the envelope carries. The Structured Outputs
    # guide agrees the `.content` field may carry the model's explanation, and
    # `tests/unit/test_raw_openai_capture.py`'s wire fixture models BOTH populated.
    #
    # An earlier rule required `content is None` and scored both-populated as an
    # ambiguous FAILED. That would have read a GENUINE refusal as a failed
    # observation — turning the one paid session INCONCLUSIVE on the very question
    # it exists to answer.
    #
    # This does NOT re-open the content-channel-refusal hole: that case is
    # `refusal is None` with refusal-shaped PROSE in content, which still fails
    # strict validation below. The distinction is which channel spoke.
    if refusal_present:
        return Observation(
            ObservationKind.VALID_REFUSAL,
            "non-empty API-owned refusal"
            + (" (content also present — permitted)" if content_present else ""),
        )

    # Whitespace-only refusal with null content lands here, and correctly fails.
    if content is None:
        return Observation(
            ObservationKind.FAILED,
            "empty envelope: neither content nor a non-blank refusal was returned",
        )
    if refusal is not None:
        # Non-null but blank refusal alongside content: not a refusal, and not a
        # clean negative either — the channel was touched but says nothing.
        return Observation(
            ObservationKind.FAILED,
            "ambiguous envelope: blank-but-present refusal alongside content",
        )
    # `content` is narrowed to `str` by the check above — deliberately NOT an
    # `assert`, which `python -O` strips, leaving a TypeError from json.loads to
    # escape as an unhandled exception instead of a FAILED observation.
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, ValueError) as exc:
        return Observation(ObservationKind.FAILED, f"content is not valid JSON: {exc}")
    if finish_reason != _COMPLETE_FINISH_REASON:
        return Observation(
            ObservationKind.FAILED,
            f"content returned with finish_reason={finish_reason!r}: the generation "
            "did not complete, so the body may be truncated or partial regardless of "
            "whether it happens to validate",
        )
    try:
        validate_strict_payload(payload)
    except Exception as exc:  # noqa: BLE001 - any validator failure is a failed observation
        return Observation(
            ObservationKind.FAILED,
            f"content did not validate against the strict schema: {type(exc).__name__}",
        )
    return Observation(ObservationKind.VALID_NEGATIVE, "schema-valid content, no refusal")


@dataclass(frozen=True, slots=True)
class EvaluatedRow:
    """ONE captured row: RAW EVIDENCE only. Every conclusion is a derived property.

    The constructor accepts what the wire actually produced — the response bytes,
    the raw `message.refusal` string, the transport status, and the parser's
    reported facts. It accepts NO conclusion: there is no `observation=`,
    `terminal_verdict=`, `wrong_tier_proof=`, or `assessment=` to set.

    `observation` was the last conclusion still injectable, and it was the most
    dangerous one: `Observation(VALID_REFUSAL, "refused")` on a row storing no
    refusal text at all satisfied the discriminator, and `classify_session` would
    return `GO` — a demonstrated refusal channel conjured from nothing. A refusal
    is now a refusal because the wire carried refusal text.

    `parser` carries FACTS (which findings were admitted, which reasons rejected),
    not judgment. Hand-built facts are UNTRUSTED: constructing an `EvaluatedRow`
    with a `ParserOutcome` you wrote yourself proves nothing, and re-hashing a
    fixture would not help — hashing shows the stored bytes are the stored bytes
    and never re-derives anything from them.

    What makes persisted evidence gradeable is reconstruction:
    `verifier.verify_and_derive` runs the REAL parser over the hash-verified
    content under the reconstructed plan, so the facts a verdict rests on are
    produced during replay rather than read from the artifact. (This docstring
    previously said replay did not do that yet. It does.)

    **Response and transport evidence are mutually exclusive.** A row is either a
    completed response (bytes and/or refusal, no status) or a transport failure
    (status, neither). Allowing both let a row claim a schema-valid response AND
    an HTTP 400, then derive `STOP-shape` from the status.
    """

    row: RowId
    #: The exact response content bytes, or None when no body was returned.
    content: str | None = None
    #: The RAW `message.refusal` string as captured.
    refusal: str | None = None
    #: The RAW `choices[0].finish_reason` as captured. Evidence, not a conclusion:
    #: `length`/`content_filter` mean the body may be truncated or partial even
    #: when it happens to validate.
    finish_reason: str | None = None
    #: True iff this row failed in TRANSPORT (no response was returned at all).
    #: An explicit flag because `transport_status=None` is AMBIGUOUS on its own:
    #: it means "timeout / no status" for a transport failure and "not a transport
    #: failure" for a completed response, so the two were indistinguishable.
    transport_failed: bool = False
    #: HTTP status of the transport failure; None for a timeout or a completed row.
    transport_status: int | None = None
    #: What the REAL parser reported about `content`. Facts, never a verdict.
    parser: ParserOutcome | None = None
    #: The scenario's pre-registered target, needed to decide "was it found?".
    expected_finding: ExpectedFinding | None = None
    #: Query ids that actually fired for the file — makes "fabricated" decidable.
    fired_query_match_ids: frozenset[str] = frozenset()

    @property
    def observation(self) -> Observation:
        """DERIVED from the raw channels, never supplied."""
        if self.transport_failed:
            return Observation(
                ObservationKind.FAILED,
                "HTTP 400 on the exact first-row strict request"
                if self.row == SCHEMA_ADMISSION_ROW and self.transport_status == 400
                else f"transport failure (status={self.transport_status}) — "
                "not evidence of anything",
            )
        return classify_envelope(
            content=self.content,
            refusal=self.refusal,
            finish_reason=self.finish_reason,
        )

    @property
    def body_digest(self) -> str | None:
        """sha256 of the captured bytes — computed, never supplied."""
        if self.content is None:
            return None
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def finding_count(self) -> int | None:
        """PROPOSALS in the body. None when there is no schema-valid body.

        Deliberately distinct from "findings": a proposal is not a finding until
        the parser admits it. See `assessment.returned_any_finding`.
        """
        if self.content is None or self.observation.kind is not ObservationKind.VALID_NEGATIVE:
            return None
        try:
            payload = json.loads(self.content)
        except (json.JSONDecodeError, ValueError):
            return None
        findings = payload.get("findings") if isinstance(payload, dict) else None
        return len(findings) if isinstance(findings, list) else None

    @property
    def wrong_tier_proof(self) -> bool:
        """Derived from the captured bytes on the CONTENT arm ONLY.

        A row whose envelope is a VALID REFUSAL is excluded: the API-owned refusal
        channel outranks content, so a refusal whose explanatory `content` happens
        to contain wrong-tier-looking JSON is a refusal, not a schema failure.
        Without this exclusion the session-level wrong-tier scan (which predates
        the refusal-precedence rule and ran over every row) turned a demonstrated
        refusal into `STOP-shape` — inverting the arc's own central finding.

        Deliberately independent of strict validation: a wrong-tier body FAILS
        validation, so deriving this only from validated bodies would make the
        `STOP-shape` branch unreachable — and a verdict nothing can reach is not
        a gate.
        """
        if self.content is None or self.observation.kind is ObservationKind.VALID_REFUSAL:
            return False
        try:
            return detect_wrong_tier_proof(json.loads(self.content))
        except (json.JSONDecodeError, ValueError):
            return False

    @property
    def terminal_verdict(self) -> Verdict | None:
        """`STOP-shape` iff THIS row is the schema-admission row and its transport
        status is exactly 400. Positional and derived."""
        if (
            self.transport_failed
            and self.row == SCHEMA_ADMISSION_ROW
            and self.transport_status == 400
        ):
            return Verdict.STOP_SHAPE
        return None

    @property
    def assessment(self) -> FindingRowAssessment | None:
        """The semantic assessment, DERIVED from the body and the parser's facts."""
        if (
            self.row != RowId.ACCEPTANCE_FINDING
            or self.observation.kind is not ObservationKind.VALID_NEGATIVE
            or self.content is None
            or self.parser is None
            or self.expected_finding is None
        ):
            return None
        return derive_assessment(
            body=self.content,
            parser=self.parser,
            fired_query_match_ids=self.fired_query_match_ids,
            expected=self.expected_finding,
        )

    def __post_init__(self) -> None:
        if self.transport_status is not None and not self.transport_failed:
            msg = (
                f"row {self.row.value!r} carries transport_status="
                f"{self.transport_status!r} without transport_failed: a status only "
                "exists for a row that actually failed in transport"
            )
            raise ValueError(msg)
        if self.transport_failed and (self.content is not None or self.refusal is not None):
            msg = (
                f"row {self.row.value!r} carries BOTH a transport status and response "
                "evidence: a row is either a completed response or a transport "
                "failure, never both"
            )
            raise ValueError(msg)
        if self.parser is not None and self.row != RowId.ACCEPTANCE_FINDING:
            msg = (
                f"parser facts supplied for row {self.row.value!r}: only "
                f"{RowId.ACCEPTANCE_FINDING.value} is assessed"
            )
            raise ValueError(msg)


def evaluate_row(
    *,
    row: RowId,
    content: str | None,
    refusal: str | None,
    finish_reason: str | None,
    run_parser: Callable[[str], ParserOutcome] | None = None,
    fired_query_match_ids: frozenset[str] = frozenset(),
    expected_finding: ExpectedFinding | None = None,
) -> EvaluatedRow:
    """Build an `EvaluatedRow` from ONE raw response.

    Attaches the EVIDENCE — raw content, raw refusal, the validator, and (for the
    finding row) the parser's reported facts. Every conclusion is derived by the
    row itself, so this function cannot smuggle one in either.
    """
    row = RowId(row)
    is_finding_row = row == RowId.ACCEPTANCE_FINDING
    parser: ParserOutcome | None = None
    if is_finding_row and content is not None and run_parser is not None:
        probe_row = EvaluatedRow(
            row=row, content=content, refusal=refusal, finish_reason=finish_reason
        )
        if probe_row.observation.kind is ObservationKind.VALID_NEGATIVE:
            parser = run_parser(content)
    return EvaluatedRow(
        row=row,
        content=content,
        refusal=refusal,
        finish_reason=finish_reason,
        parser=parser,
        expected_finding=expected_finding if is_finding_row else None,
        fired_query_match_ids=fired_query_match_ids,
    )


def classify_transport(*, row: RowId, status: int | None) -> EvaluatedRow:
    """Classify a TRANSPORT failure. Total over every status, by construction.

    TOTAL RULE: every non-2xx outcome other than an exact HTTP 400 on the
    schema-admission row is a failed observation. That covers 401/403/429/5xx and
    timeouts (`status is None`) AND the residual statuses (404/408/409/422/...);
    there is no unspecified branch to fall through.
    """
    # `RowId` is a `StrEnum`, so a plain string key compares and hashes equal to
    # its member everywhere EXCEPT under `is`. Coerce, don't identity-check.
    return EvaluatedRow(row=RowId(row), transport_failed=True, transport_status=status)


def route_finding_assessment(assessment: FindingRowAssessment) -> Verdict | None:
    """Route the `acceptance_finding` row. Returns None when the row is clean.

    **Global safety outranks recall.** An earlier order returned `INCONCLUSIVE`
    for a recall miss FIRST, which buried two safety facts behind it: a response
    whose expected finding was rejected but which still induced a retained trace
    candidate scored `INCONCLUSIVE` instead of `STOP-fabrication`. "We did not
    learn whether it finds the defect" is a weaker statement than "it fabricated
    content", and the weaker one must not mask the stronger.

    `wrong_tier_proof` is NOT routed here — it is a row-level fact checked
    session-wide before this point, because a wrong-tier body fails strict
    validation and would never reach an assessment.
    """
    # Authenticity first: fabricated proof that SURVIVED into an admitted finding.
    if assessment.fabricated_proof_survived_parser:
        return Verdict.STOP_AUTHENTICITY
    # Then fabrication pressure: ANY retained candidate, admitted or not.
    if assessment.nonempty_trace_candidates:
        return Verdict.STOP_FABRICATION
    # Only then recall. Recall is an explicit non-goal, so a missed finding leaves
    # the populated-branch question unanswered — it is not a schema failure.
    if not assessment.returned_any_finding:
        return Verdict.INCONCLUSIVE
    return None


@dataclass(frozen=True, slots=True)
class SessionVerdict:
    """The session's terminal verdict plus why, and the discriminator state."""

    verdict: Verdict
    reason: str
    discriminator_met: bool
    observations: tuple[tuple[RowId, Observation], ...]


def classify_session(*, rows: dict[RowId, EvaluatedRow]) -> SessionVerdict:
    """Compose evaluated rows into ONE terminal verdict from the closed vocabulary.

    Consumes ONLY `EvaluatedRow` objects — there are no side-channel booleans, so
    every fact the verdict rests on is bound to the bytes it came from.

    `rows` may be partial: the session aborts on the first transport error, so
    later rows are simply absent. Absence is "not observed", never a negative
    observation.

    Mapping keys are checked against each row's own `row` field first: without
    that, a refusal row filed under `ACCEPTANCE_CLEAN` would be read as the
    schema-admission observation, and every positional guarantee below rests on
    the key meaning what it says.
    """
    for key, evaluated in rows.items():
        if RowId(key) != evaluated.row:
            msg = (
                f"row mapping is inconsistent: key {RowId(key).value!r} holds an "
                f"EvaluatedRow for {evaluated.row.value!r}. Positional classification "
                "requires the key and the row to agree."
            )
            raise ValueError(msg)

    observations = tuple((row, rows[row].observation) for row in ROW_ORDER if row in rows)

    # The discriminator SHORT-CIRCUITS and cannot be retracted: once any refusal
    # row yields a VALID refusal observation, the condition is met, and a later
    # row erroring does not un-demonstrate an observation already made. The
    # complete predicate rejects BLANK refusals; a non-blank API-owned refusal
    # satisfies the discriminator whether or not content is also present.
    discriminator_met = any(
        row in rows and rows[row].observation.kind is ObservationKind.VALID_REFUSAL
        for row in REFUSAL_ROWS
    )

    def _verdict(
        v: Verdict, reason: str, *, discriminator: bool = discriminator_met
    ) -> SessionVerdict:
        return SessionVerdict(
            verdict=v,
            reason=reason,
            discriminator_met=discriminator,
            observations=observations,
        )

    # A terminal transport verdict (400 on the schema-admission row) outranks
    # everything: nothing downstream was observed under an accepted request.
    for row in ROW_ORDER:
        outcome = rows.get(row)
        if outcome is not None and outcome.terminal_verdict is not None:
            # `discriminator_met` is forced False here. "The strict request was
            # rejected" and "the refusal channel is demonstrated" cannot both be
            # true of one run, and a manifest reader would take a True as a
            # positive result about the refusal channel.
            return _verdict(
                outcome.terminal_verdict,
                f"{row.value}: {outcome.observation.detail}",
                discriminator=False,
            )

    # Wrong-tier proof anywhere falsifies the arc's central claim — that the strict
    # schema makes wrong-tier proof UNREPRESENTABLE at the wire. Checked before the
    # envelope gates precisely because such a body fails strict validation: gating
    # it behind "did this row yield a valid negative" would make the branch dead.
    for row in ROW_ORDER:
        wrong_tier_row = rows.get(row)
        if wrong_tier_row is not None and wrong_tier_row.wrong_tier_proof:
            return _verdict(
                Verdict.STOP_SHAPE,
                f"{row.value}: response carried proof fields that contradict its "
                "declared evidence_tier — the strict schema did not make wrong-tier "
                "proof unrepresentable",
            )

    clean = rows.get(SCHEMA_ADMISSION_ROW)
    if clean is None or clean.observation.kind is not ObservationKind.VALID_NEGATIVE:
        detail = "not run" if clean is None else clean.observation.detail
        return _verdict(
            Verdict.INCONCLUSIVE,
            f"schema-admission row did not yield a valid negative observation ({detail})",
        )
    # DERIVED from the body, not supplied: the clean row must carry exactly zero
    # findings, and `finding_count` came from the same bytes the envelope did.
    if clean.finding_count != 0:
        return _verdict(
            Verdict.INCONCLUSIVE,
            f"schema-admission row carried {clean.finding_count} finding(s), expected exactly 0",
        )

    # The finding row is gated on its own OUTCOME, the same way the clean row is.
    finding = rows.get(RowId.ACCEPTANCE_FINDING)
    if finding is None or finding.observation.kind is not ObservationKind.VALID_NEGATIVE:
        detail = "not run" if finding is None else finding.observation.detail
        return _verdict(
            Verdict.INCONCLUSIVE,
            f"acceptance_finding did not yield an assessable body ({detail})",
        )
    if finding.assessment is None:
        return _verdict(
            Verdict.INCONCLUSIVE,
            "acceptance_finding returned a body but produced no parser assessment",
        )
    routed = route_finding_assessment(finding.assessment)
    if routed is Verdict.INCONCLUSIVE:
        return _verdict(
            Verdict.INCONCLUSIVE,
            "acceptance_finding returned no finding — populated-branch question unanswered",
        )
    if routed is not None:
        return _verdict(routed, f"acceptance_finding routed to {routed.value}")

    if discriminator_met:
        return _verdict(Verdict.GO, "strict request accepted; refusal channel demonstrated")

    # PARK requires a COMPLETE negative set: all three refusal rows completed as
    # VALID negative observations. Two valid negatives plus one error (or one
    # malformed envelope) is not evidence of absence.
    complete_negative_set = all(
        row in rows and rows[row].observation.kind is ObservationKind.VALID_NEGATIVE
        for row in REFUSAL_ROWS
    )
    if complete_negative_set:
        return _verdict(
            Verdict.PARK,
            # Deliberately NOT "none refused": what was observed is that no row used
            # the API refusal CHANNEL. A model that declined inside the content
            # channel would fail schema validation and never reach here, but the
            # wording must not overclaim past what the taxonomy establishes.
            "schema works; all three refusal rows completed as valid negatives — "
            "no row used the API refusal channel",
        )
    return _verdict(
        Verdict.INCONCLUSIVE,
        "incomplete refusal evidence — fewer than three valid negative observations, "
        "and no row used the API refusal channel",
    )
