"""Compare externally adjudicated reviews on one frozen, paired sample.

This module does not judge claims or infer truth from pipeline admission. Callers
must supply every emitted claim, including rejected proposals, and freeze factual
judgments separately from location/severity assessments. Stable mechanism IDs
match the same real defect across differently worded findings. A claim's surfaced
flag records delivery to the declared review surface, such as the draft/HITL
result. Posting to GitHub is not required. Rejected proposals remain in the raw
inventory but cannot supply a useful delivered discovery or a product false alarm.

The settings digest identifies fixed transport, scoring and workload settings,
including declared intervention exclusions. Deliberately varied prompts, context
or schemas need not have equal digests themselves. Actual visibility remains
explicit and never excuses losing a baseline discovery.

The command-line entry point reads a complete JSON export of reviewed arms.
It validates the export, not the source captures or the factual judgments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal


class ComparisonInputError(ValueError):
    """The supplied arms do not describe a comparable, paired experiment."""


@dataclass(frozen=True)
class MeasurementIdentity:
    dataset_sha256: str
    model: str  # Host-qualified, for example "anthropic:claude-sonnet-5-5".
    measurement_settings_sha256: str


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    known_present_mechanisms: frozenset[str]
    fixed_mechanisms: frozenset[str]
    # A fixed label concerns only these named mechanisms, never the entire PR.


@dataclass(frozen=True)
class ClaimJudgment:
    claim_id: str
    verdict: Literal["valid", "incorrect", "unresolved"] | None
    mechanism_id: str | None = None
    surfaced: bool | None = None
    location_assessment: str | None = None
    severity_assessment: str | None = None
    # None means missing adjudication; conditional claims remain unresolved.


@dataclass(frozen=True)
class TrialOutcome:
    case_id: str
    repetition: int
    claims: tuple[ClaimJudgment, ...]
    expected_claim_ids: frozenset[str]
    completed: bool
    claims_complete: bool
    visibility_context: str | None
    # Freeze the raw/rejected/admitted ID inventory before assigning judgments.
    # claims_complete also rules out unreadable or otherwise unenumerated claims.
    # None visibility means actual served context has not yet been inspected.


@dataclass(frozen=True)
class AdjudicatedArm:
    arm_id: str
    identity: MeasurementIdentity
    cases: tuple[CaseSpec, ...]
    repetitions: int
    outcomes: tuple[TrialOutcome, ...]


@dataclass(frozen=True)
class ArmMetrics:
    valid_claims: int
    incorrect_claims: int
    unresolved_claims: int
    missing_judgments: int
    unknown_surface_claims: int
    surfaced_valid_claims: int
    surfaced_incorrect_claims: int
    raw_known_mechanisms_found: int
    known_mechanisms_found: int
    known_mechanism_opportunities: int
    precision: float | None
    # Known discoveries and precision concern the declared delivered surface;
    # raw discoveries remain separate. Silence has undefined precision.


@dataclass(frozen=True)
class FloorComparison:
    status: Literal["regression", "no_observed_regression", "inconclusive"]
    baseline: AdjudicatedArm
    candidate: AdjudicatedArm
    baseline_metrics: ArmMetrics
    candidate_metrics: ArmMetrics
    regressions: tuple[str, ...]
    uncertainties: tuple[str, ...]
    limitation: str = (
        "This paired sample cannot establish broader noninferiority, prove that "
        "the workflow does not hinder the model, or authorize model promotion."
    )


def _validate_arm(arm: AdjudicatedArm) -> dict[tuple[str, int], TrialOutcome]:
    """Require a complete trial roster; incomplete work stays inside its row."""
    if not arm.arm_id or not arm.identity.model:
        raise ComparisonInputError("arm and host-qualified model identities are required")
    for digest in (
        arm.identity.dataset_sha256,
        arm.identity.measurement_settings_sha256,
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ComparisonInputError("dataset and measurement settings require SHA-256 digests")
    if type(arm.repetitions) is not int or arm.repetitions < 1 or not arm.cases:
        raise ComparisonInputError("declare at least one case and repetition")
    cases = {case.case_id: case for case in arm.cases}
    if len(cases) != len(arm.cases) or any(not case_id for case_id in cases):
        raise ComparisonInputError("case identities must be nonempty and unique")
    for case in arm.cases:
        if case.known_present_mechanisms & case.fixed_mechanisms:
            raise ComparisonInputError("a named mechanism cannot be both present and fixed")
        if any(
            not mechanism for mechanism in case.known_present_mechanisms | case.fixed_mechanisms
        ):
            raise ComparisonInputError("mechanism identities must be nonempty")
    expected = {
        (case.case_id, repetition) for case in arm.cases for repetition in range(arm.repetitions)
    }
    rows: dict[tuple[str, int], TrialOutcome] = {}
    for outcome in arm.outcomes:
        key = (outcome.case_id, outcome.repetition)
        if type(outcome.repetition) is not int or key not in expected or key in rows:
            raise ComparisonInputError("unexpected or duplicate case/repetition outcome")
        if type(outcome.completed) is not bool or type(outcome.claims_complete) is not bool:
            raise ComparisonInputError("completion and claim coverage require explicit booleans")
        if outcome.visibility_context is not None and not outcome.visibility_context.strip():
            raise ComparisonInputError("visibility context must be descriptive or explicitly None")
        claim_ids: set[str] = set()
        for claim in outcome.claims:
            if not claim.claim_id or claim.claim_id in claim_ids:
                raise ComparisonInputError("claim identities must be nonempty and unique per trial")
            claim_ids.add(claim.claim_id)
            if claim.surfaced is not None and type(claim.surfaced) is not bool:
                raise ComparisonInputError("claim surface must be an explicit boolean or None")
            if claim.verdict not in {"valid", "incorrect", "unresolved", None}:
                raise ComparisonInputError(
                    "unknown factual verdict; conditional claims are unresolved"
                )
            if claim.verdict == "valid" and not claim.mechanism_id:
                raise ComparisonInputError(
                    "valid claims require a stable defect mechanism identity"
                )
            if (
                claim.verdict == "valid"
                and claim.mechanism_id in cases[outcome.case_id].fixed_mechanisms
            ):
                raise ComparisonInputError(
                    "valid claim contradicts the named fixed-mechanism label"
                )
        if claim_ids != outcome.expected_claim_ids:
            raise ComparisonInputError("judgments must cover the frozen claim ID inventory exactly")
        rows[key] = outcome
    if rows.keys() != expected:
        raise ComparisonInputError(
            "missing case/repetition coverage; retain explicit incomplete rows"
        )
    return rows


def _surfaced_discoveries(outcome: TrialOutcome) -> set[str]:
    return {
        claim.mechanism_id
        for claim in outcome.claims
        if claim.verdict == "valid" and claim.surfaced is True and claim.mechanism_id is not None
    }


def _metrics(arm: AdjudicatedArm) -> ArmMetrics:
    claims = [claim for outcome in arm.outcomes for claim in outcome.claims]
    valid = sum(claim.verdict == "valid" for claim in claims)
    incorrect = sum(claim.verdict == "incorrect" for claim in claims)
    unresolved = sum(claim.verdict == "unresolved" for claim in claims)
    missing = sum(claim.verdict is None for claim in claims)
    unknown_surface = sum(claim.surfaced is None for claim in claims)
    surfaced = [claim for claim in claims if claim.surfaced is True]
    surfaced_valid = sum(claim.verdict == "valid" for claim in surfaced)
    surfaced_incorrect = sum(claim.verdict == "incorrect" for claim in surfaced)
    complete = all(outcome.completed and outcome.claims_complete for outcome in arm.outcomes)
    cases = {case.case_id: case for case in arm.cases}
    known_found = 0
    raw_known_found = 0
    for outcome in arm.outcomes:
        found = {claim.mechanism_id for claim in outcome.claims if claim.verdict == "valid"}
        known = cases[outcome.case_id].known_present_mechanisms
        raw_known_found += len(found & known)
        known_found += len(_surfaced_discoveries(outcome) & known)
    precision = (
        surfaced_valid / len(surfaced)
        if surfaced and complete and not unresolved and not missing and not unknown_surface
        else None
    )
    return ArmMetrics(
        valid_claims=valid,
        incorrect_claims=incorrect,
        unresolved_claims=unresolved,
        missing_judgments=missing,
        unknown_surface_claims=unknown_surface,
        surfaced_valid_claims=surfaced_valid,
        surfaced_incorrect_claims=surfaced_incorrect,
        raw_known_mechanisms_found=raw_known_found,
        known_mechanisms_found=known_found,
        known_mechanism_opportunities=(
            sum(len(case.known_present_mechanisms) for case in arm.cases) * arm.repetitions
        ),
        precision=precision,
    )


def compare_review_floor(baseline: AdjudicatedArm, candidate: AdjudicatedArm) -> FloorComparison:
    """Preserve baseline discoveries and block an increase in incorrect warnings.

    Each case/repetition must retain every valid surfaced baseline mechanism.
    Extra bugs elsewhere cannot compensate for a loss. Surfaced incorrect warnings
    are also paired, so aggregate improvements cannot hide a newly worse trial.
    A partial or unresolved sample cannot receive no_observed_regression.
    """
    baseline_rows = _validate_arm(baseline)
    candidate_rows = _validate_arm(candidate)
    if baseline.identity != candidate.identity:
        raise ComparisonInputError("dataset, model and fixed measurement settings must match")
    if baseline.arm_id == candidate.arm_id:
        raise ComparisonInputError("declare distinct baseline and candidate arms")
    baseline_cases = {case.case_id: case for case in baseline.cases}
    candidate_cases = {case.case_id: case for case in candidate.cases}
    if baseline_cases != candidate_cases or baseline.repetitions != candidate.repetitions:
        raise ComparisonInputError("case labels and repetition coverage must match")

    regressions: list[str] = []
    uncertainties: list[str] = []
    for key, baseline_row in baseline_rows.items():
        candidate_row = candidate_rows[key]
        label = f"{key[0]} repetition {key[1]}"
        for arm, row in ((baseline, baseline_row), (candidate, candidate_row)):
            prefix = f"{arm.arm_id}: {label}"
            if not row.completed:
                uncertainties.append(f"{prefix}: review incomplete")
            if not row.claims_complete:
                uncertainties.append(f"{prefix}: emitted claim coverage incomplete")
            if row.visibility_context is None:
                uncertainties.append(f"{prefix}: actual served visibility uninspected")
            pending = any(claim.verdict in {None, "unresolved"} for claim in row.claims)
            if pending:
                uncertainties.append(f"{prefix}: unresolved or missing factual judgments")
            if any(claim.surfaced is None for claim in row.claims):
                uncertainties.append(f"{prefix}: claim delivery surface unknown")

        baseline_found = _surfaced_discoveries(baseline_row)
        candidate_found = _surfaced_discoveries(candidate_row)
        if candidate_row.completed and candidate_row.claims_complete:
            for mechanism in sorted(baseline_found - candidate_found):
                could_still_match = any(
                    claim.surfaced is not False
                    and claim.verdict != "incorrect"
                    and claim.mechanism_id in {None, mechanism}
                    for claim in candidate_row.claims
                )
                if not could_still_match:
                    regressions.append(f"{label}: lost baseline discovery {mechanism}")
        baseline_incorrect_upper = sum(
            claim.surfaced is not False and claim.verdict != "valid"
            for claim in baseline_row.claims
        )
        candidate_incorrect = sum(
            claim.verdict == "incorrect" and claim.surfaced is True
            for claim in candidate_row.claims
        )
        if (
            baseline_row.completed
            and baseline_row.claims_complete
            and candidate_incorrect > baseline_incorrect_upper
        ):
            regressions.append(
                f"{label}: incorrect warnings increased from "
                f"at most {baseline_incorrect_upper} to {candidate_incorrect}"
            )
    status: Literal["regression", "no_observed_regression", "inconclusive"] = (
        "regression"
        if regressions
        else "inconclusive"
        if uncertainties
        else "no_observed_regression"
    )
    return FloorComparison(
        status=status,
        baseline=baseline,
        candidate=candidate,
        baseline_metrics=_metrics(baseline),
        candidate_metrics=_metrics(candidate),
        regressions=tuple(regressions),
        uncertainties=tuple(uncertainties),
    )


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ComparisonInputError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _record(value: Any, record_type: type) -> dict[str, Any]:
    expected = {field.name for field in fields(record_type)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ComparisonInputError(f"{record_type.__name__} requires exactly its declared fields")
    return value


def _array(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ComparisonInputError("collection fields require JSON arrays")
    return value


def _text(value: Any, *, optional: bool = False) -> None:
    if not isinstance(value, str) and not (optional and value is None):
        raise ComparisonInputError("text fields require strings, or null where optional")


def _string_set(value: Any) -> frozenset[str]:
    items = _array(value)
    for item in items:
        _text(item)
    if len(set(items)) != len(items):
        raise ComparisonInputError("set fields cannot contain duplicate strings")
    return frozenset(items)


def _arm_from_json(value: Any) -> AdjudicatedArm:
    """Restore the complete dataclass representation without coercing assessments."""
    raw = _record(value, AdjudicatedArm)
    _text(raw["arm_id"])
    identity = _record(raw["identity"], MeasurementIdentity)
    for text in identity.values():
        _text(text)
    cases = []
    for value in _array(raw["cases"]):
        case = _record(value, CaseSpec)
        _text(case["case_id"])
        cases.append(
            CaseSpec(
                case["case_id"],
                _string_set(case["known_present_mechanisms"]),
                _string_set(case["fixed_mechanisms"]),
            )
        )
    outcomes = []
    for value in _array(raw["outcomes"]):
        outcome = _record(value, TrialOutcome)
        _text(outcome["case_id"])
        _text(outcome["visibility_context"], optional=True)
        claims = []
        for value in _array(outcome["claims"]):
            claim = _record(value, ClaimJudgment)
            _text(claim["claim_id"])
            for name in ("verdict", "mechanism_id", "location_assessment", "severity_assessment"):
                _text(claim[name], optional=True)
            claims.append(ClaimJudgment(**claim))
        outcomes.append(
            TrialOutcome(
                **{
                    **outcome,
                    "claims": tuple(claims),
                    "expected_claim_ids": _string_set(outcome["expected_claim_ids"]),
                }
            )
        )
    arm = AdjudicatedArm(
        raw["arm_id"],
        MeasurementIdentity(**identity),
        tuple(cases),
        raw["repetitions"],
        tuple(outcomes),
    )
    _validate_arm(arm)
    return arm


def main(argv: list[str] | None = None) -> int:
    """Check supplied saved assessments; never generate or certify model answers."""
    parser = argparse.ArgumentParser(description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON containing complete arms")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.inputs.read_text(), object_pairs_hook=_json_object)
        if not isinstance(document, dict) or not isinstance(document.get("arms"), dict):
            raise ComparisonInputError("inputs require an arms object")
        selected = []
        for name in (args.baseline, args.candidate):
            if name not in document["arms"]:
                raise ComparisonInputError(f"unknown comparison arm: {name}")
            arm = _arm_from_json(document["arms"][name])
            if arm.arm_id != name:
                raise ComparisonInputError("arm key differs from its declared arm_id")
            selected.append(arm)
        result = compare_review_floor(*selected)
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))

    print(f"Sample result: {result.status}")
    for arm, metrics in (
        (result.baseline, result.baseline_metrics),
        (result.candidate, result.candidate_metrics),
    ):
        print(
            f"{arm.arm_id}: final output has {metrics.surfaced_valid_claims} supported findings "
            f"and {metrics.surfaced_incorrect_claims} incorrect findings"
        )
        print(
            f"  Full inventory: {metrics.unresolved_claims} unresolved, "
            f"{metrics.missing_judgments} unjudged, "
            f"{metrics.unknown_surface_claims} with unknown delivery"
        )
        print(
            "  Known case checks (separate from relative floor): "
            f"{metrics.known_mechanisms_found}/{metrics.known_mechanism_opportunities} found"
        )
    if (
        result.candidate_metrics.known_mechanisms_found
        < result.candidate_metrics.known_mechanism_opportunities
    ):
        print(
            "Candidate misses declared known checks; a clean relative result is not a quality pass."
        )
    for regression in result.regressions:
        print(f"Regression: {regression}")
    for uncertainty in result.uncertainties:
        print(f"Unresolved: {uncertainty}")
    print("Counts are finding occurrences, not independent bugs.")
    print(
        "Input basis: supplied assessments; raw captures, provenance and their completeness "
        "are not independently verified by this command."
    )
    print(result.limitation)
    return {"no_observed_regression": 0, "regression": 1, "inconclusive": 2}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
