"""Exercise the sample floor without providers, database access or paid results."""

import json
from dataclasses import asdict, replace

import pytest

from .review_floor import (
    AdjudicatedArm,
    CaseSpec,
    ClaimJudgment,
    ComparisonInputError,
    MeasurementIdentity,
    TrialOutcome,
    compare_review_floor,
    main,
)

IDENTITY = MeasurementIdentity("a" * 64, "anthropic:example-model", "b" * 64)
BUG = CaseSpec("bug", frozenset({"rollover-blocks-event-loop"}), frozenset())
FIXED = CaseSpec("fixed", frozenset(), frozenset({"rollover-blocks-event-loop"}))
VALID = ClaimJudgment("valid", "valid", "rollover-blocks-event-loop", surfaced=True)


def _arm(
    arm_id: str,
    claims: tuple[ClaimJudgment, ...] = (),
) -> AdjudicatedArm:
    return AdjudicatedArm(
        arm_id=arm_id,
        identity=IDENTITY,
        cases=(BUG,),
        repetitions=1,
        outcomes=(
            TrialOutcome(
                "bug",
                0,
                claims,
                frozenset(claim.claim_id for claim in claims),
                True,
                True,
                "Captured complete source request",
            ),
        ),
    )


@pytest.mark.parametrize("candidate_surface", ["silent", "filtered"])
def test_lost_baseline_issue_is_a_regression_despite_visibility_gap(candidate_surface: str) -> None:
    baseline = _arm("direct", (VALID, ClaimJudgment("disputed", "unresolved", surfaced=True)))
    claims = () if candidate_surface == "silent" else (replace(VALID, surfaced=False),)
    candidate = _arm("workflow", claims)
    candidate = replace(
        candidate,
        outcomes=(replace(candidate.outcomes[0], visibility_context=None),),
    )

    result = compare_review_floor(baseline, candidate)

    assert result.status == "regression"
    assert result.regressions == (
        "bug repetition 0: lost baseline discovery rollover-blocks-event-loop",
    )
    assert result.baseline_metrics.known_mechanisms_found == 1
    assert result.candidate_metrics.known_mechanisms_found == 0
    assert result.candidate_metrics.raw_known_mechanisms_found == (candidate_surface == "filtered")
    assert result.candidate_metrics.precision is None
    assert len(result.uncertainties) == 2


def test_location_and_severity_stay_separate_and_fixed_cases_can_have_other_bugs() -> None:
    baseline = _arm(
        "direct", (replace(VALID, location_assessment="line 10", severity_assessment="high"),)
    )
    unrelated = ClaimJudgment("other", "valid", "unrelated-resource-leak", surfaced=True)
    baseline = replace(
        baseline,
        cases=(BUG, FIXED),
        outcomes=baseline.outcomes
        + (
            TrialOutcome(
                "fixed", 0, (unrelated,), frozenset({"other"}), True, True, "Captured fixed source"
            ),
        ),
    )
    relocated = replace(VALID, location_assessment="line 11", severity_assessment="medium")
    rejected_wrong = ClaimJudgment("rejected-wrong", "incorrect", surfaced=False)
    candidate = replace(
        baseline,
        arm_id="workflow",
        outcomes=(
            replace(
                baseline.outcomes[0],
                claims=(relocated, rejected_wrong),
                expected_claim_ids=frozenset({"valid", "rejected-wrong"}),
            ),
            baseline.outcomes[1],
        ),
    )

    result = compare_review_floor(baseline, candidate)

    assert result.status == "no_observed_regression"
    assert result.candidate_metrics.valid_claims == 2
    assert result.candidate_metrics.incorrect_claims == 1
    assert result.candidate_metrics.surfaced_incorrect_claims == 0
    assert result.candidate_metrics.known_mechanisms_found == 1
    assert result.candidate_metrics.known_mechanism_opportunities == 1
    assert result.candidate_metrics.precision == 1.0
    assert result.baseline.outcomes[0].claims[0].location_assessment == "line 10"
    assert result.candidate.outcomes[0].claims[0].location_assessment == "line 11"
    assert "broader noninferiority" in result.limitation


def test_aggregate_warning_reduction_cannot_hide_an_increase_in_another_repetition() -> None:
    wrong_one = ClaimJudgment("wrong-one", "incorrect", surfaced=True)
    wrong_two = ClaimJudgment("wrong-two", "incorrect", surfaced=True)
    baseline = _arm("direct", (wrong_one, wrong_two))
    baseline = replace(
        baseline,
        repetitions=2,
        outcomes=baseline.outcomes
        + (replace(baseline.outcomes[0], repetition=1, claims=(), expected_claim_ids=frozenset()),),
    )
    candidate = replace(
        baseline,
        arm_id="workflow",
        outcomes=(
            replace(baseline.outcomes[0], claims=(), expected_claim_ids=frozenset()),
            replace(
                baseline.outcomes[1],
                claims=(wrong_one,),
                expected_claim_ids=frozenset({"wrong-one"}),
            ),
        ),
    )

    result = compare_review_floor(baseline, candidate)

    assert result.candidate_metrics.incorrect_claims < result.baseline_metrics.incorrect_claims
    assert result.status == "regression"
    assert result.regressions == (
        "bug repetition 1: incorrect warnings increased from at most 0 to 1",
    )


@pytest.mark.parametrize(
    "gap", ["unresolved", "unjudged", "unfinished", "claims", "visibility", "surface"]
)
def test_unresolved_or_incomplete_evidence_cannot_pass(gap: str) -> None:
    baseline = _arm("direct", (VALID,))
    candidate = _arm("workflow", (VALID,))
    row = candidate.outcomes[0]
    if gap == "unresolved":
        row = replace(row, claims=(replace(VALID, verdict="unresolved"),))
    elif gap == "unjudged":
        row = replace(row, claims=(replace(VALID, verdict=None),))
    elif gap == "unfinished":
        row = replace(row, completed=False)
    elif gap == "claims":
        row = replace(row, claims_complete=False)
    elif gap == "visibility":
        row = replace(row, visibility_context=None)
    else:
        row = replace(row, claims=(replace(VALID, surfaced=None),))
    candidate = replace(candidate, outcomes=(row,))

    result = compare_review_floor(baseline, candidate)

    assert result.status == "inconclusive"
    assert result.uncertainties
    if gap != "visibility":
        assert result.candidate_metrics.precision is None


@pytest.mark.parametrize(
    "mismatch",
    [
        "dataset",
        "model",
        "settings",
        "coverage",
        "repetitions",
        "labels",
        "duplicate",
        "mechanism",
        "omitted-claim",
        "uninventoried-claim",
    ],
)
def test_mismatched_or_malformed_experiments_are_rejected(mismatch: str) -> None:
    baseline = _arm("direct", (VALID,))
    candidate = _arm("workflow", (VALID,))
    if mismatch == "dataset":
        candidate = replace(candidate, identity=replace(IDENTITY, dataset_sha256="c" * 64))
    elif mismatch == "model":
        candidate = replace(candidate, identity=replace(IDENTITY, model="other:model"))
    elif mismatch == "settings":
        candidate = replace(
            candidate, identity=replace(IDENTITY, measurement_settings_sha256="c" * 64)
        )
    elif mismatch == "coverage":
        candidate = replace(candidate, outcomes=())
    elif mismatch == "repetitions":
        candidate = replace(
            candidate,
            repetitions=2,
            outcomes=candidate.outcomes + (replace(candidate.outcomes[0], repetition=1),),
        )
    elif mismatch == "labels":
        candidate = replace(candidate, cases=(replace(BUG, known_present_mechanisms=frozenset()),))
    elif mismatch == "duplicate":
        candidate = replace(candidate, outcomes=candidate.outcomes * 2)
    elif mismatch == "mechanism":
        candidate = replace(
            candidate,
            outcomes=(replace(candidate.outcomes[0], claims=(replace(VALID, mechanism_id=None),)),),
        )
    elif mismatch == "omitted-claim":
        candidate = replace(candidate, outcomes=(replace(candidate.outcomes[0], claims=()),))
    else:
        candidate = replace(
            candidate, outcomes=(replace(candidate.outcomes[0], expected_claim_ids=frozenset()),)
        )

    with pytest.raises(ComparisonInputError):
        compare_review_floor(baseline, candidate)


@pytest.mark.parametrize(
    ("candidate_verdict", "status", "exit_code"),
    [
        ("valid", "no_observed_regression", 0),
        ("unresolved", "inconclusive", 2),
        (None, "regression", 1),
        ("both-silent", "no_observed_regression", 0),
    ],
)
def test_offline_cli_restores_assessments_and_reports_sample_limits(
    tmp_path, capsys, candidate_verdict, status, exit_code
) -> None:
    baseline = _arm("direct-one-shot", () if candidate_verdict == "both-silent" else (VALID,))
    claims = (
        ()
        if candidate_verdict in {None, "both-silent"}
        else (replace(VALID, verdict=candidate_verdict),)
    )
    candidate = _arm("workflow", claims)
    path = tmp_path / "inputs.json"
    original = json.dumps(
        {"arms": {arm.arm_id: asdict(arm) for arm in (baseline, candidate)}}, default=sorted
    )
    path.write_text(original)

    assert (
        main(
            ["--inputs", str(path), "--baseline", baseline.arm_id, "--candidate", candidate.arm_id]
        )
        == exit_code
    )

    output = capsys.readouterr().out
    assert f"Sample result: {status}" in output
    assert (
        "raw captures, provenance and their completeness are not independently verified" in output
    )
    assert "broader noninferiority" in output
    known_found = int(candidate_verdict == "valid")
    assert f"Known case checks (separate from relative floor): {known_found}/1 found" in output
    assert ("Candidate misses declared known checks" in output) == (known_found == 0)
    assert path.read_text() == original


@pytest.mark.parametrize(
    "malformed",
    ["extra-field", "duplicate-key", "boolean", "duplicate-set", "oversized-integer", "deep-array"],
)
def test_offline_cli_rejects_ambiguous_or_coerced_input(tmp_path, capsys, malformed) -> None:
    arms = {name: asdict(_arm(name, (VALID,))) for name in ("direct", "workflow")}
    row = arms["workflow"]["outcomes"][0]
    if malformed == "extra-field":
        arms["workflow"]["claims_complete"] = True
    elif malformed == "boolean":
        row["completed"] = "true"
    elif malformed == "duplicate-set":
        row["expected_claim_ids"] = ["valid", "valid"]
    text = json.dumps({"arms": arms}, default=sorted)
    if malformed == "duplicate-key":
        text = text.replace('"arms":', '"arms": {}, "arms":', 1)
    elif malformed == "oversized-integer":
        text = text.replace('"repetitions": 1', '"repetitions": ' + "9" * 5000, 1)
    elif malformed == "deep-array":
        text = "[" * 15000 + "0" + "]" * 15000
    path = tmp_path / "inputs.json"
    path.write_text(text)

    with pytest.raises(SystemExit) as raised:
        main(["--inputs", str(path), "--baseline", "direct", "--candidate", "workflow"])

    assert raised.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_two_silent_arms_have_undefined_precision_and_still_miss_the_named_bug() -> None:
    result = compare_review_floor(_arm("direct"), _arm("workflow"))

    assert result.status == "no_observed_regression"
    assert result.baseline_metrics.precision is None
    assert result.candidate_metrics.precision is None
    assert result.baseline_metrics.known_mechanisms_found == 0
    assert result.baseline_metrics.known_mechanism_opportunities == 1
    assert "does not hinder the model" in result.limitation
