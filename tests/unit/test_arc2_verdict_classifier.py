"""Arc 2 — the verdict classifier is TOTAL, prose-blind, and bound to its bytes.

Three properties under test, each of which was a review finding before it was code:

1. **Totality over BOTH arms.** Transport (any HTTP status, any timeout) AND
   capture (any envelope, including malformed and ambiguous ones). An earlier
   draft closed the transport arm and left the response arm open, so a both-null
   envelope could have counted toward `PARK` — evidence of absence manufactured
   from a malformed response.
2. **Structure-only.** Request rejection is decided POSITIONALLY (which row),
   never by reading vendor prose, because an invalid-schema 400 and any other 400
   are indistinguishable from the fields the boundary exposes.
3. **Every fact bound to the bytes it came from.** A row is one `EvaluatedRow`.
   The previous shape passed the envelope, the finding assessment, and the
   zero-findings flag as independent arguments, so a caller could pair
   `{"findings":[]}` with `returned_any_finding=True` — an impossible observation
   that still produced a positive verdict.

See `specs/2026-07-20-arc2-strict-schema-feasibility.md`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from spikes.openai.arc2.classifier import (
    REFUSAL_ROWS,
    AdmittedFinding,
    EvaluatedRow,
    ExpectedFinding,
    FindingRowAssessment,
    Observation,
    ObservationKind,
    ParserOutcome,
    RowId,
    Verdict,
    classify_envelope,
    classify_session,
    classify_transport,
    derive_assessment,
    detect_wrong_tier_proof,
    evaluate_row,
    route_finding_assessment,
)
from spikes.openai.arc2.strict_schema import derive_strict_analyze_schema

_EMPTY = '{"findings":[]}'


def _validator() -> Any:
    from jsonschema import Draft202012Validator

    return Draft202012Validator(derive_strict_analyze_schema()).validate


def _finding_body(**overrides: Any) -> str:
    finding: dict[str, Any] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "line_start": 1,
        "line_end": 2,
        "trace_candidates": [],
    }
    finding.update(overrides)
    return json.dumps({"findings": [finding]})


_EXPECTED = ExpectedFinding(finding_type="sql_injection", line_start=1, line_end=2)
#: A REGISTERED id that genuinely fires for python sources. Asserting a
#: hand-written string here is what let the probe call fabricated proof
#: "authentic" — the test supplied the authority that made its value real.
_FIRED = frozenset({"python.function_definition"})


def _admitted(
    *,
    query_match_id: str | None = None,
    trace_candidate_count: int = 0,
    finding_type: str = "sql_injection",
    line_start: int = 1,
    line_end: int = 2,
) -> AdmittedFinding:
    return AdmittedFinding(
        finding_type=finding_type,
        line_start=line_start,
        line_end=line_end,
        query_match_id=query_match_id,
        trace_candidate_count=trace_candidate_count,
    )


def _parser_outcome(
    admitted: tuple[AdmittedFinding, ...] = (),
    rejected: tuple[str, ...] = (),
    retained: int = 0,
) -> ParserOutcome:
    """`retained` is explicit at every call site — the field has no default, so a
    caller cannot silently omit the absolute fabrication fact."""
    return ParserOutcome(
        admitted=admitted, rejection_reasons=rejected, retained_trace_candidate_count=retained
    )


def _row(
    row: RowId,
    *,
    content: str | None = None,
    refusal: str | None = None,
    parser: Any = None,
    fired: frozenset[str] = _FIRED,
) -> EvaluatedRow:
    """Build a row from BYTES. Every semantic flag is derived from those bytes
    plus the parser's ADMITTED findings — no flag can be hand-supplied here."""
    return evaluate_row(
        row=row,
        content=content,
        refusal=refusal,
        finish_reason="stop",
        run_parser=parser,
        fired_query_match_ids=fired,
        expected_finding=_EXPECTED,
    )


def _negatives(*rows: RowId) -> dict[RowId, EvaluatedRow]:
    return {row: _row(row, content=_EMPTY) for row in rows}


def _session(
    *,
    refusals: dict[RowId, EvaluatedRow] | None = None,
    finding: EvaluatedRow | None = None,
    clean: EvaluatedRow | None = None,
):
    rows: dict[RowId, EvaluatedRow] = {
        RowId.ACCEPTANCE_CLEAN: clean
        if clean is not None
        else _row(RowId.ACCEPTANCE_CLEAN, content=_EMPTY),
        RowId.ACCEPTANCE_FINDING: finding
        if finding is not None
        else _row(
            RowId.ACCEPTANCE_FINDING,
            content=_finding_body(),
            parser=lambda _b: _parser_outcome(admitted=(_admitted(),)),
        ),
    }
    rows.update(refusals or {})
    return classify_session(rows=rows)


# --------------------------------------------------------------------------
# Transport arm — positional, and total.
# --------------------------------------------------------------------------


def test_classification_is_positional_not_prose() -> None:
    """A 400 classifies `STOP-shape` ONLY on the schema-admission row; the same
    status on a later row is a failed observation. No branch reads the excerpt."""
    first = classify_transport(row=RowId.ACCEPTANCE_CLEAN, status=400)
    assert first.terminal_verdict is Verdict.STOP_SHAPE

    for row in (RowId.ACCEPTANCE_FINDING, *REFUSAL_ROWS):
        later = classify_transport(row=row, status=400)
        assert later.terminal_verdict is None
        assert later.observation.kind is ObservationKind.FAILED


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503, None])
def test_non_400_statuses_are_failed_observations_on_every_row(status: int | None) -> None:
    """401/403/429/5xx and timeouts (`status is None`) never produce a verdict."""
    for row in RowId:
        outcome = classify_transport(row=row, status=status)
        assert outcome.terminal_verdict is None
        assert outcome.observation.kind is ObservationKind.FAILED


@pytest.mark.parametrize("status", [404, 408, 409, 418, 422, 451, 502, 504, 599, 200, 201])
def test_classifier_is_total_over_residual_statuses(status: int) -> None:
    """The residual statuses an enumerated list would have missed. The rule is
    closed by construction: anything other than an exact 400 on the
    schema-admission row is a failed observation."""
    for row in RowId:
        outcome = classify_transport(row=row, status=status)
        assert outcome.observation.kind is ObservationKind.FAILED
        assert outcome.terminal_verdict is None


def test_400_after_schema_acceptance_is_not_a_schema_rejection() -> None:
    """Once the admission row returns 2xx the request was accepted, so a later
    400 cannot retroactively mean the request was rejected."""
    result = _session(
        refusals={RowId.REFUSAL_1: classify_transport(row=RowId.REFUSAL_1, status=400)}
    )
    # The ACTUAL verdict, not merely "not STOP-shape" — the negative form would
    # also pass on GO or PARK, which would be far worse outcomes here.
    assert result.verdict is Verdict.INCONCLUSIVE


def test_string_row_keys_do_not_silently_lose_the_schema_rejection() -> None:
    """`RowId` is a `StrEnum`, so a plain-string row key compares and hashes equal
    everywhere — an `is` check would degrade ONLY here, turning a genuine request
    rejection into INCONCLUSIVE on the most important branch."""
    outcome = classify_transport(row="acceptance_clean", status=400)  # type: ignore[arg-type]
    assert outcome.terminal_verdict is Verdict.STOP_SHAPE


# --------------------------------------------------------------------------
# Response arm — the envelope taxonomy.
# --------------------------------------------------------------------------


def test_api_owned_refusal_takes_precedence_over_content_and_finish_reason() -> None:
    """Matches the RECORDED PRODUCTION contract, not a reading of the docs.

    `OpenAICompatibleProvider` (FUP-203, adjudicated by the openai-native-host
    spec) normalizes "a non-empty structured `message.refusal` field (which OpenAI
    can send with `finish_reason=\"stop\"`)" to a refusal whatever else the
    envelope carries. `tests/unit/test_raw_openai_capture.py`'s wire fixture models
    content AND refusal both populated.

    The previous rule demanded `content is None` and scored both-populated as
    ambiguous — which would have read a GENUINE refusal as a failed observation,
    turning the one paid session INCONCLUSIVE on the very question it exists to
    answer.
    """
    for content, finish in (
        (None, "stop"),
        (_EMPTY, "stop"),
        (None, "length"),
        (None, "content_filter"),
        (_EMPTY, "content_filter"),
    ):
        assert (
            classify_envelope(content=content, refusal="I can't help", finish_reason=finish).kind
            is ObservationKind.VALID_REFUSAL
        ), (content, finish)

    # A BLANK refusal is not a refusal, in any combination.
    for content, refusal in ((None, "   "), (None, ""), (_EMPTY, " ")):
        assert (
            classify_envelope(content=content, refusal=refusal, finish_reason="stop").kind
            is not ObservationKind.VALID_REFUSAL
        )


def test_content_channel_refusal_stays_closed_under_the_new_precedence() -> None:
    """The round-3 hole must NOT reopen. Refusal-shaped PROSE in `content` with
    `refusal=None` is a different thing from the API-owned channel firing: it
    still fails strict validation and is never a negative observation."""
    for body in ('"I cannot help with that."', '{"totally":"wrong"}', "null"):
        assert (
            classify_envelope(content=body, refusal=None, finish_reason="stop").kind
            is ObservationKind.FAILED
        ), body


def test_ambiguous_envelopes_are_failed_observations() -> None:
    """Both-null is unscoreable. (Both-POPULATED is no longer here: production
    resolves that case in favour of the API-owned refusal channel.)"""
    for content, refusal in ((None, None), ("not json", None)):
        assert (
            classify_envelope(content=content, refusal=refusal, finish_reason="stop").kind
            is ObservationKind.FAILED
        )


def test_content_channel_refusal_is_not_a_negative_observation() -> None:
    """THE cardinal-sin case, and the reason `strict_validator` is required.

    A model declining inside the CONTENT channel returns valid JSON with no API
    refusal. With schema validation skipped that scored as VALID_NEGATIVE and
    counted toward `PARK` — a refusal becoming evidence that refusals don't fire.
    """
    for body in ('"I cannot help with that."', '{"totally":"wrong"}', "null", "0", "[]"):
        observation = classify_envelope(content=body, refusal=None, finish_reason="stop")
        assert observation.kind is ObservationKind.FAILED, body

    attempted_park = _session(
        refusals={row: _row(row, content='"I cannot help with that."') for row in REFUSAL_ROWS}
    )
    assert attempted_park.verdict is Verdict.INCONCLUSIVE


def test_strict_validator_is_required_not_silently_skippable() -> None:
    """There is NO validator injection point, and validation is not skippable.

    The previous version raised `TypeError` by omitting `finish_reason` — a
    required kwarg that has nothing to do with validation — so it passed no matter
    what the validator seam looked like. It asserted the wrong impossibility.

    Two real claims instead: (1) no parameter exists through which a caller could
    supply, replace, or disable the validator; (2) with EVERY required argument
    present, an off-schema body is still `FAILED`. The `UNCHECKED` sentinel stays
    banned — it returned `VALID_NEGATIVE`, so an unvalidated body could reach
    `GO`/`PARK` with the caveat living only in a detail string nothing stored.
    """
    import inspect

    import spikes.openai.arc2.classifier as classifier_module

    params = set(inspect.signature(classify_envelope).parameters)
    assert params == {"content", "refusal", "finish_reason"}, (
        f"unexpected parameter on classify_envelope: {params}. A validator seam here "
        "is how the silent-downgrade defect returns."
    )
    assert not hasattr(classifier_module, "UNCHECKED")

    # Fully-specified call, off-schema body: wrong-tier proof (JUDGED carrying a
    # query_match_id) is well-formed JSON that the strict schema forbids.
    off_schema = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "sql_injection",
                    "evidence_tier": "judged",
                    "query_match_id": "python.function_definition",
                    "trace_path": None,
                    "title": "t",
                    "description": "d",
                    "evidence": "e",
                    "line_start": 1,
                    "line_end": 1,
                    "trace_candidates": [],
                }
            ]
        }
    )
    assert (
        classify_envelope(content=off_schema, refusal=None, finish_reason="stop").kind
        is ObservationKind.FAILED
    )


def test_non_json_content_is_a_failed_observation() -> None:
    assert (
        classify_envelope(content="not json", refusal=None, finish_reason="stop").kind
        is ObservationKind.FAILED
    )


# --------------------------------------------------------------------------
# EvaluatedRow — facts bound to bytes, impossible fixtures fail CLOSED.
# --------------------------------------------------------------------------


def test_no_conclusion_is_a_constructor_input() -> None:
    """`EvaluatedRow` accepts RAW EVIDENCE only.

    Each of these was, at some point, a settable field that `classify_session`
    trusted before looking at the evidence. They are all derived properties now,
    so the forgeries are `TypeError`s rather than values that must survive a check.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(EvaluatedRow)}
    for conclusion in ("observation", "terminal_verdict", "wrong_tier_proof", "assessment"):
        assert conclusion not in fields, conclusion
    assert {"content", "refusal", "transport_status", "parser"} <= fields

    for kwargs in (
        {"observation": Observation(ObservationKind.VALID_REFUSAL, "refused")},
        {"terminal_verdict": Verdict.GO},
        {"wrong_tier_proof": True},
        {"assessment": FindingRowAssessment(True, True, False, False)},
    ):
        with pytest.raises(TypeError):
            EvaluatedRow(row=RowId.REFUSAL_1, **kwargs)  # type: ignore[arg-type]


def test_a_refusal_requires_actual_refusal_text() -> None:
    """THE forged-discriminator case. `Observation(VALID_REFUSAL, "refused")` on a
    row storing no refusal string satisfied the discriminator and reached `GO` — a
    demonstrated refusal channel conjured from nothing."""
    empty = EvaluatedRow(row=RowId.REFUSAL_1, content=None, refusal=None)
    assert empty.observation.kind is ObservationKind.FAILED

    real = EvaluatedRow(
        row=RowId.REFUSAL_1,
        content=None,
        refusal="I can't help with that.",
        finish_reason="stop",
    )
    assert real.observation.kind is ObservationKind.VALID_REFUSAL

    # And a session built only from evidence-free rows cannot reach GO.
    rows = {
        RowId.ACCEPTANCE_CLEAN: _row(RowId.ACCEPTANCE_CLEAN, content=_EMPTY),
        RowId.ACCEPTANCE_FINDING: _row(
            RowId.ACCEPTANCE_FINDING,
            content=_finding_body(),
            parser=lambda _b: _parser_outcome(admitted=(_admitted(),)),
        ),
        RowId.REFUSAL_1: EvaluatedRow(row=RowId.REFUSAL_1),
        RowId.REFUSAL_2: EvaluatedRow(row=RowId.REFUSAL_2),
        RowId.REFUSAL_3: EvaluatedRow(row=RowId.REFUSAL_3),
    }
    assert classify_session(rows=rows).verdict is not Verdict.GO


def test_parser_facts_only_attach_to_the_assessed_row() -> None:
    with pytest.raises(ValueError, match="only"):
        EvaluatedRow(
            row=RowId.ACCEPTANCE_CLEAN,
            content=_EMPTY,
            finish_reason="stop",
            parser=_parser_outcome(),
        )


def test_response_and_transport_evidence_are_disjoint() -> None:
    """A row is either a completed response or a transport failure, never both."""
    with pytest.raises(ValueError, match="BOTH a transport status and response evidence"):
        EvaluatedRow(row=RowId.ACCEPTANCE_CLEAN, content=_EMPTY, transport_failed=True)
    with pytest.raises(ValueError, match="BOTH a transport status and response evidence"):
        EvaluatedRow(row=RowId.REFUSAL_1, refusal="I cannot", transport_failed=True)


def test_a_status_requires_an_actual_transport_failure() -> None:
    """`transport_status=None` is AMBIGUOUS on its own — it means "timeout" for a
    failed row and "not a transport failure" for a completed one, so the two were
    indistinguishable. `transport_failed` disambiguates, and a status without it
    is refused."""
    with pytest.raises(ValueError, match="without transport_failed"):
        EvaluatedRow(row=RowId.ACCEPTANCE_CLEAN, transport_status=400)

    timeout = classify_transport(row=RowId.REFUSAL_1, status=None)
    completed = EvaluatedRow(row=RowId.REFUSAL_1, content=None, refusal=None)
    assert timeout.transport_failed is True
    assert completed.transport_failed is False
    assert timeout.transport_status == completed.transport_status is None


def test_terminal_verdict_is_positional_and_status_derived() -> None:
    assert (
        classify_transport(row=RowId.ACCEPTANCE_CLEAN, status=400).terminal_verdict
        is Verdict.STOP_SHAPE
    )
    for row in (RowId.ACCEPTANCE_FINDING, *REFUSAL_ROWS):
        assert classify_transport(row=row, status=400).terminal_verdict is None
    assert classify_transport(row=RowId.ACCEPTANCE_CLEAN, status=500).terminal_verdict is None


def test_mapping_key_must_match_the_embedded_row() -> None:
    with pytest.raises(ValueError, match="key .* holds an EvaluatedRow"):
        classify_session(rows={RowId.ACCEPTANCE_CLEAN: _row(RowId.REFUSAL_1, content=_EMPTY)})


def test_body_digest_is_bound_to_the_actual_bytes() -> None:
    a = _row(RowId.ACCEPTANCE_CLEAN, content=_EMPTY)
    b = _row(RowId.ACCEPTANCE_CLEAN, content=_EMPTY)
    c = _row(RowId.ACCEPTANCE_FINDING, content=_finding_body(), parser=lambda _b: _parser_outcome())
    assert a.body_digest == b.body_digest
    assert a.body_digest != c.body_digest
    assert a.finding_count == 0
    assert c.finding_count == 1


def test_assessment_is_not_derived_for_a_failed_body() -> None:
    """The parser runs ONLY when the body validated, so an assessment can never
    describe a body that failed."""
    called: list[str] = []

    def _p(body: str) -> ParserOutcome:
        called.append(body)
        return _parser_outcome()

    row = _row(RowId.ACCEPTANCE_FINDING, content="not json", parser=_p)
    assert called == []
    assert row.assessment is None
    assert row.observation.kind is ObservationKind.FAILED


# --------------------------------------------------------------------------
# PARK completeness + the short-circuit.
# --------------------------------------------------------------------------


def test_park_requires_three_complete_valid_negatives() -> None:
    assert _session(refusals=_negatives(*REFUSAL_ROWS)).verdict is Verdict.PARK


def test_failed_observation_never_counts_toward_park() -> None:
    """Two valid negatives plus one error is `INCONCLUSIVE`. Absence of evidence
    from an incomplete run is not evidence of absence."""
    outcomes = _negatives(RowId.REFUSAL_1, RowId.REFUSAL_2)
    outcomes[RowId.REFUSAL_3] = classify_transport(row=RowId.REFUSAL_3, status=429)
    result = _session(refusals=outcomes)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.discriminator_met is False


def test_missing_refusal_rows_are_not_negatives() -> None:
    """A row that never ran is 'not observed', never a negative observation."""
    assert _session(refusals=_negatives(RowId.REFUSAL_1)).verdict is Verdict.INCONCLUSIVE


def test_demonstrated_refusal_short_circuits_and_cannot_be_retracted() -> None:
    """A VALID refusal on the SECOND refusal row satisfies the discriminator even
    when the THIRD errors and aborts the session."""
    outcomes = _negatives(RowId.REFUSAL_1)
    outcomes[RowId.REFUSAL_2] = _row(RowId.REFUSAL_2, refusal="I can't help with that.")
    outcomes[RowId.REFUSAL_3] = classify_transport(row=RowId.REFUSAL_3, status=500)
    result = _session(refusals=outcomes)
    assert result.discriminator_met is True
    assert result.verdict is Verdict.GO


def test_blank_refusal_cannot_short_circuit() -> None:
    """Only a BLANK refusal fails to satisfy the discriminator.

    The docstring previously said both-populated envelopes could not short-circuit
    while the assertions tested only blank refusals — it asserted the opposite of
    the behaviour under test, and of production's precedence rule.
    """
    for bad_content, bad_refusal in ((None, "   "), (None, "")):
        outcomes = _negatives(RowId.REFUSAL_1)
        outcomes[RowId.REFUSAL_2] = _row(RowId.REFUSAL_2, content=bad_content, refusal=bad_refusal)
        outcomes[RowId.REFUSAL_3] = classify_transport(row=RowId.REFUSAL_3, status=500)
        result = _session(refusals=outcomes)
        assert result.discriminator_met is False
        assert result.verdict is Verdict.INCONCLUSIVE


def test_discriminator_is_not_reported_met_under_a_rejected_request() -> None:
    """ "The request was rejected" and "the refusal channel is demonstrated" cannot
    both be true of one run; a manifest reader would take the latter as positive."""
    rows = {
        RowId.ACCEPTANCE_CLEAN: classify_transport(row=RowId.ACCEPTANCE_CLEAN, status=400),
        RowId.REFUSAL_1: _row(RowId.REFUSAL_1, refusal="I can't help with that."),
    }
    result = classify_session(rows=rows)
    assert result.verdict is Verdict.STOP_SHAPE
    assert result.discriminator_met is False


# --------------------------------------------------------------------------
# acceptance_finding routing — the three STOP categories stay distinct.
# --------------------------------------------------------------------------


def test_zero_findings_is_inconclusive_not_a_stop() -> None:
    """Recall is an explicit non-goal: a missed finding leaves the
    populated-branch question unanswered, which is not a schema failure."""
    missed = _row(RowId.ACCEPTANCE_FINDING, content=_EMPTY, parser=lambda _b: _parser_outcome())
    assert _session(refusals=_negatives(*REFUSAL_ROWS), finding=missed).verdict is (
        Verdict.INCONCLUSIVE
    )


def test_wrong_tier_proof_in_a_response_is_stop_shape() -> None:
    """A JUDGED finding carrying a `query_match_id`. It FAILS strict validation —
    which is why detection reads the parsed JSON directly. Gating this behind
    "did the row yield a valid negative" would make the branch UNREACHABLE, and a
    verdict nothing can reach is not a gate."""
    body = _finding_body(evidence_tier="judged", query_match_id="python.function_definition")
    row = _row(RowId.ACCEPTANCE_FINDING, content=body, parser=lambda _b: _parser_outcome())
    assert row.wrong_tier_proof is True
    assert row.observation.kind is ObservationKind.FAILED  # strict validation rejects it too

    result = _session(refusals=_negatives(*REFUSAL_ROWS), finding=row)
    assert result.verdict is Verdict.STOP_SHAPE


def test_wrong_tier_detection_is_independent_of_strict_validation() -> None:
    """Detection works on parsed JSON alone, so it fires on bodies the validator
    never sees as valid."""
    assert detect_wrong_tier_proof(
        {"findings": [{"evidence_tier": "judged", "query_match_id": "q1"}]}
    )
    assert detect_wrong_tier_proof(
        {"findings": [{"evidence_tier": "observed", "query_match_id": None}]}
    )
    assert detect_wrong_tier_proof({"findings": [{"evidence_tier": "inferred", "trace_path": []}]})
    assert detect_wrong_tier_proof({"findings": [{"evidence_tier": "not_a_tier"}]})
    # Legitimate shapes are not flagged.
    assert not detect_wrong_tier_proof(
        {"findings": [{"evidence_tier": "observed", "query_match_id": "q1", "trace_path": None}]}
    )
    assert not detect_wrong_tier_proof({"findings": []})


def test_wrong_tier_detection_matches_the_schema_on_empty_strings() -> None:
    """The exact case truthiness got wrong in BOTH directions.

    Derived from `TIER_PROOF_SHAPES`, so the semantic check and the wire encoding
    cannot disagree:
    - OBSERVED `query_match_id=""` is schema-VALID (the branch is a bare
      `{"type": "string"}`), so it must NOT be flagged. Emptiness testing flagged it.
    - INFERRED/JUDGED `query_match_id=""` VIOLATES their null-only branch, so it
      MUST be flagged. Emptiness testing read it as "absent" and missed it.
    """
    validator = _validator()

    observed_empty = {
        "findings": [{"evidence_tier": "observed", "query_match_id": "", "trace_path": None}]
    }
    assert not detect_wrong_tier_proof(observed_empty)  # schema-valid, must not flag
    validator({"findings": [_full(evidence_tier="observed", query_match_id="")]})

    for tier in ("judged", "inferred"):
        empty_on_null_branch = {
            "findings": [{"evidence_tier": tier, "query_match_id": "", "trace_path": None}]
        }
        assert detect_wrong_tier_proof(empty_on_null_branch), tier


def test_wrong_tier_detection_matches_the_schema_on_trace_path_shapes() -> None:
    """INFERRED needs a non-empty array of non-empty strings; the null-only
    branches need exactly None."""
    assert detect_wrong_tier_proof(
        {"findings": [{"evidence_tier": "inferred", "query_match_id": None, "trace_path": []}]}
    )
    assert detect_wrong_tier_proof(
        {"findings": [{"evidence_tier": "inferred", "query_match_id": None, "trace_path": [""]}]}
    )
    assert detect_wrong_tier_proof(
        {"findings": [{"evidence_tier": "judged", "query_match_id": None, "trace_path": []}]}
    )
    assert not detect_wrong_tier_proof(
        {"findings": [{"evidence_tier": "inferred", "query_match_id": None, "trace_path": ["a"]}]}
    )


def _full(**overrides: Any) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "line_start": 1,
        "line_end": 2,
        "trace_candidates": [],
    }
    finding.update(overrides)
    return finding


def test_fabrication_pressure_is_its_own_stop() -> None:
    """Non-empty `trace_candidates` STOPs even though the schema VALIDATES it and
    the parser ADMITS the finding — which is exactly why the third category
    exists: neither a shape failure nor an authenticity escape.

    Counted over ALL retained candidates — production keeps valid candidates from
    REJECTED proposals too, and the spec's rule is absolute.
    """
    body = _finding_body(trace_candidates=[{"import_string_raw": "app.db", "reason": "invented"}])
    row = _row(
        RowId.ACCEPTANCE_FINDING,
        content=body,
        parser=lambda _b: _parser_outcome(admitted=(_admitted(trace_candidate_count=1),)),
    )
    assert row.assessment is not None
    assert row.assessment.nonempty_trace_candidates is True

    assert _session(refusals=_negatives(*REFUSAL_ROWS), finding=row).verdict is (
        Verdict.STOP_FABRICATION
    )


def test_parser_rejected_expected_finding_is_inconclusive_not_a_pass() -> None:
    """THE admitted-vs-proposed finding. A schema-valid proposal the parser
    REJECTED means the expected finding was never admitted, so the
    populated-branch question is unanswered.

    Reading the raw `findings` array instead let this read PARK: a rejected
    proposal silently satisfying the acceptance row while every STOP flag stayed
    false.
    """
    body = _finding_body(evidence_tier="observed", query_match_id="py.totally.invented")
    row = _row(
        RowId.ACCEPTANCE_FINDING,
        content=body,
        parser=lambda _b: _parser_outcome(rejected=("query_match_id_not_in_registry",)),
    )
    assert row.assessment is not None
    assert row.assessment.returned_any_finding is False  # nothing was ADMITTED
    assert row.assessment.fabricated_proof_rejected_by_parser is True
    assert row.finding_count == 1  # the body DID propose one

    assert _session(refusals=_negatives(*REFUSAL_ROWS), finding=row).verdict is (
        Verdict.INCONCLUSIVE
    )


def test_admission_not_proposal_drives_returned_any_finding() -> None:
    """The same body reads differently depending on what the parser ADMITTED, and
    an admitted finding at the WRONG span does not satisfy the expected identity."""
    body = _finding_body()
    admitted = _row(
        RowId.ACCEPTANCE_FINDING,
        content=body,
        parser=lambda _b: _parser_outcome(admitted=(_admitted(),)),
    )
    assert admitted.assessment is not None
    assert admitted.assessment.returned_any_finding is True

    wrong_span = _row(
        RowId.ACCEPTANCE_FINDING,
        content=body,
        parser=lambda _b: _parser_outcome(admitted=(_admitted(line_start=40, line_end=41),)),
    )
    assert wrong_span.assessment is not None
    assert wrong_span.assessment.returned_any_finding is False

    wrong_type = _row(
        RowId.ACCEPTANCE_FINDING,
        content=body,
        parser=lambda _b: _parser_outcome(admitted=(_admitted(finding_type="hardcoded_secret"),)),
    )
    assert wrong_type.assessment is not None
    assert wrong_type.assessment.returned_any_finding is False


def test_parser_surviving_fabrication_is_stop_authenticity() -> None:
    """The escape: an id the file's queries never produced, ADMITTED anyway.

    Reachable only if the parser regresses — the real parser rejects it every
    time, which is the negative control passing. Driven here by a parser outcome
    that reports admission, with the body still supplying the claimed id.
    """
    body = _finding_body(evidence_tier="observed", query_match_id="py.totally.invented")
    row = _row(
        RowId.ACCEPTANCE_FINDING,
        content=body,
        parser=lambda _b: _parser_outcome(
            admitted=(_admitted(query_match_id="py.totally.invented"),)
        ),
    )
    assert row.assessment is not None
    assert row.assessment.fabricated_proof_survived_parser is True

    assert _session(refusals=_negatives(*REFUSAL_ROWS), finding=row).verdict is (
        Verdict.STOP_AUTHENTICITY
    )


def test_derive_assessment_never_invents_a_flag() -> None:
    """A clean body yields all-False flags regardless of what the parser says
    about unrelated proposals."""
    assessment = derive_assessment(
        body=_finding_body(),
        parser=_parser_outcome(admitted=(_admitted(),), rejected=("some_other_reason",)),
        fired_query_match_ids=_FIRED,
        expected=_EXPECTED,
    )
    assert assessment.returned_any_finding is True
    assert assessment.fabricated_proof_survived_parser is False
    assert assessment.fabricated_proof_rejected_by_parser is False
    assert assessment.nonempty_trace_candidates is False


def test_routing_precedence_is_authenticity_then_fabrication() -> None:
    """PURE routing over synthetic flags — the only place they are permitted, and
    scoped to `route_finding_assessment` rather than a session verdict."""
    both = FindingRowAssessment(
        returned_any_finding=True,
        fabricated_proof_survived_parser=True,
        fabricated_proof_rejected_by_parser=False,
        nonempty_trace_candidates=True,
    )
    assert route_finding_assessment(both) is Verdict.STOP_AUTHENTICITY


# --------------------------------------------------------------------------
# Session-level guards.
# --------------------------------------------------------------------------


def test_schema_admission_400_outranks_everything() -> None:
    """Nothing downstream was observed under a rejected request."""
    rows: dict[RowId, EvaluatedRow] = {
        RowId.ACCEPTANCE_CLEAN: classify_transport(row=RowId.ACCEPTANCE_CLEAN, status=400)
    }
    rows.update(_negatives(*REFUSAL_ROWS))
    assert classify_session(rows=rows).verdict is Verdict.STOP_SHAPE


def test_clean_row_with_findings_is_inconclusive() -> None:
    """`acceptance_clean` must carry EXACTLY zero findings — DERIVED from the body
    now, so a caller cannot assert zero over a body that has one."""
    noisy = _row(RowId.ACCEPTANCE_CLEAN, content=_finding_body())
    result = _session(refusals=_negatives(*REFUSAL_ROWS), clean=noisy)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "1 finding" in result.reason


def test_finding_row_outcome_is_gated_like_the_clean_row() -> None:
    """A row that returned no body cannot be assessed."""
    errored = classify_transport(row=RowId.ACCEPTANCE_FINDING, status=429)
    result = _session(refusals=_negatives(*REFUSAL_ROWS), finding=errored)
    assert result.verdict is Verdict.INCONCLUSIVE


def test_every_verdict_is_reachable() -> None:
    """No terminal verdict may exist in prose only — the check that caught
    `STOP-fabrication` being unreachable when it was first added."""
    reached = {
        _session(refusals=_negatives(*REFUSAL_ROWS)).verdict,  # PARK
        _session(refusals=_negatives(RowId.REFUSAL_1)).verdict,  # INCONCLUSIVE
        classify_session(
            rows={
                RowId.ACCEPTANCE_CLEAN: classify_transport(row=RowId.ACCEPTANCE_CLEAN, status=400)
            }
        ).verdict,  # STOP-shape
        _session(
            refusals=_negatives(*REFUSAL_ROWS),
            finding=_row(
                RowId.ACCEPTANCE_FINDING,
                content=_finding_body(
                    evidence_tier="observed", query_match_id="py.totally.invented"
                ),
                parser=lambda _b: _parser_outcome(
                    admitted=(_admitted(query_match_id="py.totally.invented"),)
                ),
            ),
        ).verdict,  # STOP-authenticity
        _session(
            refusals=_negatives(*REFUSAL_ROWS),
            finding=_row(
                RowId.ACCEPTANCE_FINDING,
                content=_finding_body(
                    trace_candidates=[{"import_string_raw": "a.b", "reason": "x"}]
                ),
                parser=lambda _b: _parser_outcome(admitted=(_admitted(trace_candidate_count=1),)),
            ),
        ).verdict,  # STOP-fabrication
    }
    refused = _negatives(RowId.REFUSAL_1)
    refused[RowId.REFUSAL_2] = _row(RowId.REFUSAL_2, refusal="no")
    reached.add(_session(refusals=refused).verdict)  # GO

    assert reached == set(Verdict)


def test_a_valid_refusal_is_not_scored_as_wrong_tier() -> None:
    """The interaction the refusal-precedence fix created.

    Session-level wrong-tier detection predates that rule and scanned EVERY row's
    content, so a genuine refusal whose explanatory `content` happened to contain
    wrong-tier-looking JSON returned `STOP-shape` — inverting the arc's own central
    finding. Wrong-tier is a CONTENT-arm fact.
    """
    wrong_tier_json = json.dumps(
        {"findings": [{"evidence_tier": "judged", "query_match_id": "q1"}]}
    )

    refused = EvaluatedRow(
        row=RowId.REFUSAL_1,
        content=wrong_tier_json,
        refusal="I cannot help with that.",
        finish_reason="stop",
    )
    assert refused.observation.kind is ObservationKind.VALID_REFUSAL
    assert refused.wrong_tier_proof is False

    # The IDENTICAL content with no refusal is still a shape failure.
    not_refused = EvaluatedRow(
        row=RowId.REFUSAL_1, content=wrong_tier_json, refusal=None, finish_reason="stop"
    )
    assert not_refused.wrong_tier_proof is True

    rows = dict(_negatives(RowId.REFUSAL_2, RowId.REFUSAL_3))
    rows[RowId.ACCEPTANCE_CLEAN] = _row(RowId.ACCEPTANCE_CLEAN, content=_EMPTY)
    rows[RowId.ACCEPTANCE_FINDING] = _row(
        RowId.ACCEPTANCE_FINDING,
        content=_finding_body(),
        parser=lambda _b: _parser_outcome(admitted=(_admitted(),)),
    )
    rows[RowId.REFUSAL_1] = refused
    assert classify_session(rows=rows).verdict is Verdict.GO


def test_retained_candidate_from_a_rejected_proposal_is_stop_fabrication() -> None:
    """The composition the previous round claimed closed but never pinned.

    The expected finding was REJECTED, so recall missed — but the response still
    induced a retained candidate. Global safety outranks recall, so this is
    `STOP-fabrication`, not `INCONCLUSIVE`.
    """
    row = _row(
        RowId.ACCEPTANCE_FINDING,
        content=_finding_body(),
        parser=lambda _b: _parser_outcome(rejected=("span_outside_scope_unit",), retained=1),
    )
    assert row.assessment is not None
    assert row.assessment.returned_any_finding is False
    assert row.assessment.nonempty_trace_candidates is True
    assert _session(refusals=_negatives(*REFUSAL_ROWS), finding=row).verdict is (
        Verdict.STOP_FABRICATION
    )


def test_unrelated_admitted_fabricated_proof_is_stop_authenticity() -> None:
    """Also claimed closed but never pinned. The expected finding is clean; an
    UNRELATED admitted finding carries an invented id. The spec says ANY fabricated
    proof surviving into an admitted finding is an escape, so span-scoping the
    check hid it."""
    row = _row(
        RowId.ACCEPTANCE_FINDING,
        content=_finding_body(),
        parser=lambda _b: _parser_outcome(
            admitted=(
                _admitted(),
                _admitted(
                    finding_type="hardcoded_secret",
                    line_start=9,
                    line_end=9,
                    query_match_id="python.not_registered",
                ),
            )
        ),
    )
    assert row.assessment is not None
    assert row.assessment.returned_any_finding is True
    assert row.assessment.fabricated_proof_survived_parser is True
    assert _session(refusals=_negatives(*REFUSAL_ROWS), finding=row).verdict is (
        Verdict.STOP_AUTHENTICITY
    )


def test_retained_candidate_count_is_required_and_non_negative() -> None:
    """No default: a defaulted 0 let a caller silently omit the absolute
    fabrication fact."""
    with pytest.raises(TypeError):
        ParserOutcome(admitted=(), rejection_reasons=())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="negative"):
        ParserOutcome(admitted=(), rejection_reasons=(), retained_trace_candidate_count=-1)
