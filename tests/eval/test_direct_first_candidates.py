"""Check lossless candidate accounting and the mechanical assessment boundary."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from .direct_first_candidates import (
    CandidateAssessment,
    CandidateLedgerError,
    DirectCandidate,
    SourceCitation,
    build_ledger,
    record_assessment,
)
from .model_refresh import write_json


def _proposal(**changes):
    return {
        "file_path": "app.py",
        "title": "A compatibility issue",
        "description": "The changed argument can break this caller.",
        "evidence": "call(value)",
        "line_start": 2,
        "line_end": 2,
        "category": "compatibility",
        **changes,
    }


def test_ledger_preserves_occurrences_invalid_entries_and_native_incompatible_fields(tmp_path):
    proposal = _proposal(title="T" * 180, description="D" * 3500, evidence="E" * 3500)
    entries = [
        proposal,
        proposal,
        {"title": "incomplete"},
        None,
        _proposal(title="\ud800"),
        *[_proposal()] * 97,
    ]
    raw = json.dumps({"findings": entries}, indent=2)
    ledger = build_ledger(raw, source_id="capture:one", finish_reason="end_turn")

    assert ledger.raw_response == raw
    assert len(ledger.candidates) == len(entries) == 102
    assert len({item.candidate_id for item in ledger.candidates}) == 102
    assert ledger.diagnostics == ("direct-finding-limit-exceeded",)
    assert ledger.candidates[0].proposal.title == "T" * 180
    assert ledger.candidates[0].proposal.description == "D" * 3500
    assert ledger.candidates[0].proposal.evidence == "E" * 3500
    for candidate, entry in zip(ledger.candidates, entries, strict=True):
        assert json.loads(candidate.raw_proposal_json) == entry
    for index in (2, 3, 4):
        assert ledger.candidates[index].proposal is None
        assert ledger.candidates[index].errors
    exported = tmp_path / "seed.json"
    write_json(exported, ledger.model_dump(mode="json"))
    saved = json.loads(exported.read_text())
    assert json.loads(saved["candidates"][4]["raw_proposal_json"]) == entries[4]
    assert saved["raw_response"] == raw
    assert (
        build_ledger(raw, source_id="capture:one", finish_reason="stop").candidates
        == ledger.candidates
    )
    assert (
        build_ledger(raw, source_id="capture:two", finish_reason="stop").candidates
        != ledger.candidates
    )
    with pytest.raises(ValidationError):
        ledger.candidates[0].proposal.title = "rewritten"
    with pytest.raises(ValidationError):
        ledger.candidates = ()


@pytest.mark.parametrize(
    ("raw", "finish", "diagnostic", "count"),
    [
        ("not-json", "end_turn", "invalid-json-response", 0),
        ('{"findings":[],"findings":[]}', "end_turn", "invalid-json-response", 0),
        ('{"findings":[NaN]}', "end_turn", "invalid-json-response", 0),
        ('{"findings":null}', "end_turn", "invalid-response-envelope", 0),
        ('{"findings":[],"unexpected":true}', "end_turn", "unexpected-response-fields", 0),
        ('{"findings":[]}', "max_tokens", "discovery-did-not-finish-normally", 0),
        ('{"findings":[null]}', "max_tokens", "discovery-did-not-finish-normally", 1),
    ],
)
def test_unfinished_and_malformed_responses_never_become_clean(raw, finish, diagnostic, count):
    ledger = build_ledger(raw, source_id="saved", finish_reason=finish)
    assert ledger.raw_response == raw
    assert diagnostic in ledger.diagnostics
    assert len(ledger.candidates) == count


def test_assessments_append_without_changing_or_hiding_the_original_candidate():
    ledger = build_ledger(
        json.dumps({"findings": [_proposal(), None]}), source_id="saved", finish_reason="stop"
    )
    candidate = ledger.candidates[0]
    citation = SourceCitation(file_path="app.py", line_start=2, line_end=2, quote="call(value)")
    refutation = CandidateAssessment(
        candidate_id=candidate.candidate_id,
        status="refutes",
        reason="The investigator disputes the trigger.",
        citations=(citation,),
        origin="model",
    )
    updated = record_assessment(
        ledger,
        candidate.candidate_id,
        refutation,
        contents={"app.py": "first\ncall(value)\nlast\n"},
    )
    uncertain = CandidateAssessment(
        candidate_id=candidate.candidate_id,
        status="uncertain",
        reason="Caller behavior is not captured.",
        origin="scripted",
    )
    final = record_assessment(updated, candidate.candidate_id, uncertain, contents={})

    assert ledger.candidates[0].assessments == ()
    assert final.candidates[0].assessments == (refutation, uncertain)
    assert len(final.candidates) == 2
    assert final.candidates[1] == ledger.candidates[1]
    assert final.raw_response == ledger.raw_response
    assert final.candidates[0].model_dump(exclude={"assessments"}) == candidate.model_dump(
        exclude={"assessments"}
    )
    with pytest.raises(CandidateLedgerError, match="already recorded"):
        record_assessment(updated, candidate.candidate_id, refutation, contents={})


@pytest.mark.parametrize(
    ("path", "start", "end", "quote", "message"),
    [
        ("missing.py", 2, 2, "call(value)", "absent"),
        ("app.py", 4, 4, "call(value)", "exceeds"),
        ("app.py", 1, 1, "call(value)", "differs"),
        ("app.py", 2, 2, "invented(value)", "differs"),
        ("app.py", 2, 2, "call", "differs"),
    ],
)
def test_assessment_rejects_missing_or_mislocated_citations_without_mutation(
    path, start, end, quote, message
):
    ledger = build_ledger(
        json.dumps({"findings": [_proposal()]}), source_id="saved", finish_reason="stop"
    )
    candidate_id = ledger.candidates[0].candidate_id
    assessment = CandidateAssessment(
        candidate_id=candidate_id,
        status="supports",
        reason="A model claim, not proof.",
        citations=(SourceCitation(file_path=path, line_start=start, line_end=end, quote=quote),),
        origin="model",
    )
    with pytest.raises(CandidateLedgerError, match=message):
        record_assessment(
            ledger, candidate_id, assessment, contents={"app.py": "first\ncall(value)\nlast\n"}
        )
    assert ledger.candidates[0].assessments == ()


def test_missing_duplicate_or_changed_candidate_ids_cannot_replace_the_inventory():
    ledger = build_ledger(
        json.dumps({"findings": [_proposal(), _proposal()]}),
        source_id="saved",
        finish_reason="stop",
    )
    candidate_id = ledger.candidates[0].candidate_id
    assessment = CandidateAssessment(
        candidate_id=candidate_id,
        status="uncertain",
        reason="Not enough captured context.",
        origin="scripted",
    )
    with pytest.raises(CandidateLedgerError, match="existing candidate"):
        record_assessment(ledger, "0" * 64, assessment, contents={})
    for candidates in (
        ledger.candidates[:1],
        (ledger.candidates[0], ledger.candidates[0]),
        (
            DirectCandidate(**{**ledger.candidates[0].model_dump(), "raw_proposal_json": "null"}),
            ledger.candidates[1],
        ),
    ):
        altered = type(ledger)(**ledger.model_dump(exclude={"candidates"}), candidates=candidates)
        with pytest.raises(CandidateLedgerError, match="inventory differs"):
            record_assessment(altered, candidate_id, assessment, contents={})
    for status in ("supports", "refutes"):
        with pytest.raises(ValidationError, match="require captured-source citations"):
            CandidateAssessment(
                candidate_id=candidate_id,
                status=status,
                reason="Unsupported assertion.",
                origin="model",
            )
