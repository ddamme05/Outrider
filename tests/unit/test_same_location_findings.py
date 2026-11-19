"""Exercise admission and round aggregation for claims sharing one source span."""

import json
from datetime import UTC, datetime
from itertools import permutations
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.nodes.analyze_aggregate import fold_worker_outcomes
from outrider.agent.nodes.analyze_parser import parse_analyze_response
from outrider.audit.events import FindingEvent
from outrider.audit.persister import _lift_finding_event
from outrider.audit.replay import _verify_proof_boundary
from outrider.policy.versions import ACTIVE_POLICY_VERSION
from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
from outrider.schemas.triage_result import ReviewTier

# These two titles, descriptions and their shared quote come from the saved
# pyproject-hooks #163 response. The controls below change one field at a time.
_QUOTE = """@nox.session(name="docs-live")
def release(session: nox.Session) -> None:
    session.install("flit")
    session.run("flit", "publish")"""
_SOURCE = "\n" * 42 + _QUOTE + "\n"
_MISSING_SESSION = (
    "Publish workflow requests a nonexistent Nox session",
    "On a tag push, CI requests `nox -s release`, but the decorator registers the publishing "
    "function as `docs-live`. Nox cannot find the requested session, so the job fails before "
    "Flit can publish to PyPI.",
)
_DEFAULT_PUBLISH = (
    "Default Nox run can publish to PyPI",
    "The publishing session is included in Nox's default session set. Running plain `nox` "
    "for local checks can therefore attempt an upload and, if Flit has valid credentials, "
    "publish unintentionally. The `docs-live` name also makes an explicit invocation look "
    "like a documentation command rather than a release.",
)
_HISTORICAL_LOCATION_HASH = "3707f588012927582a9ae3eace9ab69fcde1fd2bdbfc50b9d4d88f985c9710fc"
_REVIEW_ID = uuid4()


def _review_round(*claims: tuple[str, str], previous: AnalysisRound | None = None) -> AnalysisRound:
    proposals = [
        {
            "finding_type": "correctness_bug",
            "evidence_tier": "judged",
            "query_match_id": None,
            "trace_path": None,
            "line_start": 43,
            "line_end": 46,
            "title": title,
            "description": description,
            "evidence": _QUOTE,
        }
        for title, description in claims
    ]
    pass_index = 0 if previous is None else previous.pass_index + 1
    prior = {} if previous is None else {"prior_findings": previous.findings}
    result = parse_analyze_response(
        json.dumps({"findings": proposals}),
        review_id=_REVIEW_ID,
        installation_id=1,
        file_path="noxfile.py",
        file_content=_SOURCE,
        file_byte_length=len(_SOURCE.encode()),
        included_scope_units=(),
        primary_line_ranges=((43, 46),),
        query_match_id_set=frozenset(),
        degraded_mode=False,
        pass_index=pass_index,
        active_policy_version=ACTIVE_POLICY_VERSION,
        **prior,
    )
    assert result.response_rejection is None
    assert result.proposal_rejections == ()
    assert result.counters.n_proposals_seen == len(proposals)
    outcome = AnalyzeWorkerOutcome(
        path="noxfile.py",
        pass_index=pass_index,
        source="parser",
        parse_status="clean",
        review_tier=ReviewTier.DEEP,
        n_proposals_seen=result.counters.n_proposals_seen,
        admitted_findings=result.admitted_findings,
    )
    now = datetime.now(UTC)
    return fold_worker_outcomes(
        (outcome,), pass_index=pass_index, started_at=now, ended_at=now
    ).round


def test_two_consequences_on_the_same_quote_survive_reordering_and_repetition() -> None:
    original = _review_round(_MISSING_SESSION, _DEFAULT_PUBLISH)
    reordered = _review_round(_DEFAULT_PUBLISH, _MISSING_SESSION, _DEFAULT_PUBLISH)
    hashes = {finding.title: finding.content_hash for finding in original.findings}
    assert set(hashes) == {_MISSING_SESSION[0], _DEFAULT_PUBLISH[0]}
    assert len(set(hashes.values())) == 2
    assert len(reordered.findings) == 2
    assert {finding.title: finding.content_hash for finding in reordered.findings} == hashes
    assert original.round_id == reordered.round_id


@pytest.mark.parametrize(
    "variant",
    [
        _MISSING_SESSION,
        ("A tag build cannot run the release session", _MISSING_SESSION[1]),
        (
            _MISSING_SESSION[0],
            "The tag job asks Nox for release, which is registered as docs-live.",
        ),
    ],
    ids=["exact-repeat", "title-rewritten", "description-rewritten"],
)
def test_duplicate_wording_does_not_create_another_finding(variant: tuple[str, str]) -> None:
    result = _review_round(_MISSING_SESSION, variant)
    assert len(result.findings) == 1


@pytest.mark.parametrize("siblings", [(), (_DEFAULT_PUBLISH,)], ids=["alone", "with-second-issue"])
def test_title_and_description_rewrites_in_one_reply(siblings: tuple[tuple[str, str], ...]) -> None:
    title, description = _MISSING_SESSION
    variants = (
        _MISSING_SESSION,
        ("A tag build cannot run the release session", description),
        (title, "The tag job asks Nox for release, which is registered as docs-live."),
    )
    expected = _review_round(_MISSING_SESSION, *siblings)
    for order in permutations(variants):
        result = _review_round(*order, *siblings)
        assert len(result.findings) == len(expected.findings)
        assert result.round_id == expected.round_id


def test_shared_title_cannot_merge_two_independently_repeated_descriptions() -> None:
    shared_title = "The publishing session is misconfigured"
    expected = _review_round(_MISSING_SESSION, _DEFAULT_PUBLISH)
    result = _review_round(
        (shared_title, _MISSING_SESSION[1]),
        (shared_title, _DEFAULT_PUBLISH[1]),
        _MISSING_SESSION,
        _DEFAULT_PUBLISH,
    )
    assert len(result.findings) == 2
    assert result.round_id == expected.round_id


def test_single_proposal_preserves_its_recorded_hash() -> None:
    [finding] = _review_round(_MISSING_SESSION).findings
    assert finding.content_hash == _HISTORICAL_LOCATION_HASH
    assert (
        finding.proposal_hash == "7b25e62189dc17ff074a890fc2952d754670b162df5364fd8ea2cce15a877dd8"
    )
    event = _lift_finding_event(finding, is_eval=True)
    assert "claim_key" not in event.model_dump()
    _verify_proof_boundary((FindingEvent.model_validate_json(event.model_dump_json()),))


def test_audit_replay_binds_each_retained_claim_to_its_own_hash() -> None:
    findings = _review_round(_MISSING_SESSION, _DEFAULT_PUBLISH).findings
    assert len(findings) == 2
    events = tuple(_lift_finding_event(finding, is_eval=True) for finding in findings)
    restored = tuple(FindingEvent.model_validate_json(event.model_dump_json()) for event in events)
    _verify_proof_boundary(restored)
    assert [event.finding_content_hash for event in restored] == [f.content_hash for f in findings]
    first, second = restored
    assert first.claim_key != second.claim_key
    with pytest.raises(ValidationError, match="finding_content_hash mismatch"):
        FindingEvent.model_validate({**first.model_dump(), "claim_key": second.claim_key})


def test_later_repeat_reuses_the_matching_claim_among_two_at_the_same_anchor() -> None:
    first = _review_round(_MISSING_SESSION, _DEFAULT_PUBLISH)
    assert len(first.findings) == 2
    original = next(f for f in first.findings if f.title == _MISSING_SESSION[0])
    revised = ("The release tag build cannot find its session", _MISSING_SESSION[1])
    [repeat] = _review_round(revised, previous=first).findings
    assert repeat.content_hash == original.content_hash
    assert repeat.proposal_hash != original.proposal_hash


def test_one_ambiguous_prior_claim_cannot_merge_two_current_claims() -> None:
    # One prior title matches the first claim, but its description matches the
    # second. Reusing that identity for both would discard one current warning.
    ambiguous = (_MISSING_SESSION[0], _DEFAULT_PUBLISH[1])
    other = ("Missing Flit prevents publishing", "The release command requires Flit.")
    prior = _review_round(ambiguous, other)
    assert len(prior.findings) == 2
    current = _review_round(_MISSING_SESSION, _DEFAULT_PUBLISH, previous=prior)
    assert {f.title for f in current.findings} == {_MISSING_SESSION[0], _DEFAULT_PUBLISH[0]}
