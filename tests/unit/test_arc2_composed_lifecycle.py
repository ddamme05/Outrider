"""Arc 2 — the COMPOSED lifecycle, end to end, at zero spend.

Every other Arc-2 test exercises one owner in isolation. That is exactly how a
forged `GO` shipped twice: each owner was judged locally, and the
producer→consumer composition — the only place the guarantees actually have to
hold together — was never executed.

This module drives the full path:

    LockedProbeContract -> DryRunManifest -> PaidProbeContract
        -> fixtures on disk -> AttemptLedger -> verify_and_derive -> verdict

and then attacks it. Each attack mutates ONE fact and requires the replay to
refuse or to derive a different verdict. The headline regression is
`test_forged_ledger_content_cannot_manufacture_a_go`, which reproduces the
demonstrated attack: honest fixtures containing no refusal, a ledger claiming
one, every hash matching, verdict `GO`.

No network call is possible here — the verifier only reads files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 - runtime fixture type
from typing import Any

import pytest
from spikes.openai.arc2 import plan
from spikes.openai.arc2.attempts import (
    ATTEMPT_ADAPTER,
    AttemptLedger,
    AttemptRef,
    CaptureShapeFailure,
    CompletedCapture,
    LedgerViolationError,
    TransportFailure,
)
from spikes.openai.arc2.classifier import ROW_ORDER, RowId, Verdict
from spikes.openai.arc2.contracts import (
    STRICT_PROBE_CONTRACT_VERSION,
    ContractViolationError,
    DryRunManifest,
    EvaluationContract,
    LockedProbeContract,
    PaidProbeContract,
    RowMeasurement,
    digest_of,
)
from spikes.openai.arc2.plan import registered_evaluation_contract
from spikes.openai.arc2.verifier import ReplayError, verify_and_derive

from outrider.llm.raw_openai_capture import (
    RawCapture,
    RawCaptureShapeError,
    RawOpenAICaptureError,
    RawUsage,
)

_SOURCE = (
    "def run_report(conn, user_id):\n"
    '    """Fetch a user\'s report rows."""\n'
    "    return conn.execute(\n"
    '        "SELECT * FROM reports WHERE user_id = \'" + user_id + "\'"\n'
    "    ).fetchall()\n"
)
_MODEL = "gpt-5.6-sol"
_SDK_RESPONSE_JSON = '{"note": "reserialized by the SDK; values only, not byte layout"}'
_EMPTY = '{"findings":[]}'


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _finding_body(**overrides: Any) -> str:
    f: dict[str, Any] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "SQL built by concatenation",
        "description": "User input is concatenated into a SQL string.",
        "evidence": "conn.execute(...)",
        "line_start": 4,
        "line_end": 4,
        "trace_candidates": [],
    }
    f.update(overrides)
    return json.dumps({"findings": [f]})


def _evaluation() -> EvaluationContract:
    """The REGISTERED plan — deliberately not a hand-copy of it.

    This used to restate all fourteen fields inline. That made the test file a
    SECOND authority on the experiment: it passed while agreeing with the plan and
    would have kept passing while agreeing with a stale one. Delegating means these
    tests exercise the same reconstruction `verify_and_derive` compares against.
    """
    return registered_evaluation_contract()


@pytest.fixture(autouse=True)
def _frozen_elicitations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run this whole file under a plan whose refusal prompts are FROZEN.

    `registered_locked_contract()` refuses while they hold the placeholder — that
    is Gate 1 working. These tests model the Gate-2 state, so they freeze the
    prompts and exercise everything downstream of that freeze. Patched on `plan`,
    the authority, never on the probe's re-export.
    """
    monkeypatch.setattr(
        plan,
        "_REFUSAL_ELICITATIONS",
        {
            RowId.REFUSAL_1: "frozen elicitation one",
            RowId.REFUSAL_2: "frozen elicitation two",
            RowId.REFUSAL_3: "frozen elicitation three",
        },
    )


def _contracts() -> tuple[EvaluationContract, LockedProbeContract, PaidProbeContract]:
    """The REGISTERED contracts, not fabricated stand-ins.

    An earlier version hand-built the locked contract with placeholder digests
    (`sha256(b"profile")`, `analyze_prompt_version="analyze-v10"`). That made every
    lifecycle test agree with a fiction: replay now reconstructs the generation
    plan, so a fabricated locked contract is exactly what it must reject.
    """
    ev = plan.registered_evaluation_contract()
    locked = plan.registered_locked_contract()
    measurements = plan.registered_measurements()
    dry = DryRunManifest(
        contract_version=STRICT_PROBE_CONTRACT_VERSION,
        locked_contract_digest=locked.digest,
        measurements=measurements,
    )
    paid = PaidProbeContract.from_reviewed(
        locked=locked,
        dry=dry,
        caps={m.row_id: m.prompt_bytes * 2 for m in measurements},
    )
    return ev, locked, paid


def _request_digest(row: RowId) -> str:
    """The row's REAL measured request digest — replay checks against this."""
    return next(m.request_body_digest for m in plan.registered_measurements() if m.row_id == row)


def _write(tmp: Path, attempt: Any) -> AttemptRef:
    """Persist the WHOLE attempt as its fixture and reference it by hash.

    The fixture is the evidence; the ref carries nothing but a pointer.
    """
    raw = attempt.model_dump_json().encode("utf-8")
    (tmp / f"{attempt.ordinal:02d}_{attempt.row.value}.json").write_bytes(raw)
    return AttemptRef(ordinal=attempt.ordinal, row=attempt.row, fixture_digest=_h(raw))


def _raw_capture(
    *, content: str | None, refusal: str | None, finish_reason: str | None, model: str | None
) -> RawCapture:
    """The wrapper-owned DTO a live capture would produce, built here by hand.

    Every field the probe reads is set explicitly so the projection below is
    exercised against a complete capture, not a partially-populated one.
    """
    return RawCapture(
        sdk_response_json=_SDK_RESPONSE_JSON,
        response_id="chatcmpl-abc123",
        response_model=model,
        created=1_700_000_000,
        content=content,
        refusal=refusal,
        finish_reason=finish_reason,
        service_tier="default",
        usage=RawUsage(
            prompt_tokens=2000,
            completion_tokens=50,
            total_tokens=2050,
            cached_tokens=None,
            cache_write_tokens=None,
        ),
    )


def _completed(
    i: int, row: RowId, *, content: str | None, refusal: str | None = None, **kw: Any
) -> CompletedCapture:
    """Build the attempt the way the paid path would: capture DTO first, then project.

    Deliberately NOT a direct `CompletedCapture(...)`. Routing every fixture in this
    file through `from_capture` is what keeps that constructor from being a
    plausible-looking surface nothing exercises.
    """
    return CompletedCapture.from_capture(
        _raw_capture(
            content=content,
            refusal=refusal,
            finish_reason=kw.pop("finish_reason", "stop"),
            model=kw.pop("response_model", _MODEL),
        ),
        ordinal=i,
        row=row,
        request_body_digest=kw.pop("request_body_digest", _request_digest(row)),
    )


def _park_session(tmp: Path) -> AttemptLedger:
    """All five rows complete; no refusal ever fires. Honest verdict: PARK."""
    refs = []
    for i, row in enumerate(ROW_ORDER):
        body = _finding_body() if row == RowId.ACCEPTANCE_FINDING else _EMPTY
        refs.append(_write(tmp, _completed(i, row, content=body)))
    return AttemptLedger(refs=tuple(refs))


def _replay(tmp: Path, ledger: AttemptLedger, **over: Any) -> Any:
    ev, _locked, paid = _contracts()
    return verify_and_derive(
        contract=over.pop("contract", paid),
        evaluation=over.pop("evaluation", ev),
        scenario_source=over.pop("scenario_source", _SOURCE),
        ledger=ledger,
        fixture_dir=tmp,
    )


# --------------------------------------------------------------------------
# The composition works at all.
# --------------------------------------------------------------------------


def test_full_lifecycle_derives_park_from_real_bytes(tmp_path: Path) -> None:
    """Contracts -> fixtures -> ledger -> replay -> verdict, with zero spend."""
    result = _replay(tmp_path, _park_session(tmp_path))
    assert result.verdict.verdict is Verdict.PARK
    assert result.attempts_replayed == len(ROW_ORDER)


def test_full_lifecycle_derives_go_when_a_refusal_really_fired(tmp_path: Path) -> None:
    """A refusal in the FIXTURE — the only place it can come from — yields GO,
    and the session correctly stops there."""
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY)),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=_finding_body())),
        _write(
            tmp_path,
            _completed(2, ROW_ORDER[2], content=None, refusal="I can't help with that."),
        ),
    ]
    result = _replay(tmp_path, AttemptLedger(refs=tuple(refs)))
    assert result.verdict.verdict is Verdict.GO
    assert result.verdict.discriminator_met is True


# --------------------------------------------------------------------------
# The demonstrated attack, and its siblings.
# --------------------------------------------------------------------------


def test_forged_ledger_content_cannot_manufacture_a_go(tmp_path: Path) -> None:
    """THE regression. Honest fixtures, a lying ledger, every hash matching.

    Previously the ledger carried its own copy of `content`/`refusal`, so the
    verifier hashed the fixture and graded the ledger. A `GO` was derivable from
    bytes containing no refusal anywhere. `AttemptRef` now carries no response
    facts at all, so there is nothing to forge — the claim cannot be expressed.
    """
    ledger = _park_session(tmp_path)
    assert _replay(tmp_path, ledger).verdict.verdict is Verdict.PARK

    # There is no field on a ref that could assert a refusal.
    assert set(AttemptRef.model_fields) == {"ordinal", "row", "fixture_digest"}

    # Now forge a fixture that is SEMANTICALLY COHERENT — it must still decode
    # through the closed union, or this test degenerates into "arbitrary byte
    # changes break the hash", which proves nothing about forgery. The fields live
    # under `capture`; an earlier version of this test edited top-level `content`
    # and `refusal`, which no longer exist, so it silently became that weaker test
    # the moment the capture was nested.
    victim = tmp_path / "02_refusal_elicit_1.json"
    forged = json.loads(victim.read_text())
    forged["capture"]["refusal"] = "I can't help with that."
    forged["capture"]["content"] = None
    forged_bytes = json.dumps(forged).encode()

    decoded = ATTEMPT_ADAPTER.validate_json(forged_bytes)
    assert isinstance(decoded, CompletedCapture)
    assert decoded.refusal == "I can't help with that."  # a coherent lie, not garbage

    # And it is refused anyway, because the ref pins the ORIGINAL bytes.
    victim.write_bytes(forged_bytes)
    with pytest.raises(ReplayError, match="do not match the ledger"):
        _replay(tmp_path, ledger)

    # Re-pointing the ref at the forgery's own hash and KEEPING the later attempts
    # is refused — but for the automaton's reason, not the hash's: a demonstrated
    # refusal is terminal, so attempts after it are unexplained spend. See
    # `test_hashes_bind_evidence_to_the_ledger_not_to_its_origin` for what this
    # does NOT establish.
    repointed = AttemptLedger(
        refs=tuple(
            r.model_copy(update={"fixture_digest": _h(forged_bytes)}) if r.ordinal == 2 else r
            for r in ledger.refs
        )
    )
    with pytest.raises(ReplayError, match="follows a terminal state"):
        _replay(tmp_path, repointed)


def test_fixture_that_disagrees_with_its_reference_is_refused(tmp_path: Path) -> None:
    """A fixture may not identify itself as a different attempt than its ref."""
    refs = [_write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY))]
    swapped = _completed(1, ROW_ORDER[1], content=_EMPTY)
    raw = swapped.model_dump_json().encode()
    (tmp_path / "01_acceptance_finding.json").write_bytes(raw)
    refs.append(AttemptRef(ordinal=1, row=ROW_ORDER[1], fixture_digest=_h(raw)))
    # Re-point the ref at a fixture whose ordinal disagrees.
    bad = _completed(4, ROW_ORDER[1], content=_EMPTY)
    raw2 = bad.model_dump_json().encode()
    (tmp_path / "01_acceptance_finding.json").write_bytes(raw2)
    refs[1] = AttemptRef(ordinal=1, row=ROW_ORDER[1], fixture_digest=_h(raw2))
    with pytest.raises(ReplayError, match="identifies itself as attempt"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_response_to_a_different_request_is_refused(tmp_path: Path) -> None:
    """A response must answer the REVIEWED request for its row."""
    refs = [
        _write(
            tmp_path,
            _completed(
                0, ROW_ORDER[0], content=_EMPTY, request_body_digest=_h(b"some-other-request")
            ),
        )
    ]
    with pytest.raises(ContractViolationError, match="does not belong to this contract"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_response_from_a_different_model_is_refused(tmp_path: Path) -> None:
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY, response_model="gpt-5.6-luna"))
    ]
    with pytest.raises(ReplayError, match="was answered by"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_mismatched_evaluation_identity_is_refused(tmp_path: Path) -> None:
    """The grader must be the one the capture was taken under."""
    other = _evaluation().model_copy(update={"active_policy_version": "9.9.9"})
    with pytest.raises(ContractViolationError, match=r"differs on \['active_policy_version'\]"):
        _replay(tmp_path, _park_session(tmp_path), evaluation=other)


def test_scenario_source_must_match_the_evaluation_digest(tmp_path: Path) -> None:
    """Replaying against a different file would parse a different program."""
    with pytest.raises(ContractViolationError, match="not the registered defect scenario"):
        _replay(tmp_path, _park_session(tmp_path), scenario_source=_SOURCE + "# edited\n")


def test_attempt_after_a_terminal_state_is_refused(tmp_path: Path) -> None:
    """A demonstrated refusal ends the session; a later call is unexplained spend.

    This is SEMANTIC — the ledger cannot see it, because it requires the decoded
    envelope. It is enforced during sequential replay.
    """
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY)),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=_finding_body())),
        _write(tmp_path, _completed(2, ROW_ORDER[2], content=None, refusal="I can't.")),
        _write(tmp_path, _completed(3, ROW_ORDER[3], content=_EMPTY)),
    ]
    with pytest.raises(ReplayError, match="follows a terminal state"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_attempt_after_a_failed_acceptance_row_is_refused(tmp_path: Path) -> None:
    """An unmet prerequisite cannot recover — the ledger permits the shape, the
    replay refuses it once the envelope is known."""
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content="not json")),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=_finding_body())),
    ]
    with pytest.raises(ReplayError, match="follows a terminal state"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_transport_failure_then_another_call_is_refused(tmp_path: Path) -> None:
    first = _completed(0, ROW_ORDER[0], content=_EMPTY)
    fail = TransportFailure(
        ordinal=1,
        row=ROW_ORDER[1],
        request_body_digest=_request_digest(ROW_ORDER[1]),
        status=429,
    )
    refs = [
        _write(tmp_path, first),
        _write(tmp_path, fail),
        _write(tmp_path, _completed(2, ROW_ORDER[2], content=_EMPTY)),
    ]
    with pytest.raises(ReplayError, match="follows a terminal state"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_capture_shape_failure_on_a_refusal_row_may_continue(tmp_path: Path) -> None:
    """The legal partial: a projection failure on a refusal row does not end the
    session, because later refusal rows are independent elicitations."""
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY)),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=_finding_body())),
        _write(
            tmp_path,
            CaptureShapeFailure(
                ordinal=2,
                row=ROW_ORDER[2],
                request_body_digest=_request_digest(ROW_ORDER[2]),
                reason="empty choices",
            ),
        ),
        _write(tmp_path, _completed(3, ROW_ORDER[3], content=None, refusal="I can't.")),
    ]
    result = _replay(tmp_path, AttemptLedger(refs=tuple(refs)))
    assert result.verdict.verdict is Verdict.GO


def test_out_of_order_or_retried_ledger_is_refused_structurally(tmp_path: Path) -> None:
    a = _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY))
    b = _write(tmp_path, _completed(1, ROW_ORDER[1], content=_finding_body()))
    with pytest.raises(LedgerViolationError, match="frozen prefix"):
        AttemptLedger(
            refs=(b.model_copy(update={"ordinal": 0}), a.model_copy(update={"ordinal": 1}))
        )
    with pytest.raises(LedgerViolationError, match="retries are forbidden"):
        AttemptLedger(refs=(a, a.model_copy(update={"ordinal": 1})))


# --------------------------------------------------------------------------
# Contract lineage cannot be bypassed.
# --------------------------------------------------------------------------


def test_paid_contract_cannot_be_forged_by_direct_construction() -> None:
    """`from_reviewed` was the intended sole constructor, but direct construction
    stayed callable with arbitrary identity. Lineage is now re-derived on every
    load, so a swapped field fails validation rather than being accepted."""
    _ev, _locked, paid = _contracts()
    with pytest.raises(Exception, match="identity is copied, not supplied"):
        PaidProbeContract(**{**paid.model_dump(), "model": "gpt-5.6-luna"})


def test_paid_contract_rejects_an_unrelated_dry_run() -> None:
    ev, locked, _paid = _contracts()
    other_locked = locked.model_copy(update={"model": "gpt-5.6-luna"})
    dry = DryRunManifest(
        contract_version=STRICT_PROBE_CONTRACT_VERSION,
        locked_contract_digest=other_locked.digest,
        measurements=tuple(
            RowMeasurement(
                row_id=r,
                prompt_bytes=32_000,
                estimated_tokens=8_000,
                request_body_digest=_request_digest(r),
            )
            for r in ROW_ORDER
        ),
    )
    with pytest.raises(ContractViolationError, match="does not authorize"):
        PaidProbeContract.from_reviewed(locked=locked, dry=dry, caps={r: 40_000 for r in ROW_ORDER})
    assert ev is not None


def test_unknown_contract_version_fails_to_load() -> None:
    """An artifact carrying an unrecognised version must not merely compare
    unequal — it must fail to load at all."""
    _ev, locked, _paid = _contracts()
    with pytest.raises(Exception, match="contract_version"):
        LockedProbeContract(**{**locked.model_dump(), "contract_version": "arc2-strict-schema:2"})


def test_a_capture_that_cannot_identify_itself_is_refused() -> None:
    """Identity is a precondition for grading, not a nice-to-have.

    `RawCapture` types `response_model` as `str | None` because the wrapper reports
    what the API sent, absence included. The attempt narrows it to non-empty: a
    response that will not say which model produced it cannot be checked against a
    model-bound contract, so it is refused at projection rather than reaching the
    verifier as an attempt whose model comparison silently has nothing to compare.
    """
    for missing in ("response_model", "response_id"):
        capture = _raw_capture(
            content=_EMPTY, refusal=None, finish_reason="stop", model=_MODEL
        ).model_copy(update={missing: None})
        with pytest.raises(LedgerViolationError, match="cannot identify itself"):
            CompletedCapture.from_capture(
                capture,
                ordinal=0,
                row=RowId.ACCEPTANCE_CLEAN,
                request_body_digest=_request_digest(RowId.ACCEPTANCE_CLEAN),
            )


def test_projection_preserves_a_null_refusal_rather_than_blanking_it() -> None:
    """The one collapse that would silently destroy the arc's whole signal.

    A `None` refusal and an empty-string refusal are different observations; the
    probe exists to tell them apart. `from_capture` must pass `None` through.
    """
    attempt = CompletedCapture.from_capture(
        _raw_capture(content=_EMPTY, refusal=None, finish_reason="stop", model=_MODEL),
        ordinal=0,
        row=RowId.ACCEPTANCE_CLEAN,
        request_body_digest=_request_digest(RowId.ACCEPTANCE_CLEAN),
    )
    assert attempt.refusal is None


def test_shape_error_is_a_failed_observation_not_park_evidence(tmp_path: Path) -> None:
    """`PARK` needs three COMPLETED valid negatives — a projection failure is not one.

    This is the composed form of the claim: the spec deferred it because there was no
    `CaptureShapeFailure` representation to test against. Now there is. A session
    where one refusal row failed to project and the other two came back clean has
    only TWO demonstrated negatives, so it cannot conclude that refusals do not fire
    for this schema; it is `INCONCLUSIVE`.

    The failure mode this forbids is the tempting one: counting "no refusal was seen
    on that row" as evidence that no refusal occurred, when in fact nothing about
    that row was observed at all.
    """
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY)),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=_finding_body())),
        _write(
            tmp_path,
            CaptureShapeFailure(
                ordinal=2,
                row=ROW_ORDER[2],
                request_body_digest=_request_digest(ROW_ORDER[2]),
                reason="empty choices",
            ),
        ),
        _write(tmp_path, _completed(3, ROW_ORDER[3], content=_EMPTY)),
        _write(tmp_path, _completed(4, ROW_ORDER[4], content=_EMPTY)),
    ]
    result = _replay(tmp_path, AttemptLedger(refs=tuple(refs)))
    assert result.verdict.verdict is Verdict.INCONCLUSIVE

    # Control: swap the unprojectable row for a clean one and the SAME session parks.
    # Without this the assertion above would also pass if the fixture were simply
    # unable to reach PARK for an unrelated reason.
    parked = _replay(tmp_path, _park_session(tmp_path))
    assert parked.verdict.verdict is Verdict.PARK


def _contracts_for(ev: EvaluationContract) -> PaidProbeContract:
    """A self-consistent chain citing `ev` — otherwise identical to the real plan.

    Only the evaluation digest differs, so a refusal can only be attributed to the
    evaluation arm. Fabricating the generation digests too would let the test pass
    off the generation check instead.
    """
    locked = plan.registered_locked_contract().model_copy(
        update={"evaluation_contract_digest": ev.digest}
    )
    measurements = plan.registered_measurements()
    dry = DryRunManifest(
        contract_version=STRICT_PROBE_CONTRACT_VERSION,
        locked_contract_digest=locked.digest,
        measurements=measurements,
    )
    return PaidProbeContract.from_reviewed(
        locked=locked, dry=dry, caps={m.row_id: m.prompt_bytes * 2 for m in measurements}
    )


def test_a_self_consistent_forged_evaluation_cannot_whitelist_a_fabricated_query_id(
    tmp_path: Path,
) -> None:
    """The attack the digest check alone could NOT stop.

    Every earlier identity test mutated a field and left the chain inconsistent, so
    the mismatch was caught by a digest comparison. This attacker is careful: they
    author an evaluation contract naming a `query_match_id` that no query produces,
    then build the locked / dry / paid chain citing THAT contract's digest. Nothing
    disagrees with anything — the document authenticates itself.

    It matters because `fired_query_match_ids` becomes the real parser's OBSERVED
    allowlist. A fabricated id inside it is admitted as authentic structural proof,
    which is an `evidence-tier-schema-enforced` violation manufactured through the
    instrument built to detect exactly that.

    The defense is that the plan is RECONSTRUCTED, so there is nothing to forge
    against — only something to differ from.
    """
    forged = _evaluation().model_copy(
        update={"fired_query_match_ids": ("python.no_such_rule_exists",)}
    )
    assert forged.digest != _evaluation().digest
    paid = _contracts_for(forged)  # self-consistent chain, citing the forgery

    with pytest.raises(ContractViolationError, match=r"differs on \['fired_query_match_ids'\]"):
        verify_and_derive(
            contract=paid,
            evaluation=forged,
            scenario_source=_SOURCE,
            ledger=_park_session(tmp_path),
            fixture_dir=tmp_path,
        )


def test_the_forged_allowlist_would_have_admitted_fabricated_proof(tmp_path: Path) -> None:
    """Why the test above is a security test and not a consistency test.

    Runs the REAL parser under the forged allowlist and shows the fabricated
    `query_match_id` is admitted as an OBSERVED finding. Without this, the refusal
    above proves only that two objects differ; with it, the thing being refused is
    demonstrably a fabricated-proof admission.
    """
    from spikes.openai.arc2.verifier import ParserAdapter

    fabricated = "python.no_such_rule_exists"
    forged = _evaluation().model_copy(update={"fired_query_match_ids": (fabricated,)})
    outcome = ParserAdapter(forged, _SOURCE)(
        _finding_body(evidence_tier="observed", query_match_id=fabricated)
    )
    assert [f.query_match_id for f in outcome.admitted] == [fabricated], (
        "the forged allowlist must actually admit the fabricated id — otherwise the "
        "refusal test above is guarding a threat that does not exist"
    )

    # And the registered plan rejects the very same body.
    honest = ParserAdapter.for_registered_plan()(
        _finding_body(evidence_tier="observed", query_match_id=fabricated)
    )
    assert honest.admitted == ()
    assert honest.rejection_reasons


def test_clean_row_returning_findings_is_terminal(tmp_path: Path) -> None:
    """An acceptance row that already determined the verdict ends the session.

    The automaton used to check only that the envelope was gradeable, so a clean
    row that came back WITH findings — the schema-admission premise already
    failed — was followed by three more paid refusal calls. The final verdict was
    still right, which is exactly why this was invisible without a spend-shaped test.
    """
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_finding_body())),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=_finding_body())),
    ]
    with pytest.raises(ReplayError, match="follows a terminal state"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_finding_row_missing_the_expected_defect_is_terminal(tmp_path: Path) -> None:
    """Zero admitted expected findings routes INCONCLUSIVE — no later call is owed."""
    off_target = _finding_body(line_start=1, line_end=1)
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY)),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=off_target)),
        _write(tmp_path, _completed(2, ROW_ORDER[2], content=_EMPTY)),
    ]
    with pytest.raises(ReplayError, match="follows a terminal state"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_finding_row_inducing_a_trace_candidate_is_terminal(tmp_path: Path) -> None:
    """Fabrication pressure is a STOP; continuing to spend after it is unexplained."""
    # The canonical trace-candidate shape is `import_string_raw` + `reason`. An
    # earlier draft invented `target`, which failed strict validation and made the
    # row terminal through the pre-existing "not gradeable" branch — so the test
    # passed while exercising nothing. Reverting the fold now fails it.
    with_candidate = _finding_body(
        trace_candidates=[{"import_string_raw": "app.db.session", "reason": "callee not in diff"}]
    )
    refs = [
        _write(tmp_path, _completed(0, ROW_ORDER[0], content=_EMPTY)),
        _write(tmp_path, _completed(1, ROW_ORDER[1], content=with_candidate)),
        _write(tmp_path, _completed(2, ROW_ORDER[2], content=_EMPTY)),
    ]
    with pytest.raises(ReplayError, match="follows a terminal state"):
        _replay(tmp_path, AttemptLedger(refs=tuple(refs)))


def test_retained_diagnostics_survive_the_round_trip(tmp_path: Path) -> None:
    """The wrapper's evidence reaches the fixture instead of being projected away.

    Driven through `from_error` / `from_capture` — the constructors the paid runner
    uses — against KNOWN literals. An earlier version built the models directly and
    compared each field to the object it came from, so it asserted round-trip
    identity rather than preservation: it passed unchanged when the projection
    dropped a field on both sides, and left `from_error` entirely unexercised. A
    mutation that nulled every diagnostic did not fail it.
    """
    attempt = _completed(0, ROW_ORDER[0], content=_EMPTY)
    decoded = ATTEMPT_ADAPTER.validate_json(attempt.model_dump_json())
    assert isinstance(decoded, CompletedCapture)
    assert decoded.capture.sdk_response_json == _SDK_RESPONSE_JSON
    assert decoded.capture.usage.prompt_tokens == 2000
    assert decoded.capture.usage.completion_tokens == 50
    assert decoded.capture.service_tier == "default"
    assert decoded.capture.created == 1_700_000_000

    shape = CaptureShapeFailure.from_error(
        RawCaptureShapeError(
            reason="malformed openai response shape: TypeError: expected exactly one choice, got 2",
            sdk_response_json='{"choices": [{}, {}]}',
        ),
        ordinal=0,
        row=ROW_ORDER[0],
        request_body_digest=_request_digest(ROW_ORDER[0]),
    )
    round_tripped = ATTEMPT_ADAPTER.validate_json(shape.model_dump_json())
    assert isinstance(round_tripped, CaptureShapeFailure)
    assert round_tripped.sdk_response_json == '{"choices": [{}, {}]}'
    assert "expected exactly one choice" in round_tripped.reason

    transport = TransportFailure.from_error(
        RawOpenAICaptureError(status=503, request_id="req_abc123", message="upstream unavailable"),
        ordinal=0,
        row=ROW_ORDER[0],
        request_body_digest=_request_digest(ROW_ORDER[0]),
    )
    decoded_t = ATTEMPT_ADAPTER.validate_json(transport.model_dump_json())
    assert isinstance(decoded_t, TransportFailure)
    assert (decoded_t.status, decoded_t.request_id) == (503, "req_abc123")
    assert decoded_t.message == "upstream unavailable"


def test_transport_diagnostics_do_not_move_the_verdict(tmp_path: Path) -> None:
    """Retained vendor prose is inert. `STOP-shape` is positional and status-derived.

    Two sessions differing ONLY in the vendor message and request id — one of them
    prose engineered to read like a schema rejection — must produce the same
    verdict. Without this, adding diagnostics would have widened the classifier's
    input surface, which is the opposite of what retaining them is for.
    """
    verdicts = []
    for request_id, message in (
        (None, None),
        ("req_deadbeef", "invalid_request_error: response_format schema is not supported"),
    ):
        d = tmp_path / f"s-{request_id}"
        d.mkdir()
        ref = _write(
            d,
            TransportFailure(
                ordinal=0,
                row=ROW_ORDER[0],
                request_body_digest=_request_digest(ROW_ORDER[0]),
                status=400,
                request_id=request_id,
                message=message,
            ),
        )
        result = _replay(d, AttemptLedger(refs=(ref,)))
        verdicts.append(result.verdict.verdict)

    assert verdicts[0] is verdicts[1] is Verdict.STOP_SHAPE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_registry_digest", _h(b"a different registry")),
        ("parser_version", "analyze-parser-v0"),
        ("assessment_procedure_version", "arc2-strict-schema:0"),
        ("pass_index", 3),
        ("trace_candidate_form", "specifier"),
        ("fired_query_match_ids", ("python.no_such_rule_exists",)),
        ("scope_line_end", 99),
        ("expected_line_start", 1),
    ],
)
def test_every_previously_inert_identity_field_is_now_checked(
    tmp_path: Path, field: str, value: object
) -> None:
    """Each of these was recorded on the contract and read by nothing.

    `query_registry_digest`, `parser_version`, and `assessment_procedure_version`
    were never compared at all; `pass_index` and `trace_candidate_form` were stored
    and then dropped, so the parser ran under its own defaults while the contract
    claimed otherwise. A field that no code reads is not a guarantee — it is a
    comment with a type annotation.

    Parametrized rather than asserted in aggregate so a future field that slips out
    of the comparison fails on its own name.
    """
    forged = _evaluation().model_copy(update={field: value})
    with pytest.raises(ContractViolationError, match=rf"differs on \['{field}'\]"):
        verify_and_derive(
            contract=_contracts_for(forged),
            evaluation=forged,
            scenario_source=_SOURCE,
            ledger=_park_session(tmp_path),
            fixture_dir=tmp_path,
        )


def test_a_self_consistent_stale_generation_chain_is_refused(tmp_path: Path) -> None:
    """Evaluation currentness was consolidated; generation currentness was not.

    The attacker presents a chain built under an OLD strict schema (or old profile
    contract, or old prompt version). Every internal relationship holds: the dry run
    cites that locked contract, the paid contract embeds both, and `from_reviewed`
    re-derives the lineage cleanly — because lineage is internal, and every link in
    a stale chain agrees with every other link in the same stale chain.

    Graded by today's parser, that capture would answer a question nobody asked: it
    measures whether the API accepted a schema the probe no longer sends.
    """
    for field, value in (
        ("strict_schema_digest", _h(b"a strict schema from last week")),
        ("profile_contract_digest", _h(b"a profile contract from last week")),
        ("analyze_prompt_version", "analyze-v9"),
        ("max_completion_tokens", 4096),
    ):
        stale_locked = plan.registered_locked_contract().model_copy(update={field: value})
        measurements = plan.registered_measurements()
        stale_dry = DryRunManifest(
            contract_version=STRICT_PROBE_CONTRACT_VERSION,
            locked_contract_digest=stale_locked.digest,
            measurements=measurements,
        )
        # Self-consistent: this construction is accepted, lineage and all.
        stale_paid = PaidProbeContract.from_reviewed(
            locked=stale_locked,
            dry=stale_dry,
            caps={m.row_id: m.prompt_bytes * 2 for m in measurements},
        )
        with pytest.raises(ContractViolationError, match=rf"differs on \['{field}'\]"):
            verify_and_derive(
                contract=stale_paid,
                evaluation=_evaluation(),
                scenario_source=_SOURCE,
                ledger=_park_session(tmp_path),
                fixture_dir=tmp_path,
            )


def test_fabricated_request_measurements_are_refused(tmp_path: Path) -> None:
    """A manifest can cite the CURRENT locked contract and still lie about requests.

    `from_reviewed` binds `measurements == dry_run.measurements` and the dry run to
    the locked digest — all internal. Nothing recomputed the request digests, and
    replay checks each attempt against exactly those, so a response to a different
    prompt would be graded under this contract. Re-measuring from the plan's own
    request builders is what closes it.
    """
    real = plan.registered_measurements()
    fabricated = (
        real[0].model_copy(update={"request_body_digest": _h(b"a request never built")}),
        *real[1:],
    )
    locked = plan.registered_locked_contract()
    dry = DryRunManifest(
        contract_version=STRICT_PROBE_CONTRACT_VERSION,
        locked_contract_digest=locked.digest,
        measurements=fabricated,
    )
    paid = PaidProbeContract.from_reviewed(
        locked=locked, dry=dry, caps={m.row_id: m.prompt_bytes * 2 for m in fabricated}
    )
    with pytest.raises(ContractViolationError, match="do not match the requests the plan builds"):
        verify_and_derive(
            contract=paid,
            evaluation=_evaluation(),
            scenario_source=_SOURCE,
            ledger=_park_session(tmp_path),
            fixture_dir=tmp_path,
        )


def test_registry_digest_is_the_canonical_one_not_an_id_hash() -> None:
    """Ids are stable while query bodies are not.

    The plan hashed `sorted(structural_query_ids_for("python"))`, so editing a
    `.scm` body — changing which spans a query matches, and therefore which
    OBSERVED claims are admissible — left this field untouched. The registry
    publishes a digest over the actual content; that is the one identity has to
    cite.
    """
    from outrider.queries import registry

    assert _evaluation().query_registry_digest == registry.QUERY_REGISTRY_DIGEST
    assert _evaluation().query_registry_digest != digest_of(
        sorted(registry.structural_query_ids_for("python"))
    ), "an id-only hash would not move when a query body changes"


@pytest.mark.parametrize("removed", [{"defect_line": None}, {"expected_finding_type": None}])
def test_a_scenario_without_a_planted_defect_cannot_build_a_plan(
    removed: dict[str, object],
) -> None:
    """The expected finding is derived from the scenario or the plan refuses to exist.

    `EXPECTED_FINDING` used to fall back to a hand-authored
    `ExpectedFinding("sql_injection", 1, 1)` when the scenario declared none — a
    second authority hiding under a module whose claim is "derived, never
    asserted", and one whose coordinates were wrong (the planted defect is on line
    4). A scenario that lost either half of its defect declaration would have been
    graded against a defect that is not in the file.

    Both halves are required: a defect line with no finding type, and a finding
    type with no defect line, are each an incomplete plan.
    """
    from spikes.openai.arc2.plan import _require_expected

    crippled = plan.DEFECT_SCENARIO.__class__(
        **{
            **{
                f: getattr(plan.DEFECT_SCENARIO, f)
                for f in (
                    "file_path",
                    "source",
                    "scope_name",
                    "scope_line_start",
                    "scope_line_end",
                    "defect_line",
                    "expected_finding_type",
                )
            },
            **removed,
        }
    )
    with pytest.raises(ValueError, match="declares no expected finding"):
        _require_expected(crippled)

    # The real scenario still yields the real coordinates — not the old fallback's.
    assert (plan.EXPECTED_FINDING.line_start, plan.EXPECTED_FINDING.line_end) == (4, 4)


def test_hashes_bind_evidence_to_the_ledger_not_to_its_origin(tmp_path: Path) -> None:
    """The limit of what any of this proves. Stated as a test so it cannot be forgotten.

    A local operator who reseals a forged fixture AND shortens the ledger to the
    legal prefix `(0, 1, 2)` gets a clean `GO`. Nothing here prevents that, and no
    amount of hashing could: the digests establish that the evidence graded is the
    evidence referenced, not that the bytes came from OpenAI. Origin authenticity
    would need a signature from the party that produced the response, which this
    arc does not have and does not claim.

    What the machinery does buy is that every step between capture and verdict is
    mechanical and re-derivable — an artifact cannot drift, be regraded under a
    different plan, or carry a verdict nobody recomputed. The operator remains the
    trust anchor, which is acceptable precisely because the operator is the only
    consumer (`DECISIONS.md#066`: one deployment, one org, operator equals user).

    A previous comment claimed re-pointing the hash "does not rescue" a forgery.
    That held only for the ledger it was tested against, which retained attempts
    after the newly-terminal refusal. Shorten it and the forgery succeeds.
    """
    ledger = _park_session(tmp_path)
    assert _replay(tmp_path, ledger).verdict.verdict is Verdict.PARK

    victim = tmp_path / "02_refusal_elicit_1.json"
    forged = json.loads(victim.read_text())
    forged["capture"]["refusal"] = "I can't help with that."
    forged["capture"]["content"] = None
    forged_bytes = json.dumps(forged).encode()
    victim.write_bytes(forged_bytes)

    resealed = AttemptLedger(
        refs=(
            ledger.refs[0],
            ledger.refs[1],
            ledger.refs[2].model_copy(update={"fixture_digest": _h(forged_bytes)}),
        )
    )
    assert _replay(tmp_path, resealed).verdict.verdict is Verdict.GO, (
        "an operator-authored artifact CAN produce a GO — if this ever stops being "
        "true, the arc has gained an origin-authenticity guarantee it never claimed, "
        "and the claim should be documented rather than discovered"
    )
