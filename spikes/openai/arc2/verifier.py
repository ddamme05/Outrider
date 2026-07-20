"""The SOLE verdict authority: decode the evidence, derive the conclusion.

For each ordered reference this module re-hashes the fixture, DECODES it through
the closed attempt union, and derives every response fact from that decoded
object. Then it advances one semantic automaton, rebuilds each `EvaluatedRow`,
and calls `classify_session`. The verdict is the RESULT of that replay.

**What went wrong before, twice.**

1. `verify_paid_capture` authenticated hashes and identity and then accepted
   whatever `verdict` the manifest carried — a resealed false `GO` passed.
2. Its replacement hashed the fixture but graded `attempt.content` supplied
   separately by the ledger, so honest bytes containing no refusal could be
   accompanied by a ledger claiming one. A forged `GO` was demonstrated.

Both had the same shape: two copies of one fact, only one of them checked. There
is now exactly one source for captured evidence (the fixture), one for generation
identity (the contract lineage), and one for grading (this module).

**Nothing verdict-affecting is a parameter.** The parser adapter and the expected
finding are DERIVED from the verified `EvaluationContract`, not accepted from the
caller — both change the verdict, so accepting them would reintroduce the
injectable-authority defect at the outermost layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - runtime keyword-only signature type
from typing import TYPE_CHECKING
from uuid import uuid4

from spikes.openai.arc2.attempts import (
    ATTEMPT_ADAPTER,
    AttemptLedger,
    CaptureShapeFailure,
    TransportFailure,
)
from spikes.openai.arc2.classifier import (
    AdmittedFinding,
    EvaluatedRow,
    ExpectedFinding,
    ObservationKind,
    ParserOutcome,
    RowId,
    SessionVerdict,
    classify_session,
    route_finding_assessment,
)
from spikes.openai.arc2.contracts import ContractViolationError

if TYPE_CHECKING:
    from spikes.openai.arc2.contracts import EvaluationContract, PaidProbeContract

__all__ = ["ParserAdapter", "ReplayError", "VerifiedSession", "verify_and_derive"]


class ReplayError(Exception):
    """The evidence could not be replayed, so no verdict can be derived from it."""


@dataclass(frozen=True, slots=True)
class VerifiedSession:
    """A verdict DERIVED from replayed evidence, plus what it was derived from."""

    verdict: SessionVerdict
    contract_digest: str
    evaluation_digest: str
    attempts_replayed: int


class ParserAdapter:
    """Runs the REAL production parser under a VERIFIED evaluation identity.

    Module-owned and constructed FROM the contract, so a caller cannot hand the
    verifier a parser that reports whatever it likes. Every input the parser needs
    comes from the evaluation contract the artifact was captured under, and the
    scenario source is checked against that contract's digest first.
    """

    def __init__(self, evaluation: EvaluationContract, source: str) -> None:
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != evaluation.scenario_source_digest:
            msg = (
                "scenario source does not match the evaluation contract's digest: the "
                "parser would run against a different file than the capture was taken on"
            )
            raise ContractViolationError(msg)
        self._ev = evaluation
        self._source = source

    @classmethod
    def for_registered_plan(cls) -> ParserAdapter:
        """Build against the RECONSTRUCTED plan, never a supplied one."""
        from spikes.openai.arc2.plan import DEFECT_SCENARIO, registered_evaluation_contract

        return cls(registered_evaluation_contract(), DEFECT_SCENARIO.source)

    def __call__(self, body: str) -> ParserOutcome:
        from outrider.agent.nodes.analyze_parser import parse_analyze_response
        from outrider.ast_facts import ScopeUnit, compute_unit_id

        ev = self._ev
        byte_len = len(self._source.encode("utf-8"))
        scope = ScopeUnit(
            unit_id=compute_unit_id(
                ev.scenario_file_path, kind="function", qualified_name=ev.scope_name
            ),
            kind="function",
            name=ev.scope_name,
            qualified_name=ev.scope_name,
            file_path=ev.scenario_file_path,
            line_start=ev.scope_line_start,
            line_end=ev.scope_line_end,
            byte_start=0,
            byte_end=byte_len,
        )
        result = parse_analyze_response(
            body,
            review_id=uuid4(),
            installation_id=0,
            file_path=ev.scenario_file_path,
            file_content=self._source,
            file_byte_length=byte_len,
            included_scope_units=(scope,),
            query_match_id_set=frozenset(ev.fired_query_match_ids),
            degraded_mode=ev.degraded_mode,
            active_policy_version=ev.active_policy_version,
            # Previously recorded on the contract and then dropped — the parser ran
            # under its own defaults while the contract claimed these values, so the
            # two could disagree silently. Threaded through so the contract describes
            # the run that actually happened.
            pass_index=ev.pass_index,
            trace_candidate_form=ev.trace_candidate_form,
        )
        return ParserOutcome(
            admitted=tuple(
                AdmittedFinding(
                    finding_type=str(f.finding_type),
                    line_start=f.line_start,
                    line_end=f.line_end,
                    query_match_id=f.query_match_id,
                    trace_candidate_count=sum(
                        1
                        for c in result.trace_candidates
                        if c.source_proposal_hash == f.proposal_hash
                    ),
                )
                for f in result.admitted_findings
            ),
            rejection_reasons=tuple(r.rejection_reason for r in result.proposal_rejections),
            retained_trace_candidate_count=len(result.trace_candidates),
        )


def _check_counts(outcome: ParserOutcome) -> None:
    """Strict observed-count rules, enforced where counts are actually observed.

    `bool` is excluded explicitly: it subclasses `int`, so an annotation alone
    admits `True` as a count — one of the holes a plain dataclass left open.
    """
    counts = [outcome.retained_trace_candidate_count] + [
        f.trace_candidate_count for f in outcome.admitted
    ]
    for c in counts:
        if isinstance(c, bool) or type(c) is not int:
            msg = f"observed count {c!r} is not a plain int"
            raise ReplayError(msg)
        if c < 0:
            msg = f"observed count {c} is negative"
            raise ReplayError(msg)
    admitted_total = sum(f.trace_candidate_count for f in outcome.admitted)
    if admitted_total > outcome.retained_trace_candidate_count:
        msg = (
            f"admitted trace candidates ({admitted_total}) exceed the total retained "
            f"({outcome.retained_trace_candidate_count}): parts cannot exceed the whole"
        )
        raise ReplayError(msg)


_ACCEPTANCE = (RowId.ACCEPTANCE_CLEAN, RowId.ACCEPTANCE_FINDING)


def _terminal_state(row: RowId, evaluated: EvaluatedRow, negatives: int) -> str | None:
    """What, if anything, this DERIVED row makes terminal.

    Semantic — it reads the classified envelope, which the ledger cannot see. An
    acceptance row that came back malformed, refused, or errored ends the session
    because its prerequisite is unmet; a demonstrated refusal ends it because the
    discriminator short-circuits and later calls would be unexplained spend.
    """
    if evaluated.transport_failed:
        return "transport failure"
    kind = evaluated.observation.kind
    if row in _ACCEPTANCE:
        if kind is not ObservationKind.VALID_NEGATIVE:
            return f"{row.value} did not yield a gradeable response"
        # A gradeable response is NOT automatically a passing one. Checking only
        # envelope validity let an acceptance row that already DETERMINED the
        # verdict — a clean row that returned findings, a finding row whose expected
        # defect was never admitted, a finding row that fabricated proof — be
        # followed by three more paid refusal calls. The final classifier still
        # reported the right verdict, so the bug was invisible in the verdict and
        # visible only in the spend.
        if row == RowId.ACCEPTANCE_CLEAN:
            if evaluated.finding_count != 0:
                return (
                    f"{row.value} returned {evaluated.finding_count} finding(s) on clean "
                    "code: the schema-admission premise already failed"
                )
            return None
        assessment = evaluated.assessment
        if assessment is None:
            return f"{row.value} yielded no assessment"
        routed = route_finding_assessment(assessment)
        if routed is not None:
            return f"{row.value} determined {routed.value}"
        return None
    if kind is ObservationKind.VALID_REFUSAL:
        return "valid refusal (GO)"
    if kind is ObservationKind.VALID_NEGATIVE and negatives + 1 == 3:
        return "three valid negatives (PARK)"
    return None


def verify_and_derive(
    *,
    contract: PaidProbeContract,
    evaluation: EvaluationContract,
    scenario_source: str,
    ledger: AttemptLedger,
    fixture_dir: Path,
) -> VerifiedSession:
    """Replay the session and DERIVE its verdict. Every step is mandatory.

    Identity before bytes, bytes before decoding, decoding before grading. Each
    stage refuses rather than degrading, because a partially verified artifact
    cannot support a durable `GO` or `PARK`.
    """
    # 0. The EXPERIMENT is reconstructed, not accepted. Checking only that the
    #    contract's digest matches the supplied evaluation would authenticate a
    #    document against itself: a self-consistent forgery whose
    #    `fired_query_match_ids` names a query that never fires would become the
    #    parser's OBSERVED allowlist and admit a fabricated `query_match_id`. Both
    #    the supplied evaluation AND the contract's citation are therefore compared
    #    against a plan rebuilt here from live sources.
    from spikes.openai.arc2.plan import (
        DEFECT_SCENARIO,
        registered_evaluation_contract,
        registered_locked_contract,
        registered_measurements,
    )

    registered = registered_evaluation_contract()
    if evaluation != registered:
        differing = sorted(
            name
            for name in type(registered).model_fields
            if getattr(evaluation, name) != getattr(registered, name)
        )
        msg = (
            f"evaluation contract does not match the registered plan (differs on "
            f"{differing}): the grader cannot be chosen by whoever presents the evidence"
        )
        raise ContractViolationError(msg)
    if scenario_source != DEFECT_SCENARIO.source:
        msg = "scenario source is not the registered defect scenario"
        raise ContractViolationError(msg)
    if contract.evaluation_contract_digest != registered.digest:
        msg = (
            f"contract cites evaluation identity {contract.evaluation_contract_digest!r} "
            f"but the registered plan is {registered.digest!r}: the capture was taken "
            "under a different grader than the one in force"
        )
        raise ContractViolationError(msg)

    # 0b. GENERATION currentness, reconstructed the same way. Evaluation identity
    #     alone was not enough: the locked contract's schema/profile/prompt digests
    #     and the reviewed request measurements were only ever checked against each
    #     OTHER. A self-consistent chain built under a STALE strict schema — or a
    #     manifest citing the current locked contract while carrying fabricated
    #     `request_body_digest` values — validated cleanly and was then graded by
    #     today's parser. `from_reviewed` cannot catch that; it verifies internal
    #     lineage, and every link in a stale chain agrees with every other.
    current_locked = registered_locked_contract()
    if contract.locked != current_locked:
        differing = sorted(
            name
            for name in type(current_locked).model_fields
            if getattr(contract.locked, name) != getattr(current_locked, name)
        )
        msg = (
            f"locked contract does not match the registered generation plan (differs on "
            f"{differing}): the capture was generated under a different experiment"
        )
        raise ContractViolationError(msg)
    current_measurements = registered_measurements()
    if contract.measurements != current_measurements:
        differing_rows = sorted(
            a.row_id.value
            for a, b in zip(contract.measurements, current_measurements, strict=False)
            if a != b
        )
        msg = (
            f"reviewed measurements do not match the requests the plan builds "
            f"(rows {differing_rows or ['<count mismatch>']}): replay would check each "
            "attempt against a request identity nothing recomputed"
        )
        raise ContractViolationError(msg)

    parser = ParserAdapter(evaluation, scenario_source)
    expected = ExpectedFinding(
        finding_type=evaluation.expected_finding_type,
        line_start=evaluation.expected_line_start,
        line_end=evaluation.expected_line_end,
    )
    reviewed_requests = {m.row_id: m.request_body_digest for m in contract.measurements}

    rows: dict[RowId, EvaluatedRow] = {}
    terminal: str | None = None
    negatives = 0

    for ref in ledger.refs:
        if terminal is not None:
            msg = (
                f"attempt {ref.ordinal} on {ref.row.value!r} follows a terminal state "
                f"({terminal}): the session should have stopped"
            )
            raise ReplayError(msg)

        # 1. Re-hash, then DECODE. The bytes are the only evidence; nothing about
        #    the response is read from the ledger.
        path = fixture_dir / f"{ref.ordinal:02d}_{ref.row.value}.json"
        if not path.is_file():
            msg = f"fixture for attempt {ref.ordinal} is missing at {path}"
            raise ReplayError(msg)
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != ref.fixture_digest:
            msg = (
                f"fixture bytes for attempt {ref.ordinal} do not match the ledger: "
                f"recorded {ref.fixture_digest!r}, on disk {actual!r}"
            )
            raise ReplayError(msg)
        try:
            attempt = ATTEMPT_ADAPTER.validate_json(raw)
        except Exception as exc:  # noqa: BLE001 - any decode failure is unreplayable
            msg = f"fixture {ref.ordinal} is not a valid attempt: {type(exc).__name__}"
            raise ReplayError(msg) from None
        if attempt.ordinal != ref.ordinal or attempt.row != ref.row:
            msg = (
                f"fixture {ref.ordinal} identifies itself as attempt {attempt.ordinal} on "
                f"{attempt.row.value!r}: the reference and the evidence disagree"
            )
            raise ReplayError(msg)

        # 2. The response must answer the REVIEWED request for this row.
        reviewed = reviewed_requests.get(attempt.row)
        if reviewed is None or attempt.request_body_digest != reviewed:
            msg = (
                f"attempt {attempt.ordinal} answered request "
                f"{attempt.request_body_digest!r}, but the reviewed contract row is "
                f"{reviewed!r}: this response does not belong to this contract"
            )
            raise ContractViolationError(msg)
        if attempt.row not in contract.per_row_prompt_byte_cap:
            msg = f"no reviewed cap for row {attempt.row.value!r}"
            raise ContractViolationError(msg)

        if isinstance(attempt, TransportFailure):
            evaluated = EvaluatedRow(
                row=attempt.row, transport_failed=True, transport_status=attempt.status
            )
        elif isinstance(attempt, CaptureShapeFailure):
            evaluated = EvaluatedRow(row=attempt.row)
        else:
            # 3. Refuse a response from a different model before grading it.
            if attempt.response_model != contract.model:
                msg = (
                    f"attempt {attempt.ordinal} was answered by "
                    f"{attempt.response_model!r}, but the contract is for "
                    f"{contract.model!r}"
                )
                raise ReplayError(msg)

            outcome: ParserOutcome | None = None
            if attempt.row == RowId.ACCEPTANCE_FINDING and attempt.content is not None:
                try:
                    json.loads(attempt.content)
                except (json.JSONDecodeError, ValueError):
                    outcome = None
                else:
                    outcome = parser(attempt.content)
                    _check_counts(outcome)

            evaluated = EvaluatedRow(
                row=attempt.row,
                content=attempt.content,
                refusal=attempt.refusal,
                finish_reason=attempt.finish_reason,
                parser=outcome,
                expected_finding=expected if attempt.row == RowId.ACCEPTANCE_FINDING else None,
                fired_query_match_ids=frozenset(evaluation.fired_query_match_ids),
            )

        rows[attempt.row] = evaluated
        terminal = _terminal_state(attempt.row, evaluated, negatives)
        if (
            attempt.row not in _ACCEPTANCE
            and evaluated.observation.kind is ObservationKind.VALID_NEGATIVE
        ):
            negatives += 1

    # 4. The verdict is PRODUCED here — never read from the artifact.
    return VerifiedSession(
        verdict=classify_session(rows=rows),
        contract_digest=contract.digest,
        evaluation_digest=evaluation.digest,
        attempts_replayed=len(ledger.refs),
    )
