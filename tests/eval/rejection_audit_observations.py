"""Read native admission and draft facts from saved workflow records.

Persisted candidate events establish the original parser disposition. Analysis
selection and completed review drafts are separate observations. Missing records
stay unknown, and an omitted finding never establishes a cap or publication loss.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from outrider.agent.native_candidates import candidate_event_id
from outrider.audit.events import AnalyzeCompletedEvent, NativeCandidateEvent
from outrider.policy.canonical import compute_response_hash
from outrider.schemas.native_candidate import NativeCandidateRef
from outrider.schemas.review_finding import ReviewFinding


def _payload(
    wrapper: dict[str, Any], event_type: str, request: dict[str, Any]
) -> dict[str, Any] | None:
    """Validate the persisted envelope without filling missing identity fields."""
    if not isinstance(wrapper, dict) or wrapper.get("kind") != "persisted-audit-event":
        return None
    payload = wrapper.get("payload")
    if not isinstance(payload, dict):
        if wrapper.get("event_type") == event_type:
            raise ValueError("persisted event has no payload")
        return None
    if event_type not in {wrapper.get("event_type"), payload.get("event_type")}:
        return None
    for key in ("event_type", "event_id", "review_id", "is_eval"):
        if key in wrapper and key in payload and wrapper[key] != payload[key]:
            raise ValueError(f"persisted event wrapper differs from its payload: {key}")
    for key in ("review_id", "is_eval"):
        for record in (wrapper, payload):
            if key in record and record[key] != request[key]:
                raise ValueError(f"persisted event differs from its source request: {key}")
    if any(
        key not in record
        for record in (wrapper, payload)
        for key in ("event_type", "event_id", "review_id", "is_eval", "timestamp")
    ):
        return None
    if type(wrapper["is_eval"]) is not bool or type(payload["is_eval"]) is not bool:
        raise ValueError("persisted event is_eval must be a Boolean")
    # Historical exports leave the outer phase column NULL. The payload carries
    # the worker phase; comparing those two columns would reject real records.
    return payload


def _exchange_identity(exchange: dict[str, Any]) -> bool:
    """Require the recorded request, response and call to identify one exchange."""
    request, response, call = (exchange.get(key) for key in ("request", "response", "event"))
    if not all(isinstance(item, dict) for item in (request, response, call)):
        return False
    assert isinstance(request, dict) and isinstance(response, dict) and isinstance(call, dict)
    if any(key not in request or key not in call for key in ("review_id", "node_id", "is_eval")):
        return False
    if any(key not in item for item in (request, call) for key in ("phase_key",)):
        return False
    if any(key not in response for key in ("text", "exchange_event_id")) or "event_id" not in call:
        return False
    for key in ("review_id", "node_id", "is_eval", "phase_key"):
        if request[key] != call[key]:
            raise ValueError(f"recorded call differs from its request: {key}")
    if type(request["is_eval"]) is not bool or type(call["is_eval"]) is not bool:
        raise ValueError("recorded request and call is_eval must be Boolean")
    if request["node_id"] != "analyze" or call.get("event_type") != "llm_call":
        raise ValueError("native candidate source must be an analyzer call")
    if response["exchange_event_id"] != call["event_id"]:
        raise ValueError("response exchange identity differs from its recorded call")
    UUID(request["review_id"])
    UUID(call["event_id"])
    return request["phase_key"] is not None and isinstance(response["text"], str)


def _analysis_record(
    row: dict[str, Any], request: dict[str, Any], pass_index: int
) -> tuple[AnalyzeCompletedEvent, dict[str, Any]] | None:
    """Read one pass aggregate, retaining whether optional fields were present."""
    found = None
    for wrapper in row.get("diagnostic_events", []):
        payload = _payload(wrapper, "analyze_completed", request)
        if payload is None:
            continue
        event = AnalyzeCompletedEvent.model_validate(payload)
        if event.pass_index != pass_index:
            continue
        if found is not None and found[1] != payload:
            raise ValueError("saved workflow has conflicting analysis aggregates for one pass")
        found = event, payload
    return found


def _draft_findings(
    row: dict[str, Any], report_schema: str, request: dict[str, Any]
) -> list[ReviewFinding] | None:
    """Use the comparison runner's completed native report export only."""
    if (
        report_schema != "review-comparison-v1"
        or row.get("finding_format") != "native-analyze"
        or row.get("status") != "completed"
        or "admitted_findings" not in row
    ):
        return None
    if not isinstance(row["admitted_findings"], list):
        raise ValueError("completed native draft findings must be an array")
    findings = []
    by_id: dict[UUID, ReviewFinding] = {}
    for raw in row["admitted_findings"]:
        if not isinstance(raw, dict) or "finding_id" not in raw:
            return None
        finding = ReviewFinding.model_validate(raw)
        if str(finding.review_id) != request["review_id"]:
            raise ValueError("draft finding belongs to a different review")
        if finding.finding_id in by_id and by_id[finding.finding_id] != finding:
            raise ValueError("draft repeats a finding identity with conflicting content")
        by_id[finding.finding_id] = finding
        findings.append(finding)
    return findings


def _unknown() -> dict[str, Any]:
    return {
        "admission": "unknown",
        "reason": None,
        "detail": None,
        "observation": "not_observed",
        "native_candidate_id": None,
        "native_candidate_event_id": None,
        "source_exchange_event_id": None,
        "analysis_selected": None,
        "analysis_selection_event_id": None,
        "aggregate_event_id": None,
        "n_proposals_dropped": None,
        "n_findings_dropped_over_cap": None,
        "delivery": "unknown",
        "draft_match_finding_ids": [],
        "draft_source": None,
    }


def recorded_observations(
    row: dict[str, Any],
    exchange: dict[str, Any],
    *,
    report_schema: str,
    file_path: str | None,
    source_side: str,
    proposal_hashes: list[str | None],
) -> list[dict[str, Any]] | None:
    """Join persisted native events to exact raw proposal occurrences, without replay."""
    if not any(
        isinstance(wrapper, dict)
        and wrapper.get("kind") == "persisted-audit-event"
        and (
            wrapper.get("event_type") == "native_candidate"
            or isinstance(wrapper.get("payload"), dict)
            and wrapper["payload"].get("event_type") == "native_candidate"
        )
        for wrapper in row.get("diagnostic_events", [])
    ):
        return None
    if file_path is None or source_side not in {"head", "base"} or not _exchange_identity(exchange):
        return None
    request, response = exchange["request"], exchange["response"]
    response_hash = compute_response_hash(response["text"])
    candidates: dict[int, NativeCandidateEvent] = {}
    for wrapper in row.get("diagnostic_events", []):
        payload = _payload(wrapper, "native_candidate", request)
        if payload is None:
            continue
        if payload.get("phase_key") is None or "node_id" not in payload:
            continue
        event = NativeCandidateEvent.model_validate(payload)
        ref = event.candidate
        if event.event_id != candidate_event_id(ref.candidate_id):
            raise ValueError("native candidate event identity differs from its candidate")
        if str(ref.source_exchange_event_id) != response["exchange_event_id"]:
            continue
        if (
            ref.response_hash != response_hash
            or ref.file_path != file_path
            or ref.source_side != source_side
            or event.phase_key != request["phase_key"]
            or request["phase_key"] != f"file:{file_path}#{ref.pass_index}"
            or ref.proposal_index >= len(proposal_hashes)
            or ref.proposal_hash != proposal_hashes[ref.proposal_index]
        ):
            raise ValueError("native candidate differs from its exact saved occurrence")
        previous = candidates.get(ref.proposal_index)
        if previous is not None and previous != event:
            raise ValueError("saved occurrence has conflicting native candidate events")
        candidates[ref.proposal_index] = event
    if not candidates:
        return None

    # Snapshot copies can corroborate persisted events, but cannot replace them.
    for raw in row.get("native_candidates", []):
        ref = NativeCandidateRef.model_validate(raw)
        if str(ref.review_id) != request["review_id"]:
            raise ValueError("native candidate snapshot belongs to a different review")
        if str(ref.source_exchange_event_id) != response["exchange_event_id"]:
            continue
        recorded = candidates.get(ref.proposal_index)
        if recorded is not None and ref != recorded.candidate:
            raise ValueError("native candidate snapshot differs from its persisted event")

    draft = _draft_findings(row, report_schema, request)
    observations = [_unknown() for _ in proposal_hashes]
    for index, event in candidates.items():
        ref = event.candidate
        observation = observations[index]
        observation.update(
            admission=ref.initial_disposition,
            reason=ref.rejection_reason,
            observation="recorded_native_candidate",
            native_candidate_id=ref.candidate_id,
            native_candidate_event_id=str(event.event_id),
            source_exchange_event_id=str(ref.source_exchange_event_id),
        )
        aggregate = _analysis_record(row, request, ref.pass_index)
        if aggregate is not None:
            analysis, payload = aggregate
            observation["aggregate_event_id"] = str(analysis.event_id)
            for key in ("n_proposals_dropped", "n_findings_dropped_over_cap"):
                if key in payload and type(payload[key]) is not int:
                    raise ValueError("recorded aggregate drop counters must be integers")
                observation[key] = payload.get(key)
            if ref.initial_disposition == "admitted" and analysis.selected_finding_ids is not None:
                observation["analysis_selected"] = ref.finding_id in analysis.selected_finding_ids
                observation["analysis_selection_event_id"] = str(analysis.event_id)
        if ref.initial_disposition != "admitted" or draft is None:
            continue
        matches = [finding for finding in draft if finding.finding_id == ref.finding_id]
        if matches and any(
            finding.content_hash != ref.content_hash or finding.proposal_hash != ref.proposal_hash
            for finding in matches
        ):
            raise ValueError("draft finding differs from its original candidate identity")
        if matches:
            observation["delivery"] = "included_in_draft"
        else:
            matches = [finding for finding in draft if finding.content_hash == ref.content_hash]
            observation["delivery"] = "duplicate_in_draft" if matches else "not_in_recorded_draft"
        observation["draft_match_finding_ids"] = sorted({str(f.finding_id) for f in matches})
        observation["draft_source"] = "completed_native_review_comparison"
    return observations
