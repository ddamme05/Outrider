"""Check saved native admission observations without new review generation.

The frozen values come from case 0 of review-comparison report
20261102T030642Z-2b6d6416e278 and concise-with-modules/calls/000002.exchange.json.
The envelope retains only relevant metadata. Raw answer bytes and canonical
metadata bytes are pinned separately. Later mutations are authored protocol
controls, not observed failures or model-quality results.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from outrider.agent.native_candidates import candidate_event_id
from outrider.policy.canonical import compute_response_hash
from outrider.schemas.native_candidate import native_candidate_id

from .rejected_findings_audit import prepare
from .rejection_audit_observations import recorded_observations
from .review_adjudication import write_packet

SOURCE_FILE = "flask_security/oauth_glue.py"
RECORDED_RESPONSE = '{"findings":[{"finding_type":"auth_bypass","evidence_tier":"judged","query_match_id":null,"trace_path":null,"title":"OAuth verify compares users by email regardless of the identity attribute used","description":"The OAuth verify response treats the re-authentication as successful only if `user.email == current_user.email`, but the user was looked up via whichever identity attribute the provider returned (e.g. username or phone). If that attribute is not email, comparing emails is not a reliable identity check: users sharing an email would pass verification via another account, or a legitimately matching user can be rejected. If email is None on both sides, `None == None` also passes. Compare user identity (e.g. `user.fs_uniquifier == current_user.fs_uniquifier` or `user == current_user`) instead.","evidence":"    if user and user.email == current_user.email:","line_start":211,"line_end":211}]}'  # noqa: E501
RECORDED_RESPONSE_SHA256 = "f53a2134c23aae30bea4970c00126d0fbbb518c24d2979d026f0329fcd045656"
RECORDED_METADATA_SHA256 = "74b303e55378b1c2afba35bbfd974692575b452e57351df0fc31aa2da1584af0"
RECORDED_METADATA = r"""{
  "exchange": {
    "event": {
      "event_id": "ab2194c2-7ad6-419c-b43f-cb60efb52338",
      "event_type": "llm_call",
      "is_eval": true,
      "node_id": "analyze",
      "phase_key": "file:flask_security/oauth_glue.py#0",
      "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340"
    },
    "request": {
      "is_eval": true,
      "node_id": "analyze",
      "phase_key": "file:flask_security/oauth_glue.py#0",
      "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340"
    },
    "response": {
      "exchange_event_id": "ab2194c2-7ad6-419c-b43f-cb60efb52338"
    }
  },
  "report_schema": "review-comparison-v1",
  "row": {
    "admitted_findings": [
      {
        "confidence": 0.5,
        "content_hash": "bf5ad4db81dfdf39596b3c6c48698e5d0954a00d8e79a6100459560495609784",
        "description": "The OAuth verify response treats the re-authentication as successful only if `user.email == current_user.email`, but the user was looked up via whichever identity attribute the provider returned (e.g. username or phone). If that attribute is not email, comparing emails is not a reliable identity check: users sharing an email would pass verification via another account, or a legitimately matching user can be rejected. If email is None on both sides, `None == None` also passes. Compare user identity (e.g. `user.fs_uniquifier == current_user.fs_uniquifier` or `user == current_user`) instead.",
        "dimension": "security",
        "evidence": "    if user and user.email == current_user.email:",
        "evidence_tier": "judged",
        "file_path": "flask_security/oauth_glue.py",
        "finding_id": "6c3bcbcc-3716-4e10-a0e7-3cd14b04577a",
        "finding_type": "auth_bypass",
        "installation_id": 9740,
        "line_end": 211,
        "line_start": 211,
        "original_severity": null,
        "override_reason": null,
        "overrider_id": null,
        "policy_version": "1.2.0",
        "proposal_hash": "9b6aedc883939042b2e52674d8c6d97422f4630792a4fd755efe440837bee958",
        "publish_destination": null,
        "query_match_id": null,
        "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340",
        "severity": "critical",
        "suggested_fix": null,
        "title": "OAuth verify compares users by email regardless of the identity attribute used",
        "trace_path": null
      }
    ],
    "arm": "concise-with-modules",
    "case_id": "flask-security-1216",
    "diagnostic_events": [
      {
        "event_id": "6335fe90-5fea-5522-b261-abffa7c1c0a9",
        "event_type": "native_candidate",
        "is_eval": true,
        "kind": "persisted-audit-event",
        "payload": {
          "candidate": {
            "candidate_id": "9f9569a76fcf4161b49041081470386f0ce2eee33b5412d7238513e7e2a7767c",
            "content_hash": "bf5ad4db81dfdf39596b3c6c48698e5d0954a00d8e79a6100459560495609784",
            "evidence_tier": "judged",
            "file_path": "flask_security/oauth_glue.py",
            "finding_id": "6c3bcbcc-3716-4e10-a0e7-3cd14b04577a",
            "finding_type": "auth_bypass",
            "initial_disposition": "admitted",
            "pass_index": 0,
            "policy_version": "1.2.0",
            "proposal_hash": "9b6aedc883939042b2e52674d8c6d97422f4630792a4fd755efe440837bee958",
            "proposal_index": 0,
            "rejection_reason": null,
            "response_hash": "f53a2134c23aae30bea4970c00126d0fbbb518c24d2979d026f0329fcd045656",
            "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340",
            "severity": "critical",
            "source_exchange_event_id": "ab2194c2-7ad6-419c-b43f-cb60efb52338"
          },
          "event_id": "6335fe90-5fea-5522-b261-abffa7c1c0a9",
          "event_type": "native_candidate",
          "is_eval": true,
          "node_id": "analyze",
          "phase_key": "file:flask_security/oauth_glue.py#0",
          "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340",
          "timestamp": "2026-11-02T03:06:50.334274Z"
        },
        "phase_key": null,
        "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340",
        "sequence_number": 15,
        "timestamp": "2026-11-02T03:06:50.334274+00:00"
      },
      {
        "event_id": "96c830d6-0b02-4c6b-b96a-3446006d073f",
        "event_type": "analyze_completed",
        "is_eval": true,
        "kind": "persisted-audit-event",
        "payload": {
          "analyze_model": "claude-sonnet-5-5",
          "event_id": "96c830d6-0b02-4c6b-b96a-3446006d073f",
          "event_type": "analyze_completed",
          "is_eval": true,
          "n_files_analyzed": 2,
          "n_files_skipped": 0,
          "n_findings_dropped_over_cap": 0,
          "n_findings_emitted": 1,
          "n_findings_observed": 0,
          "n_findings_served": 0,
          "n_llm_calls": 2,
          "n_proposals_dropped": 0,
          "n_proposals_rejected": 0,
          "n_proposals_seen": 1,
          "n_proposals_superseded_by_observed": 0,
          "n_responses_rejected": 0,
          "n_trace_candidates_dropped_malformed": 0,
          "n_trace_candidates_emitted": 0,
          "node_id": "analyze",
          "pass_index": 0,
          "phase_key": "aggregate#0",
          "policy_version": "1.2.0",
          "pricing_version": "v8",
          "profile_contract_digest": "e5a53cca1a05f1b8e37ee64ef7c42a967abbf4e2d0a9a7a0e8834580680c543b",
          "profile_id": "anthropic",
          "reasoning_enabled": false,
          "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340",
          "selected_finding_ids": [
            "6c3bcbcc-3716-4e10-a0e7-3cd14b04577a"
          ],
          "standard_analyze_model": "claude-sonnet-5-5",
          "subsumed_matches": [],
          "timestamp": "2026-11-02T03:06:52.216806Z",
          "total_cache_read_tokens": 1549,
          "total_cache_write_tokens": 1549,
          "total_cost_usd": 0.0601814,
          "total_input_tokens": 26537,
          "total_output_tokens": 308
        },
        "phase_key": null,
        "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340",
        "sequence_number": 24,
        "timestamp": "2026-11-02T03:06:52.216806+00:00"
      }
    ],
    "finding_format": "native-analyze",
    "native_candidates": [
      {
        "candidate_id": "9f9569a76fcf4161b49041081470386f0ce2eee33b5412d7238513e7e2a7767c",
        "content_hash": "bf5ad4db81dfdf39596b3c6c48698e5d0954a00d8e79a6100459560495609784",
        "evidence_tier": "judged",
        "file_path": "flask_security/oauth_glue.py",
        "finding_id": "6c3bcbcc-3716-4e10-a0e7-3cd14b04577a",
        "finding_type": "auth_bypass",
        "initial_disposition": "admitted",
        "pass_index": 0,
        "policy_version": "1.2.0",
        "proposal_hash": "9b6aedc883939042b2e52674d8c6d97422f4630792a4fd755efe440837bee958",
        "proposal_index": 0,
        "rejection_reason": null,
        "response_hash": "f53a2134c23aae30bea4970c00126d0fbbb518c24d2979d026f0329fcd045656",
        "review_id": "58bd735c-1734-42df-8dbe-ac68d40c7340",
        "severity": "critical",
        "source_exchange_event_id": "ab2194c2-7ad6-419c-b43f-cb60efb52338"
      }
    ],
    "rep": 0,
    "status": "completed"
  }
}
"""  # noqa: E501


def _capture() -> tuple[dict[str, Any], dict[str, Any]]:
    assert hashlib.sha256(RECORDED_RESPONSE.encode()).hexdigest() == RECORDED_RESPONSE_SHA256
    assert hashlib.sha256(RECORDED_METADATA.encode()).hexdigest() == RECORDED_METADATA_SHA256
    capture = json.loads(RECORDED_METADATA)
    exchange = capture["exchange"]
    exchange["response"]["text"] = RECORDED_RESPONSE
    return capture["row"], exchange


def _observe(
    row: dict[str, Any], exchange: dict[str, Any], **overrides: Any
) -> list[dict[str, Any]] | None:
    options = {
        "report_schema": json.loads(RECORDED_METADATA)["report_schema"],
        "file_path": SOURCE_FILE,
        "source_side": "head",
        "proposal_hashes": [row["native_candidates"][0]["proposal_hash"]],
        **overrides,
    }
    return recorded_observations(row, exchange, **options)


def _retarget_response(row: dict[str, Any], exchange: dict[str, Any]) -> None:
    """Rebind an authored response control with the real identity recipes."""
    wrapper = row["diagnostic_events"][0]
    ref = wrapper["payload"]["candidate"]
    ref["response_hash"] = compute_response_hash(exchange["response"]["text"])
    ref["candidate_id"] = native_candidate_id(
        review_id=UUID(ref["review_id"]),
        source_exchange_event_id=UUID(ref["source_exchange_event_id"]),
        response_hash=ref["response_hash"],
        proposal_index=ref["proposal_index"],
        file_path=ref["file_path"],
        pass_index=ref["pass_index"],
    )
    event_id = str(candidate_event_id(ref["candidate_id"]))
    wrapper["event_id"] = wrapper["payload"]["event_id"] = event_id
    row["native_candidates"] = [deepcopy(ref)]


def test_recorded_pilot_admission_analysis_and_draft_join_without_replay() -> None:
    row, exchange = _capture()
    original = deepcopy((row, exchange))
    observations = _observe(row, exchange)
    assert observations is not None
    observation = observations[0]
    ref = row["native_candidates"][0]
    assert observation["admission"] == "admitted"
    assert observation["observation"] == "recorded_native_candidate"
    assert observation["native_candidate_id"] == ref["candidate_id"]
    assert observation["native_candidate_event_id"] == row["diagnostic_events"][0]["event_id"]
    assert observation["source_exchange_event_id"] == exchange["event"]["event_id"]
    assert observation["analysis_selected"] is True
    assert observation["analysis_selection_event_id"] == row["diagnostic_events"][1]["event_id"]
    assert observation["delivery"] == "included_in_draft"
    assert observation["draft_match_finding_ids"] == [ref["finding_id"]]
    assert observation["draft_source"] == "completed_native_review_comparison"
    assert observation["n_proposals_dropped"] == 0
    assert observation["n_findings_dropped_over_cap"] == 0
    assert (row, exchange) == original


@pytest.mark.parametrize(
    "damage",
    [
        "wrapper-event",
        "wrapper-review",
        "wrapper-eval",
        "payload-review",
        "payload-eval",
        "event-id",
        "response-hash",
        "proposal-hash",
        "path",
        "side",
        "pass",
        "phase",
        "call-id",
        "call-review",
        "call-phase",
        "snapshot",
    ],
)
def test_authored_conflicting_source_identities_fail_visibly(damage: str) -> None:
    row, exchange = _capture()
    wrapper = row["diagnostic_events"][0]
    payload = wrapper["payload"]
    ref = payload["candidate"]
    options: dict[str, Any] = {}
    if damage == "wrapper-event":
        wrapper["event_id"] = str(uuid4())
    elif damage == "wrapper-review":
        wrapper["review_id"] = str(uuid4())
    elif damage == "wrapper-eval":
        wrapper["is_eval"] = False
    elif damage == "payload-review":
        payload["review_id"] = str(uuid4())
    elif damage == "payload-eval":
        payload["is_eval"] = False
    elif damage == "event-id":
        wrapper["event_id"] = payload["event_id"] = str(uuid4())
    elif damage == "response-hash":
        exchange["response"]["text"] += " "
    elif damage == "proposal-hash":
        options["proposal_hashes"] = ["a" * 64]
    elif damage == "path":
        options["file_path"] = "other.py"
    elif damage == "side":
        options["source_side"] = "base"
    elif damage == "pass":
        ref["pass_index"] = 1
        _retarget_response(row, exchange)
    elif damage == "phase":
        payload["phase_key"] = "file:other.py#0"
    elif damage == "call-id":
        exchange["event"]["event_id"] = str(uuid4())
    elif damage == "call-review":
        exchange["event"]["review_id"] = str(uuid4())
    elif damage == "call-phase":
        exchange["event"]["phase_key"] = "file:other.py#0"
    else:
        row["native_candidates"][0]["content_hash"] = "a" * 64
    with pytest.raises(ValueError):
        _observe(row, exchange, **options)


@pytest.mark.parametrize("field", ["event_id", "review_id", "is_eval", "timestamp"])
def test_authored_missing_candidate_envelope_fields_do_not_use_model_defaults(field: str) -> None:
    row, exchange = _capture()
    del row["diagnostic_events"][0]["payload"][field]
    assert _observe(row, exchange) is None


def test_authored_snapshot_without_persisted_event_does_not_invent_admission() -> None:
    row, exchange = _capture()
    row["diagnostic_events"] = row["diagnostic_events"][1:]
    assert (
        _observe(row, exchange, proposal_hashes=[row["native_candidates"][0]["proposal_hash"]])
        is None
    )


def test_authored_legacy_exchange_without_source_id_stays_unknown() -> None:
    row, exchange = _capture()
    del exchange["response"]["exchange_event_id"]
    assert _observe(row, exchange) is None


def test_authored_unrelated_call_metadata_cannot_reject_a_replay_only_row() -> None:
    row, exchange = _capture()
    row["diagnostic_events"] = []
    exchange["event"]["node_id"] = "triage"
    assert _observe(row, exchange) is None


def test_authored_numeric_eval_flag_is_not_a_matching_boolean() -> None:
    row, exchange = _capture()
    exchange["event"]["is_eval"] = 1
    with pytest.raises(ValueError, match="must be Boolean"):
        _observe(row, exchange)


def test_authored_missing_selection_and_counters_remain_unknown() -> None:
    row, exchange = _capture()
    payload = row["diagnostic_events"][1]["payload"]
    for key in ("selected_finding_ids", "n_proposals_dropped", "n_findings_dropped_over_cap"):
        del payload[key]
    observations = _observe(row, exchange)
    assert observations is not None
    observation = observations[0]
    assert observation["admission"] == "admitted"
    assert observation["analysis_selected"] is None
    assert observation["analysis_selection_event_id"] is None
    assert observation["n_proposals_dropped"] is None
    assert observation["n_findings_dropped_over_cap"] is None
    assert observation["aggregate_event_id"] == row["diagnostic_events"][1]["event_id"]
    assert observation["delivery"] == "included_in_draft"


@pytest.mark.parametrize("damage", ["failed", "legacy-schema", "missing-format", "missing-draft"])
def test_authored_unproved_final_drafts_remain_unknown(damage: str) -> None:
    row, exchange = _capture()
    options: dict[str, Any] = {}
    if damage == "failed":
        row["status"] = "failed"
    elif damage == "legacy-schema":
        options["report_schema"] = "captured-review-diagnostics-v1"
    elif damage == "missing-format":
        del row["finding_format"]
    else:
        del row["admitted_findings"]
    observations = _observe(row, exchange, **options)
    assert observations is not None
    assert observations[0]["analysis_selected"] is True
    assert observations[0]["delivery"] == "unknown"
    assert observations[0]["draft_source"] is None


def test_authored_duplicate_draft_and_missing_draft_are_separate_from_cap_claims() -> None:
    row, exchange = _capture()
    replacement_id = str(uuid4())
    row["admitted_findings"][0]["finding_id"] = replacement_id
    observations = _observe(row, exchange)
    assert observations is not None
    assert observations[0]["delivery"] == "duplicate_in_draft"
    assert observations[0]["draft_match_finding_ids"] == [replacement_id]
    row["admitted_findings"] = []
    observations = _observe(row, exchange)
    assert observations is not None
    assert observations[0]["analysis_selected"] is True
    assert observations[0]["delivery"] == "not_in_recorded_draft"
    assert observations[0]["n_findings_dropped_over_cap"] == 0


def test_authored_same_id_draft_cannot_change_the_original_proposal_hash() -> None:
    row, exchange = _capture()
    row["admitted_findings"][0]["proposal_hash"] = "a" * 64
    with pytest.raises(ValueError, match="original candidate identity"):
        _observe(row, exchange)


def test_authored_repeated_identical_event_is_not_another_occurrence() -> None:
    row, exchange = _capture()
    row["diagnostic_events"].append(deepcopy(row["diagnostic_events"][0]))
    observations = _observe(row, exchange)
    assert observations is not None
    assert len(observations) == 1


def test_authored_conflicting_aggregate_is_not_resolved_by_event_order() -> None:
    row, exchange = _capture()
    conflicting = deepcopy(row["diagnostic_events"][1])
    conflicting["event_id"] = conflicting["payload"]["event_id"] = str(uuid4())
    row["diagnostic_events"].append(conflicting)
    with pytest.raises(ValueError, match="conflicting analysis aggregates"):
        _observe(row, exchange)


def test_authored_partial_candidate_events_keep_unobserved_siblings_unknown() -> None:
    row, exchange = _capture()
    raw = json.loads(RECORDED_RESPONSE)
    raw["findings"].append(deepcopy(raw["findings"][0]))
    exchange["response"]["text"] = json.dumps(raw)
    _retarget_response(row, exchange)
    proposal_hash = row["native_candidates"][0]["proposal_hash"]
    observations = _observe(row, exchange, proposal_hashes=[proposal_hash, proposal_hash])
    assert observations is not None
    assert [item["admission"] for item in observations] == ["admitted", "unknown"]
    assert observations[1]["native_candidate_id"] is None
    assert observations[1]["delivery"] == "unknown"


@pytest.mark.parametrize("disposition", ["rejected", "malformed"])
def test_authored_unadmitted_candidates_make_no_selection_or_draft_claim(disposition: str) -> None:
    row, exchange = _capture()
    ref = row["diagnostic_events"][0]["payload"]["candidate"]
    ref["initial_disposition"] = disposition
    ref["rejection_reason"] = (
        "raw_proposal_invalid" if disposition == "malformed" else "finding_type_not_in_enum"
    )
    for key in (
        "finding_id",
        "content_hash",
        "finding_type",
        "severity",
        "evidence_tier",
        "policy_version",
    ):
        ref[key] = None
    if disposition == "malformed":
        exchange["response"]["text"] = '{"findings":[{"title":"authored malformed control"}]}'
        ref["proposal_hash"] = None
    _retarget_response(row, exchange)
    observations = _observe(row, exchange)
    assert observations is not None
    assert observations[0]["admission"] == disposition
    assert observations[0]["analysis_selected"] is None
    assert observations[0]["delivery"] == "unknown"


def _report(tmp_path: Path, row: dict[str, Any], exchange: dict[str, Any]) -> Path:
    """Wrap the recorded pilot in a reduced, receipt-bound adapter export."""
    # Source content remains the original captured PR; this small request only
    # supplies the adapter's primary-file header, not a regenerated model input.
    exchange["request"]["user_prompt"] = f"File: {SOURCE_FILE}\n"
    row["raw_exchange_files"] = ["exchange.json"]
    report = tmp_path / "source"
    (report / "generation").mkdir(parents=True)
    fixture = Path(__file__).parent / "corpus/review_comparison/flask-security-1216/fixture.json"
    data = fixture.read_bytes()
    assert hashlib.sha256(data).hexdigest() == (
        "4f3defac19f6e3e05d41dd66eed7aca743c0094d0d1f76e5f3387816aea80983"
    )
    (report / "generation/fixture.json").write_bytes(data)
    plan = {
        "cases": [
            {
                "id": row["case_id"],
                "fixture": "fixture.json",
                "fixture_sha256": hashlib.sha256(data).hexdigest(),
            }
        ]
    }
    results = {
        "schema": json.loads(RECORDED_METADATA)["report_schema"],
        "status": "completed",
        "plan": plan,
        "cases": [row],
    }
    for name, value in (
        ("plan.json", plan),
        ("results.json", results),
        ("exchange.json", exchange),
    ):
        (report / name).write_text(json.dumps(value, sort_keys=True))
    write_packet(results, report)
    receipts = {
        str(path.relative_to(report)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in report.rglob("*")
        if path.is_file() and path != report / "adjudication/review.json"
    }
    (report / "receipts.json").write_text(json.dumps(receipts))
    return report


@pytest.mark.parametrize("failed", [False, True])
def test_recorded_admission_reaches_prepare_without_replay(tmp_path: Path, failed: bool) -> None:
    row, exchange = _capture()
    if failed:
        row["status"] = "failed"
    report = _report(tmp_path, row, exchange)
    output = tmp_path / "prepared"
    summary = prepare(report, output=output)
    identity = json.loads((output / "identity-map.json").read_text())
    observation = identity["roster"][0]
    assert summary["counts"]["admitted_occurrences"] == 1
    assert summary["counts"]["unknown_dispositions"] == 0
    assert summary["counts"]["eligible_rejected_occurrences"] == 0
    assert summary["counts"]["eligible_selection_gaps"] == 0
    assert summary["admissions"] == {"admitted": 1}
    assert summary["observation_sources"] == {"recorded_native_candidate": 1}
    assert summary["analysis_selection"] == {"selected": 1}
    delivery = "unknown" if failed else "included_in_draft"
    assert summary["native_draft_outcomes"] == {delivery: 1}
    assert observation["analysis_selected"] is True
    assert observation["delivery"] == delivery
    assert observation["observation"] == "recorded_native_candidate"
    assert observation["audit_reason"] is None


def test_authored_parser_replay_cannot_replace_recorded_native_admission(tmp_path: Path) -> None:
    row, exchange = _capture()
    report = _report(tmp_path, row, exchange)
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "case_id": "flask-security-1216",
                        "rep": 0,
                        "arm": "concise-with-modules",
                        "exchange": "exchange.json",
                        "proposal_rejections": [],
                    }
                ]
            }
        )
    )
    with pytest.raises(
        ValueError, match="audit recorded native admission separately from parser replay"
    ):
        prepare(report, output=tmp_path / "prepared", admission_replay=replay)


@pytest.mark.parametrize("stage", ["analysis", "draft"])
def test_authored_observed_selection_gaps_are_eligible_without_inventing_cap_reasons(
    tmp_path: Path, stage: str
) -> None:
    row, exchange = _capture()
    row["admitted_findings"] = []
    if stage == "analysis":
        aggregate = row["diagnostic_events"][1]["payload"]
        aggregate.update(
            selected_finding_ids=[],
            n_findings_emitted=0,
            n_proposals_dropped=1,
            n_findings_dropped_over_cap=1,
        )
    report = _report(tmp_path, row, exchange)
    output = tmp_path / "prepared"
    summary = prepare(report, output=output)
    observation = json.loads((output / "identity-map.json").read_text())["roster"][0]
    expected_reason = "not_selected_in_analysis" if stage == "analysis" else "not_in_recorded_draft"
    assert summary["counts"]["eligible_occurrences"] == 1
    assert summary["counts"]["eligible_selection_gaps"] == 1
    assert summary["counts"]["eligible_rejected_occurrences"] == 0
    assert summary["counts"]["selected_occurrences"] == 1
    assert summary["reasons"] == {expected_reason: 1}
    assert observation["reason"] is None
    assert observation["audit_reason"] == expected_reason
    assert observation["analysis_selected"] is (stage != "analysis")
    assert observation["delivery"] == "not_in_recorded_draft"


def test_authored_missing_events_do_not_become_rejections_from_an_empty_draft(
    tmp_path: Path,
) -> None:
    row, exchange = _capture()
    row["diagnostic_events"] = []
    row["admitted_findings"] = []
    report = _report(tmp_path, row, exchange)
    summary = prepare(report, output=tmp_path / "prepared")
    assert summary["counts"]["unknown_dispositions"] == 1
    assert summary["counts"]["eligible_occurrences"] == 0
    assert summary["analysis_selection"] == {}
    assert summary["native_draft_outcomes"] == {}
