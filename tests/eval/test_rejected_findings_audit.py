"""Exercise the audit protocol with an exact saved answer and captured source.

The answer is calls/000003.exchange.json from the 20261101T234842Z-87dc02b1f0fd
diagnostic report. Its two unknown categories were rejected in the recorded
parser replay. The request envelope below retains only adapter metadata; these
tests do not claim a fresh model run or execute that review pipeline. Mutations
and judgment values are authored protocol controls, never quality results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from .rejected_findings_audit import complete, main, prepare
from .review_adjudication import write_packet

# Exact paid response bytes. Do not rewrite the saved model's prose.
RECORDED_RESPONSE = r"""{"findings":[{"finding_type":"logic error","evidence_tier":"judged","query_match_id":null,"trace_path":null,"title":"Verify compares hardcoded email, ignoring the identity attribute used for lookup","description":"The identity is looked up via an arbitrary configured identity attribute (field_name), but ownership is verified by comparing `user.email` to `current_user.email`. If the attribute is not email (e.g. username or phone), distinct users can share an email or a user may lack one, so verification can wrongly succeed or fail. Also, `user.email` is compared without normalization. Compare user identity (e.g. `user.fs_uniquifier` / `user == current_user`) instead.","evidence":"    if user and user.email == current_user.email:","line_start":211,"line_end":211,"trace_candidates":[]},{"finding_type":"missing validation","evidence_tier":"judged","query_match_id":null,"trace_path":null,"title":"Verify callback does not require an authenticated current_user","description":"oauth_verify_response has no @auth_required decorator and accesses current_user.email directly. If the callback is hit by an anonymous user (session expired between start and callback), current_user is an AnonymousUser, which has no `email` attribute, raising AttributeError (500) when `user` is truthy. Check is_user_authenticated first or decorate the view.","evidence":"    if user and user.email == current_user.email:","line_start":211,"line_end":211,"trace_candidates":[]}]}"""  # noqa: E501
RECORDED_RESPONSE_SHA256 = "619a701a751f5d10fc64f0c23193dd53a1334e1f8b12958962b19bfd79262feb"
RECORDED_PROPOSAL_HASHES = (
    "6987fbdeda34ec078be7e657e361a6fdd85486e1ef1d2bea939757213a2775ab",
    "42db12c5a78941e19e8d55d0bbb0c8fbbcaf4560c06dcb1df01c5ef02ec17067",
)
SOURCE_FILE = "flask_security/oauth_glue.py"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _source(tmp_path: Path, *, duplicate_control: bool = False) -> tuple[Path, Path]:
    """Build a reduced report export; original source and answer bytes are pinned."""
    report = tmp_path / "source-report"
    (report / "generation").mkdir(parents=True)
    fixture = Path(__file__).parent / "corpus/review_comparison/flask-security-1216/fixture.json"
    data = fixture.read_bytes()
    assert hashlib.sha256(data).hexdigest() == (
        "4f3defac19f6e3e05d41dd66eed7aca743c0094d0d1f76e5f3387816aea80983"
    )
    (report / "generation/fixture.json").write_bytes(data)
    assert hashlib.sha256(RECORDED_RESPONSE.encode()).hexdigest() == RECORDED_RESPONSE_SHA256
    text = RECORDED_RESPONSE
    hashes = list(RECORDED_PROPOSAL_HASHES)
    if duplicate_control:
        raw = json.loads(text)
        raw["findings"][1] = raw["findings"][0]
        text = json.dumps(raw)
        hashes[1] = hashes[0]
    _json(
        report / "exchange.json",
        {
            "request": {"node_id": "analyze", "user_prompt": f"File: {SOURCE_FILE}\n"},
            "response": {"text": text},
        },
    )
    plan = {
        "cases": [
            {
                "id": "flask-security-1216",
                "fixture": "fixture.json",
                "fixture_sha256": hashlib.sha256(data).hexdigest(),
            }
        ]
    }
    row = {
        "case_id": "flask-security-1216",
        "rep": 0,
        "arm": "open-categories",
        "status": "completed",
        "analysis_file": SOURCE_FILE,
        "raw_exchange_files": ["exchange.json"],
        "admitted_findings": [],
    }
    results = {
        "schema": "captured-review-diagnostics-v1",
        "status": "completed",
        "plan": plan,
        "cases": [row],
    }
    _json(report / "plan.json", plan)
    _json(report / "results.json", results)
    write_packet(results, report)
    _json(
        report / "receipts.json",
        {
            str(path.relative_to(report)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in report.rglob("*")
            if path.is_file() and path != report / "adjudication/review.json"
        },
    )
    replay = tmp_path / "parser-replay.json"
    _json(
        replay,
        {
            "rows": [
                {
                    "case_id": row["case_id"],
                    "rep": 0,
                    "arm": row["arm"],
                    "exchange": "exchange.json",
                    "admitted_findings": [],
                    "proposal_rejections": [
                        {
                            "proposal_hash": digest,
                            "file_path": SOURCE_FILE,
                            "rejection_reason": "finding_type_not_in_enum",
                            "rejection_detail": "no_near_enum_match",
                        }
                        for digest in hashes
                    ],
                    "counters": {
                        "n_proposals_seen": 2,
                        "n_findings_emitted": 0,
                        "n_proposals_rejected": 2,
                    },
                    "response_rejection": None,
                }
            ]
        },
    )
    return report, replay


def _prepared(tmp_path: Path, *, duplicate_control: bool = False) -> Path:
    report, replay = _source(tmp_path, duplicate_control=duplicate_control)
    output = tmp_path / "prepared"
    prepare(report, output=output, admission_replay=replay)
    return output


def _review(audit: Path, path: Path, *, verdict: str = "unresolved") -> Path:
    review = json.loads((audit / "blind/review.json").read_text())
    for judgment in review["judgments"]:
        judgment.update(
            verdict=verdict,
            rationale="Authored protocol control; no truth claim.",
            pr_relevance="uncertain",
            location_accuracy="uncertain",
            evidence_accuracy="uncertain",
            explanation_accuracy="uncertain",
        )
    _json(path, review)
    return path


def test_recorded_unknown_categories_join_exact_hashes_and_hide_origins(tmp_path: Path) -> None:
    audit = _prepared(tmp_path)
    summary = json.loads((audit / "audit.json").read_text())
    assert summary["counts"]["raw_occurrences"] == 2
    assert summary["counts"]["selected_unique_claims"] == 2
    assert summary["reasons"] == {"finding_type_not_in_enum": 2}
    origins = json.loads((audit / "identity-map.json").read_text())
    assert [item["proposal_hash"] for item in origins["roster"]] == list(RECORDED_PROPOSAL_HASHES)
    assert [item["proposal_index"] for item in origins["roster"]] == [0, 1]
    assert all(item["delivery"] == "not_run" for item in origins["roster"])
    blind = (audit / "blind/candidates.json").read_text()
    for hidden in (
        "open-categories",
        "finding_type_not_in_enum",
        "proposal_index",
        "occurrence_id",
    ):
        assert hidden not in blind
    assert all(
        item["verdict"] == "pending"
        for item in json.loads((audit / "blind/review.json").read_text())["judgments"]
    )


def test_authored_duplicate_occurrences_share_one_card_without_losing_counts(
    tmp_path: Path,
) -> None:
    audit = _prepared(tmp_path, duplicate_control=True)
    summary = json.loads((audit / "audit.json").read_text())
    assert summary["counts"]["selected_occurrences"] == 2
    assert summary["counts"]["selected_unique_claims"] == 1
    origins = json.loads((audit / "identity-map.json").read_text())
    assert len({item["occurrence_id"] for item in origins["roster"]}) == 2


def test_missing_parser_observations_remain_unknown_despite_empty_final_findings(
    tmp_path: Path,
) -> None:
    report, _ = _source(tmp_path)
    summary = prepare(report, output=tmp_path / "prepared")
    assert summary["observation_status"] == "no_observed_rejections"
    assert summary["counts"]["unknown_dispositions"] == 2
    assert summary["counts"]["eligible_rejected_occurrences"] == 0


@pytest.mark.parametrize("missing", ["receipts.json", "exchange.json"])
def test_prepare_cli_distinguishes_missing_receipts_from_missing_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    report, _ = _source(tmp_path)
    missing_path = report / missing
    missing_path.unlink()
    output = tmp_path / "prepared"

    with pytest.raises(SystemExit) as error:
        main(["prepare", "--report", str(report), "--output", str(output)])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(missing_path) in message
    if missing == "receipts.json":
        assert "saved review comparison" in message
        assert "captured review diagnostic" in message
    else:
        assert "saved review comparison" not in message
    assert not output.exists()


def test_selection_is_frozen_before_grading_and_reports_unsampled_occurrences(
    tmp_path: Path,
) -> None:
    report, replay = _source(tmp_path)
    for name in ("first", "second"):
        summary = prepare(report, output=tmp_path / name, admission_replay=replay, limit=1)
        assert summary["counts"]["not_sampled"] == 1
    assert (tmp_path / "first/identity-map.json").read_bytes() == (
        tmp_path / "second/identity-map.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "damage", ["receipt", "proposal", "response", "exchange", "counter", "source", "request"]
)
def test_authored_input_mutations_cannot_create_an_audit(tmp_path: Path, damage: str) -> None:
    report, replay = _source(tmp_path)
    if damage == "receipt":
        (report / "exchange.json").write_text("{}")
    elif damage == "request":
        exchange = json.loads((report / "exchange.json").read_text())
        exchange["request"]["user_prompt"] = "File: different.py\n"
        _json(report / "exchange.json", exchange)
        receipts = json.loads((report / "receipts.json").read_text())
        receipts["exchange.json"] = hashlib.sha256(
            (report / "exchange.json").read_bytes()
        ).hexdigest()
        _json(report / "receipts.json", receipts)
    else:
        data = json.loads(replay.read_text())
        row = data["rows"][0]
        if damage == "proposal":
            row["proposal_rejections"][0]["proposal_hash"] = "f" * 64
        elif damage == "response":
            row["response_hash"] = "f" * 64
        elif damage == "exchange":
            row["exchange"] = "missing.json"
        elif damage == "source":
            row["proposal_rejections"][0]["file_path"] = "different.py"
        else:
            row["counters"]["n_proposals_seen"] = 1
        _json(replay, data)
    with pytest.raises(ValueError):
        prepare(report, output=tmp_path / "prepared", admission_replay=replay)
    assert not (tmp_path / "prepared").exists()


def test_external_judgments_freeze_before_unblinded_report_without_policy_inference(
    tmp_path: Path,
) -> None:
    audit = _prepared(tmp_path)
    judgment_path = _review(audit, tmp_path / "external.json")
    output = tmp_path / "completed"
    summary = complete(audit, judgment_path, output=output)
    assert summary["selected_occurrence_outcomes"] == {"unresolved": 2}
    assert summary["current_policy_assessment"] == "unassessed"
    assert (output / "judgments.frozen.json").read_bytes() == judgment_path.read_bytes()
    rows = json.loads((output / "unblinded.json").read_text())
    assert all(item["reason"] == "finding_type_not_in_enum" for item in rows)
    assert all(item["current_policy_assessment"] == "unassessed" for item in rows)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong-packet", "bad-verdict"])
def test_authored_judgment_mutations_reject_incomplete_identity_coverage(
    tmp_path: Path, damage: str
) -> None:
    audit = _prepared(tmp_path)
    path = _review(audit, tmp_path / "external.json")
    review = json.loads(path.read_text())
    if damage == "missing":
        review["judgments"].pop()
    elif damage == "duplicate":
        review["judgments"][1] = review["judgments"][0]
    elif damage == "wrong-packet":
        review["candidates_sha256"] = "f" * 64
    else:
        review["judgments"][0]["verdict"] = "false-negative"
    _json(path, review)
    with pytest.raises(ValueError):
        complete(audit, path, output=tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_pending_judgments_stay_incomplete_and_cli_requires_no_provider(
    tmp_path: Path, capsys: Any
) -> None:
    audit = _prepared(tmp_path)
    assert (
        main(
            [
                "complete",
                "--audit",
                str(audit),
                "--judgments",
                str(audit / "blind/review.json"),
                "--output",
                str(tmp_path / "completed"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
    assert json.loads((tmp_path / "completed/audit.json").read_text())[
        "selected_occurrence_outcomes"
    ] == {"pending": 2}


@pytest.mark.parametrize("removed", ["audit.json", "blind/candidates.json", "source"])
def test_authored_receipt_removal_cannot_bypass_required_evidence(
    tmp_path: Path, removed: str
) -> None:
    audit = _prepared(tmp_path)
    path = _review(audit, tmp_path / "external.json")
    receipts = json.loads((audit / "receipts.json").read_text())
    if removed == "source":
        removed = next(name for name in receipts if name.startswith("blind/sources/"))
    receipts.pop(removed)
    _json(audit / "receipts.json", receipts)
    with pytest.raises(ValueError, match="required immutable receipts"):
        complete(audit, path, output=tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_authored_valid_judgment_does_not_infer_policy_or_delivery(tmp_path: Path) -> None:
    audit = _prepared(tmp_path)
    path = _review(audit, tmp_path / "external.json", verdict="valid")
    packet = json.loads((audit / "blind/candidates.json").read_text())
    source = json.loads((audit / "blind" / packet["sources"][0]["path"]).read_text())
    content = next(file["content_head"] for file in source["files"] if file["path"] == SOURCE_FILE)
    review = json.loads(path.read_text())
    for item in review["judgments"]:
        item.update(
            trigger="Authored protocol trigger.", consequence="Authored protocol consequence."
        )
        item["sources"] = [
            {
                "source": packet["sources"][0]["path"],
                "file_path": SOURCE_FILE,
                "source_side": "head",
                "line_start": 211,
                "line_end": 211,
                "quote": content.splitlines()[210],
            }
        ]
    _json(path, review)
    summary = complete(audit, path, output=tmp_path / "completed")
    assert summary["selected_occurrence_outcomes"] == {"valid_claim_rejected_by_parser": 2}
    assert summary["current_policy_assessment"] == "unassessed"
    assert all(
        item["delivery"] == "not_run"
        for item in json.loads((tmp_path / "completed/unblinded.json").read_text())
    )
    review["judgments"][0]["sources"][0]["quote"] = "authored mismatched source quote"
    _json(path, review)
    with pytest.raises(ValueError, match="complete captured lines"):
        complete(audit, path, output=tmp_path / "bad-citation")


def test_pending_accuracy_and_original_report_output_remain_explicit(tmp_path: Path) -> None:
    audit = _prepared(tmp_path)
    path = _review(audit, tmp_path / "external.json")
    review = json.loads(path.read_text())
    review["judgments"][0]["evidence_accuracy"] = "pending"
    _json(path, review)
    summary = complete(audit, path, output=tmp_path / "completed")
    assert summary["status"] == "incomplete"
    assert summary["pending_assessments"] == {"evidence_accuracy": 1}
    with pytest.raises(ValueError, match="outside the original report"):
        complete(audit, path, output=tmp_path / "source-report/completed")
