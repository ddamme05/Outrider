"""Keep factual review separate from model/arm identity and admission policy."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from .review_adjudication import write_packet

if TYPE_CHECKING:
    from pathlib import Path


def _capture(directory: Path) -> dict:
    fixture = {
        "pr_title": "Update calculation",
        "pr_body": "PRIVATE_BODY_SENTINEL",
        "author": "PRIVATE_AUTHOR_SENTINEL",
        "files": [{"path": "app.py", "content_head": "def calculate():\n    return 1\n"}],
    }
    data = json.dumps(fixture).encode()
    source = directory / "generation/fixture.json"
    source.parent.mkdir()
    source.write_bytes(data)
    finding = {
        "file_path": "app.py",
        "title": "Check the result",
        "description": "This candidate needs a factual judgment.",
        "evidence": "return 1",
        "line_start": 2,
        "line_end": 2,
    }
    exchanges = []
    for arm in ("context-on", "direct-one-shot"):
        raw = dict(finding)
        if arm == "context-on":
            raw.pop("file_path")
            raw["finding_type"] = "outside_the_native_enum"
        else:
            raw["category"] = "arbitrary correctness category"
        name = arm + ".exchange.json"
        (directory / name).write_text(
            json.dumps(
                {
                    "request": {
                        "node_id": "analyze",
                        "model": "MODEL_IDENTITY_SENTINEL",
                        "user_prompt": "File: app.py\n",
                    },
                    "response": {"text": json.dumps({"findings": [raw]})},
                }
            )
        )
        exchanges.append((arm, name))
    return {
        "plan": {
            "cases": [
                {
                    "id": "CASE_IDENTITY_SENTINEL",
                    "fixture": "fixture.json",
                    "fixture_sha256": hashlib.sha256(data).hexdigest(),
                    "oracle_manifest": "labels.json",
                }
            ]
        },
        "cases": [
            {
                "arm": arm,
                "case_id": "CASE_IDENTITY_SENTINEL",
                "rep": 0,
                "status": "completed",
                "raw_exchange_files": [name],
                "admitted_findings": [{**finding, "severity": "high", "confidence": 0.9}]
                if arm == "context-on"
                else [],
                "diagnostic_events": [{"event_type": "finding_proposal_rejected"}],
            }
            for arm, name in exchanges
        ],
    }


def test_raw_and_admitted_candidates_share_a_blind_pending_judgment(tmp_path: Path) -> None:
    report = _capture(tmp_path)
    write_packet(report, tmp_path)
    folder = tmp_path / "adjudication"
    candidates = json.loads((folder / "candidates.json").read_text())
    assert hashlib.sha256((folder / "candidates.json").read_bytes()).hexdigest() == (
        "418338f21331d1f08107c9fb7e81a866b6d9c45e8a7730f167a584e6b6115537"
    )
    assert len(candidates["candidates"]) == 1
    encoded = (folder / "candidates.json").read_text()
    for hidden in (
        "context-on",
        "direct-one-shot",
        "MODEL_IDENTITY_SENTINEL",
        "CASE_IDENTITY_SENTINEL",
        "severity",
        "confidence",
        "outside_the_native_enum",
    ):
        assert hidden not in encoded
    source = (folder / candidates["sources"][0]["path"]).read_text()
    assert "PRIVATE_BODY_SENTINEL" not in source
    assert "PRIVATE_AUTHOR_SENTINEL" not in source
    origins = json.loads((folder / "identity-map.json").read_text())
    card_id = candidates["candidates"][0]["id"]
    assert len(origins["candidate_origins"][card_id]) == 3
    assert {item["stage"] for item in origins["candidate_origins"][card_id]} == {
        "raw-proposal",
        "final-finding",
    }
    assert origins["actual_visibility"][0]["diagnostic_events"]
    review = json.loads((folder / "review.json").read_text())
    assert review["status"] == "pending"
    assert review["judgments"][0]["verdict"] == "pending"
    assert (
        review["candidates_sha256"]
        == hashlib.sha256((folder / "candidates.json").read_bytes()).hexdigest()
    )


def test_malformed_raw_responses_remain_explicit_gaps(tmp_path: Path) -> None:
    report = _capture(tmp_path)
    path = tmp_path / "direct-one-shot.exchange.json"
    exchange = json.loads(path.read_text())
    exchange["response"]["text"] = "not JSON"
    path.write_text(json.dumps(exchange))
    write_packet(report, tmp_path)
    origins = json.loads((tmp_path / "adjudication/identity-map.json").read_text())
    assert origins["collection_gaps"][0]["reason"] == "unparseable-raw-findings"
    assert origins["collection_gaps"][0]["arm"] == "direct-one-shot"


def test_unlocated_direct_claim_does_not_inherit_the_first_file(tmp_path: Path) -> None:
    report = _capture(tmp_path)
    path = tmp_path / "direct-one-shot.exchange.json"
    exchange = json.loads(path.read_text())
    response = json.loads(exchange["response"]["text"])
    response["findings"][0].pop("file_path")
    exchange["response"]["text"] = json.dumps(response)
    exchange["request"]["user_prompt"] = 'File: "app.py"\n\nFile: "other.py"\n'
    path.write_text(json.dumps(exchange))
    write_packet(report, tmp_path)
    cards = json.loads((tmp_path / "adjudication/candidates.json").read_text())["candidates"]
    assert {card["file_path"] for card in cards} == {None, "app.py"}
    unknown = next(card for card in cards if card["file_path"] is None)
    assert unknown["source_side"] == "unknown"
    origins = json.loads((tmp_path / "adjudication/identity-map.json").read_text())
    assert any(gap["reason"] == "unknown-source-frame" for gap in origins["collection_gaps"])


def _replace_files(directory: Path, report: dict, files: list[dict]) -> None:
    path = directory / "generation/fixture.json"
    fixture = json.loads(path.read_text())
    fixture["files"] = files
    data = json.dumps(fixture).encode()
    path.write_bytes(data)
    report["plan"]["cases"][0]["fixture_sha256"] = hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("bad_patch", [False, True])
def test_removed_raw_and_final_claims_keep_code_assigned_base_provenance(tmp_path: Path, bad_patch):
    report = _capture(tmp_path)
    base = "def calculate():\n    return 1\n"
    _replace_files(
        tmp_path,
        report,
        [
            {
                "path": "app.py",
                "status": "removed",
                "content_head": None,
                "content_base": base,
                "patch": "@@ -1,2 +0,0 @@\n-def calculate():\n-    return 1\n"
                if not bad_patch
                else "@@ -1,1 +0,0 @@\n-def calculate():\n",
            }
        ],
    )
    report["cases"][0]["admitted_findings"][0]["source_side"] = "base"
    for arm in ("context-on", "direct-one-shot"):
        path = tmp_path / f"{arm}.exchange.json"
        exchange = json.loads(path.read_text())
        raw = json.loads(exchange["response"]["text"])
        raw["findings"][0]["source_side"] = "head"  # Raw claims cannot choose their frame.
        exchange["response"]["text"] = json.dumps(raw)
        if arm == "context-on":
            exchange["request"].update(
                degradation_reason="file_removal",
                context_summary=[
                    {
                        "file_path": "app.py",
                        "source_side": "base",
                        "inclusion_reason": "changed_region",
                    }
                ],
            )
        path.write_text(json.dumps(exchange))
    write_packet(report, tmp_path)
    packet = json.loads((tmp_path / "adjudication/candidates.json").read_text())
    origins = json.loads((tmp_path / "adjudication/identity-map.json").read_text())
    if bad_patch:
        assert {card["source_side"] for card in packet["candidates"]} == {"base", "unknown"}
        assert len(origins["collection_gaps"]) == 2
    else:
        (card,) = packet["candidates"]
        assert card["source_side"] == "base"
        assert len(origins["candidate_origins"][card["id"]]) == 3
        assert origins["collection_gaps"] == []


def test_same_text_and_coordinates_from_head_and_base_have_different_cards(tmp_path: Path):
    report = _capture(tmp_path)
    source = "def calculate():\n    return 1\n"
    _replace_files(
        tmp_path,
        report,
        [
            {
                "path": "app.py",
                "status": "modified",
                "content_head": source,
                "content_base": source,
            }
        ],
    )
    head = report["cases"][0]["admitted_findings"][0]
    report["cases"][0]["admitted_findings"].append({**head, "source_side": "base"})
    write_packet(report, tmp_path)
    cards = json.loads((tmp_path / "adjudication/candidates.json").read_text())["candidates"]
    assert len(cards) == 2
    assert {card.get("source_side", "head") for card in cards} == {"head", "base"}


def test_captured_unchanged_repository_file_keeps_its_head_frame(tmp_path: Path):
    report = _capture(tmp_path)
    fixture_path = tmp_path / "generation/fixture.json"
    fixture = json.loads(fixture_path.read_text())
    fixture["repository_contents_head"] = {"helpers.py": "def helper():\n    return 1\n"}
    data = json.dumps(fixture).encode()
    fixture_path.write_bytes(data)
    report["plan"]["cases"][0]["fixture_sha256"] = hashlib.sha256(data).hexdigest()
    path = tmp_path / "direct-one-shot.exchange.json"
    exchange = json.loads(path.read_text())
    response = json.loads(exchange["response"]["text"])
    response["findings"][0]["file_path"] = "helpers.py"
    exchange["response"]["text"] = json.dumps(response)
    path.write_text(json.dumps(exchange))
    write_packet(report, tmp_path)
    cards = json.loads((tmp_path / "adjudication/candidates.json").read_text())["candidates"]
    helper = next(card for card in cards if card["file_path"] == "helpers.py")
    assert "source_side" not in helper


@pytest.mark.parametrize("analysis_file", ["app.py", "other.py"])
def test_native_experiment_file_must_match_its_saved_request(
    tmp_path: Path, analysis_file: str
) -> None:
    report = _capture(tmp_path)
    report["cases"][0].update(arm="concise-scoped", analysis_file=analysis_file)
    if analysis_file != "app.py":
        with pytest.raises(ValueError, match="declared analysis file"):
            write_packet(report, tmp_path)
        assert not (tmp_path / "adjudication").exists()
        return
    write_packet(report, tmp_path)
    cards = json.loads((tmp_path / "adjudication/candidates.json").read_text())["candidates"]
    assert len(cards) == 1
    assert cards[0]["file_path"] == "app.py"


def test_declared_native_format_retains_workflow_file_locations(tmp_path: Path) -> None:
    report = _capture(tmp_path)
    report["cases"][0].update(arm="concise-workflow", finding_format="native-analyze")
    write_packet(report, tmp_path)
    cards = json.loads((tmp_path / "adjudication/candidates.json").read_text())["candidates"]
    assert len(cards) == 1
    assert cards[0]["file_path"] == "app.py"


@pytest.mark.parametrize("damage", ["fixture-drift", "exchange-escape"])
def test_wrong_or_escaping_evidence_cannot_create_a_packet(tmp_path: Path, damage: str) -> None:
    report = _capture(tmp_path)
    if damage == "fixture-drift":
        (tmp_path / "generation/fixture.json").write_text("{}")
    else:
        report["cases"][0]["raw_exchange_files"] = ["../outside.exchange.json"]
    with pytest.raises(ValueError):
        write_packet(report, tmp_path)
    assert not (tmp_path / "adjudication").exists()


def test_existing_human_judgments_are_never_overwritten(tmp_path: Path) -> None:
    report = _capture(tmp_path)
    write_packet(report, tmp_path)
    path = tmp_path / "adjudication/review.json"
    path.write_text('{"status":"reviewed"}')
    with pytest.raises(FileExistsError):
        write_packet(report, tmp_path)
    assert path.read_text() == '{"status":"reviewed"}'
