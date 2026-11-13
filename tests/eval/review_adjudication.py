"""Prepare anonymous candidate cards and a separate origin map for review.

Raw proposals and final findings share the same factual review. Neither format
validity nor admission establishes truth. Human judgments stay outside the
immutable candidate/source artifacts so editing a verdict cannot alter evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from outrider.coordinates import validate_removed_file
from outrider.coordinates.errors import CoordinateError
from outrider.llm.parsing import strip_outer_json_fence


def _artifact(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError("review artifact paths must be normalized and relative")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("review artifact escapes the experiment directory")
    return path


def _common_finding(
    value: Any, file_path: str | None = None, *, source_side: str = "head"
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not any(
        isinstance(value.get(key), str) and value[key]
        for key in ("title", "description", "evidence")
    ):
        return None
    common = {
        "file_path": value.get("file_path", file_path),
        **{
            key: value.get(key)
            for key in ("title", "description", "evidence", "line_start", "line_end")
        },
    }
    if source_side != "head":
        common["source_side"] = source_side
    return common


def _raw_source_side(
    request: dict[str, Any], files: dict[str, dict[str, Any]], file_path: Any, *, native: bool
) -> str:
    """Bind the primary frame to saved metadata/source, never a model-provided side."""
    if not isinstance(file_path, str) or file_path not in files:
        return "unknown"
    file = files[file_path]
    head_available = isinstance(file.get("content_head"), str)
    removed = False
    if file.get("status") == "removed":
        if not isinstance(file.get("content_base"), str) or not isinstance(file.get("patch"), str):
            return "unknown"
        try:
            validate_removed_file(
                file_path=file_path,
                status="removed",
                base_content=file.get("content_base"),
                head_content=file.get("content_head"),
                patch=file.get("patch"),
            )
        except CoordinateError:
            return "unknown"
        removed = True
    if not native:
        return "base" if removed else "head" if head_available else "unknown"
    context = request.get("context_summary", [])
    if not isinstance(context, list):
        return "unknown"
    matching = [
        entry
        for entry in context
        if isinstance(entry, dict) and entry.get("file_path") == file_path
    ]
    primary = [
        entry
        for entry in matching
        if entry.get("inclusion_reason") in ("changed_scope", "changed_region")
    ]
    frames = [entry.get("source_side", "head") for entry in primary or matching]
    if any(side not in ("head", "base") for side in frames):
        return "unknown"
    sides = set(frames)
    if len(sides) > 1 or context and not matching:
        return "unknown"
    if request.get("degradation_reason") == "file_removal":
        return "base" if removed and not sides - {"base"} else "unknown"
    if sides == {"base"}:
        return "base" if removed else "unknown"
    return "head" if head_available else "unknown"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def write_packet(report: dict[str, Any], directory: Path) -> None:
    """Write pending judgments without arm/model/policy or frequency hints.

    Full captured source supports the initial factual judgment. Only the separate
    origin map links actual served contexts, admission and known-mechanism labels.
    Raw unknown-category/rejected proposals remain eligible for factual review.
    """
    output = directory / "adjudication"
    if output.exists():
        raise FileExistsError("adjudication already exists; preserve existing judgments")
    sources: dict[str, dict[str, Any]] = {}
    case_keys: dict[str, str] = {}
    case_origins: dict[str, Any] = {}
    case_files: dict[str, dict[str, dict[str, Any]]] = {}
    for case in report["plan"]["cases"]:
        data = _artifact(directory, "generation/" + case["fixture"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != case["fixture_sha256"]:
            raise ValueError("adjudication source differs from the captured fixture")
        case_key = "case-" + digest[:24]
        if case_key in sources:
            raise ValueError("duplicate adjudication source identity")
        fixture = json.loads(data)
        sources[case_key] = {
            "fixture_sha256": digest,
            "pr_title": fixture.get("pr_title"),
            "files": fixture["files"],
            "repository_contents_head": fixture.get("repository_contents_head", {}),
        }
        case_keys[case["id"]] = case_key
        case_files[case_key] = {
            file["path"]: file
            for file in fixture["files"]
            if isinstance(file, dict) and isinstance(file.get("path"), str)
        }
        for path, content in fixture.get("repository_contents_head", {}).items():
            case_files[case_key].setdefault(path, {"content_head": content})
        case_origins[case_key] = {
            "case_id": case["id"],
            "fixture": case["fixture"],
            "provenance_source": case.get("provenance_source"),
            "capture_receipt": case.get("capture_receipt"),
            "oracle_manifest": case.get("oracle_manifest"),
        }

    cards: dict[str, dict[str, Any]] = {}
    origins: dict[str, list[dict[str, Any]]] = {}
    gaps: list[dict[str, Any]] = []
    visibility: list[dict[str, Any]] = []

    def add(
        value: Any,
        case_key: str,
        origin: dict[str, Any],
        file_path: str | None = None,
        *,
        source_side: str = "head",
    ) -> None:
        common = _common_finding(value, file_path, source_side=source_side)
        if common is None:
            gaps.append({**origin, "reason": "no-readable-candidate-fields"})
            return
        if source_side == "unknown":
            gaps.append({**origin, "reason": "unknown-source-frame"})
        card = {"case_key": case_key, **common}
        card_id = "candidate-" + hashlib.sha256(_json_bytes(card)).hexdigest()
        cards[card_id] = {"id": card_id, **card}
        origins.setdefault(card_id, []).append(origin)

    for row in report["cases"]:
        case_key = case_keys[row["case_id"]]
        slot = {key: row[key] for key in ("arm", "case_id", "rep", "status")}
        visibility.append(
            {
                **slot,
                "raw_exchange_files": row.get("raw_exchange_files", []),
                "diagnostic_events": row.get("diagnostic_events", []),
                "context_support": "pending; inspect actual requests, not full-source approval",
            }
        )
        for index, finding in enumerate(row.get("admitted_findings", [])):
            side = finding.get("source_side", "head") if isinstance(finding, dict) else "unknown"
            add(
                finding,
                case_key,
                {**slot, "stage": "final-finding", "index": index},
                source_side=side if side in ("head", "base") else "unknown",
            )
        for name in row.get("raw_exchange_files", []):
            exchange = json.loads(_artifact(directory, name).read_bytes())
            request = exchange["request"]
            if request.get("node_id") != "analyze":
                continue
            origin = {**slot, "stage": "raw-proposal", "exchange": name}
            try:
                response = json.loads(strip_outer_json_fence(exchange["response"]["text"]))
                proposals = response["findings"]
                if not isinstance(proposals, list):
                    raise ValueError("findings is not an array")
            except (ValueError, TypeError, KeyError, RecursionError):
                gaps.append({**origin, "reason": "unparseable-raw-findings"})
                continue
            file_path = row.get("analysis_file")
            native = (
                file_path is not None
                or row.get("finding_format") == "native-analyze"
                or row["arm"] in {"context-on", "context-off"}
            )
            if native:
                header = re.search(r"(?m)^File: ([^\r\n]+)$", request.get("user_prompt", ""))
                if file_path is not None and (header is None or header.group(1) != file_path):
                    raise ValueError("declared analysis file differs from the saved request")
                if file_path is None and header is not None:
                    file_path = header.group(1)
            for index, proposal in enumerate(proposals):
                path = (
                    proposal.get("file_path", file_path)
                    if isinstance(proposal, dict)
                    else file_path
                )
                side = _raw_source_side(request, case_files[case_key], path, native=native)
                if native and path != file_path:
                    side = "unknown"
                add(proposal, case_key, {**origin, "index": index}, file_path, source_side=side)

    candidates: dict[str, Any] = {
        "schema": "review-adjudication-v1",
        "scope": (
            "Arm/model labels, policy fields and occurrence counts withheld; not perfect blinding."
        ),
        "instructions": [
            "First judge each claim against the complete captured source, without the origin map.",
            "Record valid, incorrect, conditional or unresolved, with a rationale.",
            "Assess the issue separately from location, evidence and explanatory mistakes.",
            "After freezing factual judgments, use the origin map to match named mechanisms "
            "and inspect actual served context and pipeline drops.",
            "A full-source label does not establish what a workflow model actually saw.",
        ],
        "sources": [{"case_key": key, "path": f"sources/{key}.json"} for key in sorted(sources)],
        "candidates": [cards[key] for key in sorted(cards)],
    }
    if any(card.get("source_side") in ("base", "unknown") for card in cards.values()):
        candidates["instructions"].append(
            "source_side=base names captured BASE; unknown has no established frame. "
            "An omitted source_side names HEAD. Judge the claim against its stated source side."
        )
    candidate_bytes = _json_bytes(candidates)
    judgments = {
        "schema": "review-judgments-v1",
        "candidates_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "status": "pending",
        "judgments": [
            {
                "candidate_id": key,
                "verdict": "pending",
                "rationale": "",
                "location_accuracy": "pending",
                "evidence_accuracy": "pending",
                "explanation_accuracy": "pending",
                "named_mechanism_match": "pending-after-unblinding",
            }
            for key in sorted(cards)
        ],
    }
    identity_map = {
        "schema": "review-origins-v1",
        "cases": case_origins,
        "candidate_origins": origins,
        "actual_visibility": visibility,
        "collection_gaps": gaps,
        "quality": "pending independent adjudication; no automatic quality scores or winner",
    }
    output.mkdir()
    (output / "sources").mkdir()
    for key, source in sources.items():
        (output / "sources" / f"{key}.json").write_bytes(_json_bytes(source))
    (output / "candidates.json").write_bytes(candidate_bytes)
    (output / "review.json").write_bytes(_json_bytes(judgments))
    (output / "identity-map.json").write_bytes(_json_bytes(identity_map))
