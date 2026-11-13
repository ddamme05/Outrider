"""Audit recorded admission and selection decisions without new model calls.

Truth cards reuse the existing anonymous adjudication packet. Parser replay is
a separate observation from saved native workflow events. Neither establishes
truth. Draft selection remains separate from analysis selection and publication.
No provider, database or automatic policy judgment is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy.canonical import compute_proposal_hash, compute_response_hash
from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw

from .direct_first_candidates import decode_candidate_json
from .rejection_audit_observations import recorded_observations
from .review_diagnostics_inputs import artifact, sha

SCHEMA = "rejected-findings-audit-v1"
LIMITATION = (
    "Selected recorded claims only. Admission and selection are not truth verdicts or gate bugs. "
    "Offline parser replay does not exercise delivery. A completed draft is not publication. "
    "Missing draft membership does not establish a cap loss. "
    "Counts do not estimate a broader false-negative rate or authorize promotion. "
    "Receipts establish local consistency, not independent authenticity."
)


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _read(path: Path) -> Any:
    return decode_candidate_json(path.read_text())


def _verify(directory: Path) -> dict[str, str]:
    receipts = _read(directory / "receipts.json")
    if not isinstance(receipts, dict) or not receipts:
        raise ValueError("a nonempty receipt inventory is required")
    for name, digest in receipts.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or sha(artifact(directory, name)) != digest
        ):
            raise ValueError(f"artifact differs from its receipt: {name}")
    return receipts


def _report(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipts = _verify(directory)
    for name in (
        "plan.json",
        "results.json",
        "adjudication/candidates.json",
        "adjudication/identity-map.json",
    ):
        if name not in receipts:
            raise ValueError(f"report input lacks a receipt: {name}")
    report = _read(directory / "results.json")
    plan = _read(directory / "plan.json")
    if (
        report.get("schema") not in {"review-comparison-v1", "captured-review-diagnostics-v1"}
        or report["plan"] != plan
    ):
        raise ValueError("a matching saved comparison or diagnostic report is required")
    for name, digest in plan.get("inputs", {}).items():
        if sha(artifact(directory, "generation/" + name)) != digest:
            raise ValueError("captured generation input differs from its plan")
    for case in plan["cases"]:
        if sha(artifact(directory, "generation/" + case["fixture"])) != case["fixture_sha256"]:
            raise ValueError("captured fixture differs from its plan")
    cards = _read(directory / "adjudication/candidates.json")
    for source in cards["sources"]:
        if "adjudication/" + source["path"] not in receipts:
            raise ValueError("adjudication source lacks a receipt")
    for row in report["cases"]:
        if any(name not in receipts for name in row.get("raw_exchange_files", [])):
            raise ValueError("saved exchange lacks a receipt")
    return report, cards, _read(directory / "adjudication/identity-map.json")


def _replay_rows(
    replay: dict[str, Any] | None,
) -> dict[tuple[Any, ...], tuple[int, dict[str, Any]]]:
    rows: dict[tuple[Any, ...], tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate([] if replay is None else replay["rows"]):
        key = (row["case_id"], row["rep"], row["arm"], row["exchange"])
        if key in rows:
            raise ValueError("parser replay repeats a saved exchange")
        rows[key] = (index, row)
    return rows


def _dispositions(row: dict[str, Any], hashes: list[str | None]) -> list[dict[str, Any]]:
    """Join only exact, validated proposal identities inside one saved response."""
    rejected: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for entry in row["proposal_rejections"]:
        rejected[entry["proposal_hash"]].append(entry)
    admitted = Counter(finding["proposal_hash"] for finding in row["admitted_findings"])
    if None in hashes or Counter(hashes) != admitted + Counter(
        {key: len(items) for key, items in rejected.items()}
    ):
        raise ValueError("parser replay proposal inventory does not match the exact saved response")
    counters = row["counters"]
    if (
        counters["n_proposals_seen"],
        counters["n_findings_emitted"],
        counters["n_proposals_rejected"],
    ) != (len(hashes), len(row["admitted_findings"]), len(row["proposal_rejections"])) or row.get(
        "response_rejection"
    ) is not None:
        raise ValueError("parser replay counters do not match its occurrence inventory")
    results = []
    for digest in hashes:
        assert digest is not None
        if rejected[digest]:
            if admitted[digest] or len({item["rejection_reason"] for item in rejected[digest]}) > 1:
                raise ValueError("identical proposals have ambiguous parser dispositions")
            entry = rejected[digest].popleft()
            results.append(
                {
                    "admission": "rejected",
                    "reason": entry["rejection_reason"],
                    "detail": entry["rejection_detail"],
                }
            )
        else:
            admitted[digest] -= 1
            results.append({"admission": "admitted", "reason": None, "detail": None})
    return results


def _inventory(
    directory: Path, report: dict[str, Any], origins: dict[str, Any], replay: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    card_ids: dict[tuple[Any, ...], str] = {}
    for candidate_id, occurrences in origins["candidate_origins"].items():
        for origin in occurrences:
            if origin["stage"] != "raw-proposal":
                continue
            occurrence_key = (
                origin["case_id"],
                origin["rep"],
                origin["arm"],
                origin["exchange"],
                origin["index"],
            )
            if occurrence_key in card_ids:
                raise ValueError("adjudication repeats an occurrence identity")
            card_ids[occurrence_key] = candidate_id
    replays = _replay_rows(replay)
    roster, gaps, seen = [], list(origins["collection_gaps"]), set()
    source_receipts_hash = sha(directory / "receipts.json")
    for row in report["cases"]:
        for name in row.get("raw_exchange_files", []):
            exchange = _read(artifact(directory, name))
            request, response = exchange["request"], exchange["response"]
            if request["node_id"] != "analyze":
                continue
            if name in seen:
                raise ValueError("report repeats an analyzer exchange")
            seen.add(name)
            key = (row["case_id"], row["rep"], row["arm"], name)
            try:
                raw = decode_candidate_json(strip_outer_json_fence(response["text"]))
                proposals = raw["findings"]  # type: ignore[index]
                if not isinstance(proposals, list):
                    raise ValueError("findings is not an array")
            except (ValueError, TypeError, KeyError, RecursionError):
                gaps.append({"exchange": name, "reason": "unreadable-response-envelope"})
                if key in replays:
                    raise ValueError(
                        "parser replay cannot identify an unreadable saved response"
                    ) from None
                continue
            headers = [
                line[6:]
                for line in request["user_prompt"].splitlines()
                if line.startswith("File: ")
            ]
            source_file = row.get("analysis_file")
            if source_file is not None and headers != [source_file]:
                raise ValueError("analysis file differs from the exact saved request")
            source_file = headers[0] if len(headers) == 1 else None
            sides = {
                entry.get("source_side", "head")
                for entry in request.get("context_summary", [])
                if entry.get("file_path") == source_file
                and entry.get("inclusion_reason") in {"changed_scope", "changed_region"}
            }
            source_side = (
                next(iter(sides)) if len(sides) == 1 else "head" if not sides else "unknown"
            )
            response_hash = compute_response_hash(response["text"])
            hashes: list[str | None] = []
            for proposal in proposals:
                try:
                    if not isinstance(source_file, str) or source_side not in {"head", "base"}:
                        raise ValueError("proposal has no established primary source frame")
                    parsed = AnalyzeFindingProposalRaw.model_validate(proposal)
                    digest = compute_proposal_hash(
                        source_file_path=source_file,
                        source_side="base" if source_side == "base" else "head",
                        **parsed.model_dump(exclude={"trace_candidates"}),
                    )
                except (ValueError, TypeError, ValidationError):
                    digest = None
                hashes.append(digest)
            replay_index, replay_row = replays.pop(key, (None, None))
            if replay_row is not None and any(
                entry["file_path"] != source_file or entry.get("source_side", "head") != source_side
                for entry in replay_row["proposal_rejections"]
            ):
                raise ValueError("parser rejection source differs from its saved request")
            if (
                replay_row is not None
                and replay_row.get("response_hash", response_hash) != response_hash
            ):
                raise ValueError("parser replay response hash differs from the saved exchange")
            recorded = recorded_observations(
                row,
                exchange,
                report_schema=report["schema"],
                file_path=source_file,
                source_side=source_side,
                proposal_hashes=hashes,
            )
            if recorded is not None and replay_row is not None:
                raise ValueError("audit recorded native admission separately from parser replay")
            if recorded is not None:
                dispositions = recorded
            elif replay_row is not None:
                dispositions = _dispositions(replay_row, hashes)
            else:
                dispositions = [
                    {"admission": "unknown", "reason": None, "detail": None} for _ in proposals
                ]
            for index, disposition in enumerate(dispositions):
                identity = [source_receipts_hash, name, response_hash, index]
                roster.append(
                    {
                        "occurrence_id": hashlib.sha256(_bytes(identity)).hexdigest(),
                        "candidate_id": card_ids.get((*key, index)),
                        "case_id": row["case_id"],
                        "rep": row["rep"],
                        "arm": row["arm"],
                        "exchange": name,
                        "proposal_index": index,
                        "response_hash": response_hash,
                        "proposal_hash": hashes[index],
                        "file_path": source_file,
                        "source_side": source_side,
                        "replay_row_index": replay_index,
                        "observation": "not_observed"
                        if replay_row is None
                        else "offline_parser_replay",
                        "delivery": "unknown" if replay_row is None else "not_run",
                        **disposition,
                    }
                )
    if replays:
        raise ValueError("parser replay names an exchange outside the saved report")
    return roster, gaps


def _write(output: Path, files: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = artifact(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    receipts = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()
        if name != "blind/review.json"
    }
    (output / "receipts.json").write_bytes(_bytes(receipts))


def _audit_reason(occurrence: dict[str, Any]) -> str | None:
    """Keep original rejection reasons separate from observed selection gaps."""
    if occurrence["admission"] in {"rejected", "malformed"}:
        reason = occurrence["reason"]
        if not isinstance(reason, str):
            raise ValueError("observed rejection lacks its recorded reason")
        return reason
    if occurrence["delivery"] in {"included_in_draft", "duplicate_in_draft"}:
        return None
    if occurrence.get("analysis_selected") is False:
        return "not_selected_in_analysis"
    if occurrence["delivery"] == "not_in_recorded_draft":
        return "not_in_recorded_draft"
    return None


def prepare(
    report: Path, *, output: Path, admission_replay: Path | None = None, limit: int = 20
) -> dict[str, Any]:
    """Freeze the full roster before selecting occurrences by reason and identity."""
    if type(limit) is not int or limit < 1:
        raise ValueError("sample limit must be a positive integer")
    if output.resolve().is_relative_to(report.resolve()):
        raise ValueError("audit output must be outside the original report")
    results, packet, origins = _report(report)
    replay = None if admission_replay is None else _read(admission_replay)
    if replay is not None and replay.get(
        "original_report_receipts_sha256", sha(report / "receipts.json")
    ) != sha(report / "receipts.json"):
        raise ValueError("parser replay belongs to different report receipts")
    roster, gaps = _inventory(report, results, origins, replay)
    cards = {card["id"]: card for card in packet["candidates"]}
    for item in roster:
        item["audit_reason"] = _audit_reason(item)
    eligible = [item for item in roster if item["audit_reason"] and item["candidate_id"] in cards]
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for item in sorted(eligible, key=lambda item: item["occurrence_id"]):
        groups[item["audit_reason"]].append(item)
    selected: list[str] = []
    while len(selected) < min(limit, len(eligible)):
        for reason in sorted(groups):
            if groups[reason] and len(selected) < limit:
                selected.append(groups[reason].popleft()["occurrence_id"])
    selected_ids = {item["candidate_id"] for item in roster if item["occurrence_id"] in selected}
    selected_cards = [cards[key] for key in sorted(selected_ids)]
    cases = {card["case_key"] for card in selected_cards}
    sources = [source for source in packet["sources"] if source["case_key"] in cases]
    blind = {**packet, "candidates": selected_cards, "sources": sources}
    blind["instructions"] = [
        "Judge these claims against captured source before opening identity-map.json.",
        "Assess truth, location, evidence, explanation and PR relevance separately.",
        "Use pending or unresolved for missing evidence. Cite exact complete source lines.",
        "Freeze external judgments with complete before inspecting origins. Blinding is partial.",
    ]
    candidate_bytes = _bytes(blind)
    judgments = {
        "schema": "review-judgments-v1",
        "status": "pending",
        "candidates_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "judgments": [
            {
                "candidate_id": card["id"],
                "verdict": "pending",
                "rationale": "",
                "trigger": "",
                "consequence": "",
                "sources": [],
                "location_accuracy": "pending",
                "evidence_accuracy": "pending",
                "explanation_accuracy": "pending",
                "pr_relevance": "pending",
                "named_mechanism_match": "pending-after-unblinding",
            }
            for card in selected_cards
        ],
    }
    summary = {
        "schema": SCHEMA,
        "status": "prepared",
        "model_calls_made": 0,
        "database_calls": 0,
        "report_schema": results["schema"],
        "report_status": results["status"],
        "observation_status": "observed_rejections_or_selection_gaps"
        if eligible
        else "no_observed_rejections",
        "counts": {
            "raw_occurrences": len(roster),
            "admitted_occurrences": sum(item["admission"] == "admitted" for item in roster),
            "eligible_occurrences": len(eligible),
            "eligible_rejected_occurrences": sum(
                item["admission"] in {"rejected", "malformed"} for item in eligible
            ),
            "eligible_selection_gaps": sum(item["admission"] == "admitted" for item in eligible),
            "selected_occurrences": len(selected),
            "selected_unique_claims": len(selected_ids),
            "not_sampled": len(eligible) - len(selected),
            "unjoined_rejections": sum(
                item["admission"] in {"rejected", "malformed"} and item["candidate_id"] not in cards
                for item in roster
            ),
            "unjoined_selection_gaps": sum(
                item["admission"] == "admitted"
                and item["audit_reason"] is not None
                and item["candidate_id"] not in cards
                for item in roster
            ),
            "unknown_dispositions": sum(item["admission"] == "unknown" for item in roster),
            "collection_gaps": len(gaps),
            "incomplete_rows": sum(row["status"] != "completed" for row in results["cases"]),
        },
        "reasons": dict(Counter(item["audit_reason"] for item in eligible)),
        "admissions": dict(Counter(item["admission"] for item in roster)),
        "observation_sources": dict(Counter(item["observation"] for item in roster)),
        "analysis_selection": dict(
            Counter(
                "selected"
                if item.get("analysis_selected") is True
                else "not_selected"
                if item.get("analysis_selected") is False
                else "unknown"
                for item in roster
                if item["observation"] == "recorded_native_candidate"
            )
        ),
        "native_draft_outcomes": dict(
            Counter(
                item["delivery"]
                for item in roster
                if item["observation"] == "recorded_native_candidate"
            )
        ),
        "limitation": LIMITATION,
    }
    identity = {
        "source_report": str(report.resolve()),
        "source_receipts_sha256": sha(report / "receipts.json"),
        "admission_replay": None if admission_replay is None else str(admission_replay.resolve()),
        "admission_replay_sha256": None if admission_replay is None else sha(admission_replay),
        "selection": (
            "round-robin sorted audit reasons, then sorted occurrence identities; "
            "no truth labels used"
        ),
        "selected_occurrence_ids": selected,
        "roster": roster,
        "collection_gaps": gaps,
    }
    files = {
        "audit.json": _bytes(summary),
        "identity-map.json": _bytes(identity),
        "blind/candidates.json": candidate_bytes,
        "blind/review.json": _bytes(judgments),
        "inputs/report-receipts.json": (report / "receipts.json").read_bytes(),
    }
    if admission_replay is not None:
        files["inputs/admission-replay.json"] = admission_replay.read_bytes()
    for source in sources:
        files["blind/" + source["path"]] = artifact(
            report, "adjudication/" + source["path"]
        ).read_bytes()
    _write(output, files)
    return summary


def _judgments(audit: Path, review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packet = _read(audit / "blind/candidates.json")
    if review.get("candidates_sha256") != sha(audit / "blind/candidates.json"):
        raise ValueError("judgments name a different blind packet")
    cards = {card["id"]: card for card in packet["candidates"]}
    judgments = {}
    for item in review["judgments"]:
        candidate_id = item["candidate_id"]
        if candidate_id not in cards or candidate_id in judgments:
            raise ValueError("judgments repeat or name an unselected claim")
        if item["verdict"] not in {"pending", "valid", "incorrect", "conditional", "unresolved"}:
            raise ValueError("unknown factual verdict")
        if item["pr_relevance"] not in {
            "pending",
            "introduced_or_worsened",
            "fix_attempt",
            "unrelated_preexisting",
            "uncertain",
        }:
            raise ValueError("unknown PR relevance judgment")
        for key in ("location_accuracy", "evidence_accuracy", "explanation_accuracy"):
            if item[key] not in {"accurate", "inaccurate", "uncertain", "not_assessed", "pending"}:
                raise ValueError("unknown accuracy judgment")
        if item["verdict"] != "pending" and not item["rationale"].strip():
            raise ValueError("a factual judgment requires a rationale")
        if item["verdict"] in {"valid", "incorrect", "conditional"} and not item["sources"]:
            raise ValueError("definite and conditional judgments require source references")
        if item["verdict"] in {"valid", "conditional"} and not all(
            item[key].strip() for key in ("trigger", "consequence")
        ):
            raise ValueError("valid and conditional claims require a trigger and consequence")
        for reference in item["sources"]:
            if reference["source"] != "sources/" + cards[candidate_id]["case_key"] + ".json":
                raise ValueError("judgment citation belongs to a different captured case")
            source = _read(artifact(audit / "blind", reference["source"]))
            side = reference["source_side"]
            if side not in {"head", "base"}:
                raise ValueError("source citation must name HEAD or BASE")
            contents = {file["path"]: file.get("content_" + side) for file in source["files"]}
            if side == "head":
                contents = {
                    **source.get("repository_contents_head", {}),
                    **{key: value for key, value in contents.items() if value is not None},
                }
            content = contents.get(reference["file_path"])
            start, end = reference["line_start"], reference["line_end"]
            if (
                not isinstance(content, str)
                or type(start) is not int
                or type(end) is not int
                or not 1 <= start <= end <= len(content.splitlines())
            ):
                raise ValueError("source citation has no captured line range")
            if reference["quote"] != "\n".join(content.splitlines()[start - 1 : end]):
                raise ValueError("source citation differs from complete captured lines")
        judgments[candidate_id] = item
    if judgments.keys() != cards.keys():
        raise ValueError("judgments must cover every selected claim exactly")
    return judgments


def complete(audit: Path, judgments: Path, *, output: Path) -> dict[str, Any]:
    """Freeze external judgments before writing any unblinded derived report."""
    if output.resolve().is_relative_to(audit.resolve()):
        raise ValueError("completion output must be outside the prepared audit")
    receipts = _verify(audit)
    required = {
        "audit.json",
        "identity-map.json",
        "blind/candidates.json",
        "inputs/report-receipts.json",
    }
    packet = _read(audit / "blind/candidates.json")
    required.update("blind/" + source["path"] for source in packet["sources"])
    if not required <= receipts.keys():
        raise ValueError("prepared audit lacks required immutable receipts")
    review_bytes = judgments.read_bytes()
    review = decode_candidate_json(review_bytes.decode())
    if not isinstance(review, dict):
        raise ValueError("judgments must be an object")
    reviewed = _judgments(audit, review)
    origins = _read(audit / "identity-map.json")
    report = Path(origins["source_report"])
    if output.resolve().is_relative_to(report.resolve()):
        raise ValueError("completion output must be outside the original report")
    if sha(report / "receipts.json") != origins["source_receipts_sha256"]:
        raise ValueError("original report receipts changed after preparation")
    _report(report)
    if sha(audit / "inputs/report-receipts.json") != origins["source_receipts_sha256"]:
        raise ValueError("prepared report receipt copy differs from its pinned source")
    if origins["admission_replay"] is not None and "inputs/admission-replay.json" not in receipts:
        raise ValueError("prepared parser replay lacks a receipt")
    if (
        origins["admission_replay"] is not None
        and sha(Path(origins["admission_replay"])) != origins["admission_replay_sha256"]
    ):
        raise ValueError("original parser replay changed after preparation")
    rows = []
    outcomes: Counter[str] = Counter()
    for item in origins["roster"]:
        if item["occurrence_id"] not in origins["selected_occurrence_ids"]:
            continue
        judgment = reviewed[item["candidate_id"]]
        verdict = judgment["verdict"]
        outcome = verdict
        if verdict == "valid":
            if item["admission"] in {"rejected", "malformed"}:
                outcome = "valid_claim_rejected_by_parser"
            elif item.get("audit_reason") == "not_selected_in_analysis":
                outcome = "valid_claim_not_selected_in_analysis"
            else:
                outcome = "valid_claim_not_in_recorded_draft"
        outcomes[outcome] += 1
        rows.append(
            {
                **item,
                "judgment": judgment,
                "audit_outcome": outcome,
                "parser_outcome": outcome
                if item["admission"] in {"rejected", "malformed"}
                else None,
                "current_policy_assessment": "unassessed",
                "policy_rationale": "",
            }
        )
    pending_assessments = Counter(
        key
        for item in reviewed.values()
        for key in (
            "verdict",
            "pr_relevance",
            "location_accuracy",
            "evidence_accuracy",
            "explanation_accuracy",
        )
        if item[key] == "pending"
    )
    prepared = _read(audit / "audit.json")
    if reviewed:
        status = "incomplete" if pending_assessments else "judgments_frozen"
    elif prepared.get("observation_sources", {}).get("recorded_native_candidate"):
        status = "no_selected_claims"
    else:
        status = "no_observations"
    summary = {
        **prepared,
        "status": status,
        "judgments_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "unique_claim_verdicts": dict(Counter(item["verdict"] for item in reviewed.values())),
        "selected_occurrence_outcomes": dict(outcomes),
        "current_policy_assessment": "unassessed",
        "pending_assessments": dict(pending_assessments),
    }
    lines = [
        "# Recorded rejection audit",
        "",
        LIMITATION,
        "",
        f"Status: {summary['status']}",
        "",
        "Count | Value",
        "--- | ---",
        *[f"{key} | {value}" for key, value in summary["counts"].items()],
        "",
        "Outcome | Selected occurrences",
        "--- | ---",
        *[f"{key} | {value}" for key, value in sorted(outcomes.items())],
        "",
        "Current-policy assessments remain unassessed. A valid source claim alone does not "
        "establish an unjustified rejection.",
        "Inspect unblinded.json beside the frozen judgments for original reasons and "
        "admission, analysis selection and draft limits.",
    ]
    _write(
        output,
        {
            "judgments.frozen.json": review_bytes,
            "audit.json": _bytes(summary),
            "unblinded.json": _bytes(rows),
            "REPORT.md": ("\n".join(lines) + "\n").encode(),
            "prepared-receipts.json": (audit / "receipts.json").read_bytes(),
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", allow_abbrev=False)
    preparing.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Saved review comparison or captured review diagnostic directory with receipts.json.",
    )
    preparing.add_argument("--admission-replay", type=Path)
    preparing.add_argument("--limit", type=int, default=20)
    preparing.add_argument("--output", type=Path, required=True)
    completing = commands.add_parser("complete", allow_abbrev=False)
    completing.add_argument("--audit", type=Path, required=True)
    completing.add_argument("--judgments", type=Path, required=True)
    completing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            prepare(
                args.report,
                output=args.output,
                admission_replay=args.admission_replay,
                limit=args.limit,
            )
            if args.command == "prepare"
            else complete(args.audit, args.judgments, output=args.output)
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        if (
            args.command == "prepare"
            and isinstance(exc, FileNotFoundError)
            and exc.filename == str(args.report / "receipts.json")
        ):
            preparing.error(
                f"Report receipt file {exc.filename} is missing. Pass a saved review comparison "
                "or captured review diagnostic directory containing receipts.json."
            )
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
