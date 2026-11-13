"""Measure visible-context quality and reanalyze saved refresh exchanges offline.

Original paid artifacts remain immutable. Archived production code reproduces
each request and admitted result; current, explicitly versioned eval checks then
describe what the visible evidence supports. No model client is constructed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
V1_CONTRACT = "refresh-judged-origin-excluded-three-reps-v1"
V2_CONTRACT = "refresh-grounded-visible-context-v2"
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_CALL_BYTES = 8 * 1024 * 1024
MAX_CALLS = 2_000
MAX_INPUT_FILES = 1_000
MAX_REPETITIONS = 10
MEASUREMENT_FILES = (
    "tests/eval/review_quality.py",
    "tests/eval/fixtures/review_quality_context.json",
    "tests/eval/model_refresh_regrade.py",
    "tests/eval/model_refresh_nodes.py",
    "tests/eval/model_comparison.py",
    "tests/eval/grading.py",
    "tests/eval/model_refresh_report.py",
    "tests/eval/model_refresh.py",
    "tests/eval/grounded_cases.py",
    "tests/eval/real_pr_cases.py",
)


class ReplayIntegrityError(ValueError):
    """Saved evidence or replay identity differs; never an original model failure."""


def measurement_inputs(suite: str = "legacy") -> dict[str, str]:
    """Pin the current v2 criteria separately from original generation inputs."""
    inputs = {name: _sha(ROOT / name) for name in MEASUREMENT_FILES}
    if suite == "grounded":
        from .grounded_cases import dataset_inputs

        inputs.update(dataset_inputs())
    elif suite == "real-pr":
        from .real_pr_cases import dataset_inputs

        inputs.update(dataset_inputs())
    elif suite != "legacy":
        raise ValueError("unknown model comparison suite")
    return inputs


def score_case_v2(result, scenario, *, fixture_path: Path, request) -> dict[str, Any]:
    """Keep original producer coverage, grounded matches and unknowns separate."""
    from .grading import finding_matches
    from .review_quality import MEASUREMENT_CONTRACT_V2, assess_case

    coverage = result.producer_coverage_findings
    if coverage is None:
        raise ValueError("v2 requires independently collected original producer coverage")
    if len(scenario.expected) > 1:
        raise ValueError("v2 currently supports one expected issue per fixture")
    producer_hashes = {finding.content_hash for finding in coverage}
    model_findings = tuple(
        finding for finding in result.findings if finding.content_hash not in producer_hashes
    )
    expected = tuple(
        item
        for item in scenario.expected
        if not any(finding_matches(finding, item) for finding in coverage)
    )
    assessments = assess_case(model_findings, expected, fixture_path=fixture_path, request=request)
    product = assess_case(
        result.findings, scenario.expected, fixture_path=fixture_path, request=request
    )
    supported = [item for item in assessments if item.status != "unscored"]
    product_supported = [item for item in product if item.status != "unscored"]
    unscored = [item for item in assessments if item.status == "unscored"]
    product_unscored = [item for item in product if item.status == "unscored"]
    pending = sum(item.status == "needs-adjudication" for item in supported)
    product_pending = sum(item.status == "needs-adjudication" for item in product_supported)
    safe = not scenario.expected
    return {
        "model_dependent": {
            "matched": sum(item.status == "matched" for item in supported),
            "expected": len(supported),
            "pending": pending,
            "extras_advisory": len(model_findings)
            - len({index for item in supported for index in item.matching_indices}),
            "false_alarms": None,
        },
        "producer_covered_expected": len(scenario.expected) - len(expected),
        "product_grounded": {
            "matched": sum(item.status == "matched" for item in product_supported),
            "expected": len(product_supported),
            "pending": product_pending,
        },
        "response_rejections": result.response_rejections,
        "proposal_rejections": result.proposal_rejections,
        "findings": [finding.model_dump(mode="json") for finding in result.findings],
        "producer_hashes": sorted(producer_hashes),
        "measurement_v2": {
            "measurement_contract": MEASUREMENT_CONTRACT_V2,
            "grounded_issue_assessments": [asdict(item) for item in assessments],
            "product_assessments": [asdict(item) for item in product],
            "unscored_expected": len(product_unscored),
            "unscored_model_expected": len(unscored),
            "pending_expected": product_pending,
            "pending_model_expected": pending,
            "safe_model_emissions": len(model_findings) if safe else None,
            "safe_product_emissions": len(result.findings) if safe else None,
            "model_precision_status": "requires-adjudication"
            if safe and model_findings
            else "zero-emissions-observed"
            if safe
            else "not-assessed",
            "semantic_adjudication": "pending",
            "product_precision_status": "requires-adjudication"
            if safe and result.findings
            else "zero-emissions-observed"
            if safe
            else "not-assessed",
            "scope": (
                "Bounded source grounding and declared locations; "
                "not a semantic correctness oracle."
            ),
        },
    }


def _read_json(path: Path, limit: int = MAX_CALL_BYTES) -> Any:
    if not path.is_file() or path.stat().st_size > limit:
        raise ReplayIntegrityError("missing or oversized replay input")
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ReplayIntegrityError("replay input path must stay within its declared root")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ReplayIntegrityError("replay input path escapes its declared root")
    return path


def _request_identity(request: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request.items() if key != "review_id"}


def _finding_identity(finding: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in finding.items() if key not in {"finding_id", "review_id"}}


def case_slot(case: dict[str, Any]) -> str:
    if case["phase"] == "compat":
        return f"compat:{case['fixture']}"
    if case["phase"] == "analyze":
        return f"analyze:{case['rep']}:{case['fixture']}"
    if case["phase"] == "nodes":
        return f"nodes:{case['rep']}:{case['name']}"
    raise ReplayIntegrityError("unsupported saved case phase")


def planned_slots(plan: dict[str, Any]) -> tuple[str, ...]:
    """Describe the bounded refresh protocol independently of observed calls."""
    reps = plan["repetitions"]
    if type(reps) is not int or not 1 <= reps <= MAX_REPETITIONS:
        raise ReplayIntegrityError("unsupported refresh repetition count")
    if plan["phase"] not in {"compat", "analyze", "nodes", "all"}:
        raise ReplayIntegrityError("unsupported refresh phase")
    paths = [row["fixture"] for row in plan["scenarios"]]
    names = plan["node_cases"]
    compatibility = plan["compatibility_cases"]
    if (
        len(paths) > 100
        or len(names) > 50
        or len(set(paths)) != len(paths)
        or len(set(names)) != len(names)
        or len(set(compatibility)) != len(compatibility)
        or (plan.get("suite", "legacy") == "legacy" and not set(compatibility) <= set(paths))
    ):
        raise ReplayIntegrityError("invalid or duplicate declared refresh cases")
    slots = [f"compat:{path}" for path in compatibility]
    for rep in range(reps):
        if plan["phase"] in {"analyze", "all"}:
            slots.extend(f"analyze:{rep}:{path}" for path in paths)
        if plan["phase"] in {"nodes", "all"}:
            slots.extend(f"nodes:{rep}:{name}" for name in names)
    if not slots or len(slots) > MAX_CALLS:
        raise ReplayIntegrityError("unsupported refresh cohort size")
    if (
        plan.get("suite", "legacy") not in {"legacy", "grounded", "real-pr"}
        or (plan.get("suite") == "grounded" and plan["phase"] not in {"compat", "analyze"})
        or (plan.get("suite") == "real-pr" and plan["phase"] != "analyze")
    ):
        raise ReplayIntegrityError("unsupported saved comparison suite or phase")
    return tuple(slots)


def _verify_original(report_path: Path, source: Path, receipt_manifest: Path):
    report_path, source = report_path.resolve(), source.resolve()
    if not source.is_dir():
        raise ReplayIntegrityError("an existing original source archive is required")
    report_dir = report_path.parent
    manifest = _read_json(receipt_manifest, MAX_REPORT_BYTES)
    entries = [files for name, files in manifest.items() if (ROOT / name).resolve() == report_dir]
    if len(entries) != 1:
        raise ReplayIntegrityError("receipt manifest must identify exactly this original report")
    receipts = entries[0]
    if (
        report_path.name != "results.json"
        or "results.json" not in receipts
        or "plan.json" not in receipts
    ):
        raise ReplayIntegrityError(
            "the canonical original results and plan must be receipt-covered"
        )
    if _sha(report_path) != receipts["results.json"]:
        raise ReplayIntegrityError("the selected original report is not the pinned results receipt")
    if len(receipts) > 2 * MAX_CALLS + 8:
        raise ReplayIntegrityError("too many original receipt files")
    for name, digest in receipts.items():
        if _sha(_contained(report_dir, name)) != digest:
            raise ReplayIntegrityError("original receipt hash mismatch")
    report = _read_json(report_path, MAX_REPORT_BYTES)
    if (
        report.get("schema") != "model-refresh-v1"
        or report.get("evidence_kind") != "live-provider"
        or report["plan"].get("measurement_contract") not in {V1_CONTRACT, V2_CONTRACT}
    ):
        raise ReplayIntegrityError(
            "offline reanalysis requires an original live refresh report with a known contract"
        )
    if report["plan"] != _read_json(report_dir / "plan.json", MAX_REPORT_BYTES):
        raise ReplayIntegrityError("saved plan and result identities differ")
    inputs = report["plan"]["inputs"]
    if not inputs or len(inputs) > MAX_INPUT_FILES:
        raise ReplayIntegrityError("invalid original source input set")
    for name, digest in inputs.items():
        if _sha(_contained(source, name)) != digest:
            raise ReplayIntegrityError(f"original source input hash mismatch: {name}")
    if report["plan"]["measurement_contract"] == V2_CONTRACT:
        criteria = report["plan"].get("measurement_inputs")
        if not criteria:
            raise ReplayIntegrityError("the original v2 measurement criteria are missing")
        for name, digest in criteria.items():
            if _sha(_contained(source, name)) != digest:
                raise ReplayIntegrityError("the original v2 measurement criteria changed")
    elif report["plan"].get("suite", "legacy") != "legacy":
        raise ReplayIntegrityError("v1 collection supports only the legacy suite")
    calls = [call for candidate in report["candidates"] for call in candidate["calls"]]
    if not calls or len(calls) > MAX_CALLS or len({call["index"] for call in calls}) != len(calls):
        raise ReplayIntegrityError("empty, oversized or duplicate saved call cohort")
    actual_files = {
        str(path.relative_to(report_dir))
        for path in (report_dir / "calls").iterdir()
        if path.suffix == ".json"
    }
    if actual_files != {
        name for name in receipts if name.startswith("calls/") and name.endswith(".json")
    }:
        raise ReplayIntegrityError("unexpected or missing original call receipts")
    expected_slots = set(planned_slots(report["plan"]))
    for candidate in report["candidates"]:
        if len({case_slot(case) for case in candidate["cases"]}) != len(candidate["cases"]):
            raise ReplayIntegrityError("duplicate saved case slots")
        if len({call["scenario"] for call in candidate["calls"]}) != len(candidate["calls"]):
            raise ReplayIntegrityError("duplicate saved response slots")
        case_slots = {case_slot(case) for case in candidate["cases"]}
        call_slots = {call["scenario"] for call in candidate["calls"]}
        if not case_slots <= call_slots <= expected_slots:
            raise ReplayIntegrityError("saved calls or results lie outside the declared slots")
        if candidate["status"] == "completed" and (
            case_slots != expected_slots or call_slots != expected_slots
        ):
            raise ReplayIntegrityError("completed target has partial saved coverage")
        for call in candidate["calls"]:
            stem = f"calls/{call['index']:06d}"
            if call != _read_json(report_dir / f"{stem}.attempt.json"):
                raise ReplayIntegrityError("saved attempt differs from reported call")
            target = candidate["target"]
            if call["target"] != f"{target['host']}:{target['model']}:{target['role']}":
                raise ReplayIntegrityError("saved attempt target differs from its candidate")
            if call["persisted_exchange"]:
                exchange = _read_json(report_dir / f"{stem}.exchange.json")
                if (
                    exchange["scenario"] != call["scenario"]
                    or exchange["call_index"] != call["index"]
                    or exchange["target"] != candidate["target"]
                    or exchange["request"]["model"] != candidate["target"]["model"]
                    or exchange["response"]["model"] != exchange["event"]["model"]
                    or exchange["event"]["event_id"] != call["audit_event_id"]
                ):
                    raise ReplayIntegrityError("saved response/request slot identity mismatch")
    return report, receipts


def _load_instrumentation(path: Path):
    spec = importlib.util.spec_from_file_location(
        "tests.eval._refresh_coverage_instrumentation", path
    )
    if spec is None or spec.loader is None:
        raise ReplayIntegrityError("eval instrumentation is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    source = Path(payload["source"]).resolve()
    original, _receipts = _verify_original(
        Path(payload["report_path"]), source, Path(payload["receipt_manifest"])
    )
    if original != payload["report"]:
        raise ReplayIntegrityError("worker report identity changed before imports")
    instrumentation_path = Path(__file__).with_name("model_comparison.py").resolve()
    if Path(payload["instrumentation"]).resolve() != instrumentation_path:
        raise ReplayIntegrityError("only the current eval instrumentation can be loaded")
    if (source / ".env").exists():
        raise ReplayIntegrityError("the replay archive must not contain an environment file")
    runtime = {
        "python": sys.version,
        "packages": {
            name: package_version(name) for name in original["plan"]["runtime"]["packages"]
        },
    }
    if (
        runtime["packages"] != original["plan"]["runtime"]["packages"]
        or runtime["python"].split()[0] != original["plan"]["runtime"]["python"].split()[0]
    ):
        raise ReplayIntegrityError(
            "offline replay needs the original dependency and Python versions"
        )
    sys.path[:0] = [str(source / "src"), str(source)]
    from pydantic import ValidationError
    from tests.eval import model_refresh as original_runner
    from tests.eval.model_refresh_nodes import run_node_case

    from outrider.llm.base import INCLUDE_TEXT_OPT_IN, LLMResponse

    instrumentation = _load_instrumentation(Path(payload["instrumentation"]))
    suite = original["plan"].get("suite", "legacy")
    analysis = (
        original_runner.scenarios(suite) if suite != "legacy" else original_runner.scenarios()
    )
    compatibility = original_runner.compatibility_scenarios(
        original_runner.scenarios() if suite == "legacy" else None
    )
    scenarios = {row.path: row for row in (*analysis, *compatibility)}
    if "dataset" in original["plan"]:
        rebuilt = original_runner.dataset_identity(suite, analysis)
        if json.loads(json.dumps(rebuilt)) != original["plan"]["dataset"]:
            raise ReplayIntegrityError("the selected dataset differs from its original provenance")
    output = []
    for candidate in original["candidates"]:
        cases = {case_slot(case): case for case in candidate["cases"]}
        rows = []
        for call in candidate["calls"]:
            saved = cases.get(call["scenario"])
            if not call["persisted_exchange"]:
                rows.append(
                    {
                        "scenario": call["scenario"],
                        "replay_status": "original-provider-failure-without-response",
                    }
                )
                continue
            exchange = _read_json(
                Path(payload["report_dir"]) / "calls" / f"{call['index']:06d}.exchange.json"
            )
            try:
                normalized_response = LLMResponse.model_validate(exchange["response"])
            except ValidationError:
                raise ReplayIntegrityError("saved normalized response payload is invalid") from None

            class RecordedResponse:
                def __init__(self, saved, response):
                    self.saved = saved
                    self.response = response
                    self.calls = 0

                async def complete(self, request):
                    self.calls += 1
                    if "prompt_control_variant" in original["plan"]:
                        from tests.eval.prompt_control import request_for_variant

                        request = request_for_variant(
                            request, original["plan"]["prompt_control_variant"]
                        )
                    actual = request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
                    if self.calls != 1 or _request_identity(actual) != _request_identity(
                        self.saved["request"]
                    ):
                        raise ReplayIntegrityError(
                            "regenerated original request differs from its saved slot"
                        )
                    return self.response

            provider = RecordedResponse(exchange, normalized_response)
            try:
                if call["scenario"].startswith("nodes:"):
                    name = call["scenario"].split(":", 2)[2]
                    kwargs = {"provider": provider, "model": candidate["target"]["model"]}
                    if original["plan"]["measurement_contract"] == V2_CONTRACT:
                        kwargs["measurement_contract"] = V2_CONTRACT
                    result = await run_node_case(name, **kwargs)
                    if saved is None or json.loads(json.dumps(result)) != saved["result"]:
                        raise ReplayIntegrityError("original node result did not reproduce")
                    row = {
                        "scenario": call["scenario"],
                        "replay_status": "reproduced",
                        "node_result": result,
                    }
                else:
                    path = call["scenario"].split(":", 2)[-1]
                    state = (
                        original_runner.state_for_scenario(scenarios[path], suite=suite)
                        if suite != "legacy" and call["scenario"].startswith("analyze:")
                        else instrumentation.state_from_eval_fixture(source / path)
                    )
                    result = await instrumentation.run_analyze_instrumented(
                        state,
                        provider=provider,
                        model=candidate["target"]["model"],
                        collect_producer_coverage=True,
                    )
                    score = (
                        original_runner.score_run(result, scenarios[path])
                        if original["plan"]["measurement_contract"] == V1_CONTRACT
                        else original_runner._score_collected_case(
                            result, scenarios[path], plan=original["plan"], exchanges=[exchange]
                        )
                    )
                    if saved is None:
                        raise ReplayIntegrityError("original admitted result has no saved case")
                    for key, value in score.items():
                        expected = saved[key]
                        if key == "findings":
                            value, expected = (
                                [_finding_identity(item) for item in value],
                                [_finding_identity(item) for item in expected],
                            )
                        if json.loads(json.dumps(value)) != expected:
                            raise ReplayIntegrityError(
                                "original admitted findings or v1 counters did not reproduce"
                            )
                    row = {
                        "scenario": call["scenario"],
                        "replay_status": "reproduced",
                        "producer_coverage_findings": [
                            finding.model_dump(mode="json")
                            for finding in result.producer_coverage_findings
                        ],
                    }
            except ReplayIntegrityError:
                raise
            except Exception as exc:
                if (
                    saved is not None
                    or type(exc).__name__ != candidate.get("error_type")
                    or call is not candidate["calls"][-1]
                ):
                    raise
                row = {
                    "scenario": call["scenario"],
                    "replay_status": "original-downstream-failure-reproduced",
                    "error_type": type(exc).__name__,
                }
            if provider.calls != 1:
                raise ReplayIntegrityError(
                    "saved case did not consume exactly one original response"
                )
            rows.append(row)
        output.append({"target": candidate["target"], "slots": rows})
    return {
        "candidates": output,
        "runtime": runtime,
        "scenarios": [
            {
                "path": row.path,
                "track": row.track,
                "expected": [item.model_dump(mode="json") for item in row.expected],
            }
            for row in scenarios.values()
        ],
    }


def regrade_saved(
    report_path: Path, *, original_source: Path, receipt_manifest: Path, output: Path
) -> dict[str, Any]:
    """Create separate v2 artifacts after exact, offline original-source replay."""
    from types import SimpleNamespace

    from outrider.llm.base import LLMRequest
    from outrider.schemas.review_finding import ReviewFinding

    from .grading import ExpectedFinding
    from .model_refresh import write_json
    from .model_refresh_nodes import assess_saved_node_case
    from .model_refresh_report import REPORT_SCHEMA, comparisons, markdown
    from .review_quality import MEASUREMENT_CONTRACT_V2

    report_path = report_path / "results.json" if report_path.is_dir() else report_path
    report_path, original_source, receipt_manifest = (
        path.resolve() for path in (report_path, original_source, receipt_manifest)
    )
    original, receipts = _verify_original(report_path, original_source, receipt_manifest)
    suite = original["plan"].get("suite", "legacy")
    criteria = measurement_inputs(suite)
    original_hash = _sha(report_path)
    manifest_hash = _sha(receipt_manifest)
    payload = {
        "source": str(original_source.resolve()),
        "report_dir": str(report_path.resolve().parent),
        "report_path": str(report_path.resolve()),
        "receipt_manifest": str(receipt_manifest.resolve()),
        "instrumentation": str(Path(__file__).with_name("model_comparison.py")),
        "report": original,
    }
    env = {"PATH": os.defpath, "OUTRIDER_EVAL_REAL_MODELS": "0", "PYTHONNOUSERSITE": "1"}
    process = subprocess.run(  # noqa: S603 — fixed interpreter/script; archive hashes verified above
        [sys.executable, str(Path(__file__).resolve()), "_worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=original_source,
        timeout=180,
    )
    if process.returncode:
        try:
            failure = json.loads(process.stderr.strip().splitlines()[-1])
        except (ValueError, IndexError):
            failure = {}
        if failure.get("error_type") == "ReplayIntegrityError":
            raise ReplayIntegrityError("original-source replay: " + failure["check"])
        raise ReplayIntegrityError(
            "offline original-source replay failed; no derived quality result was written"
        )
    replay = json.loads(process.stdout)
    rows = {
        row["path"]: SimpleNamespace(
            path=row["path"],
            track=row["track"],
            expected=tuple(ExpectedFinding.model_validate(item) for item in row["expected"]),
        )
        for row in replay["scenarios"]
    }
    candidates = []
    for candidate, reproduced in zip(original["candidates"], replay["candidates"], strict=True):
        slots = {slot["scenario"]: slot for slot in reproduced["slots"]}
        calls = {call["scenario"]: call for call in candidate["calls"]}
        derived = {
            key: value for key, value in candidate.items() if key not in {"cases", "summary"}
        }
        derived["cases"] = []
        for case in candidate["cases"]:
            slot = case_slot(case)
            if slots[slot]["replay_status"] != "reproduced":
                raise ValueError("saved result slot was not reproduced")
            call = calls[slot]
            exchange = _read_json(
                report_path.parent / "calls" / f"{call['index']:06d}.exchange.json"
            )
            if case["phase"] == "nodes":
                result = dict(case["result"])
                result["measurement_v2"] = assess_saved_node_case(
                    case["name"],
                    request=exchange["request"],
                    response=exchange["response"],
                    prior_result=result,
                )
                derived["cases"].append({**case, "result": result})
                continue
            replay_result = SimpleNamespace(
                findings=tuple(
                    ReviewFinding.model_validate(finding) for finding in case["findings"]
                ),
                producer_coverage_findings=tuple(
                    ReviewFinding.model_validate(finding)
                    for finding in slots[slot]["producer_coverage_findings"]
                ),
                response_rejections=case["response_rejections"],
                proposal_rejections=case["proposal_rejections"],
            )
            scored = score_case_v2(
                replay_result,
                rows[case["fixture"]],
                fixture_path=original_source / case["fixture"],
                request=LLMRequest.model_validate(exchange["request"]),
            )
            derived["cases"].append(
                {
                    **{
                        key: value
                        for key, value in case.items()
                        if key
                        not in {
                            "model_judged",
                            "product",
                            "producer_covered_expected",
                            "findings",
                            "producer_hashes",
                            "response_rejections",
                            "proposal_rejections",
                        }
                    },
                    "original_v1_score"
                    if original["plan"]["measurement_contract"] == V1_CONTRACT
                    else "original_v2_score": {
                        key: case[key]
                        for key in (
                            ("model_judged", "product", "producer_covered_expected")
                            if original["plan"]["measurement_contract"] == V1_CONTRACT
                            else (
                                "model_dependent",
                                "product_grounded",
                                "producer_covered_expected",
                                "measurement_v2",
                            )
                        )
                    },
                    **scored,
                }
            )
        derived["replay_slots"] = reproduced["slots"]
        candidates.append(derived)
    # Recheck immutable inputs after the worker and before creating any derivative.
    final_original, final_receipts = _verify_original(
        report_path, original_source, receipt_manifest
    )
    if (
        final_original != original
        or final_receipts != receipts
        or _sha(receipt_manifest) != manifest_hash
    ):
        raise ReplayIntegrityError("original report or receipt anchor changed during replay")
    if measurement_inputs(suite) != criteria:
        raise ReplayIntegrityError("v2 measurement inputs changed during replay or scoring")
    report = {
        "schema": REPORT_SCHEMA,
        "evidence_kind": "saved-response-reanalysis",
        "plan": {
            **original["plan"],
            "measurement_contract": MEASUREMENT_CONTRACT_V2,
            "mode": "offline-reanalysis",
            "measurement_inputs": criteria,
        },
        "original_plan": original["plan"],
        "candidates": candidates,
        "budget": original["budget"],
        "cost_note": (
            "Original recorded usage-based USD and timing only; no calls, new charges or repricing."
        ),
        "reanalysis": {
            "original_results_sha256": original_hash,
            "original_receipts": receipts,
            "receipt_manifest_sha256": manifest_hash,
            "original_source": str(original_source.resolve()),
            "runtime": replay["runtime"],
            "original_runtime": original["plan"]["runtime"],
            "scope": (
                "Exact original requests/admission reproduced with original production; "
                "bounded current v2 assessments, semantic adjudication pending."
            ),
        },
    }
    report["comparisons"] = comparisons(report)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", report)
    (output / "report.md").write_text(
        "# Offline v2 reanalysis\n\nOriginal paid costs are retained, not rebilled. "
        "Semantic/context gaps prevent automatic qualification.\n\n" + markdown(report)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["_worker"]:

        def offline(event, _args):
            if event in {"socket.connect", "socket.getaddrinfo"}:
                raise RuntimeError("network is disabled during saved-response replay")

        sys.addaudithook(offline)
        try:
            print(json.dumps(asyncio.run(_worker(json.load(sys.stdin)))))
        except ReplayIntegrityError as exc:
            print(
                json.dumps({"error_type": type(exc).__name__, "check": str(exc)}), file=sys.stderr
            )
            return 1
        except Exception as exc:
            print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr)
            return 1
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--original-source", type=Path, required=True)
    parser.add_argument("--receipt-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    destination = args.output or ROOT / "reports/model-refresh-reanalysis" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    )
    try:
        report = regrade_saved(
            args.report,
            original_source=args.original_source,
            receipt_manifest=args.receipt_manifest,
            output=destination,
        )
    except ReplayIntegrityError as exc:
        print(
            json.dumps(
                {"status": "not-regraded", "error_type": type(exc).__name__, "check": str(exc)}
            )
        )
        return 1
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "not-regraded", "error_type": type(exc).__name__}))
        return 1
    print(
        json.dumps(
            {
                "report": str(destination),
                "evidence_kind": report["evidence_kind"],
                "model_calls_made": 0,
                "comparisons": report["comparisons"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
