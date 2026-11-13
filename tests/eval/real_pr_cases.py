"""Load frozen public PR captures for a bounded, one-file analysis comparison.

The full PR remains available to the harness. A reviewed manifest selects one
changed file and narrow labels; it does not certify the rest of the PR as clean.
These historical examples are development-selected, not held-out data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from outrider.agent.eval_driver import EvalFixture
from outrider.coordinates import added_line_byte_ranges, lookup_patched_file
from outrider.policy import FindingType, lookup_severity

from .grading import ExpectedFinding

if TYPE_CHECKING:
    from .review_quality import ContextStatus

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = "tests/eval/corpus/real_pr_refresh"
MANIFEST = f"{DIRECTORY}/manifest.json"
DATASET_VERSION = "real-pr-commonprefix-development-v1"
EXPOSURE = "development-selected-historical-prs"


@dataclass(frozen=True, slots=True)
class CounterpartScope:
    file_path: str
    finding_type: FindingType
    line_start: int
    line_end: int
    reason: str


@dataclass(frozen=True, slots=True)
class RealPRCase:
    id: str
    fixture_path: str
    fixture_sha256: str
    receipt_path: str
    receipt_sha256: str
    license_path: str
    license_sha256: str
    analysis_file: str
    expected: tuple[ExpectedFinding, ...]
    scoped_counterparts: tuple[CounterpartScope, ...]
    track: str
    context_status: ContextStatus
    context_reason: str
    provenance_family: str
    provenance_source: str
    exposure: str = EXPOSURE


def _path(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError("real PR corpus paths must be normalized and repository-relative")
    path = ROOT / name
    if not path.resolve().is_relative_to((ROOT / DIRECTORY).resolve()):
        raise ValueError("real PR corpus path escapes its directory")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    manifest = json.loads(_path(MANIFEST).read_text())
    if (
        manifest.get("schema") != "real-pr-refresh-v1"
        or manifest.get("dataset_id") != DATASET_VERSION
        or manifest.get("exposure") != EXPOSURE
        or manifest.get("held_out") is not False
        or not manifest.get("cases")
    ):
        raise ValueError("unsupported real PR comparison manifest")
    return manifest


def load_cases() -> tuple[RealPRCase, ...]:
    """Validate authentic capture hashes and the selected analysis/label bounds."""
    cases = []
    for row in _manifest()["cases"]:
        for key in ("fixture", "receipt", "license"):
            if _sha(_path(row[key])) != row[f"{key}_sha256"]:
                raise ValueError(f"real PR {key} changed after review: {row['id']}")
        receipt = json.loads(_path(row["receipt"]).read_text())
        directory = PurePosixPath(row["receipt"]).parent
        if receipt.get("schema") != "pr-capture-v1":
            raise ValueError("unsupported real PR capture receipt")
        required = {"fixture.json", "LICENSE", "raw/pr.json", "raw/files.json", "raw/compare.json"}
        if not required <= receipt["raw_files"].keys():
            raise ValueError("real PR capture is missing raw provenance files")
        for name, digest in receipt["raw_files"].items():
            if _sha(_path(str(directory / name))) != digest:
                raise ValueError(f"real PR raw capture changed: {row['id']}")
        if receipt["fixture_sha256"] != row["fixture_sha256"] or (
            receipt["license"]["sha256"] != row["license_sha256"]
        ):
            raise ValueError("real PR receipt and manifest hashes disagree")
        fixture = EvalFixture.model_validate_json(_path(row["fixture"]).read_bytes())
        source = receipt["source"]
        if (
            source["repo"] != f"{fixture.owner}/{fixture.repo}"
            or source["pr"] != fixture.pr_number
            or source["base_sha"] != fixture.base_sha
            or source["head_sha"] != fixture.head_sha
            or fixture.llm_responses
            or fixture.analyze_responses_by_path is not None
        ):
            raise ValueError("real PR fixture identity or response scripts differ from capture")
        files = {file.path: file for file in fixture.files}
        if len(files) != len(fixture.files):
            raise ValueError("real PR capture has duplicate changed paths")
        captured = {file["path"]: file for file in receipt["files"]}
        if files.keys() != captured.keys():
            raise ValueError("real PR receipt and fixture changed-file inventories differ")
        for file in files.values():
            saved = captured[file.path]
            if saved["status"] != file.status or saved["previous_path"] != file.previous_path:
                raise ValueError("real PR captured file status differs from its receipt")
            for side in ("base", "head"):
                content = getattr(file, f"content_{side}")
                identity = saved[side]
                if (content is None) != (identity is None) or (
                    content is not None
                    and hashlib.sha256(content.encode()).hexdigest() != identity["sha256"]
                ):
                    raise ValueError("real PR source content differs from its capture receipt")
        selected = files.get(row["analysis_file"])
        if (
            selected is None
            or selected.status == "removed"
            or not selected.path.endswith(".py")
            or selected.content_head is None
            or not selected.patch
        ):
            raise ValueError("real PR analysis requires one changed Python head with a patch")
        patch = lookup_patched_file(selected.patch, selected.path)
        if patch is None or not added_line_byte_ranges(patch, selected.content_head):
            raise ValueError("real PR analysis file has no validated added source lines")
        expected = tuple(ExpectedFinding.model_validate(item) for item in row["expected"])
        counterparts = tuple(
            CounterpartScope(**{**item, "finding_type": FindingType(item["finding_type"])})
            for item in row["scoped_counterparts"]
        )
        if len(expected) > 1 or not (expected or counterparts):
            raise ValueError("real PR cases need one expected issue or narrow counterpart labels")
        for label in (*expected, *counterparts):
            if label.file_path != selected.path or not (
                1 <= label.line_start <= label.line_end <= len(selected.content_head.splitlines())
            ):
                raise ValueError("real PR label is outside the selected captured source")
        if any(item.severity != lookup_severity(item.finding_type) for item in expected):
            raise ValueError("real PR expected severity differs from current policy")
        required = row.get("required_context_ranges", [])
        if not required or any(
            len(bounds) != 2
            or any(type(line) is not int for line in bounds)
            or not 1 <= bounds[0] <= bounds[1] <= len(selected.content_head.splitlines())
            for bounds in required
        ):
            raise ValueError("real PR context approval requires bounded source premises")
        if row["context_status"] not in {"supported", "unknown", "unsupported"}:
            raise ValueError("unsupported real PR context review status")
        cases.append(
            RealPRCase(
                id=row["id"],
                fixture_path=row["fixture"],
                fixture_sha256=row["fixture_sha256"],
                receipt_path=row["receipt"],
                receipt_sha256=row["receipt_sha256"],
                license_path=row["license"],
                license_sha256=row["license_sha256"],
                analysis_file=row["analysis_file"],
                expected=expected,
                scoped_counterparts=counterparts,
                track="real-pr-positive" if expected else "real-pr-counterpart",
                context_status=row["context_status"],
                context_reason=row["context_reason"],
                provenance_family=row["provenance_family"],
                provenance_source=source["url"],
            )
        )
    if len({case.id for case in cases}) != len(cases) or len(
        {case.fixture_path for case in cases}
    ) != len(cases):
        raise ValueError("duplicate real PR comparison cases")
    return tuple(cases)


def dataset_inputs() -> dict[str, str]:
    """Pin labels, route, capture procedure, licenses and every saved capture file."""
    names = {
        MANIFEST,
        f"{DIRECTORY}/README.md",
        "tests/eval/real_pr_cases.py",
        "tests/eval/test_model_refresh.py",
        "tests/eval/capture_pr.py",
        "tests/eval/test_capture_pr.py",
        "tests/eval/test_real_pr_semantics.py",
    }
    for case in load_cases():
        receipt = json.loads((ROOT / case.receipt_path).read_text())
        directory = PurePosixPath(case.receipt_path).parent
        names.add(case.receipt_path)
        names.update(str(directory / name) for name in receipt["raw_files"])
    return {name: _sha(ROOT / name) for name in sorted(names)}


def context_for_case(
    fixture_path: str | Path,
    expected: ExpectedFinding,
    *,
    verified_ranges: tuple[tuple[int, int], ...] = (),
) -> tuple[ContextStatus, str] | None:
    """Keep byte-bound premise approval separate from source quote matching."""
    path = Path(fixture_path)
    if f"/{DIRECTORY}/" not in "/" + path.as_posix():
        return None
    rows = [
        row
        for row in _manifest()["cases"]
        if path.as_posix() == row["fixture"] or path.as_posix().endswith("/" + row["fixture"])
    ]
    if not rows:
        return None
    row = rows[0]
    if _sha(path) != row["fixture_sha256"]:
        return "unknown", "The real PR capture changed after visible-context review."
    if expected not in tuple(ExpectedFinding.model_validate(item) for item in row["expected"]):
        return "unknown", "The real PR expectation changed after visible-context review."
    required = row.get("required_context_ranges", [])
    if not required or not all(
        any(lo <= line <= hi for lo, hi in verified_ranges)
        for start, end in required
        for line in range(start, end + 1)
    ):
        return "unknown", "The request omits source required for the reviewed real PR premise."
    return row["context_status"], row["context_reason"]
