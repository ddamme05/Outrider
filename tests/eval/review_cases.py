"""Load exposed public PR pairs with independently executable mechanism labels.

The complete changed-file captures are model inputs. Labels and provenance stay
outside them; a counterpart certifies only its named repaired mechanism.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from outrider.agent.eval_driver import EvalFixture
from outrider.policy import FindingType, lookup_severity

from .grading import ExpectedFinding

if TYPE_CHECKING:
    from .review_quality import ContextStatus

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = "tests/eval/corpus/review_comparison"
MANIFEST = f"{DIRECTORY}/manifest.json"
DATASET_VERSION = "review-comparison-development-v1"
EXPANDED_DATASET_VERSION = "review-comparison-expanded-development-v1"
EXPOSURE = "development-exposed"


@dataclass(frozen=True, slots=True)
class ReviewMechanismScope:
    file_path: str
    finding_type: FindingType
    line_start: int
    line_end: int
    reason: str


@dataclass(frozen=True, slots=True)
class ReviewCase:
    id: str
    fixture_path: str
    fixture_sha256: str
    analysis_file: str
    expected: tuple[ExpectedFinding, ...]
    scoped_counterparts: tuple[ReviewMechanismScope, ...]
    context_status: ContextStatus
    context_reason: str
    provenance_family: str
    provenance_source: str
    capture_manifest_path: str
    exposure: str = EXPOSURE


def _path(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError("review comparison paths must be normalized and relative")
    path = ROOT / name
    if not path.resolve().is_relative_to((ROOT / DIRECTORY).resolve()):
        raise ValueError("review comparison path escapes the corpus")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(*, include_path_cases: bool = False) -> tuple[ReviewCase, ...]:
    """Validate captures, optionally including the existing historical path cases.

    The expanded set has seven exposed development captures across three bug
    families. Reusing their reviewed labels adds no held-out or oracle evidence.
    """
    manifest = json.loads(_path(MANIFEST).read_text())
    if (
        manifest.get("schema") != "review-comparison-v1"
        or manifest.get("dataset_id") != DATASET_VERSION
        or manifest.get("exposure") != EXPOSURE
        or manifest.get("held_out") is not False
        or not manifest.get("cases")
    ):
        raise ValueError("unsupported review comparison manifest")
    cases = []
    for row in manifest["cases"]:
        fixture_path = _path(row["fixture"])
        receipt_path = _path(row["receipt"])
        if _sha(fixture_path) != row["fixture_sha256"] or (
            _sha(receipt_path) != row["receipt_sha256"]
        ):
            raise ValueError(f"review capture changed after label review: {row['id']}")
        receipt = json.loads(receipt_path.read_text())
        directory = PurePosixPath(row["receipt"]).parent
        required = {"fixture.json", "LICENSE", "raw/pr.json", "raw/files.json", "raw/compare.json"}
        if receipt.get("schema") != "pr-capture-v1" or not required <= receipt["raw_files"].keys():
            raise ValueError("review capture lacks complete raw provenance")
        for name, digest in receipt["raw_files"].items():
            if _sha(_path(str(directory / name))) != digest:
                raise ValueError("review raw capture changed after capture")
        provenance_path = _path(row["provenance"])
        if _sha(provenance_path) != row["provenance_sha256"]:
            raise ValueError("review commit provenance changed after label review")
        provenance = json.loads(provenance_path.read_text())
        if receipt["fixture_sha256"] != row["fixture_sha256"]:
            raise ValueError("review fixture and receipt hashes disagree")
        fixture = EvalFixture.model_validate_json(fixture_path.read_bytes())
        source = receipt["source"]
        if (
            f"{fixture.owner}/{fixture.repo}" != source["repo"]
            or fixture.pr_number != source["pr"]
            or fixture.base_sha != source["base_sha"]
            or fixture.head_sha != source["head_sha"]
            or fixture.llm_responses
            or fixture.analyze_responses_by_path is not None
        ):
            raise ValueError("review fixture identity or scripted responses differ from capture")
        files = {file.path: file for file in fixture.files}
        captured = {file["path"]: file for file in receipt["files"]}
        if len(files) != len(fixture.files) or files.keys() != captured.keys():
            raise ValueError("review capture changed-file inventory differs from receipt")
        raw_pr = json.loads(_path(str(directory / "raw/pr.json")).read_text())
        if (
            len(files) != raw_pr["changed_files"]
            or fixture.total_additions != raw_pr["additions"]
            or fixture.total_deletions != raw_pr["deletions"]
            or fixture.head_sha != provenance["head_commit"]["sha"]
            or row["selected_file_last_touch_sha"]
            != provenance["selected_file_last_touch"][0]["sha"]
        ):
            raise ValueError("review inventory or immutable commit provenance differs")
        for file in files.values():
            saved = captured[file.path]
            if file.status != saved["status"] or file.previous_path != saved["previous_path"]:
                raise ValueError("review file status differs from capture")
            for side in ("base", "head"):
                content = getattr(file, f"content_{side}")
                identity = saved[side]
                if (content is None) != (identity is None) or (
                    content is not None
                    and hashlib.sha256(content.encode()).hexdigest() != identity["sha256"]
                ):
                    raise ValueError("review source bytes differ from capture")
        selected = files.get(row["analysis_file"])
        if selected is None or selected.content_head is None or not selected.patch:
            raise ValueError("review diagnosis requires a changed head source with its patch")
        expected = tuple(ExpectedFinding.model_validate(item) for item in row["expected"])
        counterparts = tuple(
            ReviewMechanismScope(**{**item, "finding_type": FindingType(item["finding_type"])})
            for item in row["scoped_counterparts"]
        )
        if len(expected) > 1 or bool(expected) == bool(counterparts):
            raise ValueError("review cases need one positive or narrow counterpart labels")
        for item in (*expected, *counterparts):
            if item.file_path != selected.path or not (
                1 <= item.line_start <= item.line_end <= len(selected.content_head.splitlines())
            ):
                raise ValueError("review label is outside the selected captured source")
        if any(item.severity != lookup_severity(item.finding_type) for item in expected):
            raise ValueError("review label severity differs from current policy")
        if row["context_status"] not in {"supported", "unknown", "unsupported"}:
            raise ValueError("invalid reviewed context status")
        cases.append(
            ReviewCase(
                id=row["id"],
                fixture_path=row["fixture"],
                fixture_sha256=row["fixture_sha256"],
                analysis_file=row["analysis_file"],
                expected=expected,
                scoped_counterparts=counterparts,
                context_status=row["context_status"],
                context_reason=row["context_reason"],
                provenance_family=row["provenance_family"],
                provenance_source=source["url"],
                capture_manifest_path=row["receipt"],
            )
        )
    if include_path_cases:
        from . import real_pr_cases

        for case in real_pr_cases.load_cases():
            cases.append(
                ReviewCase(
                    id=case.id,
                    fixture_path=case.fixture_path,
                    fixture_sha256=case.fixture_sha256,
                    analysis_file=case.analysis_file,
                    expected=case.expected,
                    scoped_counterparts=tuple(
                        ReviewMechanismScope(
                            file_path=scope.file_path,
                            finding_type=scope.finding_type,
                            line_start=scope.line_start,
                            line_end=scope.line_end,
                            reason=scope.reason,
                        )
                        for scope in case.scoped_counterparts
                    ),
                    context_status=case.context_status,
                    context_reason=case.context_reason,
                    provenance_family=case.provenance_family,
                    provenance_source=case.provenance_source,
                    capture_manifest_path=case.receipt_path,
                    exposure=case.exposure,
                )
            )
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate review comparison case identity")
    return tuple(cases)


def dataset_inputs(*, include_path_cases: bool = False) -> dict[str, str]:
    """Pin each selected corpus through its existing validated input inventory."""
    paths = {
        MANIFEST,
        f"{DIRECTORY}/README.md",
        "tests/eval/review_cases.py",
        "tests/eval/test_review_case_semantics.py",
        "tests/eval/capture_pr.py",
    }
    for case in load_cases():
        receipt = json.loads((ROOT / case.capture_manifest_path).read_text())
        directory = PurePosixPath(case.capture_manifest_path).parent
        paths.add(case.capture_manifest_path)
        paths.update(str(directory / name) for name in receipt["raw_files"])
    paths.update(
        str(path.relative_to(ROOT)) for path in (ROOT / DIRECTORY / "provenance").glob("*.json")
    )
    inputs = {name: _sha(ROOT / name) for name in sorted(paths)}
    if include_path_cases:
        from . import real_pr_cases

        inputs.update(real_pr_cases.dataset_inputs())
    return dict(sorted(inputs.items()))
