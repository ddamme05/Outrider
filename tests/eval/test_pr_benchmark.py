"""Check capture integrity and grading; test findings are not model-quality evidence."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingType, lookup_severity
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.review_finding import ReviewFinding

from . import pr_benchmark

if TYPE_CHECKING:
    from pathlib import Path


def _finding(file_path: str, line: int, *, line_end: int | None = None) -> ReviewFinding:
    end = line if line_end is None else line_end
    finding_type = FindingType.PATH_TRAVERSAL
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=lookup_severity(finding_type),
        file_path=file_path,
        line_start=line,
        line_end=end,
        title="Traversal",
        description="Test grading input",
        evidence="Test grading input",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path, line_start=line, line_end=end, finding_type=finding_type
        ),
        proposal_hash="a" * 64,
    )


def test_upstream_corpus_has_bug_clean_and_beyond_diff_cases() -> None:
    corpus = pr_benchmark.load_corpus()
    assert {case.id for case in corpus.cases} == {"aiohttp-1327", "aiohttp-8079", "aiohttp-8063"}
    assert sum(bool(case.expected) for case in corpus.cases) == 2
    assert sum(bool(case.clean_scopes) for case in corpus.cases) == 1
    assert any("aiohttp/web_urldispatcher.py" in case.context_paths for case in corpus.cases)
    for case in corpus.cases:
        fixture = pr_benchmark.EvalFixture.model_validate_json(
            pr_benchmark.fixture_path(case).read_bytes()
        )
        assert fixture.head_sha == case.provenance["head_sha"]
        assert fixture.base_sha == case.provenance["diff_merge_base_sha"]
        assert case.evidence_urls


def test_changed_capture_is_refused(tmp_path: Path) -> None:
    shutil.copytree(pr_benchmark.CORPUS_ROOT, tmp_path / "corpus")
    path = tmp_path / "corpus/aiohttp-1327.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fixture hash mismatch"):
        pr_benchmark.load_corpus(tmp_path / "corpus")


def test_grading_records_a_miss_and_does_not_call_unlabeled_extras_false_positives() -> None:
    case = next(case for case in pr_benchmark.load_corpus().cases if case.id == "aiohttp-1327")
    expected = case.expected[0]
    extra = _finding("another_file.py", 1)
    missed = pr_benchmark.grade_case(case, [extra])
    assert missed["matched_count"] == 0
    assert len(missed["missed"]) == 1
    assert missed["false_positives"] == []
    assert len(missed["needs_review"]) == 1
    caught = pr_benchmark.grade_case(case, [_finding(expected.file_path, expected.line_start)])
    assert caught["matched_count"] == 1
    assert caught["missed"] == []


def test_only_findings_within_adjudicated_clean_scope_count_as_false_positives() -> None:
    case = next(case for case in pr_benchmark.load_corpus().cases if case.id == "aiohttp-8079")
    scope = case.clean_scopes[0]
    result = pr_benchmark.grade_case(
        case,
        [
            _finding(scope.file_path, scope.line_start),
            _finding(scope.file_path, scope.line_start, line_end=scope.line_end + 1),
        ],
    )
    assert len(result["false_positives"]) == 1
    assert len(result["needs_review"]) == 1


def test_validation_report_never_claims_model_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pr_benchmark, "run_case", lambda *args, **kwargs: pytest.fail("model called")
    )
    output = tmp_path / "report.json"
    assert pr_benchmark.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["mode"] == "validation_only"
    assert report["models"] is None
    assert all(row["status"] == "not_run" and "matched_count" not in row for row in report["cases"])


def test_live_mode_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTRIDER_EVAL_REAL_MODELS", raising=False)
    monkeypatch.setattr(
        pr_benchmark, "run_case", lambda *args, **kwargs: pytest.fail("model called")
    )
    with pytest.raises(SystemExit) as exc_info:
        pr_benchmark.main(["--live"])
    assert exc_info.value.code == 2


def test_failed_live_run_is_recorded_with_null_quality_and_without_error_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("OUTRIDER_IS_EVAL", "OUTRIDER_EVAL_REAL_MODELS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setenv("TEST_DATABASE_URL", "test-placeholder")

    def fail(*args, **kwargs):
        raise RuntimeError("content that must not appear in the report")

    monkeypatch.setattr(pr_benchmark, "run_case", fail)
    output = tmp_path / "errored.json"
    assert pr_benchmark.main(["--live", "--case", "aiohttp-1327", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["cases"][0]["status"] == "errored"
    assert "matched_count" not in report["cases"][0]
    assert "content that must not appear" not in output.read_text()
