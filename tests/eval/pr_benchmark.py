"""Validate frozen upstream PRs or grade a live, isolated full-graph review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from outrider.agent.eval_driver import EvalFixture, run_review
from outrider.coordinates import (
    COORDINATES_IMPORT_PATH_RESOLVER,
    added_line_byte_ranges,
    lookup_patched_file,
)
from outrider.llm.anthropic_provider import AnthropicProvider
from outrider.llm.config import ModelConfig
from outrider.llm.pricing import PRICING_VERSION
from outrider.policy.severity import ACTIVE_POLICY_VERSION, SEVERITY_POLICY, FindingType
from outrider.prompts import analyze, synthesize, trace, triage

from .grading import ExpectedFinding, grade

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.agent.eval_driver import EvalRunResult
    from outrider.llm.base import LLMExchangePersister, LLMProvider
    from outrider.schemas.review_finding import ReviewFinding

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "tests/eval/corpus/real_prs"


class CleanScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file_path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    finding_types: tuple[FindingType, ...]
    rationale: str

    def contains(self, finding: ReviewFinding) -> bool:
        return (
            finding.file_path == self.file_path
            and finding.finding_type in self.finding_types
            and self.line_start <= finding.line_start <= finding.line_end <= self.line_end
        )


class PRCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    fixture: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_url: str
    evidence_urls: tuple[str, ...]
    rationale: str
    expected: tuple[ExpectedFinding, ...]
    clean_scopes: tuple[CleanScope, ...]
    context_paths: tuple[str, ...]
    provenance: dict[str, Any]


class PRCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1, le=1)
    cases: tuple[PRCase, ...]


def fixture_path(case: PRCase, corpus_root: Path = CORPUS_ROOT) -> Path:
    path = (corpus_root / case.fixture).resolve()
    if path.parent != corpus_root.resolve() or path.suffix != ".json":
        raise ValueError(f"{case.id}: fixture must be a JSON file directly in the corpus")
    return path


def load_corpus(corpus_root: Path = CORPUS_ROOT) -> PRCorpus:
    corpus = PRCorpus.model_validate_json((corpus_root / "manifest.json").read_bytes())
    ids = [case.id for case in corpus.cases]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("PR cases must have unique ids and the corpus must not be empty")
    for case in corpus.cases:
        data = fixture_path(case, corpus_root).read_bytes()
        if hashlib.sha256(data).hexdigest() != case.sha256:
            raise ValueError(f"{case.id}: fixture hash mismatch")
        fixture = EvalFixture.model_validate_json(data)
        if fixture.llm_responses or fixture.analyze_responses_by_path is not None:
            raise ValueError(f"{case.id}: benchmark inputs must not contain scripted answers")
        if not all(
            re.fullmatch(r"[a-f0-9]{40}", sha) for sha in (fixture.base_sha, fixture.head_sha)
        ):
            raise ValueError(f"{case.id}: base and head must be immutable commit SHAs")
        files = {file.path: file for file in fixture.files}
        if len(files) != len(fixture.files):
            raise ValueError(f"{case.id}: duplicate changed paths")
        for file in fixture.files:
            if file.patch and file.content_head is not None:
                patch = lookup_patched_file(file.patch, file.path)
                if patch is None:
                    raise ValueError(f"{case.id}: missing patch for {file.path}")
                added_line_byte_ranges(patch, file.content_head)
        if not set(case.context_paths) <= fixture.repository_contents_head.keys():
            raise ValueError(f"{case.id}: required beyond-diff context is missing")
        if not case.expected and not case.clean_scopes:
            raise ValueError(f"{case.id}: no human expectations")
        for label in (*case.expected, *case.clean_scopes):
            file = files.get(label.file_path)
            head = file.content_head if file is not None else None
            if head is None or not 1 <= label.line_start <= label.line_end <= len(
                head.splitlines()
            ):
                raise ValueError(f"{case.id}: expectation outside captured head: {label.file_path}")
        for expected in case.expected:
            if expected.severity != SEVERITY_POLICY[expected.finding_type]:
                raise ValueError(f"{case.id}: expected severity differs from active policy")
        if any(not scope.finding_types for scope in case.clean_scopes):
            raise ValueError(f"{case.id}: clean scopes must name the finding types adjudicated")
    return corpus


def grade_case(case: PRCase, findings: Sequence[ReviewFinding]) -> dict[str, Any]:
    result = grade(findings, case.expected, line_window=0)
    false_positives = []
    needs_review = []
    for finding in result.extra:
        if any(scope.contains(finding) for scope in case.clean_scopes):
            false_positives.append(finding.model_dump(mode="json"))
        else:
            needs_review.append(finding.model_dump(mode="json"))
    return {
        "expected_count": len(case.expected),
        "matched_count": result.n_matched,
        "missed": [finding.model_dump(mode="json") for finding in result.missed],
        "false_positives": false_positives,
        "needs_review": needs_review,
    }


def run_case(case: PRCase, *, model_config: ModelConfig) -> EvalRunResult:
    def provider_factory(persister: LLMExchangePersister) -> LLMProvider:
        return AnthropicProvider(
            api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
            model_config=model_config,
            persister=persister,
        )

    return run_review(
        fixture_path(case),
        model_config=model_config,
        host="anthropic",
        provider_factory=provider_factory,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src/outrider").rglob("*")):
        if path.suffix in {".py", ".scm"}:
            digest.update(path.relative_to(REPO_ROOT).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call real models; incurs API charges.")
    parser.add_argument(
        "--case", action="append", dest="case_ids", help="Select a case id; repeatable."
    )
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args(argv)
    corpus = load_corpus()
    selected = set(args.case_ids or [case.id for case in corpus.cases])
    unknown = selected - {case.id for case in corpus.cases}
    if unknown:
        parser.error(f"unknown case ids: {sorted(unknown)}")
    if args.live:
        for name in ("OUTRIDER_IS_EVAL", "OUTRIDER_EVAL_REAL_MODELS"):
            if os.environ.get(name) != "1":
                parser.error(f"--live requires {name}=1")
        if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("TEST_DATABASE_URL"):
            parser.error("--live requires ANTHROPIC_API_KEY and TEST_DATABASE_URL")
    model_config = ModelConfig.for_host("anthropic") if args.live else None
    rows = []
    had_errors = False
    for case in corpus.cases:
        if case.id not in selected:
            continue
        row: dict[str, Any] = {"id": case.id, "fixture_sha256": case.sha256, "status": "not_run"}
        if model_config is not None:
            try:
                result = run_case(case, model_config=model_config)
                row.update(
                    status="completed",
                    **grade_case(case, result.findings),
                    findings=[finding.model_dump(mode="json") for finding in result.findings],
                    trace_decisions=[
                        decision.model_dump(mode="json") for decision in result.trace_decisions
                    ],
                    hitl_gated=result.hitl_gated,
                    metrics=result.review_metrics.model_dump(mode="json")
                    if result.review_metrics
                    else None,
                )
            except Exception as exc:
                # Persist failure status without copying provider content or credentials.
                row.update(status="errored", error_type=type(exc).__name__)
                had_errors = True
        rows.append(row)
    report = {
        "mode": "live" if args.live else "validation_only",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_sha256": hashlib.sha256((CORPUS_ROOT / "manifest.json").read_bytes()).hexdigest(),
        "outrider_source_sha256": source_fingerprint(),
        "benchmark_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("grading.py").read_bytes()
        ).hexdigest(),
        "models": model_config.model_dump(mode="json") if model_config else None,
        "prompt_versions": {
            "triage": triage.VERSION,
            "analyze": analyze.VERSION,
            "trace": trace.VERSION,
            "synthesize": synthesize.VERSION,
        },
        "severity_policy_version": ACTIVE_POLICY_VERSION,
        "pricing_version": PRICING_VERSION,
        "patches_enabled": False,
        "cases": rows,
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
