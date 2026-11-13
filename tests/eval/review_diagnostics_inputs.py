"""Verify a saved review and prepare controlled single-call comparisons.

Context controls vary instructions and appended reference source. Discovery
controls vary the review opening or category guidance on identical source.
Both retain native schemas, settings and finding bounds. The captured direct
request is verified against its fixture before either comparison is prepared.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from outrider.agent.eval_driver import EvalFixture
from outrider.llm.base import INCLUDE_TEXT_OPT_IN, LLMRequest
from outrider.prompts import analyze
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA_JSON

from . import direct_review
from .discovery_control import DISCOVERY_ARMS as DISCOVERY_ARMS
from .discovery_control import discovery_identity as discovery_identity
from .discovery_control import requests_for_discovery
from .model_refresh import Target
from .prompt_control import CONCISE_PROMPT, CONCISE_PROMPTS, CONCISE_VERSION

ARMS = (
    "current-scoped",
    "concise-scoped",
    "current-full-reference",
    "concise-full-reference",
    "clarified-scoped",
    "direct-control",
)
APPEND_RECIPE = "append-complete-captured-pr-reference-v1"
CLARIFIED_VERSION = "captured-source-clarification-v1"
SOURCE_CLAUSE = "Use only the explicitly supplied source and additional context;"
CLARIFIED_CLAUSE = (
    "Use programming language and standard library semantics to reason about visible operations. "
    "Ground claims about this repository in the supplied source and additional context;"
)
MAX_REQUEST_TEXT_BYTES = 2_000_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    path = directory / name
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != name
        or not path.resolve().is_relative_to(directory.resolve())
    ):
        raise ValueError("source artifact paths must be normalized and inside the report")
    return path


def fields(request: LLMRequest) -> dict[str, Any]:
    return request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)


def request_identity(request: LLMRequest, *, omit: tuple[str, ...] = ()) -> str:
    value = fields(request)
    for key in ("review_id", *omit):
        value.pop(key)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _check_request_size(request: LLMRequest) -> None:
    text_bytes = sum(
        len(value.encode())
        for value in (
            request.system_prompt,
            request.user_prompt,
            request.response_schema_json or "",
        )
    )
    if text_bytes > MAX_REQUEST_TEXT_BYTES:
        raise ValueError("diagnostic request exceeds its byte cap; no source is truncated")


@dataclass(frozen=True)
class CapturedCase:
    metadata: dict[str, Any]
    fixture: EvalFixture
    native: LLMRequest
    direct: LLMRequest
    native_exchange: str
    direct_exchange: str


@dataclass(frozen=True)
class CapturedSource:
    directory: Path
    target: Target
    hashes: dict[str, str]
    cases: tuple[CapturedCase, ...]
    dataset: dict[str, Any]
    profile_identity: dict[str, Any] | None = None


def _matching_request(
    directory: Path,
    rows: list[dict[str, Any]],
    case_id: str,
    arm: str,
    file_path: str,
    *,
    compare_discovery: bool = False,
    analyze_version: str = analyze.VERSION,
) -> tuple[LLMRequest, str]:
    matches = []
    for row in rows:
        if row["case_id"] != case_id or row["arm"] != arm:
            continue
        for name in row["raw_exchange_files"]:
            exchange = json.loads(artifact(directory, name).read_bytes())
            request = LLMRequest.model_validate(exchange["request"])
            if request.node_id != "analyze":
                continue
            header = re.search(r"(?m)^File: ([^\r\n]+)$", request.user_prompt)
            if arm == "direct-one-shot" or (
                (
                    request.prompt_template_version in CONCISE_PROMPTS
                    if compare_discovery
                    else request.prompt_template_version == analyze_version
                )
                and header
                and header.group(1) == file_path
            ):
                matches.append((request, name))
    if len(matches) != 1:
        raise ValueError(f"expected one captured {arm} analysis for {case_id}:{file_path}")
    return matches[0]


def _captured_profile(plan: dict[str, Any]) -> dict[str, Any]:
    profile = {
        name: plan.get(name)
        for name in ("profile_id", "reasoning_enabled", "profile_contract_digest")
    }
    if (
        not isinstance(profile["profile_id"], str)
        or not profile["profile_id"]
        or type(profile["reasoning_enabled"]) is not bool
        or not isinstance(profile["profile_contract_digest"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", profile["profile_contract_digest"])
    ):
        raise ValueError("discovery source lacks a complete recorded provider identity")
    return profile


def _check_exchange_profile(
    directory: Path, name: str, target: Target, profile: dict[str, Any]
) -> None:
    exchange = json.loads(artifact(directory, name).read_bytes())
    event = exchange.get("event", {})
    if event.get("model") != target.model or any(
        event.get(key) != value for key, value in profile.items()
    ):
        raise ValueError("captured exchange provider identity differs from the source plan")


def load_source(
    directory: Path, file_path: str, *, compare_discovery: bool = False
) -> CapturedSource:
    """Fail before generation if receipts, source files or direct rendering drift."""
    directory = directory.resolve()
    receipts = json.loads((directory / "receipts.json").read_bytes())
    if not {"plan.json", "results.json"} <= receipts.keys():
        raise ValueError("original plan and results must be covered by receipts")
    for name, digest in receipts.items():
        if sha(artifact(directory, name)) != digest:
            raise ValueError(f"original report receipt differs: {name}")
    plan = json.loads((directory / "plan.json").read_bytes())
    result = json.loads((directory / "results.json").read_bytes())
    if result.get("schema") != "review-comparison-v1" or result.get("status") != "completed":
        raise ValueError("diagnostics require a completed review-comparison-v1 report")
    if result["plan"] != plan:
        raise ValueError("original result and plan differ")
    profile = None
    if compare_discovery:
        if plan.get("comparison_mode") != "module-context" or (
            plan.get("module_context_control", {})
            .get("enabled_by_arm", {})
            .get("concise-with-modules")
            is not True
        ):
            raise ValueError("discovery controls require a recorded module-context candidate")
        profile = _captured_profile(plan)
    hashes = {"receipts.json": sha(directory / "receipts.json"), **receipts}
    for name, digest in plan["inputs"].items():
        relative = "generation/" + name
        if sha(artifact(directory, relative)) != digest:
            raise ValueError(f"original generation source differs: {name}")
        hashes[relative] = digest
    cases = []
    target = Target(**plan["target"])
    for metadata in plan["cases"]:
        fixture_name = "generation/" + metadata["fixture"]
        if hashes.get(fixture_name) != metadata["fixture_sha256"]:
            raise ValueError("original fixture must be hash-bound to generation inputs")
        fixture = EvalFixture.model_validate_json(artifact(directory, fixture_name).read_bytes())
        if not any(file.path == file_path for file in fixture.files):
            continue
        native, native_name = _matching_request(
            directory,
            result["cases"],
            metadata["id"],
            "concise-with-modules" if compare_discovery else "context-off",
            file_path,
            compare_discovery=compare_discovery,
            analyze_version=plan.get("workflow_settings", {})
            .get("analyze", {})
            .get("version", analyze.VERSION),
        )
        direct, direct_name = _matching_request(
            directory, result["cases"], metadata["id"], "direct-one-shot", file_path
        )
        if native_name not in receipts or direct_name not in receipts:
            raise ValueError("selected exchanges must be covered by original receipts")
        if (
            native.model != target.model
            or direct.model != target.model
            or not native.is_eval
            or not direct.is_eval
            or native.degraded_mode
            or native.response_schema_json != ANALYZE_RESPONSE_SCHEMA_JSON
        ):
            raise ValueError("captured requests must use the original model and clean eval route")
        if compare_discovery:
            requests_for_discovery(native)
            assert profile is not None
            for name in (native_name, direct_name):
                _check_exchange_profile(directory, name, target, profile)
        regenerated = direct_review.render_request(
            fixture, model=target.model, review_id=direct.review_id
        )
        if fields(regenerated) != fields(direct):
            raise ValueError("original direct request no longer regenerates exactly")
        cases.append(CapturedCase(metadata, fixture, native, direct, native_name, direct_name))
    if not cases:
        raise ValueError("no captured cases contain the selected analysis file")
    return CapturedSource(directory, target, hashes, tuple(cases), plan["dataset"], profile)


def requests_for_case(
    case: CapturedCase, *, compare_discovery: bool = False
) -> dict[str, LLMRequest]:
    if compare_discovery:
        requests = requests_for_discovery(case.native)
        for request in requests.values():
            _check_request_size(request)
        return requests
    current = case.native
    concise = current.model_copy(
        update={"system_prompt": CONCISE_PROMPT, "prompt_template_version": CONCISE_VERSION}
    )
    reference, manifest = direct_review.render_context(case.fixture)
    full_user = current.user_prompt + "\n\n## Full captured PR reference data\n\n" + reference
    full_manifest = (*current.context_summary, *manifest)
    if current.user_prompt.count(SOURCE_CLAUSE) != 1:
        raise ValueError("captured source-use clause must occur exactly once for clarification")
    clarified_user = current.user_prompt.replace(SOURCE_CLAUSE, CLARIFIED_CLAUSE, 1)
    requests = {
        "current-scoped": current,
        "concise-scoped": concise,
        "current-full-reference": current.model_copy(
            update={"user_prompt": full_user, "context_summary": full_manifest}
        ),
        "concise-full-reference": concise.model_copy(
            update={"user_prompt": full_user, "context_summary": full_manifest}
        ),
        "clarified-scoped": current.model_copy(
            update={"user_prompt": clarified_user, "prompt_template_version": CLARIFIED_VERSION}
        ),
        "direct-control": case.direct,
    }
    for request in requests.values():
        _check_request_size(request)
    for context in ("scoped", "full-reference"):
        left = request_identity(
            requests["current-" + context], omit=("system_prompt", "prompt_template_version")
        )
        right = request_identity(
            requests["concise-" + context], omit=("system_prompt", "prompt_template_version")
        )
        if left != right:
            raise ValueError("instruction arms differ in more than system text/version")
    for instructions in ("current", "concise"):
        left = request_identity(
            requests[instructions + "-scoped"], omit=("user_prompt", "context_summary")
        )
        right = request_identity(
            requests[instructions + "-full-reference"], omit=("user_prompt", "context_summary")
        )
        if left != right:
            raise ValueError("context arms differ in more than source text/manifest")
    if request_identity(
        current, omit=("user_prompt", "prompt_template_version")
    ) != request_identity(
        requests["clarified-scoped"], omit=("user_prompt", "prompt_template_version")
    ):
        raise ValueError("clarified arm differs in more than its declared source-use instruction")
    return requests


def case_plan(case: CapturedCase, *, compare_discovery: bool = False) -> dict[str, Any]:
    requests = requests_for_case(case, compare_discovery=compare_discovery)
    primary_context = case.native.context_summary
    if compare_discovery:
        header = re.search(r"(?m)^File: ([^\r\n]+)$", case.native.user_prompt)
        if header is None:
            raise ValueError("captured discovery request lacks its reviewed file header")
        primary_context = tuple(
            item
            for item in primary_context
            if item.file_path == header.group(1)
            and item.inclusion_reason in {"changed_scope", "changed_region"}
        )
        if not primary_context:
            raise ValueError("captured discovery request has no declared primary source")
    return {
        **case.metadata,
        "native_exchange": case.native_exchange,
        "direct_exchange": case.direct_exchange,
        "primary_context_summary": [item.model_dump(mode="json") for item in primary_context],
        "request_identities": {
            name: request_identity(request) for name, request in requests.items()
        },
        "request_bytes": {
            name: {
                "system": len(request.system_prompt.encode()),
                "user": len(request.user_prompt.encode()),
            }
            for name, request in requests.items()
        },
    }


def source_identity(source: CapturedSource) -> dict[str, Any]:
    identity = {
        "source_report": str(source.directory),
        "source_inputs": source.hashes,
        "target": asdict(source.target),
        "dataset": source.dataset,
    }
    if source.profile_identity is not None:
        identity["source_profile_identity"] = source.profile_identity
    return identity
