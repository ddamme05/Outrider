"""Plan or run a capped model comparison against the current review fixtures.

Planning never constructs providers. Paid runs require both spend flags and
keep each model's compatibility, admission, quality and usage evidence separate.
Existing frozen baseline files are never written by this command.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from outrider.llm.base import INCLUDE_TEXT_OPT_IN, LLMProviderError, LLMRefusalError
from outrider.llm.host_profiles import resolve_host_identity, validate_available_models
from outrider.llm.pricing import (
    COST_UNADJUDICATED_PROFILE_IDS,
    LONG_CONTEXT_POLICY,
    PRICING_VERSION,
    RATE_TABLE,
    SERVICE_TIER_MULTIPLIERS,
    pricing_key,
)
from outrider.policy.findings import EvidenceTier

from .exemplar_baseline import PROMPT_IDENTITY_RECIPE, analyze_prompt_identity
from .grading import finding_matches, grade
from .model_comparison import run_analyze_instrumented, state_from_eval_fixture
from .model_refresh_report import REPORT_SCHEMA
from .review_quality import MEASUREMENT_CONTRACT_V2

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.llm.base import LLMProvider, LLMRequest, LLMResponse

    from .grading import ExpectedFinding
    from .model_comparison import AnalyzeRun
    from .prompt_control import ContextPairs

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_CONTRACT = MEASUREMENT_CONTRACT_V2


@dataclass(frozen=True)
class Target:
    host: str
    model: str
    role: str

    @property
    def name(self) -> str:
        return f"{self.host}:{self.model}:{self.role}"


TARGETS = {
    "anthropic-sonnet-5": Target("anthropic", "claude-sonnet-5", "baseline-deep"),
    "anthropic-haiku-4-5": Target("anthropic", "claude-haiku-4-5", "baseline-cheap"),
    "anthropic-sonnet-5-5": Target("anthropic", "claude-sonnet-5-5", "candidate-deep"),
    "anthropic-haiku-5-5": Target("anthropic", "claude-haiku-5-5", "candidate-cheap"),
    "fireworks-glm-5-2": Target("fireworks", "accounts/fireworks/models/glm-5p2", "baseline"),
    "fireworks-glm-5-3-flash": Target(
        "fireworks", "accounts/fireworks/models/glm-5p3-flash", "candidate"
    ),
    "openai-sol": Target("openai", "gpt-6.1-sol", "research"),
    "openai-luna": Target("openai", "gpt-6-luna", "research"),
}


@dataclass(frozen=True)
class Scenario:
    path: str
    expected: tuple[ExpectedFinding, ...]
    track: str
    analysis_file: str | None = None


def scenarios(suite: str = "legacy") -> tuple[Scenario, ...]:
    if suite == "real-pr":
        from .real_pr_cases import load_cases

        return tuple(
            Scenario(case.fixture_path, case.expected, case.track, case.analysis_file)
            for case in load_cases()
        )
    if suite == "grounded":
        from .grounded_cases import load_cases

        return tuple(
            Scenario(case.fixture_path, case.expected, case.track) for case in load_cases()
        )
    if suite != "legacy":
        raise ValueError("unknown model comparison suite")
    from . import test_model_comparison as fixtures

    fixtures._holdout_preamble()
    rows = [
        Scenario(path, expected, "recall")
        for path, expected in fixtures._GROUND_TRUTH_BY_FIXTURE.items()
    ]
    rows.extend(Scenario(path, (), "safe") for path in fixtures._SAFE_CODE_FIXTURES)
    rows.extend(
        Scenario(path, expected, "holdout-recall")
        for path, expected in fixtures._FRESH_RECALL_HOLDOUTS.values()
    )
    rows.extend(Scenario(path, (), "holdout-safe") for path in fixtures._FRESH_SAFE_PATHS)
    if any(len(row.expected) > 1 for row in rows):
        raise ValueError(
            "refresh majority scoring requires at most one expected finding per scenario"
        )
    return tuple(rows)


def compatibility_scenarios(rows: tuple[Scenario, ...] | None = None) -> tuple[Scenario, ...]:
    if rows is None:
        from . import test_model_comparison as fixtures

        rows = tuple(
            Scenario(path, expected, "recall")
            for path, expected in fixtures._GROUND_TRUTH_BY_FIXTURE.items()
        )
        rows += tuple(
            Scenario(path, expected, "holdout-recall")
            for path, expected in fixtures._FRESH_RECALL_HOLDOUTS.values()
        )
        rows += tuple(Scenario(path, (), "holdout-safe") for path in fixtures._FRESH_SAFE_PATHS)
    names = (
        "sqli_unseen_template.json",
        "safe_unseen_psycopg_composition.json",
        "cmd_injection_eval_indirect.json",
    )
    return tuple(next(row for row in rows if Path(row.path).name == name) for name in names)


def state_for_scenario(scenario: Scenario, *, suite: str = "legacy"):
    """Hold benchmark routing fixed without requiring scripted triage for new cases."""
    if suite == "legacy":
        return state_from_eval_fixture(ROOT / scenario.path)
    if suite not in {"grounded", "real-pr"}:
        raise ValueError("unknown model comparison suite")
    from outrider.schemas.triage_result import ReviewDimension, RiskLevel

    return state_from_eval_fixture(
        ROOT / scenario.path,
        dimensions=tuple(ReviewDimension),
        overall_risk=RiskLevel.MEDIUM,
        analysis_file=scenario.analysis_file,
    )


def dataset_identity(suite: str, rows: tuple[Scenario, ...]) -> dict[str, Any]:
    """Describe the selected analysis data and the files that establish its labels."""
    if suite == "real-pr":
        from .real_pr_cases import DATASET_VERSION, EXPOSURE, dataset_inputs, load_cases

        return {
            "id": DATASET_VERSION,
            "exposure": EXPOSURE,
            "held_out": False,
            "analysis_only": True,
            "family": "commonprefix-path-containment",
            "cases": [
                {
                    **asdict(case),
                    "expected": [item.model_dump(mode="json") for item in case.expected],
                    "scoped_counterparts": [asdict(scope) for scope in case.scoped_counterparts],
                }
                for case in load_cases()
            ],
            "inputs": dataset_inputs(),
        }
    if suite == "grounded":
        from .grounded_cases import DATASET_VERSION, EXPOSURE, dataset_inputs, load_cases

        return {
            "id": DATASET_VERSION,
            "exposure": EXPOSURE,
            "held_out": False,
            "analysis_only": True,
            "cases": [
                {
                    **asdict(case),
                    "expected": [item.model_dump(mode="json") for item in case.expected],
                }
                for case in load_cases()
            ],
            "inputs": dataset_inputs(),
        }
    names = [row.path for row in rows] + ["tests/eval/test_model_comparison.py"]
    return {
        "id": "legacy-32-plus-supplemental-4",
        "exposure": "previously-evaluated",
        "held_out": False,
        "cases": [
            {
                "path": row.path,
                "track": row.track,
                "expected": [item.model_dump(mode="json") for item in row.expected],
            }
            for row in rows
        ],
        "inputs": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
    }


def context_limit(target: Target) -> int:
    if target.host == "anthropic":
        if target.model.startswith("claude-haiku-4-5"):
            return 200_000
        if target.model.startswith(("claude-sonnet-5", "claude-haiku-5-5")):
            return 1_000_000
    if target.host in {"fireworks", "baseten"} and "glm-5" in target.model.lower():
        return 1_048_576
    if target.host == "openai" and target.model in {"gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"}:
        return 1_050_000
    raise ValueError(f"no reviewed context bound for {target.host}:{target.model}")


def maximum_rates(target: Target) -> tuple[Decimal, Decimal]:
    key = pricing_key(target.host, target.model)
    rates = RATE_TABLE[key]
    policy = LONG_CONTEXT_POLICY.get(key)
    multipliers = (
        (Decimal(1),) * 4
        if policy is None
        else (
            max(Decimal(1), policy.in_mult),
            max(Decimal(1), policy.cache_write_mult),
            max(Decimal(1), policy.cache_read_mult),
            max(Decimal(1), policy.out_mult),
        )
    )
    tier = Decimal(1)
    if target.host == "openai":
        from outrider.llm.host_profiles import resolve_host_profile

        requested = (
            resolve_host_profile(target.host).for_models((target.model,)).requested_service_tier
        )
        tier = SERVICE_TIER_MULTIPLIERS[requested]
    input_rate = max(
        rates.in_per_token * multipliers[0],
        rates.cache_write_per_token * multipliers[1],
        rates.cache_read_per_token * multipliers[2],
    )
    return input_rate * tier, rates.out_per_token * multipliers[3] * tier


def reservation_usd(target: Target, max_output_tokens: int) -> Decimal:
    input_rate, output_rate = maximum_rates(target)
    value = context_limit(target) * input_rate + max_output_tokens * output_rate
    return value.quantize(Decimal("0.00000001"), rounding=ROUND_CEILING)


def input_hashes(rows: tuple[Scenario, ...]) -> dict[str, str]:
    paths = {ROOT / row.path for row in rows}
    paths.update((ROOT / "pyproject.toml", ROOT / "uv.lock"))
    paths.update(
        path for path in (ROOT / "src/outrider").rglob("*") if path.suffix in {".py", ".scm"}
    )
    paths.update(
        path for path in (ROOT / "tests/eval").rglob("*") if path.suffix in {".py", ".json", ".txt"}
    )
    paths.update(
        path for path in (ROOT / "spikes/openai/fixtures").rglob("*") if path.suffix == ".json"
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def build_plan(
    targets: tuple[Target, ...], *, phase: str, reps: int, suite: str = "legacy"
) -> dict[str, Any]:
    from .model_refresh_regrade import MAX_REPETITIONS, measurement_inputs

    rows = scenarios(suite)
    compatibility = compatibility_scenarios()
    if suite == "real-pr" and phase != "analyze":
        raise ValueError("the real-pr suite supports analyze only")
    if suite == "grounded" and phase not in {"compat", "analyze"}:
        raise ValueError("the grounded suite currently supports compat or analyze only")
    dataset = dataset_identity(suite, rows)
    if (
        not 1 <= reps <= MAX_REPETITIONS
        or not targets
        or len({target.name for target in targets}) != len(targets)
    ):
        raise ValueError("one to ten repetitions and distinct explicit targets are required")
    from outrider.prompts import analyze, patch, synthesize, trace, triage

    from .model_refresh_nodes import node_case_names

    output_ceiling = max(
        module.MAX_TOKENS for module in (analyze, patch, synthesize, trace, triage)
    )

    calls = (
        len(compatibility)
        + (len(rows) * reps if phase in {"analyze", "all"} else 0)
        + (len(node_case_names()) * reps if phase in {"nodes", "all"} else 0)
    )
    version, digest = analyze_prompt_identity()
    planned = []
    for target in targets:
        entry: dict[str, Any] = asdict(target) | {"name": target.name, "planned_calls": calls}
        try:
            validate_available_models(target.host, (target.model,))
            profile_id, reasoning, contract = resolve_host_identity(
                target.host, reasoning=False, models=(target.model,)
            )
            reserve = reservation_usd(target, output_ceiling)
            entry.update(
                context_tokens=context_limit(target),
                per_call_max_reservation_usd=str(reserve),
                profile_id=profile_id,
                reasoning_enabled=reasoning,
                profile_contract_digest=contract,
                ready=True,
            )
            rates = RATE_TABLE[pricing_key(target.host, target.model)]
            estimate = (
                12_000
                * max(rates.in_per_token, rates.cache_write_per_token, rates.cache_read_per_token)
                + output_ceiling * rates.out_per_token
            )
            entry["planning_estimate"] = {
                "input_tokens_assumed": 12_000,
                "output_tokens_assumed": output_ceiling,
                "usd_for_scheduled_calls": str(estimate * calls),
                "basis": (
                    "Illustration using prior fixture-sized input and the full output ceiling, "
                    "at flat table rates. Not observed usage, an invoice or a spending guarantee."
                ),
                "billing_adjudicated": target.host not in COST_UNADJUDICATED_PROFILE_IDS,
            }
        except (KeyError, ValueError) as exc:
            entry.update(ready=False, issue_type=type(exc).__name__, issue=str(exc))
        planned.append(entry)
    return {
        "schema": REPORT_SCHEMA,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "measurement_inputs": measurement_inputs(suite),
        "runtime": {
            "python": sys.version,
            "packages": {
                name: package_version(name)
                for name in (
                    "anthropic",
                    "openai",
                    "langgraph",
                    "pydantic",
                    "sqlalchemy",
                    "psycopg",
                )
            },
        },
        "mode": "free-plan",
        "phase": phase,
        "suite": suite,
        "dataset": dataset,
        "analysis_state": {
            "tier": "standard",
            "risk": "medium",
            "dimensions": "all",
            "model_triage_measured": False,
        }
        | (
            {
                "selected_files": {row.path: row.analysis_file for row in rows},
                "other_changed_files": "skip",
                "full_pr_context_retained": True,
            }
            if suite == "real-pr"
            else {}
        )
        if suite in {"grounded", "real-pr"}
        else {
            "tier": "standard",
            "risk_and_dimensions": "fixture-scripted-triage",
            "model_triage_measured": False,
        },
        "repetitions": reps,
        "targets": planned,
        "planned_calls": calls * len(targets),
        "fixture_count": len(rows),
        "analysis_calls_per_target": len(rows) * reps
        if phase == "analyze" or phase == "all"
        else 0,
        "prompt_version": version,
        "prompt_digest": digest,
        "prompt_identity_recipe": PROMPT_IDENTITY_RECIPE,
        "pricing_version": PRICING_VERSION,
        "inputs": input_hashes(rows) | dataset["inputs"],
        "scenarios": [
            {"fixture": row.path, "track": row.track}
            | ({"analysis_file": row.analysis_file} if row.analysis_file is not None else {})
            for row in rows
        ],
        "node_cases": list(node_case_names()),
        "compatibility_cases": [row.path for row in compatibility],
        "reservation_note": (
            "Worst-case headroom, not expected spend. Actual usage-based cost is normally lower; "
            "unpriced usage stays an upper-bound reservation."
        ),
        "observed_cost_estimate_usd": None,
        "estimate_note": (
            "No new live usage has been measured. The maximum reservation is not an estimate "
            "of the bill."
        ),
        "quality_scope": (
            "V2 bounded visible-context grounding across admitted evidence tiers; "
            "independent producer coverage, locations and product emissions stay separate. "
            "Semantic quality and precision require adjudication. No automatic qualification."
        ),
    }


class BudgetLimitError(RuntimeError):
    """The next call cannot fit the authorized cost or call ceiling."""


class OutputLimitError(RuntimeError):
    """The persisted completion reached its output limit and cannot establish quality."""


@dataclass
class Budget:
    maximum_usd: Decimal
    maximum_calls: int
    calls: int = 0
    known_usd: Decimal = Decimal(0)
    uncertain_usd: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if not self.maximum_usd.is_finite() or self.maximum_usd <= 0 or self.maximum_calls < 1:
            raise ValueError("finite positive dollar and call caps are required")

    def reserve(self, amount: Decimal) -> int:
        if not amount.is_finite() or amount < 0:
            raise ValueError("reservation must be finite and nonnegative")
        if (
            self.calls >= self.maximum_calls
            or self.known_usd + self.uncertain_usd + amount > self.maximum_usd
        ):
            raise BudgetLimitError("next call exceeds authorized headroom; no generation requested")
        self.calls += 1
        self.uncertain_usd += amount
        return self.calls

    def settle(self, reserved: Decimal, retained: Decimal, *, priced: bool) -> None:
        self.uncertain_usd -= reserved
        if priced:
            self.known_usd += retained
        else:
            self.uncertain_usd += retained
        if retained > reserved:
            raise RuntimeError("observed charge exceeded the reviewed reservation contract")


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


@dataclass
class ExchangeRecorder:
    directory: Path
    target: Target
    scenario: str = ""
    call_index: int = 0
    exchanges: list[dict[str, Any]] = field(default_factory=list)

    async def persist(self, event: Any, request: LLMRequest, response: LLMResponse) -> None:
        record = {
            "call_index": self.call_index,
            "target": asdict(self.target),
            "scenario": self.scenario,
            "event": event.model_dump(mode="json"),
            "request": request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
            "response": response.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
            "provider_response_id": None,
            "reasoning_tokens": None,
            "unavailable_fields": (
                "Provider response IDs and reasoning-token breakdown are not retained by "
                "the normalized provider contract."
            ),
        }
        path = self.directory / f"{self.call_index:06d}.exchange.json"
        write_json(path, record)
        self.exchanges.append(record)


class BudgetedProvider:
    def __init__(
        self,
        provider: LLMProvider,
        target: Target,
        recorder: ExchangeRecorder,
        budget: Budget,
        identities: dict[str, str],
        *,
        measurement_contract: str | None = None,
    ) -> None:
        self.provider, self.target, self.recorder, self.budget = provider, target, recorder, budget
        self.identities = identities
        self.measurement_contract = measurement_contract
        self.attempts: list[dict[str, Any]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.model != self.target.model or not request.is_eval:
            raise ValueError(
                "refresh requests must use the explicit target model and evaluation scope"
            )
        for path, digest in self.identities.items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"frozen comparison input changed: {path}")
        reserve = reservation_usd(self.target, request.max_tokens)
        index = self.budget.reserve(reserve)
        self.recorder.call_index = index
        before = len(self.recorder.exchanges)
        started, error, diagnostics = time.perf_counter(), None, None
        try:
            response = await self.provider.complete(request)
            if response.finish_reason in {"length", "max_tokens"}:
                raise OutputLimitError("completion reached the output limit; no retry requested")
            if (
                self.measurement_contract == MEASUREMENT_CONTRACT_V2
                and response.finish_reason == "refusal"
            ):
                raise LLMRefusalError("recorded completion refused the request; no retry requested")
            return response
        except BaseException as exc:
            error = type(exc).__name__
            if isinstance(exc, LLMProviderError):
                diagnostics = exc.diagnostic_metadata()
            raise
        finally:
            completed = self.recorder.exchanges[before:]
            exchange = completed[0] if len(completed) == 1 else None
            actual = None if exchange is None else exchange["event"].get("cost_usd")
            priced = actual is not None and self.target.host not in COST_UNADJUDICATED_PROFILE_IDS
            retained = Decimal(str(actual)) if priced else reserve
            if exchange is not None and not priced:
                response = exchange["response"]
                input_tokens = (
                    max(
                        response.get("billed_prompt_tokens") or 0,
                        response["input_tokens"] + response["cache_read_tokens"],
                    )
                    + response["cache_write_tokens"]
                )
                input_rate, output_rate = maximum_rates(self.target)
                if self.target.host != "openai" or response.get("service_tier_actual") == "default":
                    retained = (
                        input_tokens * input_rate
                        + max(response["output_tokens"], request.max_tokens) * output_rate
                    )
            self.budget.settle(reserve, retained, priced=priced)
            truncated = exchange is not None and exchange["response"]["finish_reason"] in {
                "length",
                "max_tokens",
            }
            attempt = {
                "index": index,
                "target": self.target.name,
                "scenario": self.recorder.scenario,
                "status": "output-limit" if truncated else "error" if error else "completed",
                "error_type": error,
                "provider_diagnostics": diagnostics,
                "worst_case_reserved_usd": str(reserve),
                "usage_based_usd": str(actual) if priced else None,
                "budget_retained_upper_usd": str(retained),
                "cost_status": "usage-priced" if priced else "uncosted",
                "wall_ms": round((time.perf_counter() - started) * 1000),
                "audit_event_id": None if exchange is None else exchange["event"]["event_id"],
                "finish_reason": None
                if exchange is None
                else exchange["response"]["finish_reason"],
                "persisted_exchange": exchange is not None,
                "truncated": truncated,
            }
            self.attempts.append(attempt)
            write_json(self.recorder.directory / f"{index:06d}.attempt.json", attempt)

    async def aclose(self) -> None:
        await self.provider.aclose()


def create_provider(target: Target, recorder: ExchangeRecorder) -> LLMProvider:
    from pydantic import SecretStr

    key_name = {
        "anthropic": "ANTHROPIC_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "baseten": "BASETEN_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[target.host]
    value = os.environ.get(key_name)
    if not value or value.startswith("op://"):
        raise ValueError(f"resolved {key_name} is required for the selected host")
    if target.host == "anthropic":
        from outrider.llm.anthropic_provider import AnthropicProvider
        from outrider.llm.config import ModelConfig

        config = ModelConfig(
            **dict.fromkeys(
                (
                    "triage_model",
                    "analyze_model",
                    "standard_analyze_model",
                    "synthesize_model",
                    "trace_model",
                    "patch_model",
                ),
                target.model,
            )
        )
        return AnthropicProvider(api_key=SecretStr(value), model_config=config, persister=recorder)
    from outrider.llm.host_profiles import resolve_host_profile
    from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        api_key=SecretStr(value),
        profile=resolve_host_profile(target.host).for_models((target.model,)),
        models=(target.model,),
        persister=recorder,
    )


def score_run(result: AnalyzeRun, scenario: Scenario) -> dict[str, Any]:
    producer = result.producer_findings
    producer_hashes = {finding.content_hash for finding in producer}
    judged = tuple(
        finding
        for finding in result.findings
        if finding.evidence_tier is EvidenceTier.JUDGED
        and finding.content_hash not in producer_hashes
    )
    expected = tuple(
        item
        for item in scenario.expected
        if not any(finding_matches(finding, item) for finding in producer)
    )
    model_grade, product_grade = grade(judged, expected), grade(result.findings, scenario.expected)
    false_alarms = None
    if not scenario.expected:
        false_alarms = (
            sum(finding.finding_type.value == "sql_injection" for finding in judged)
            if scenario.track == "holdout-safe"
            else len(judged)
        )
    return {
        "model_judged": {
            "matched": model_grade.n_matched,
            "expected": len(expected),
            "extras_advisory": len(model_grade.extra),
            "false_alarms": false_alarms,
        },
        "producer_covered_expected": len(scenario.expected) - len(expected),
        "product": {"matched": product_grade.n_matched, "expected": len(scenario.expected)},
        "response_rejections": result.response_rejections,
        "proposal_rejections": result.proposal_rejections,
        "findings": [finding.model_dump(mode="json") for finding in result.findings],
        "producer_hashes": sorted(producer_hashes),
    }


def _score_collected_case(result, scenario, *, plan, exchanges):
    if plan["measurement_contract"] != MEASUREMENT_CONTRACT_V2:
        from .model_refresh_regrade import V1_CONTRACT

        if plan["measurement_contract"] != V1_CONTRACT:
            raise ValueError("unknown refresh measurement contract")
        return score_run(result, scenario)
    from outrider.llm.base import LLMRequest

    from .model_refresh_regrade import measurement_inputs, score_case_v2

    if len(exchanges) != 1:
        raise ValueError("v2 case scoring requires exactly one saved request context")
    if measurement_inputs(plan.get("suite", "legacy")) != plan["measurement_inputs"]:
        raise ValueError("v2 measurement inputs changed before scoring")
    score = score_case_v2(
        result,
        scenario,
        fixture_path=ROOT / scenario.path,
        request=LLMRequest.model_validate(exchanges[0]["request"]),
    )
    if measurement_inputs(plan.get("suite", "legacy")) != plan["measurement_inputs"]:
        raise ValueError("v2 measurement inputs changed during scoring")
    return score


async def run_refresh(
    plan: dict[str, Any],
    *,
    maximum_usd: Decimal,
    maximum_calls: int,
    output: Path,
    provider_factory: Callable[[Target, ExchangeRecorder], LLMProvider] = create_provider,
    prompt_contexts: ContextPairs | None = None,
) -> dict[str, Any]:
    if "prompt_control_variant" in plan and prompt_contexts is None:
        raise ValueError("system-prompt control requires paired context checks")
    budget = Budget(maximum_usd, maximum_calls)
    output.mkdir(parents=True, exist_ok=False)
    calls_dir = output / "calls"
    calls_dir.mkdir()
    write_json(output / "plan.json", plan)
    suite = plan.get("suite", "legacy")
    rows, candidates = scenarios(suite), []
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "evidence_kind": "live-provider"
        if provider_factory is create_provider
        else "injected-provider",
        "plan": plan,
        "candidates": candidates,
        "cost_note": (
            "Usage-priced USD is computed from reported tokens and versioned rates, not an "
            "invoice. Uncosted calls cannot establish a cost winner."
        ),
    }
    try:
        for entry in plan["targets"]:
            target = Target(entry["host"], entry["model"], entry["role"])
            candidate: dict[str, Any] = {
                "target": asdict(target),
                "status": "prepared",
                "cases": [],
                "calls": [],
            }
            candidates.append(candidate)
            if not entry["ready"]:
                candidate["status"] = "not-ready"
                continue
            recorder = ExchangeRecorder(calls_dir, target)
            try:
                inner = provider_factory(target, recorder)
            except Exception as exc:
                candidate.update(status="unavailable", error_type=type(exc).__name__)
                if isinstance(exc, LLMProviderError):
                    candidate["provider_diagnostics"] = exc.diagnostic_metadata()
                continue
            if "prompt_control_variant" in plan:
                from .prompt_control import PromptControlProvider

                assert prompt_contexts is not None
                inner = PromptControlProvider(
                    inner, plan["prompt_control_variant"], recorder, prompt_contexts
                )
            provider = BudgetedProvider(
                inner,
                target,
                recorder,
                budget,
                plan["inputs"],
                measurement_contract=plan["measurement_contract"],
            )
            try:
                for case in compatibility_scenarios():
                    recorder.scenario = f"compat:{case.path}"
                    before_calls, before_exchanges = len(provider.attempts), len(recorder.exchanges)
                    case_started = time.perf_counter()
                    result = await run_analyze_instrumented(
                        state_for_scenario(case),
                        provider=provider,
                        model=target.model,
                        collect_producer_coverage=plan["measurement_contract"]
                        == MEASUREMENT_CONTRACT_V2,
                    )
                    if (
                        result.response_rejections
                        or result.proposal_rejections
                        or len(provider.attempts) - before_calls != 1
                        or len(recorder.exchanges) - before_exchanges != 1
                        or provider.attempts[-1]["truncated"]
                    ):
                        raise ValueError(
                            "compatibility response did not satisfy real analyze admission"
                        )
                    candidate["cases"].append(
                        {
                            "phase": "compat",
                            "fixture": case.path,
                            "track": case.track,
                            **_score_collected_case(
                                result,
                                case,
                                plan=plan,
                                exchanges=recorder.exchanges[before_exchanges:],
                            ),
                        }
                    )
                    if prompt_contexts is not None:
                        candidate["cases"][-1]["harness_wall_ms"] = round(
                            (time.perf_counter() - case_started) * 1000
                        )
                for rep in range(plan["repetitions"]):
                    if plan["phase"] in {"analyze", "all"}:
                        for case in rows:
                            recorder.scenario = f"analyze:{rep}:{case.path}"
                            before_exchanges = len(recorder.exchanges)
                            case_started = time.perf_counter()
                            result = await run_analyze_instrumented(
                                state_for_scenario(case, suite=suite),
                                provider=provider,
                                model=target.model,
                                collect_producer_coverage=plan["measurement_contract"]
                                == MEASUREMENT_CONTRACT_V2,
                            )
                            candidate["cases"].append(
                                {
                                    "phase": "analyze",
                                    "rep": rep,
                                    "fixture": case.path,
                                    "track": case.track,
                                    **_score_collected_case(
                                        result,
                                        case,
                                        plan=plan,
                                        exchanges=recorder.exchanges[before_exchanges:],
                                    ),
                                }
                            )
                            if prompt_contexts is not None:
                                candidate["cases"][-1]["harness_wall_ms"] = round(
                                    (time.perf_counter() - case_started) * 1000
                                )
                    if plan["phase"] in {"nodes", "all"}:
                        from .model_refresh_nodes import node_case_names, run_node_case

                        for name in node_case_names():
                            recorder.scenario = f"nodes:{rep}:{name}"
                            result = await run_node_case(
                                name,
                                provider=provider,
                                model=target.model,
                                measurement_contract=plan["measurement_contract"]
                                if plan["measurement_contract"] == MEASUREMENT_CONTRACT_V2
                                else None,
                            )
                            candidate["cases"].append(
                                {"phase": "nodes", "rep": rep, "name": name, "result": result}
                            )
                candidate["status"] = "completed"
            except BudgetLimitError:
                candidate["status"] = "budget-stopped-before-call"
            except OutputLimitError as exc:
                candidate.update(
                    status="output-limit",
                    error_type=type(exc).__name__,
                    failed_scenario=recorder.scenario,
                )
            except Exception as exc:
                candidate.update(
                    status="incompatible-or-failed",
                    error_type=type(exc).__name__,
                    failed_scenario=recorder.scenario,
                )
                if isinstance(exc, LLMProviderError):
                    candidate["provider_diagnostics"] = exc.diagnostic_metadata()
            finally:
                candidate["calls"] = provider.attempts
                candidate["cost_eligible"] = bool(provider.attempts) and all(
                    call["cost_status"] == "usage-priced" for call in provider.attempts
                )
                try:
                    await provider.aclose()
                except Exception as exc:
                    candidate["close_error_type"] = type(exc).__name__
    finally:
        report["budget"] = {
            "authorized_maximum_usd": str(maximum_usd),
            "maximum_calls": maximum_calls,
            "calls": budget.calls,
            "known_usage_based_usd": str(budget.known_usd),
            "uncertain_budget_upper_usd": str(budget.uncertain_usd),
        }
        from .model_refresh_report import comparisons, markdown

        report["comparisons"] = comparisons(report)
        write_json(output / "results.json", report)
        with (output / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(markdown(report))
    return report


def parse_target(value: str) -> Target:
    if value in TARGETS:
        return TARGETS[value]
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("use a listed target alias or HOST:MODEL:ROLE")
    return Target(*parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", type=parse_target)
    parser.add_argument(
        "--compare", nargs=2, type=Path, metavar=("BASELINE_JSON", "CANDIDATE_JSON")
    )
    parser.add_argument(
        "--compare-prompts",
        action="store_true",
        help="analysis-only system-prompt control on identical source inputs; doubles calls",
    )
    parser.add_argument("--phase", choices=("compat", "analyze", "nodes", "all"), default="compat")
    parser.add_argument("--suite", choices=("legacy", "grounded", "real-pr"), default="legacy")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--i-authorize-spend", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cost-usd", type=Decimal)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model-refresh"))
    args = parser.parse_args(argv)
    if args.compare:
        if args.paid or args.i_authorize_spend or args.target or args.compare_prompts:
            parser.error("--compare reads saved reports without paid flags or target selection")
        from .model_refresh_report import compare_saved, markdown

        first, second = (json.loads(path.read_text()) for path in args.compare)
        try:
            combined = compare_saved(first, second)
        except ValueError as exc:
            parser.error(str(exc))
        destination = args.output_dir / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-compare-" + uuid4().hex[:12]
        )
        destination.mkdir(parents=True, exist_ok=False)
        combined["source_reports"] = [str(path.resolve()) for path in args.compare]
        write_json(destination / "results.json", combined)
        (destination / "report.md").write_text(markdown(combined))
        print(
            json.dumps(
                {"report": str(destination), "comparisons": combined["comparisons"]}, indent=2
            )
        )
        return 0
    if not args.target:
        parser.error("select explicit --target aliases or HOST:MODEL:ROLE")
    if args.paid != args.i_authorize_spend:
        parser.error("paid execution needs BOTH --paid and --i-authorize-spend")
    if args.reps < 1 or args.max_calls < 1:
        parser.error("--reps and --max-calls must be positive")
    if args.max_cost_usd is not None and (
        not args.max_cost_usd.is_finite() or args.max_cost_usd <= 0
    ):
        parser.error("--max-cost-usd must be finite and positive")
    if args.suite == "grounded" and args.phase not in {"compat", "analyze"}:
        parser.error("the grounded suite currently supports compat or analyze only")
    if args.suite == "real-pr" and args.phase != "analyze":
        parser.error("the real-pr suite supports analyze only")
    plan = build_plan(tuple(args.target), phase=args.phase, reps=args.reps, suite=args.suite)
    if args.compare_prompts:
        from .prompt_control import prepare_plan

        try:
            plan = prepare_plan(plan)
        except ValueError as exc:
            parser.error(str(exc))
        if args.max_calls < 2:
            parser.error("--compare-prompts requires --max-calls of at least two")
    display = {
        key: value
        for key, value in plan.items()
        if key not in {"inputs", "scenarios", "node_cases", "compatibility_cases"}
    }
    display.update(
        input_file_count=len(plan["inputs"]),
        authorized_maximum_usd=None if args.max_cost_usd is None else str(args.max_cost_usd),
        maximum_calls=args.max_calls,
        model_calls_made=0,
    )
    if args.compare_prompts:
        display["per_arm_caps"] = {
            "maximum_usd": None if args.max_cost_usd is None else str(args.max_cost_usd / 2),
            "maximum_calls": args.max_calls // 2,
            "unallocated_calls": args.max_calls % 2,
            "unused_cap_transferred": False,
        }
    print(json.dumps(display, indent=2))
    if not args.paid or args.dry_run:
        return 0
    if (
        args.max_cost_usd is None
        or not args.max_cost_usd.is_finite()
        or args.max_cost_usd <= 0
        or args.max_calls < 1
    ):
        parser.error("paid execution requires positive --max-cost-usd and --max-calls")
    destination = args.output_dir / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    )
    if args.compare_prompts:
        from .prompt_control import run_prompt_control

        report = asyncio.run(
            run_prompt_control(
                plan,
                maximum_usd=args.max_cost_usd,
                maximum_calls=args.max_calls,
                output=destination,
            )
        )
        print(
            json.dumps(
                {"report": str(destination), "status": report["status"], "arms": report["arms"]},
                indent=2,
            )
        )
        return 0 if report["status"] == "completed" else 1
    report = asyncio.run(
        run_refresh(
            plan, maximum_usd=args.max_cost_usd, maximum_calls=args.max_calls, output=destination
        )
    )
    print(
        json.dumps(
            {
                "report": str(destination),
                "budget": report["budget"],
                "targets": [
                    {"target": row["target"], "status": row["status"]}
                    for row in report["candidates"]
                ],
            },
            indent=2,
        )
    )
    return 0 if all(row["status"] == "completed" for row in report["candidates"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
