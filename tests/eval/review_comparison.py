"""Plan or collect full-workflow reviews and an unrestricted one-shot model review.

Free planning needs no database, credentials or provider. Collection uses the
existing isolated-database eval driver, fixture GitHub reads and captured posting.
Findings remain unadjudicated; this command never declares a quality winner.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from contextvars import ContextVar
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from outrider.agent.eval_driver import EvalFixture, run_review
from outrider.agent.nodes.analyze import ANALYZE_MAX_CONCURRENCY, DEFAULT_REVIEW_BUDGET_TOKENS
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER
from outrider.eval_support import assert_test_url_is_isolated, require_eval_mode
from outrider.llm.base import INCLUDE_TEXT_OPT_IN
from outrider.llm.config import ModelConfig
from outrider.llm.host_profiles import resolve_host_identity, validate_available_models
from outrider.llm.pricing import PRICING_VERSION
from outrider.prompts import analyze, synthesize, trace, triage

from . import direct_review
from .discovery_control import discovery_identity, requests_for_discovery
from .model_refresh import (
    ROOT,
    Budget,
    BudgetedProvider,
    BudgetLimitError,
    ExchangeRecorder,
    Scenario,
    Target,
    create_provider,
    input_hashes,
    parse_target,
    reservation_usd,
    write_json,
)
from .prompt_control import request_for_variant, variant_identity
from .review_costs import SCHEMA as COST_SCHEMA
from .review_costs import CostInputError, summarize_costs
from .review_costs import render_text as render_costs

if TYPE_CHECKING:
    from outrider.llm.base import LLMExchangePersister, LLMProvider, LLMRequest, LLMResponse

SCHEMA = "review-comparison-v1"
ARMS = ("context-off", "context-on", "direct-one-shot")
INSTRUCTION_ARMS = ("current-workflow", "concise-workflow", "direct-one-shot")
CALLED_FUNCTION_ARMS = ("concise-workflow", "concise-with-helpers", "direct-one-shot")
MODULE_CONTEXT_ARMS = ("concise-workflow", "concise-with-modules", "direct-one-shot")
REVIEW_OPENING_ARMS = ("concise-with-modules", "alternative-with-modules", "direct-one-shot")
CANDIDATE_CHECK_ARMS = ("concise-with-modules", "concise-with-checks", "direct-one-shot")
_ACTIVE_RECORDER: ContextVar[ExchangeRecorder] = ContextVar("review_comparison_recorder")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_identity(request: LLMRequest) -> str:
    fields = request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
    fields.pop("review_id")
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def _comparison_arms(plan: dict[str, Any]) -> tuple[str, ...]:
    mode = plan.get("comparison_mode", "context")
    if mode == "context":
        arms = ARMS
    elif mode == "instructions":
        arms = INSTRUCTION_ARMS
    elif mode == "called-functions":
        arms = CALLED_FUNCTION_ARMS
    elif mode == "module-context":
        arms = MODULE_CONTEXT_ARMS
    elif mode == "review-opening":
        arms = REVIEW_OPENING_ARMS
    elif mode == "candidate-checks":
        arms = CANDIDATE_CHECK_ARMS
    else:
        raise ValueError("unknown workflow comparison mode")
    if "arms" in plan and tuple(plan["arms"]) != arms:
        raise ValueError("declared arms differ from the workflow comparison mode")
    return arms


def build_plan(
    target: Target,
    *,
    reps: int = 3,
    calls_per_review: int = 32,
    analyze_concurrency: int = ANALYZE_MAX_CONCURRENCY,
    case_ids: tuple[str, ...] = (),
    compare_instructions: bool = False,
    supplemental_context_enabled: bool = False,
    compare_called_functions: bool = False,
    compare_module_context: bool = False,
    compare_review_opening: bool = False,
    compare_candidate_checks: bool = False,
    include_path_cases: bool = False,
) -> dict[str, Any]:
    from .review_cases import DATASET_VERSION, dataset_inputs, load_cases

    if compare_candidate_checks and (
        compare_instructions
        or compare_called_functions
        or compare_module_context
        or compare_review_opening
    ):
        raise ValueError("--compare-candidate-checks cannot be combined with another comparison")
    if compare_candidate_checks and supplemental_context_enabled:
        raise ValueError("--compare-candidate-checks already enables context in both arms")
    if sum((compare_instructions, compare_called_functions, compare_module_context)) > 1:
        raise ValueError("select one comparison: instructions, called functions or module context")
    if compare_review_opening and (
        compare_instructions or compare_called_functions or compare_module_context
    ):
        raise ValueError("--compare-review-opening cannot be combined with another comparison")
    if compare_review_opening and supplemental_context_enabled:
        raise ValueError(
            "--compare-review-opening already enables supplemental context in both arms"
        )
    if supplemental_context_enabled and not (
        compare_instructions or compare_called_functions or compare_module_context
    ):
        raise ValueError("--supplemental-context requires --compare-instructions")
    cases = load_cases(include_path_cases=include_path_cases)
    if case_ids:
        cases_by_id = {case.id: case for case in cases}
        if len(set(case_ids)) != len(case_ids) or set(case_ids) - cases_by_id.keys():
            raise ValueError("select distinct known review case IDs")
        cases = tuple(cases_by_id[case_id] for case_id in case_ids)
    if not 1 <= reps <= 10 or not 1 <= calls_per_review <= 256:
        raise ValueError("use 1–10 repetitions and 1–256 calls per workflow review")
    if not 1 <= analyze_concurrency <= ANALYZE_MAX_CONCURRENCY:
        raise ValueError(f"analyze concurrency must be 1–{ANALYZE_MAX_CONCURRENCY}")
    validate_available_models(target.host, (target.model,))
    profile, reasoning, contract = resolve_host_identity(
        target.host, reasoning=False, models=(target.model,)
    )
    if compare_candidate_checks:
        arms = CANDIDATE_CHECK_ARMS
    elif compare_review_opening:
        arms = REVIEW_OPENING_ARMS
    elif compare_module_context:
        arms = MODULE_CONTEXT_ARMS
    elif compare_called_functions:
        arms = CALLED_FUNCTION_ARMS
    elif compare_instructions:
        arms = INSTRUCTION_ARMS
    else:
        arms = ARMS
    rows, schedule = [], []
    for index, case in enumerate(cases):
        fixture = EvalFixture.model_validate_json((ROOT / case.fixture_path).read_bytes())
        if fixture.llm_responses or fixture.analyze_responses_by_path is not None:
            raise ValueError("comparison source must not contain scripted answers")
        request = direct_review.render_request(fixture, model=target.model, review_id=uuid4())
        rows.append(
            {
                "id": case.id,
                "fixture": case.fixture_path,
                "fixture_sha256": _sha(ROOT / case.fixture_path),
                "provenance_source": case.provenance_source,
                "provenance_family": case.provenance_family,
                "oracle_manifest": str(Path(case.fixture_path).parent.parent / "manifest.json"),
                "capture_receipt": case.capture_manifest_path,
                "changed_files": [file.path for file in fixture.files],
                "captured_repository_files": sorted(fixture.repository_contents_head),
                "direct_user_sha256": hashlib.sha256(request.user_prompt.encode()).hexdigest(),
                "direct_user_bytes": len(request.user_prompt.encode()),
            }
        )
        if include_path_cases:
            rows[-1]["exposure"] = case.exposure
        for rep in range(reps):
            offset = (index + rep) % len(arms)
            schedule.append(
                {"case_id": case.id, "rep": rep, "order": list(arms[offset:] + arms[:offset])}
            )
    reviews = len(cases) * reps
    inputs = input_hashes(tuple(Scenario(case.fixture_path, (), "review") for case in cases))
    inputs.update(dataset_inputs(include_path_cases=include_path_cases))
    inputs["alembic.ini"] = _sha(ROOT / "alembic.ini")
    inputs.update(
        {
            str(path.relative_to(ROOT)): _sha(path)
            for path in (ROOT / "db/migrations").rglob("*")
            if path.is_file() and path.suffix in {".py", ".mako"}
        }
    )
    if (ROOT / "tests/__init__.py").is_file():
        inputs["tests/__init__.py"] = _sha(ROOT / "tests/__init__.py")
    plan = {
        "schema": SCHEMA,
        "mode": "free-plan",
        "model_calls_made": 0,
        "target": asdict(target),
        "profile_id": profile,
        "reasoning_enabled": reasoning,
        "profile_contract_digest": contract,
        "pricing_version": PRICING_VERSION,
        "dataset": {"id": DATASET_VERSION, "exposure": "development-exposed", "held_out": False},
        "cases": rows,
        "repetitions": reps,
        "schedule": schedule,
        "workflow_call_cap_per_review": calls_per_review,
        "workflow_analyze_concurrency": analyze_concurrency,
        "latency_configuration": "production concurrency"
        if analyze_concurrency == ANALYZE_MAX_CONCURRENCY
        else "reduced concurrency for quality collection; not production latency",
        "workflow_default_review_budget_tokens": DEFAULT_REVIEW_BUDGET_TOKENS,
        "call_upper_bounds": {
            arms[0]: reviews * calls_per_review,
            arms[1]: reviews * calls_per_review,
            "direct-one-shot": reviews,
            "total": reviews * (2 * calls_per_review + 1),
        },
        "call_bound_note": "Upper bounds: triage and trace change the actual workflow call count.",
        "workflow_settings": {
            name: {
                "version": module.VERSION,
                "max_tokens": module.MAX_TOKENS,
                "temperature": module.TEMPERATURE,
            }
            for name, module in (
                ("triage", triage),
                ("analyze", analyze),
                ("trace", trace),
                ("synthesize", synthesize),
            )
        },
        "direct_prompt": direct_review.prompt_identity(),
        "direct_settings": {
            "max_tokens": direct_review.MAX_TOKENS,
            "temperature": direct_review.TEMPERATURE,
            "cache_control": True,
        },
        "per_call_reservation_upper_usd": str(reservation_usd(target, 8192)),
        "concurrent_reservation_upper_usd_per_workflow_arm": str(
            analyze_concurrency * reservation_usd(target, 8192)
        ),
        "budget_policy": (
            "USD caps split equally, rounded down; direct gets one call per review; remaining "
            "call capacity splits equally between workflow arms. Unused caps never transfer."
        ),
        "privacy": (
            "Model inputs contain PR title and captured file data; other PR metadata stays local."
        ),
        "source_access": "Identical frozen captures only; no live repository fallback.",
        "scope": (
            "Existing intake, triage, analysis, trace and synthesis through automated draft/HITL; "
            "patch generation disabled, human approval not resumed, publication captured locally."
        ),
        "quality": "Independent semantic adjudication pending; no automatic recall or precision.",
        "order": "Rotate first arm by case index and repetition; cache state uncontrolled.",
        "first_triage_identity_recipe": (
            "SHA-256 of model_dump JSON with text opt-in, excluding only metadata review_id; "
            "json.dumps(sort_keys=True) default separators/ASCII escaping. Prompt text unchanged."
        ),
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
        "inputs": inputs,
    }
    if include_path_cases:
        from .review_cases import EXPANDED_DATASET_VERSION

        plan["dataset"].update(
            id=EXPANDED_DATASET_VERSION,
            families=sorted({case.provenance_family for case in cases}),
            note=(
                "Historical development captures from the expanded catalog; "
                "selected families are listed; not held out."
            ),
        )
        plan["include_path_cases"] = True
    if (
        compare_instructions
        or compare_called_functions
        or compare_module_context
        or compare_review_opening
        or compare_candidate_checks
    ):
        if compare_candidate_checks:
            mode = "candidate-checks"
        elif compare_review_opening:
            mode = "review-opening"
        elif compare_module_context:
            mode = "module-context"
        elif compare_called_functions:
            mode = "called-functions"
        else:
            mode = "instructions"
        plan.update(
            comparison_mode=mode,
            arms=list(arms),
            instruction_control={
                "current": variant_identity(
                    "concise-control"
                    if compare_called_functions
                    or compare_module_context
                    or compare_review_opening
                    or compare_candidate_checks
                    else "outrider"
                ),
                "candidate": discovery_identity()["arms"]["simple-instructions"]
                if compare_review_opening
                else variant_identity("concise-control"),
                "eligible_requests": (
                    "Clean pass-0 analysis only: analyze node, production template version "
                    "and exact production system prefix. Degraded and post-trace analysis "
                    "retain their original instructions."
                ),
                "supplemental_context_enabled": supplemental_context_enabled
                or compare_called_functions
                or compare_module_context
                or compare_review_opening
                or compare_candidate_checks,
                "fixed_inputs": (
                    "User context, response schema, model/settings and native admission. "
                    "Triage, trace, synthesis and direct review are unchanged."
                ),
                "planner_limit": (
                    "Original production prompt estimates and file selection remain in use; "
                    "the instruction replacement occurs after planning, before transport. "
                    "This is not the budget behavior of a production prompt replacement."
                ),
                "result_cache": (
                    "The default isolated single-run eval driver injects no analysis-result "
                    "cache store. Provider prompt-cache state remains uncontrolled."
                ),
            },
        )
        if (
            supplemental_context_enabled
            or compare_called_functions
            or compare_module_context
            or compare_review_opening
            or compare_candidate_checks
        ):
            plan["instruction_control"]["context_setting_note"] = (
                "Both instruction arms use the production context supplement. "
                "The explicit same-model role override and isolated cache behavior remain; "
                "this is not full production-configuration qualification."
            )
    if compare_called_functions:
        plan["instruction_control"]["fixed_inputs"] = (
            "Both workflow arms use the same concise clean-pass-0 instructions, model/settings, "
            "schema, admission and base context supplement. Only bounded captured called-function "
            "references are enabled in the candidate. Base file allocations are funded first; "
            "helpers use only spare pool and per-file headroom. Later optional references, "
            "responses and routing may differ."
        )
        plan["instruction_control"]["planner_limit"] = (
            "Both concise arms still use the production prompt overhead and planning algorithm. "
            "Base file allocations are preserved; exact helper text uses remaining headroom. "
            "This does not measure the budget behavior of adopting a concise production prompt."
        )
        plan["called_function_control"] = {
            "enabled_by_arm": {arms[0]: False, arms[1]: True},
            "scope": "Evaluation-only pass-0 references from already captured changed files.",
            "source_selection": "Bounded name heuristic; no resolved receiver or proof claim.",
            "allocation_policy": "baseline-first-called-function-headroom-v1",
            "budget": (
                "Reserve base file allocations first. Add exact rendered helper text only within "
                "unused pool and per-file headroom; otherwise retain the base context unchanged."
            ),
            "defaults": "Production, other comparison modes and post-trace behavior are unchanged.",
        }
    if compare_module_context or compare_review_opening or compare_candidate_checks:
        from outrider.agent.nodes.analyze_context import (
            MAX_MODULE_CONTEXT_BYTES,
            MODULE_ALLOCATION_POLICY,
        )
        from outrider.ast_facts.review_context import MAX_MODULE_SOURCE_BYTES

        plan["instruction_control"]["fixed_inputs"] = (
            "Both workflow arms use the same concise clean-pass-0 instructions, model/settings, "
            "schema, admission and base context supplement. Only complete bounded captured-module "
            "references are enabled in the candidate. Within each review, base file allocations "
            "are reserved first; optional modules use remaining headroom. Later responses and "
            "routing may differ."
        )
        plan["instruction_control"]["planner_limit"] = (
            "Both concise arms retain production planning overhead. Module references cannot "
            "reduce the files or primary scopes funded by that review's base plan. Sampled "
            "triage can still differ across arms. This does not qualify a concise production "
            "prompt or production model mix."
        )
        plan["module_context_control"] = {
            "enabled_by_arm": {
                arms[0]: compare_review_opening or compare_candidate_checks,
                arms[1]: True,
            },
            "scope": "Evaluation-only pass-0 references from captured changed files.",
            "source_selection": (
                "Complete small primary module and at most one captured Python module connected "
                "by an explicit import or importer; no runtime receiver-resolution claim."
            ),
            "max_source_bytes_per_module": MAX_MODULE_SOURCE_BYTES,
            "max_modules": 2,
            "max_rendered_supplement_bytes": MAX_MODULE_CONTEXT_BYTES,
            "allocation_policy": MODULE_ALLOCATION_POLICY,
            "budget": (
                "Reserve base file allocations first. Add exact rendered module cost only "
                "within unused pool and per-file headroom; otherwise retain base source and "
                "record the omission."
            ),
            "causality": (
                "Identical first-triage requests do not imply identical triage responses. "
                "Use saved-request contrasts to isolate context; these runs measure the workflow."
            ),
            "release_gate": (
                "Preserve independently verified direct discoveries and prior known checks; "
                "adjudicate all warnings, measure total cost/time and omissions. Development "
                "captures alone cannot establish general parity or authorize release."
            ),
            "defaults": "Production, other modes and post-trace behavior remain unchanged.",
        }
        if compare_review_opening:
            plan["module_context_control"]["causality"] = (
                "Identical first-triage requests do not imply identical sampled responses. "
                "Saved-request diagnostics isolate the opening; these full-workflow runs "
                "measure later routing, admitted findings and total cost/time."
            )
            plan["instruction_control"]["fixed_inputs"] = (
                "Both workflow arms enable the same base context supplement and complete bounded "
                "captured-module references under the same baseline-first allocation policy. "
                "Only the clean pass-0 review opening and its template version differ; model, "
                "settings, wire schema, category guidance and native admission stay fixed. "
                "Later responses and routing may differ."
            )
            plan["instruction_control"]["planner_limit"] = (
                "Both workflow arms retain original production planning overhead. The opening "
                "replacement occurs after planning, before transport. Base file allocations are "
                "reserved before optional module references. Sampled triage can still differ "
                "across arms; this does not qualify a production prompt replacement."
            )
    if compare_candidate_checks:
        from outrider.agent.candidate_checks import (
            CANDIDATE_CHECK_CALL_LIMIT,
            MAX_CONTEXT_BYTES,
            MAX_SNIPPET_BYTES,
            MAX_SNIPPETS,
            VERSION,
        )

        plan["instruction_control"]["fixed_inputs"] = (
            "Both workflow arms use identical concise clean-pass-0 instructions, bounded module "
            "references, baseline-first allocation, model/settings and native admission. Both "
            "retain native proposal references. Only the candidate enables bounded follow-up "
            "checks using captured PR changes and HEAD references; later routing and human "
            "holds may differ."
        )
        plan["candidate_check_control"] = {
            "enabled_by_arm": {arms[0]: False, arms[1]: True},
            "retention_enabled_by_arm": {arms[0]: True, arms[1]: True},
            "maximum_checks_per_review": CANDIDATE_CHECK_CALL_LIMIT,
            "maximum_snippets": MAX_SNIPPETS,
            "maximum_snippet_bytes": MAX_SNIPPET_BYTES,
            "maximum_rendered_context_bytes": MAX_CONTEXT_BYTES,
            "prompt_version": VERSION,
            "call_budget": "Check calls consume the existing workflow call and USD caps.",
            "source": (
                "Same-review captured BASE/HEAD changes and bounded HEAD dependencies; "
                "no additional repository fetch."
            ),
            "result": (
                "Provisional supports/refutes/uncertain plus nullable canonical suggestions. "
                "Original findings remain unchanged; disputes require human review. Unknown "
                "categories remain separate unclassified candidates."
            ),
            "privacy": "Content stays in expiring retained exchanges; metadata alone enters state.",
            "quality": "Scripted checks cover plumbing; live semantic quality remains unproven.",
            "defaults": "Investigation remains opt-in; no production model or policy promotion.",
        }
    return plan


class _TeePersister:
    def __init__(self, recorder: ExchangeRecorder, database: LLMExchangePersister | None):
        self.recorder, self.database = recorder, database

    async def persist(self, event, request, response) -> None:
        current = _ACTIVE_RECORDER.get()
        await current.persist(event, request, response)
        self.recorder.exchanges.append(current.exchanges[-1])
        if self.database is not None:
            await self.database.persist(event, request, response)

    @property
    def target(self) -> Target:
        return self.recorder.target


class _ReviewProvider:
    def __init__(
        self,
        inner: LLMProvider,
        recorder: ExchangeRecorder,
        budget: Budget,
        inputs: dict[str, str],
        *,
        arm: str,
        slot: str,
        call_cap: int,
        first_requests: dict,
    ):
        self.inner, self.recorder, self.arm = inner, recorder, arm
        self.budget, self.inputs = budget, inputs
        self.slot, self.call_cap, self.first_requests = slot, call_cap, first_requests
        self.calls = 0
        self.attempts: list[dict[str, Any]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if (
            self.arm
            in {
                "concise-workflow",
                "concise-with-helpers",
                "concise-with-modules",
                "concise-with-checks",
                "alternative-with-modules",
            }
            and request.node_id == "analyze"
            and not request.degraded_mode
            and request.prompt_template_version == analyze.VERSION
            and request.system_prompt == analyze.SYSTEM_PROMPT_STABLE_PREFIX
        ):
            request = request_for_variant(request, variant_identity("concise-control"))
            if self.arm == "alternative-with-modules":
                request = requests_for_discovery(request)["simple-instructions"]
        if self.calls >= self.call_cap:
            raise BudgetLimitError("per-review call upper bound reached before generation")
        if request.node_id == "triage" and self.arm != "direct-one-shot":
            identity = _request_identity(request)
            pairs = self.first_requests.setdefault(self.slot, {})
            if pairs and identity not in pairs.values():
                raise ValueError("workflow arms have different first-triage inputs or settings")
            pairs[self.arm] = identity
        current = ExchangeRecorder(
            self.recorder.directory,
            self.recorder.target,
            scenario=f"{self.slot}:{request.node_id}:{self.calls}",
        )
        self.calls += 1
        budgeted = BudgetedProvider(
            self.inner, self.recorder.target, current, self.budget, self.inputs
        )
        token = _ACTIVE_RECORDER.set(current)
        try:
            return await budgeted.complete(request)
        finally:
            self.attempts.extend(budgeted.attempts)
            _ACTIVE_RECORDER.reset(token)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _archive_inputs(inputs: dict[str, str], directory: Path) -> None:
    for name, digest in inputs.items():
        source = (ROOT / name).resolve()
        if Path(name).is_absolute() or ".." in Path(name).parts or not source.is_relative_to(ROOT):
            raise ValueError("generation input escapes its repository")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"generation source changed before provider construction: {name}")
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)


def budget_caps(
    plan: dict[str, Any], *, maximum_usd: Decimal | None, maximum_calls: int | None
) -> dict[str, Any]:
    arms = _comparison_arms(plan)
    direct_calls = plan["call_upper_bounds"]["direct-one-shot"]
    workflow_calls = None if maximum_calls is None else (maximum_calls - direct_calls) // 2
    if workflow_calls is not None and workflow_calls < 1:
        raise ValueError("call cap must fund direct reviews and both workflow arms")
    usd_per_arm = (
        None
        if maximum_usd is None
        else (maximum_usd / 3).quantize(Decimal("0.00000001"), rounding=ROUND_FLOOR)
    )
    if usd_per_arm is not None and (not usd_per_arm.is_finite() or usd_per_arm <= 0):
        raise ValueError("each arm requires a positive finite USD cap")
    return {
        "total_maximum_usd": None if maximum_usd is None else str(maximum_usd),
        "total_maximum_calls": maximum_calls,
        "maximum_usd_per_arm": None if usd_per_arm is None else str(usd_per_arm),
        "maximum_calls_per_arm": {
            arms[0]: workflow_calls,
            arms[1]: workflow_calls,
            "direct-one-shot": direct_calls,
        },
        "unused_caps_transferred": False,
    }


async def collect(
    plan: dict[str, Any],
    *,
    maximum_usd: Decimal,
    maximum_calls: int,
    output: Path,
    provider_factory=None,
    workflow_runner=run_review,
) -> dict[str, Any]:
    arms = _comparison_arms(plan)
    if workflow_runner is run_review:
        require_eval_mode()
        if not os.environ.get("TEST_DATABASE_URL"):
            raise ValueError("workflow collection requires isolated TEST_DATABASE_URL")
        assert_test_url_is_isolated(os.environ["TEST_DATABASE_URL"])
    caps = budget_caps(plan, maximum_usd=maximum_usd, maximum_calls=maximum_calls)
    usd_per_arm = Decimal(caps["maximum_usd_per_arm"])
    direct_calls = plan["call_upper_bounds"]["direct-one-shot"]
    workflow_calls = caps["maximum_calls_per_arm"][arms[0]]
    budgets = {
        arm: Budget(usd_per_arm, direct_calls if arm == "direct-one-shot" else workflow_calls)
        for arm in arms
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    _archive_inputs(plan["inputs"], output / "generation")
    target = Target(**plan["target"])
    factory = create_provider if provider_factory is None else provider_factory
    recorders = {}
    for arm in arms:
        directory = output / arm / "calls"
        directory.mkdir(parents=True)
        recorders[arm] = ExchangeRecorder(directory, target)
    first_requests, results = {}, []
    cases = {case["id"]: case for case in plan["cases"]}
    config = ModelConfig(
        **dict.fromkeys(
            (
                "triage_model",
                "analyze_model",
                "standard_analyze_model",
                "trace_model",
                "synthesize_model",
                "patch_model",
            ),
            target.model,
        )
    )
    prerequisite_failed = False
    for scheduled in plan["schedule"]:
        case = cases[scheduled["case_id"]]
        fixture = EvalFixture.model_validate_json((ROOT / case["fixture"]).read_bytes())
        slot = f"{scheduled['rep']}:{case['id']}"
        for arm in scheduled["order"]:
            recorder = recorders[arm]
            before = len(recorder.exchanges)
            row = {
                "arm": arm,
                "case_id": case["id"],
                "rep": scheduled["rep"],
                "status": "prepared",
                "semantic_adjudication": "pending",
                "diagnostic_events": [],
            }
            if arm != "direct-one-shot":
                row["finding_format"] = "native-analyze"
            results.append(row)
            provider = None

            def make_provider(database=None, *, recorder=recorder, arm=arm, slot=slot):
                nonlocal provider
                native = factory(target, _TeePersister(recorder, database))
                provider = _ReviewProvider(
                    native,
                    recorder,
                    budgets[arm],
                    plan["inputs"],
                    arm=arm,
                    slot=slot,
                    call_cap=1
                    if arm == "direct-one-shot"
                    else plan["workflow_call_cap_per_review"],
                    first_requests=first_requests,
                )
                return provider

            started = time.perf_counter()
            try:
                if arm == "direct-one-shot":
                    provider = make_provider()
                    request = direct_review.render_request(
                        fixture, model=target.model, review_id=uuid4()
                    )
                    response = await provider.complete(request)
                    row["direct_assessment"] = direct_review.assess_response(response.text, fixture)
                    row["status"] = (
                        "completed"
                        if row["direct_assessment"]["status"] == "parsed"
                        else "invalid-response"
                    )
                else:
                    context_enabled = arm == "context-on"
                    if plan.get("comparison_mode") == "instructions":
                        context_enabled = plan.get("instruction_control", {}).get(
                            "supplemental_context_enabled", False
                        )
                    called_functions = plan.get("comparison_mode") == "called-functions"
                    review_opening = plan.get("comparison_mode") == "review-opening"
                    module_context = plan.get("comparison_mode") in {
                        "module-context",
                        "review-opening",
                        "candidate-checks",
                    }
                    candidate_checks = plan.get("comparison_mode") == "candidate-checks"
                    if called_functions or module_context:
                        context_enabled = True
                    module_options = (
                        {"module_context_enabled": True}
                        if module_context
                        and (arm == "concise-with-modules" or review_opening or candidate_checks)
                        else {}
                    )
                    check_options = (
                        {"candidate_checks_enabled": arm == "concise-with-checks"}
                        if candidate_checks
                        else {}
                    )
                    result = await asyncio.to_thread(
                        workflow_runner,
                        ROOT / case["fixture"],
                        model_config=config,
                        host=target.host,
                        analyze_max_concurrency=plan.get(
                            "workflow_analyze_concurrency", ANALYZE_MAX_CONCURRENCY
                        ),
                        provider_factory=make_provider,
                        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
                        supplemental_context_enabled=context_enabled,
                        called_function_context_enabled=called_functions
                        and arm == "concise-with-helpers",
                        **module_options,
                        **check_options,
                        diagnostic_events=row["diagnostic_events"],
                    )
                    row.update(
                        status="completed",
                        admitted_findings=[
                            finding.model_dump(mode="json") for finding in result.findings
                        ],
                        hitl_gated=result.hitl_gated,
                        automated_review_wall_ms=result.automated_review_wall_ms,
                        database_setup_wall_ms=result.database_setup_wall_ms,
                        trace_decisions=[
                            item.model_dump(mode="json") for item in result.trace_decisions
                        ],
                    )
                    if candidate_checks:
                        row.update(
                            native_candidates=[
                                item.model_dump(mode="json") for item in result.native_candidates
                            ],
                            candidate_assessments=[
                                item.model_dump(mode="json")
                                for item in result.candidate_assessments
                            ],
                        )
            except Exception as exc:
                row.update(
                    status="budget-stopped-before-call"
                    if isinstance(exc, BudgetLimitError)
                    else "failed",
                    error_type=type(exc).__name__,
                )
            finally:
                if arm == "direct-one-shot" and provider is not None:
                    try:
                        await provider.aclose()
                    except Exception as exc:
                        row["close_error_type"] = type(exc).__name__
                row["whole_driver_wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
                row["calls"] = [] if provider is None else provider.attempts
                row["diagnostics_status"] = (
                    "not-applicable"
                    if arm == "direct-one-shot"
                    else "incomplete"
                    if any(
                        event.get("kind") == "diagnostic-export-error"
                        for event in row["diagnostic_events"]
                    )
                    else "captured"
                    if row["diagnostic_events"]
                    else "unavailable"
                )
                row["raw_exchange_files"] = [
                    str(
                        (
                            recorder.directory / f"{exchange['call_index']:06d}.exchange.json"
                        ).relative_to(output)
                    )
                    for exchange in recorder.exchanges[before:]
                ]
            if arm != "direct-one-shot" and provider is None and row["status"] == "failed":
                row["status"] = "prerequisite-failed"
                prerequisite_failed = True
                break
        if prerequisite_failed:
            break
    report = {
        "schema": SCHEMA,
        "status": "completed"
        if all(
            row["status"] == "completed"
            and row["diagnostics_status"] in {"captured", "not-applicable"}
            for row in results
        )
        else "incomplete",
        "plan": plan,
        "evidence_kind": "live-provider" if factory is create_provider else "injected-provider",
        "cases": results,
        "first_triage_request_pairs": first_requests,
        "budgets": {
            arm: {
                "authorized_maximum_usd": str(budget.maximum_usd),
                "maximum_calls": budget.maximum_calls,
                "calls": budget.calls,
                "known_usage_based_usd": str(budget.known_usd),
                "uncertain_budget_upper_usd": str(budget.uncertain_usd),
            }
            for arm, budget in budgets.items()
        },
        "authorization": {
            "total_maximum_usd": str(maximum_usd),
            "total_maximum_calls": maximum_calls,
            "unallocated_usd": str(maximum_usd - 3 * usd_per_arm),
            "unallocated_calls": maximum_calls - direct_calls - 2 * workflow_calls,
        },
        "timing_note": (
            "Automated review measures graph invocation through draft/HITL. DB setup is database "
            "creation plus migrations. Whole driver also includes seeding, construction, integrity "
            "checks and cleanup. Direct wall time includes provider construction and persistence."
        ),
        "quality": "Findings and extras need independent adjudication; no winner.",
    }
    try:
        report["cost_comparison"] = summarize_costs(report)
    except (CostInputError, KeyError, TypeError) as exc:
        report["status"] = "incomplete"
        report["cost_comparison"] = {
            "schema": COST_SCHEMA,
            "status": "incomplete",
            "error_type": type(exc).__name__,
            "note": "Cost summary did not reconcile; original calls and budgets are retained.",
        }
    write_json(output / "results.json", report)
    from .review_adjudication import write_packet

    write_packet(report, output)
    receipts = {
        str(path.relative_to(output)): _sha(path)
        for path in output.rglob("*")
        if path.is_file()
        and not path.is_relative_to(output / "generation")
        and path != output / "adjudication/review.json"
    }
    write_json(output / "receipts.json", receipts)
    return report


_REPLAY_DIRECT = """
import hashlib,json,sys
from pathlib import Path
payload=json.load(sys.stdin)
source=Path(payload['source'])
sys.path[:0]=[str(source/'src'),str(source)]
def no_network(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise RuntimeError('offline replay network call')
sys.addaudithook(no_network)
from outrider.agent.eval_driver import EvalFixture
from outrider.llm.base import LLMRequest,INCLUDE_TEXT_OPT_IN
from tests.eval.direct_review import render_request,assess_response
for row in payload['rows']:
    exchange=row['exchange']
    saved=LLMRequest.model_validate(exchange['request'])
    fixture=EvalFixture.model_validate_json((source/row['fixture']).read_bytes())
    regenerated=render_request(fixture,model=saved.model,review_id=saved.review_id)
    if regenerated.model_dump(mode='json',context=INCLUDE_TEXT_OPT_IN)!=exchange['request']:
        raise ValueError('archived direct request differs from its capture')
    if assess_response(exchange['response']['text'],fixture)!=row['assessment']:
        raise ValueError('archived direct response assessment differs')
print(json.dumps({'reproduced_direct_reviews':len(payload['rows']),'model_calls_made':0,
                  'scope':'direct replay only; full workflow regeneration not claimed'}))
"""


def replay_direct(directory: Path) -> dict[str, Any]:
    report = json.loads((directory / "results.json").read_text())
    if report["schema"] != SCHEMA:
        raise ValueError("unsupported comparison report")
    receipts = json.loads((directory / "receipts.json").read_text())
    if not {"results.json", "plan.json"} <= receipts.keys():
        raise ValueError("canonical results and plan must be receipt-covered")
    for name, digest in receipts.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or _sha(path) != digest:
            raise ValueError("saved comparison receipt differs")
    source = directory / "generation"
    if (source / ".env").exists():
        raise ValueError("generation archive must not contain an environment file")
    if report["plan"] != json.loads((directory / "plan.json").read_text()):
        raise ValueError("saved plan differs from the collected result")
    for name, digest in report["plan"]["inputs"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or _sha(path) != digest:
            raise ValueError("archived generation source differs")
    cases = {case["id"]: case for case in report["plan"]["cases"]}
    rows = []
    for case in report["cases"]:
        if case["arm"] == "direct-one-shot" and "direct_assessment" in case:
            if len(case["raw_exchange_files"]) != 1:
                raise ValueError("direct review must have exactly one saved exchange")
            if case["raw_exchange_files"][0] not in receipts:
                raise ValueError("direct exchange must be receipt-covered")
            rows.append(
                {
                    "fixture": cases[case["case_id"]]["fixture"],
                    "exchange": json.loads((directory / case["raw_exchange_files"][0]).read_text()),
                    "assessment": case["direct_assessment"],
                }
            )
    if not rows:
        raise ValueError("no assessed direct responses are available to replay")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("_API_KEY") and key not in {"PYTHONPATH", "ANTHROPIC_AUTH_TOKEN"}
    }
    completed = subprocess.run(  # noqa: S603 — fixed interpreter and script; JSON data stays on stdin
        [sys.executable, "-c", _REPLAY_DIRECT],
        input=json.dumps({"source": str(source.resolve()), "rows": rows}),
        text=True,
        capture_output=True,
        cwd=source,
        env=environment,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        raise ValueError("archived direct replay did not reproduce")
    return json.loads(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--target", type=parse_target)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Select a case; repeat in execution order (default: corpus order).",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--calls-per-review", type=int, default=32)
    parser.add_argument("--analyze-concurrency", type=int, default=ANALYZE_MAX_CONCURRENCY)
    parser.add_argument("--compare-instructions", action="store_true")
    parser.add_argument(
        "--compare-called-functions",
        action="store_true",
        help="Compare the concise workflow with and without bounded captured helper context.",
    )
    parser.add_argument(
        "--compare-module-context",
        action="store_true",
        help="Compare concise review with and without complete bounded connected-file references.",
    )
    parser.add_argument(
        "--compare-review-opening",
        action="store_true",
        help="Compare concise and alternative review openings with complete modules in both arms.",
    )
    parser.add_argument(
        "--compare-candidate-checks",
        action="store_true",
        help="Compare identical module reviews with bounded candidate checks in one arm.",
    )
    parser.add_argument(
        "--include-path-cases",
        action="store_true",
        help="Add three existing path-containment captures; these remain exposed development data.",
    )
    parser.add_argument(
        "--supplemental-context",
        action="store_true",
        help="Include context in both instruction arms; requires --compare-instructions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--i-authorize-spend", action="store_true")
    parser.add_argument("--max-cost-usd", type=Decimal)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--replay-direct", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/review-comparison"))
    args = parser.parse_args(argv)
    if args.replay_direct:
        if (
            args.supplemental_context
            or args.compare_called_functions
            or args.compare_module_context
            or args.compare_review_opening
            or args.compare_candidate_checks
            or args.include_path_cases
        ):
            parser.error(
                "direct replay uses saved context; --supplemental-context needs a new plan"
            )
        if args.target or args.paid or args.i_authorize_spend:
            parser.error("direct replay uses saved source without a target or paid flags")
        print(json.dumps(replay_direct(args.replay_direct), indent=2))
        return 0
    if args.target is None:
        parser.error("select one explicit --target")
    if args.paid != args.i_authorize_spend:
        parser.error("paid collection requires both spend flags")
    if args.max_cost_usd is not None and (
        not args.max_cost_usd.is_finite() or args.max_cost_usd <= 0
    ):
        parser.error("--max-cost-usd must be finite and positive")
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive")
    try:
        plan = build_plan(
            args.target,
            reps=args.reps,
            calls_per_review=args.calls_per_review,
            analyze_concurrency=args.analyze_concurrency,
            case_ids=tuple(args.case),
            compare_instructions=args.compare_instructions,
            supplemental_context_enabled=args.supplemental_context,
            compare_called_functions=args.compare_called_functions,
            compare_module_context=args.compare_module_context,
            compare_review_opening=args.compare_review_opening,
            compare_candidate_checks=args.compare_candidate_checks,
            include_path_cases=args.include_path_cases,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        plan["authorization"] = budget_caps(
            plan, maximum_usd=args.max_cost_usd, maximum_calls=args.max_calls
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in plan.items() if key != "inputs"}, indent=2))
    if not args.paid or args.dry_run:
        return 0
    if args.max_cost_usd is None or not args.max_cost_usd.is_finite() or args.max_cost_usd <= 0:
        parser.error("paid collection requires a finite positive --max-cost-usd")
    if args.max_calls is None or args.max_calls < plan["call_upper_bounds"]["direct-one-shot"] + 2:
        parser.error(
            "paid collection requires --max-calls funding all direct calls and both workflows"
        )
    destination = args.output_dir / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    )
    report = asyncio.run(
        collect(
            plan, maximum_usd=args.max_cost_usd, maximum_calls=args.max_calls, output=destination
        )
    )
    print(json.dumps({"report": str(destination), "budgets": report["budgets"]}, indent=2))
    if "totals" in report["cost_comparison"]:
        print(render_costs(report["cost_comparison"]))
    else:
        print("Cost comparison incomplete; original calls and budgets are saved in the report.")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
