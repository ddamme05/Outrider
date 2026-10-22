"""Classify changed files and assess the pull request's overall risk.

The model supplies a TriageResult. Deterministic validation requires every
changed-file path, rejects SKIP, and anchors the active policy version.
The provider records its call before returning; failed work returns no
state update and can leave the start phase open.
"""

import hashlib
import json
from collections.abc import Iterable, Set

from outrider.agent.state import ReviewState
from outrider.audit.events import ReviewPhaseEvent
from outrider.audit.sinks import PhaseEventSink
from outrider.llm.base import LLMProvider, LLMRequest
from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy.canonical import compute_phase_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import triage as triage_prompt
from outrider.schemas.triage_result import ReviewTier, TriageResult

_POLICY_VIOLATION_SAMPLE_SIZE = 3
"""Bound path samples in policy errors; hash the full sorted set for correlation."""

_REASONING_MAX_CHARS = 500
"""Match TriageResult.reasoning's schema cap; advisory reasoning can be truncated."""


def _validate_triage_response(raw_json: str) -> TriageResult:
    """Clamp advisory reasoning to its schema cap, then validate the response.

    Let Pydantic report malformed JSON; normalize only the reasoning length.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return TriageResult.model_validate_json(raw_json)
    if isinstance(data, dict) and isinstance(data.get("reasoning"), str):
        data["reasoning"] = data["reasoning"][:_REASONING_MAX_CHARS]
    return TriageResult.model_validate(data)


class TriagePolicyViolationError(ValueError):
    """Schema-valid model output violated the deterministic triage policy.

    Reject SKIP, unknown or missing paths, and an inactive policy_version.
    Messages use bounded, escaped path samples and a hash of the full set.
    In the node flow, the provider has persisted its call before this gate runs.
    """


def _format_path_set_for_error(paths: Iterable[str]) -> str:
    """Describe violating paths without echoing an unbounded path list.

    Return the count, a 12-character SHA-256 prefix over the NUL-separated
    sorted paths, and an escaped sample of at most three paths.
    See DECISIONS.md#013 and #016 for the log-content constraint.
    """
    sorted_paths = sorted(paths)
    digest = hashlib.sha256("\x00".join(sorted_paths).encode("utf-8")).hexdigest()[:12]
    sample = sorted_paths[:_POLICY_VIOLATION_SAMPLE_SIZE]
    return f"count={len(sorted_paths)} hash={digest} sample={sample!r}"


def _enforce_triage_policy(
    result: TriageResult,
    *,
    expected_paths: Set[str],
) -> None:
    """Require the active policy and one non-SKIP tier for every expected path."""
    skip_paths: list[str] = []
    for candidate_path, candidate_tier in result.file_tiers.items():
        if candidate_tier is ReviewTier.SKIP:
            skip_paths.append(candidate_path)
    if skip_paths:
        raise TriagePolicyViolationError(
            "LLM produced SKIP tier for one or more paths "
            f"({_format_path_set_for_error(skip_paths)}); SKIP is the "
            "policy-gate scope path (per the triage-node spec's non-goal #1) "
            "and is never produced by this node. The deterministic §6.10 "
            "size-cap gate upstream of triage is the only producer of SKIP; "
            "an LLM-emitted SKIP is either a hallucination or a sign the "
            "system prompt's 'never produce skip' rule needs reinforcement."
        )

    actual_paths = frozenset(result.file_tiers.keys())
    unknown_paths = actual_paths - expected_paths
    if unknown_paths:
        raise TriagePolicyViolationError(
            "file_tiers contains unknown paths "
            f"({_format_path_set_for_error(unknown_paths)}) not in changed_files "
            f"(expected: {_format_path_set_for_error(expected_paths)}). "
            "The triage node MUST tier exactly the changed-files set — no "
            "more, no less. An unknown path indicates either an LLM "
            "hallucination of a file that doesn't exist in this PR, or a "
            "drift between intake's changed_files population and the "
            "downstream node's expected set. Either is a correctness gap "
            "the deterministic floor catches before the analyze step can "
            "consume the bad triage_result."
        )

    missing_paths = expected_paths - actual_paths
    if missing_paths:
        raise TriagePolicyViolationError(
            "file_tiers is missing paths from changed_files "
            f"({_format_path_set_for_error(missing_paths)}; expected: "
            f"{_format_path_set_for_error(expected_paths)}). Every "
            "changed file under review MUST receive a tier — DEEP, "
            "STANDARD, or SKIM. A missing path means the downstream "
            "analyze node has no instruction for that file: silent drop "
            "is the failure mode the policy-gate exists to prevent."
        )

    # Downstream finding validation trusts this snapshot (DECISIONS.md#028).
    # The model must not select a different policy version.
    if result.policy_version != ACTIVE_POLICY_VERSION:
        raise TriagePolicyViolationError(
            f"triage_result.policy_version={result.policy_version!r} "
            f"does not match the live ACTIVE_POLICY_VERSION="
            f"{ACTIVE_POLICY_VERSION!r}. The field is the snapshot "
            "anchor for synthesize's H-1 forge defense; an LLM-emitted "
            "value (training-data leakage, prompt injection, schema "
            "doc-leak) would survive the BARE_SEMVER_PATTERN floor "
            "and let a downstream forged finding match the poisoned "
            "anchor. At fresh-review time the value MUST equal live "
            "ACTIVE. Replay reconstruction reads audit rows directly "
            "and does NOT re-execute this gate; the gate fires only "
            "on the live-review path."
        )


async def triage(
    state: ReviewState,
    *,
    provider: LLMProvider,
    triage_model: str,
    phase_event_sink: PhaseEventSink,
) -> dict[str, TriageResult]:
    """Classify files, validate the model result, and return a triage state update.

    Emit the end marker only after model and policy validation succeed.
    Return no state update if request, provider, validation, or audit work fails.
    """
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="triage",
        attempt_key="triage",
    )

    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="triage",
            marker="start",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    prompt_parts = triage_prompt.render(state.pr_context)

    request = LLMRequest(
        model=triage_model,
        system_prompt=prompt_parts.system_prompt,
        user_prompt=prompt_parts.user_prompt,
        max_tokens=triage_prompt.MAX_TOKENS,
        temperature=triage_prompt.TEMPERATURE,
        review_id=state.review_id,
        node_id="triage",
        is_eval=state.is_eval,
        prompt_template_version=triage_prompt.VERSION,
        degraded_mode=False,
    )

    response = await provider.complete(request)

    # Normalize vendor output before schema and policy validation.
    triage_result = _validate_triage_response(strip_outer_json_fence(response.text))

    _enforce_triage_policy(
        triage_result,
        expected_paths={cf.path for cf in state.pr_context.changed_files},
    )

    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="triage",
            marker="end",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    return {"triage_result": triage_result}


__all__ = [
    "TriagePolicyViolationError",
    "triage",
]
