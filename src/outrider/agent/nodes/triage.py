# Triage node body per specs/2026-05-15-triage-node.md.
"""Triage node body: classify changed files into review tiers via Haiku.

Consumes a size-OK seed `ReviewState` (with `PRContext.changed_files`
populated by intake; spec assumes upstream §6.10 size-cap gate has run)
and produces a frozen `TriageResult` via a Haiku LLM pass. The node
classifies each changed file as DEEP/STANDARD/SKIM (never SKIP — that's
the policy-gate path), assesses overall_risk, and selects applicable
ReviewDimensions.

Runtime dependencies are closure-injected per `nodes-receive-deps-via-closure`:
- `provider: LLMProvider` — calls Haiku; the provider's internal persister
  writes `LLMCallEvent` + `llm_call_content` rows in one transaction.
- `triage_model: str` — the per-tier Haiku id from `ModelConfig.triage_model`
  (NOT a hardcoded model string — pins `model-strings-from-config-not-hardcoded`).
- `phase_event_sink: PhaseEventSink` — emits `ReviewPhaseEvent` start/end
  pairs per `phase-events-bound-work`. Required (no Optional, no production
  no-op default).

Failure-path semantics: the end-event is skipped if ANY of four post-start
steps raises — request construction (`pydantic.ValidationError`), provider
call (`LLMProviderError` subclasses), schema validation (`pydantic.ValidationError`
on `response.text`), or policy validation (`TriagePolicyViolationError`).
The prompt-render step is intentionally pure / non-raising (see
`prompts/triage.py`).

The post-schema policy gate (`_enforce_triage_policy`) is the deterministic
floor: `TriageResult.model_validate_json(response.text)` validates SHAPE
(enum casing, max_length, types) but admits SKIP values and arbitrary path
keys. The policy gate rejects schema-valid-but-policy-invalid output:
(a) any SKIP value, (b) `file_tiers` keys not in `state.pr_context.changed_files`,
(c) any changed-file path missing from `file_tiers`.

`TriagePolicyViolationError` is the typed exception so callers can
distinguish policy failure from transport failure (`LLMProviderError`)
and schema failure (`pydantic.ValidationError`).
"""

from collections.abc import Set
from uuid import uuid4

from outrider.agent.state import ReviewState
from outrider.audit.events import ReviewPhaseEvent
from outrider.audit.sinks import PhaseEventSink
from outrider.llm.base import LLMProvider, LLMRequest
from outrider.prompts import triage as triage_prompt
from outrider.schemas.triage_result import ReviewTier, TriageResult


class TriagePolicyViolationError(ValueError):
    """LLM produced schema-valid TriageResult that violates deterministic floor.

    Raised when (a) file_tiers contains a `ReviewTier.SKIP` value (SKIP is
    the policy-gate path per the triage-node spec's non-goal #1, never
    node output), (b) file_tiers has a key not in the changed-files-under-
    review set, or (c) the set of keys does not cover every changed file
    under review.

    The wrapper's internal persister has already written `LLMCallEvent` +
    content rows by the time this exception fires (`provider.complete()`
    returned successfully); the caller decides whether to retry or fail
    the review.
    """


def _enforce_triage_policy(
    result: TriageResult,
    *,
    expected_paths: Set[str],
) -> None:
    """Deterministic post-schema gate. Raises TriagePolicyViolationError on violation.

    `expected_paths` is the abstract `Set` (from `collections.abc`) so
    callers can pass `set`, `frozenset`, or `dict_keys` view without
    needing to wrap. Set arithmetic below works identically across all.
    """
    if any(tier is ReviewTier.SKIP for tier in result.file_tiers.values()):
        raise TriagePolicyViolationError(
            "LLM produced SKIP; SKIP is policy-gate scope, not node output"
        )
    actual_paths = frozenset(result.file_tiers.keys())
    extra = actual_paths - expected_paths
    if extra:
        raise TriagePolicyViolationError(f"file_tiers contains unknown paths: {sorted(extra)!r}")
    missing = expected_paths - actual_paths
    if missing:
        raise TriagePolicyViolationError(
            f"file_tiers missing paths from changed_files: {sorted(missing)!r}"
        )


async def triage(
    state: ReviewState,
    *,
    provider: LLMProvider,
    triage_model: str,
    phase_event_sink: PhaseEventSink,
) -> dict[str, TriageResult]:
    """Run the triage classification pass.

    Returns `{"triage_result": <validated, policy-checked TriageResult>}`
    for LangGraph's reducer to merge into state. Default reducer is
    overwrite — appropriate here because `triage_result` is a singleton
    field per the schema-foundation spec.

    Order of operations (failure-path-significant; see module docstring):
      1. Emit start phase event.
      2. Render prompts (pure, non-raising).
      3. Build LLMRequest (raises ValidationError on malformed fields).
      4. Call provider.complete() (raises LLMProviderError subclasses on transport failure).
      5. Validate response shape via TriageResult.model_validate_json (raises ValidationError).
      6. Enforce policy gate (raises TriagePolicyViolationError).
      7. Emit end phase event.
      8. Return partial state.
    """
    phase_id = str(uuid4())

    # Step 1: emit start phase event. If THIS raises (audit infra outage),
    # the node fails before any work; no dangling start.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="triage",
            marker="start",
            phase_key=None,  # V1 single-instance; V1.5 parallel-analyze populates this
        )
    )

    # Step 2: render prompts (pure, non-raising — see prompts.triage docstring).
    parts = triage_prompt.render(state.pr_context)

    # Step 3: build LLMRequest. Field validators run; ValidationError surfaces
    # if any value violates the request schema (e.g., empty prompt).
    request = LLMRequest(
        model=triage_model,
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
        max_tokens=triage_prompt.MAX_TOKENS,
        temperature=triage_prompt.TEMPERATURE,
        review_id=state.review_id,
        node_id="triage",
        prompt_template_version=triage_prompt.VERSION,
        degraded_mode=False,  # triage has no degraded path in V1
        # cache_control defaults to True per DECISIONS#013 point 4
        # is_eval defaults to False; eval-harness factories set explicitly
        # context_summary defaults to ()
    )

    # Step 4: provider call. The provider's internal persister writes
    # LLMCallEvent + llm_call_content rows BEFORE returning, per the
    # LLMProvider Protocol contract. LLMProviderError subclasses propagate.
    response = await provider.complete(request)

    # Step 5: schema validation. ValidationError on malformed JSON,
    # missing required keys, wrong enum casing, or reasoning >500 chars.
    triage_result = TriageResult.model_validate_json(response.text)

    # Step 5b: deterministic policy gate. Catches schema-valid output that
    # violates this node's contract (SKIP, unknown path, missing path).
    _enforce_triage_policy(
        triage_result,
        expected_paths={cf.path for cf in state.pr_context.changed_files},
    )

    # Step 6: success-exit phase event. Same phase_id as the start above.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="triage",
            marker="end",
            phase_key=None,
        )
    )

    # Step 7: partial-state return. LangGraph's default reducer (overwrite)
    # merges triage_result into ReviewState; pr_context survives unchanged.
    return {"triage_result": triage_result}


__all__ = [
    "TriagePolicyViolationError",
    "triage",
]
