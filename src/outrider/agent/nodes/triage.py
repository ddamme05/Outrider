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

    Log-content note: the exception messages embed file paths via
    `f"{paths!r}"` — for rule (b) those paths are LLM-controlled (a
    fabricated path not in changed_files). Python's `repr` escapes
    control chars to `\\xNN` notation, so ANSI / terminal-injection via
    control sequences is mitigated. Per DECISIONS#013 point 5
    ("Logs never contain prompt or completion content"), the policy-gate
    paths are technically LLM completion content — but the leak is
    bounded to filenames (not file CONTENTS) and `repr` neutralizes
    active exfiltration. Callers logging this exception should consider
    that paths in the message can be attacker-influenced.
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
    # Rule (a): no SKIP values.
    skip_paths = sorted(path for path, tier in result.file_tiers.items() if tier is ReviewTier.SKIP)
    if skip_paths:
        raise TriagePolicyViolationError(
            f"LLM produced SKIP tier for paths {skip_paths!r}; SKIP is the "
            "policy-gate scope path (per the triage-node spec's non-goal #1) "
            "and is never produced by this node. The deterministic §6.10 "
            "size-cap gate upstream of triage is the only producer of SKIP; "
            "an LLM-emitted SKIP is either a hallucination or a sign the "
            "system prompt's 'never produce skip' rule needs reinforcement."
        )

    # Rule (b): no unknown paths.
    actual_paths = frozenset(result.file_tiers.keys())
    extra = actual_paths - expected_paths
    if extra:
        raise TriagePolicyViolationError(
            f"file_tiers contains unknown paths {sorted(extra)!r} that are "
            f"not in changed_files (expected: {sorted(expected_paths)!r}). "
            "The triage node MUST tier exactly the changed-files set — no "
            "more, no less. An unknown path indicates either an LLM "
            "hallucination of a file that doesn't exist in this PR, or a "
            "drift between intake's changed_files population and the "
            "downstream node's expected set. Either is a correctness gap "
            "the deterministic floor catches before the analyze step can "
            "consume the bad triage_result."
        )

    # Rule (c): no missing paths.
    missing = expected_paths - actual_paths
    if missing:
        raise TriagePolicyViolationError(
            f"file_tiers is missing paths {sorted(missing)!r} from changed_files "
            f"(expected: {sorted(expected_paths)!r}). Every changed file "
            "under review MUST receive a tier — DEEP, STANDARD, or SKIM. A "
            "missing path means the downstream analyze node has no "
            "instruction for that file: silent drop is the failure mode "
            "the policy-gate exists to prevent."
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
    # `is_eval` flows from ReviewState.is_eval so eval-tagged reviews
    # produce eval-tagged phase events (parity with LLMCallEvent at step 4).
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="triage",
            marker="start",
            is_eval=state.is_eval,
            phase_key=None,  # V1 single-instance; V1.5 parallel-analyze populates this
        )
    )

    # Step 2: render prompts (pure, non-raising — see prompts.triage docstring).
    parts = triage_prompt.render(state.pr_context)

    # Step 3: build LLMRequest. Field validators run; ValidationError surfaces
    # if any value violates the request schema (e.g., empty prompt).
    # `is_eval` flows from ReviewState.is_eval so audit rows produced during
    # eval runs are correctly tagged per docs/testing.md "Eval isolation".
    request = LLMRequest(
        model=triage_model,
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
        max_tokens=triage_prompt.MAX_TOKENS,
        temperature=triage_prompt.TEMPERATURE,
        review_id=state.review_id,
        node_id="triage",
        is_eval=state.is_eval,
        prompt_template_version=triage_prompt.VERSION,
        degraded_mode=False,  # triage has no degraded path in V1
        # cache_control defaults to True per DECISIONS#013 point 4
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

    # Step 6: success-exit phase event. Same phase_id as the start above;
    # same is_eval flag for audit-isolation parity.
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

    # Step 7: partial-state return. LangGraph's default reducer (overwrite)
    # merges triage_result into ReviewState; pr_context survives unchanged.
    return {"triage_result": triage_result}


__all__ = [
    "TriagePolicyViolationError",
    "triage",
]
