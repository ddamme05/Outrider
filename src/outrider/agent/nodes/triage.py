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

Failure-path semantics: the end-event is skipped (or partially-written) if
ANY of FIVE post-start failure sources raises:

  1. Request construction (`pydantic.ValidationError`)
  2. Provider call (`LLMProviderError` subclasses)
  3. Schema validation (`pydantic.ValidationError` on `response.text`)
  4. Policy validation (`TriagePolicyViolationError`)
  5. End-phase-event emission itself (`PhaseEventSink.emit_phase` raise per
     the Protocol's "Implementations MUST persist before returning, OR
     raise" rule — durable sinks raise on persistence failure)

Sources 1-4 mean the end event is never attempted; the audit stream has
a dangling start. Source 5 means the end event WAS attempted; depending
on the sink's transaction shape the row may have partially landed or
not. In all five cases the function raises and `{"triage_result": ...}`
is never returned, so the partial state cannot reach downstream nodes.

The prompt-render step is intentionally pure / non-raising (see
`prompts/triage.py`) so it does NOT appear in this list.

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

import hashlib
from collections.abc import Iterable, Set
from uuid import uuid4

from outrider.agent.state import ReviewState
from outrider.audit.events import ReviewPhaseEvent
from outrider.audit.sinks import PhaseEventSink
from outrider.llm.base import LLMProvider, LLMRequest
from outrider.prompts import triage as triage_prompt
from outrider.schemas.triage_result import ReviewTier, TriageResult

_POLICY_VIOLATION_SAMPLE_SIZE = 3
"""How many paths from a violating set appear verbatim in error messages.

The full sorted set is hashed (`sha256[:12]`) and the count is reported,
so operators can correlate repeat violations and bound the disclosure
surface. A bounded sample preserves enough signal for debugging without
echoing arbitrary-length LLM/webhook-derived path lists into log
streams. See `_format_path_set_for_error` below for the message shape;
see DECISIONS#013 point 5 + #016 point 4 for the underlying "logs never
contain prompt or completion content" rule.
"""


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

    Log-content note: exception messages do NOT embed verbatim path
    lists. All three rule messages route the violating set through
    `_format_path_set_for_error`, which emits
    `count=N hash=<sha256-12> sample=[first-3 paths, repr-escaped]`.
    The disclosure surface is bounded to `_POLICY_VIOLATION_SAMPLE_SIZE`
    (=3) paths regardless of set size; the SHA-256 hex prefix gives
    operators a stable correlation key for repeat violations without
    echoing content. ANSI / terminal-injection mitigation is preserved
    by `repr()` on the bounded sample list — control chars escape to
    `\\xNN` notation. Per DECISIONS#013 point 5 + #016 point 4
    ("Logs never contain prompt or completion content"), only the
    bounded sample paths can appear in a log capturing this exception's
    message body; the full set is recoverable via the hash for replay
    correlation but not from the message alone. Callers logging this
    exception should still consider that the sample paths may be
    attacker-influenced (LLM hallucination for rule b, webhook payload
    for rule c). FUP-021 closed this gap; see the `_format_path_set_for_error`
    docstring for the full design rationale.
    """


def _format_path_set_for_error(paths: Iterable[str]) -> str:
    """Format a path set for embedding in `TriagePolicyViolationError`.

    Returns `count=N hash=<sha256-12> sample=[...]` where sample is up to
    `_POLICY_VIOLATION_SAMPLE_SIZE` paths formatted via `repr()` (which
    escapes control chars to `\\xNN` notation — same ANSI / terminal-
    injection mitigation the pre-mitigation code relied on). The hash is
    SHA-256 over the NUL-separated sorted full set, first 12 hex chars —
    long enough to be useful for correlating repeat violations, short
    enough to read.

    Bounds the disclosure surface per DECISIONS#013 point 5 + #016
    point 4 ("logs never contain prompt or completion content"). The
    paths come from two attacker-influenced sources: LLM output
    (hallucinated unknown paths from rule b) and webhook payload
    (`PRContext.changed_files` paths surfaced by rule c). A downstream
    `logger.exception(...)` capturing this message would historically
    have echoed the full set; the bounded-sample-plus-hash shape caps
    that leak to `_POLICY_VIOLATION_SAMPLE_SIZE` paths regardless of
    set size, while still giving operators a stable correlation key.
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
    """Deterministic post-schema gate. Raises TriagePolicyViolationError on violation.

    `expected_paths` is the abstract `Set` (from `collections.abc`) so
    callers can pass `set`, `frozenset`, or `dict_keys` view without
    needing to wrap. Set arithmetic below works identically across all.

    Error messages embed bounded path samples plus a stable hash of the
    full violating set — never the verbatim sorted list. See
    `_format_path_set_for_error` for the message shape and the
    DECISIONS#013 / #016 log-content discipline that motivates it.
    """
    # Rule (a): no SKIP values.
    skip_paths = [path for path, tier in result.file_tiers.items() if tier is ReviewTier.SKIP]
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

    # Rule (b): no unknown paths.
    actual_paths = frozenset(result.file_tiers.keys())
    extra = actual_paths - expected_paths
    if extra:
        raise TriagePolicyViolationError(
            "file_tiers contains unknown paths "
            f"({_format_path_set_for_error(extra)}) not in changed_files "
            f"(expected: {_format_path_set_for_error(expected_paths)}). "
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
            "file_tiers is missing paths from changed_files "
            f"({_format_path_set_for_error(missing)}; expected: "
            f"{_format_path_set_for_error(expected_paths)}). Every "
            "changed file under review MUST receive a tier — DEEP, "
            "STANDARD, or SKIM. A missing path means the downstream "
            "analyze node has no instruction for that file: silent drop "
            "is the failure mode the policy-gate exists to prevent."
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
