"""LLMRequest constraint + validator tests.

Covers all field-level constraints + the two cross-field validators (the
V1-messages-unset rejection and the analyze/synthesize empty-context
rejection). Per AC#20.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import ContextManifestEntry
from outrider.llm.base import LLMMessage, LLMRequest


def _entry() -> ContextManifestEntry:
    return ContextManifestEntry(
        file_path="src/foo.py",
        scope_unit_name="Foo.bar",
        line_start=1,
        line_end=10,
        inclusion_reason="changed_scope",
    )


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "system_prompt": "You are a triage classifier.",
        "user_prompt": "Classify this PR.",
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "temperature": 0.0,
        "review_id": uuid4(),
        "node_id": "triage",
        "prompt_template_version": "triage@1.0.0",
        "degraded_mode": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Transport-field constraints.
# ---------------------------------------------------------------------------


def test_system_prompt_min_length_one() -> None:
    """Empty system_prompt is a node-side bug (template render returned ''):
    fail at construction, not at the SDK billing surface."""
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(system_prompt=""))


def test_user_prompt_min_length_one() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(user_prompt=""))


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(max_tokens=0))
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(max_tokens=-1))


def test_max_tokens_upper_bound() -> None:
    """8192 is the V1 cap (cost-cliff guard)."""
    LLMRequest(**_kwargs(max_tokens=8192))  # admits at the boundary
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(max_tokens=8193))


def test_temperature_bounds() -> None:
    LLMRequest(**_kwargs(temperature=0.0))
    LLMRequest(**_kwargs(temperature=1.0))
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(temperature=-0.01))
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(temperature=1.01))


def test_temperature_required_no_default() -> None:
    """temperature has no default; required per spec (round 11 sharp-edges H3
    — None / SDK default would be 1.0, defeating replay determinism)."""
    kwargs = _kwargs()
    del kwargs["temperature"]
    with pytest.raises(ValidationError):
        LLMRequest(**kwargs)


# ---------------------------------------------------------------------------
# Audit-context fields.
# ---------------------------------------------------------------------------


def test_review_id_required() -> None:
    kwargs = _kwargs()
    del kwargs["review_id"]
    with pytest.raises(ValidationError):
        LLMRequest(**kwargs)


def test_review_id_must_be_uuid() -> None:
    """Pydantic auto-coerces UUID strings; arbitrary strings rejected."""
    LLMRequest(**_kwargs(review_id=str(uuid4())))  # admits a UUID string
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(review_id="not-a-uuid"))


@pytest.mark.parametrize("node_id", ["triage", "analyze", "synthesize", "trace"])
def test_node_id_admits_canonical_values(node_id: str) -> None:
    # analyze needs non-empty context_summary; provide one for analyze only
    extra: dict[str, object] = {"node_id": node_id}
    if node_id == "analyze":
        extra["context_summary"] = (_entry(),)
    req = LLMRequest(**_kwargs(**extra))
    assert req.node_id == node_id


def test_node_id_rejects_typo() -> None:
    """`"analyse"` (British spelling) is a real-world typo that
    `node_id: str` would have admitted; `Literal[...]` rejects."""
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(node_id="analyse"))


def test_node_id_rejects_arbitrary_string() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(node_id="custom_node"))


def test_prompt_template_version_min_length_one() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(prompt_template_version=""))


def test_prompt_template_version_admits_canonical_shapes() -> None:
    """Spec §8.3 allows `analyze@1.0.0` (Git-SHA-style) and
    `analyze_v2.1.3` (semver-style) — strict shape validation lives in
    the prompts/ registry, not at the wrapper schema layer."""
    LLMRequest(**_kwargs(prompt_template_version="analyze@1.0.0"))
    LLMRequest(**_kwargs(prompt_template_version="analyze_v2.1.3"))
    LLMRequest(**_kwargs(prompt_template_version="v1"))
    LLMRequest(**_kwargs(prompt_template_version="v1.2.3"))
    LLMRequest(**_kwargs(prompt_template_version="abc123def456"))  # Git SHA shape


def test_degraded_mode_required_no_default() -> None:
    """`degraded_mode` has no default — caller must explicitly pass.
    Silent default `False` would poison the audit row when ast_facts
    actually returned a degraded parse (round 11 sharp-edges C3)."""
    kwargs = _kwargs()
    del kwargs["degraded_mode"]
    with pytest.raises(ValidationError):
        LLMRequest(**kwargs)


def test_is_eval_defaults_false() -> None:
    req = LLMRequest(**_kwargs())
    assert req.is_eval is False


def test_context_summary_defaults_empty_tuple() -> None:
    req = LLMRequest(**_kwargs())
    assert req.context_summary == ()
    assert isinstance(req.context_summary, tuple)


def test_context_summary_admits_tuple_of_entries() -> None:
    """Two distinct entries (different scope_unit_name) admit; the
    set-semantic dup check fires only on (file_path, scope_unit_name)
    collisions."""
    e1 = _entry()
    e2 = ContextManifestEntry(
        file_path=e1.file_path,
        scope_unit_name=f"{e1.scope_unit_name}.nested",
        line_start=e1.line_start + 100,
        line_end=e1.line_end + 100,
        inclusion_reason="same_file_context",
    )
    entries = (e1, e2)
    req = LLMRequest(**_kwargs(context_summary=entries, node_id="analyze"))
    assert req.context_summary == entries


def test_context_summary_coerces_list_to_tuple() -> None:
    """Pydantic V2 auto-coerces list → tuple for `tuple[X, ...]`
    annotations. The frozen-tuple immutability guarantee is preserved on
    the stored field regardless of input shape — what matters is that
    `request.context_summary` is a tuple after construction (not a live
    list reference)."""
    req = LLMRequest(**_kwargs(context_summary=[_entry()], node_id="analyze"))
    assert isinstance(req.context_summary, tuple)
    assert len(req.context_summary) == 1


# ---------------------------------------------------------------------------
# Cross-field validators.
# ---------------------------------------------------------------------------


def test_messages_must_be_none_in_v1() -> None:
    """V1 reserves `messages` for V1.5+; non-None rejected at construction."""
    with pytest.raises(ValidationError, match="reserved for V1.5"):
        LLMRequest(**_kwargs(messages=[LLMMessage(role="user", content="hi")]))


def test_messages_default_none_admits() -> None:
    req = LLMRequest(**_kwargs())
    assert req.messages is None


def test_analyze_with_empty_context_summary_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty context_summary"):
        LLMRequest(**_kwargs(node_id="analyze", context_summary=()))


def test_synthesize_admits_empty_context_summary() -> None:
    """Synthesize legitimately calls without a scope manifest — it
    aggregates findings already produced by analyze, NOT per-file
    scope context. Synthesize is correctly excluded from the
    `_enforce_context_for_scope_nodes` allowlist (analyze is the
    only node that walks per-file scope).
    """
    req = LLMRequest(**_kwargs(node_id="synthesize", context_summary=()))
    assert req.context_summary == ()


def test_triage_admits_empty_context_summary() -> None:
    """Triage legitimately calls without a scope manifest — the
    cross-field rule only fires for analyze."""
    req = LLMRequest(**_kwargs(node_id="triage", context_summary=()))
    assert req.context_summary == ()


def test_trace_admits_empty_context_summary() -> None:
    """Trace also calls without a scope manifest in some flows."""
    req = LLMRequest(**_kwargs(node_id="trace", context_summary=()))
    assert req.context_summary == ()


def test_analyze_with_non_empty_context_summary_admits() -> None:
    req = LLMRequest(**_kwargs(node_id="analyze", context_summary=(_entry(),)))
    assert req.node_id == "analyze"
    assert len(req.context_summary) == 1


# ---------------------------------------------------------------------------
# Smoke test: well-formed construction across all four node_ids.
# ---------------------------------------------------------------------------


def test_all_node_ids_construct_with_appropriate_context() -> None:
    for node_id in ("triage", "trace"):
        req = LLMRequest(**_kwargs(node_id=node_id))
        assert req.node_id == node_id
    for node_id in ("analyze", "synthesize"):
        req = LLMRequest(**_kwargs(node_id=node_id, context_summary=(_entry(),)))
        assert req.node_id == node_id
        assert len(req.context_summary) == 1


# ---------------------------------------------------------------------------
# §0b: degradation_reason + provenance validator.
# Pins the 13-row truth table from specs/2026-05-19-analyze-foundation.md §0b.
# Three-way coupling: analyze-only scoping AND bidirectional bool/reason
# coupling AND degraded-analyze admits empty context_summary.
# ---------------------------------------------------------------------------


def test_degradation_reason_defaults_none() -> None:
    """New `degradation_reason` field defaults to None — backward compatible
    with existing callers that don't set degraded_mode."""
    req = LLMRequest(**_kwargs())
    assert req.degradation_reason is None


@pytest.mark.parametrize(
    "reason",
    [
        "parse_failed",
        "tree_has_error_in_changed_regions",
        "tree_has_error_no_scope",
        "module_level_observed_match",
    ],
)
def test_analyze_degraded_with_typed_reason_admits_empty_context(reason: str) -> None:
    """Matrix row: analyze + degraded_mode=True + reason + empty context → ADMIT.

    The §0b core path: degraded analyze with a documented provenance reason
    is the only way to bypass the context-required validator.
    """
    req = LLMRequest(
        **_kwargs(
            node_id="analyze",
            degraded_mode=True,
            degradation_reason=reason,
            context_summary=(),
        )
    )
    assert req.degraded_mode is True
    assert req.degradation_reason == reason
    assert req.context_summary == ()


def test_analyze_degraded_without_reason_raises_provenance() -> None:
    """Matrix row: analyze + degraded_mode=True + reason=None + empty → REJECT.

    Naked degraded_mode is the silent context-validator bypass §0b closes.
    """
    with pytest.raises(ValidationError, match="degraded_mode=True requires degradation_reason"):
        LLMRequest(
            **_kwargs(
                node_id="analyze",
                degraded_mode=True,
                degradation_reason=None,
                context_summary=(),
            )
        )


def test_analyze_reason_without_mode_raises_provenance() -> None:
    """Matrix row: analyze + degraded_mode=False + reason="parse_failed" → REJECT.

    Reason-without-mode is the inverse asymmetry — the bool/reason coupling
    is enforced in both directions so neither flag can drift independently.
    """
    with pytest.raises(ValidationError, match="degradation_reason requires degraded_mode=True"):
        LLMRequest(
            **_kwargs(
                node_id="analyze",
                degraded_mode=False,
                degradation_reason="parse_failed",
                context_summary=(_entry(),),
            )
        )


def test_analyze_non_degraded_with_empty_context_raises_context() -> None:
    """Matrix row: analyze + degraded_mode=False + reason=None + empty → REJECT (context).

    Non-degraded analyze still requires context — degraded_mode is the
    ONLY escape hatch.
    """
    with pytest.raises(ValidationError, match="non-empty context_summary"):
        LLMRequest(
            **_kwargs(
                node_id="analyze",
                degraded_mode=False,
                degradation_reason=None,
                context_summary=(),
            )
        )


def test_analyze_non_degraded_with_context_admits() -> None:
    """Matrix row: analyze + degraded_mode=False + reason=None + non-empty → ADMIT.

    The happy non-degraded analyze path — unchanged from pre-§0b behavior.
    """
    req = LLMRequest(
        **_kwargs(
            node_id="analyze",
            degraded_mode=False,
            degradation_reason=None,
            context_summary=(_entry(),),
        )
    )
    assert req.degraded_mode is False
    assert req.degradation_reason is None


@pytest.mark.parametrize("node_id", ["synthesize", "trace", "triage"])
def test_non_analyze_with_degraded_mode_raises_scoping(node_id: str) -> None:
    """Matrix rows: synthesize/trace/triage + degraded_mode=True → REJECT (scoping).

    Per round-2-post-split audit F3: only analyze has a degraded-mode
    contract in V1. Carrying it elsewhere is silent contract drift.
    """
    extra: dict[str, object] = {
        "node_id": node_id,
        "degraded_mode": True,
        "degradation_reason": "parse_failed",
    }
    if node_id == "synthesize":
        extra["context_summary"] = (_entry(),)
    with pytest.raises(
        ValidationError, match="degraded_mode=True only valid for node_id='analyze'"
    ):
        LLMRequest(**_kwargs(**extra))


@pytest.mark.parametrize("node_id", ["synthesize", "trace", "triage"])
def test_non_analyze_with_reason_only_raises_scoping(node_id: str) -> None:
    """Matrix rows: synthesize/trace/triage + degradation_reason set → REJECT.

    The scoping rule is symmetric: reason-on-non-analyze raises even when
    degraded_mode=False. Without this, a buggy caller could leave the
    field set across requests and silently mark non-analyze rows as
    degraded-related at the audit layer.
    """
    extra: dict[str, object] = {
        "node_id": node_id,
        "degraded_mode": False,
        "degradation_reason": "parse_failed",
    }
    if node_id == "synthesize":
        extra["context_summary"] = (_entry(),)
    with pytest.raises(
        ValidationError, match="degradation_reason is only valid for node_id='analyze'"
    ):
        LLMRequest(**_kwargs(**extra))


def test_synthesize_non_degraded_with_context_admits() -> None:
    """Matrix row: synthesize + degraded_mode=False + reason=None + non-empty → ADMIT.

    Synthesize is unaffected by degraded-mode logic and unchanged from
    pre-§0b behavior.
    """
    req = LLMRequest(
        **_kwargs(
            node_id="synthesize",
            degraded_mode=False,
            degradation_reason=None,
            context_summary=(_entry(),),
        )
    )
    assert req.node_id == "synthesize"


def test_trace_non_degraded_with_empty_context_admits() -> None:
    """Matrix row: trace + degraded_mode=False + reason=None + empty → ADMIT.

    Trace is not in the context-required nodeset; the §0b provenance
    validator's analyze-only scoping also doesn't fire when both
    degraded_mode and reason are unset.
    """
    req = LLMRequest(
        **_kwargs(
            node_id="trace",
            degraded_mode=False,
            degradation_reason=None,
            context_summary=(),
        )
    )
    assert req.node_id == "trace"


def test_degradation_reason_rejects_arbitrary_string() -> None:
    """The Literal is narrow on purpose — new degradation causes require
    explicit Literal expansion. An off-list string is a contract violation
    that V1 must reject at construction.
    """
    with pytest.raises(ValidationError):
        LLMRequest(
            **_kwargs(
                node_id="analyze",
                degraded_mode=True,
                degradation_reason="some_new_reason",
                context_summary=(),
            )
        )


def test_provenance_validator_fires_before_context_on_conflict() -> None:
    """SE-2 audit fold: declaration-order behavioral pin.

    A request with `synthesize + degraded_mode=True` fails the
    provenance validator (only analyze admits degraded_mode). The
    context validator no longer fires for synthesize (synthesize-node
    spec audit dropped it from the allowlist — synthesize doesn't pack
    per-file scope context). But the provenance-first ordering is still
    load-bearing for any future scope-context node that might re-add a
    second axis of failure on the same request. Pydantic V2 runs
    `@model_validator(mode="after")` in declaration order; this test
    surfaces a future refactor that re-orders the validators (auto-sort,
    mixin migration, model_rebuild) by behavioral assertion rather than
    introspection of internal Pydantic state.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            **_kwargs(
                node_id="synthesize",
                degraded_mode=True,
                degradation_reason="parse_failed",
                context_summary=(),
            )
        )
    # Provenance error must surface; context-validator error must not
    # "win" over it. The provenance validator should also raise here
    # because synthesize cannot carry degradation_reason at all.
    error_str = str(exc_info.value)
    assert "only valid for node_id='analyze'" in error_str, (
        f"Expected provenance scoping error to fire first; got: {error_str}"
    )
    # Negative pin: the context validator's prose must NOT appear, because
    # the provenance validator should short-circuit construction first.
    assert "non-empty context_summary" not in error_str, (
        f"Context validator fired despite provenance violation; "
        f"declaration order may have inverted. Got: {error_str}"
    )


def test_llm_request_rejects_duplicate_context_summary_entries() -> None:
    """`context_summary` is set-semantic by `(file_path, scope_unit_name)`.
    Catching at request construction means the failure fires BEFORE the
    paid SDK call rather than after, when the audit-event mirror runs.
    """
    entry = _entry()
    with pytest.raises(ValidationError, match="duplicate"):
        LLMRequest(
            **_kwargs(
                node_id="analyze",
                degraded_mode=False,
                context_summary=(entry, entry),  # same (file_path, scope_unit_name)
            )
        )


def test_llm_request_admits_distinct_context_summary_entries() -> None:
    """Two entries differing in scope_unit_name admit cleanly."""
    e1 = _entry()
    e2 = ContextManifestEntry(
        file_path="src/foo.py",
        scope_unit_name="Foo.baz",  # different scope unit, same file — OK
        line_start=20,
        line_end=30,
        inclusion_reason="same_file_context",
    )
    req = LLMRequest(
        **_kwargs(
            node_id="analyze",
            degraded_mode=False,
            context_summary=(e1, e2),
        )
    )
    assert len(req.context_summary) == 2


# Declaration-order precedence is already covered behaviorally by
# `test_provenance_validator_fires_before_context_on_conflict` above —
# a source-text introspection test would couple to private method names
# and break on harmless renames while runtime behavior is unchanged.


# ---------------------------------------------------------------------------
# Constrained decoding (FUP-096): response_schema_json + derived digest.
# ---------------------------------------------------------------------------


def test_response_schema_json_defaults_none_digest_none() -> None:
    req = LLMRequest(**_kwargs())
    assert req.response_schema_json is None
    assert req.response_format_digest is None


def test_response_format_digest_is_sha256_of_the_string() -> None:
    import hashlib

    req = LLMRequest(**_kwargs(response_schema_json='{"type":"object"}'))
    assert req.response_format_digest == hashlib.sha256(b'{"type":"object"}').hexdigest()


def test_response_format_digest_matches_pinned_constant() -> None:
    """The derivability chain: a request carrying the canonical JSON
    string recomputes exactly the module-level digest, so the audit
    event and the cache key can never disagree about the format."""
    from outrider.schemas.llm.analyze import (
        ANALYZE_RESPONSE_FORMAT_DIGEST,
        ANALYZE_RESPONSE_SCHEMA_JSON,
    )

    req = LLMRequest(**_kwargs(response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON))
    assert req.response_format_digest == ANALYZE_RESPONSE_FORMAT_DIGEST


def test_response_schema_json_rejects_sub_minimal_string() -> None:
    """min_length=2 — the shortest valid JSON-object serialization is
    `{}`; empty/one-char strings are construction bugs, not schemas."""
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(response_schema_json=""))
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(response_schema_json="{"))


def test_response_schema_json_rejects_malformed_json() -> None:
    """A malformed string must fail at construction, not as a raw
    `json.JSONDecodeError` inside the provider's kwargs-building (which
    sits outside the typed SDK error translation)."""
    with pytest.raises(ValidationError, match="must be valid JSON"):
        LLMRequest(**_kwargs(response_schema_json="{not json}"))


@pytest.mark.parametrize("non_object", ['["x"]', '"xx"', "42"])
def test_response_schema_json_rejects_non_object(non_object: str) -> None:
    """`output_config.format` schemas are JSON objects; scalars and
    arrays are construction bugs."""
    with pytest.raises(ValidationError, match="JSON object"):
        LLMRequest(**_kwargs(response_schema_json=non_object))


@pytest.mark.parametrize(
    "non_compact",
    [
        '{"a": 1}',  # whitespace after the colon
        '{"a":1} ',  # trailing whitespace
        '{"a": 1, "b": 2}',  # default-dumps spacing
    ],
)
def test_response_schema_json_rejects_non_compact_form(non_compact: str) -> None:
    """A whitespace-variant string would mint a different
    `response_format_digest` for byte-identical wire intent, fragmenting
    the request-format identity the audit stream and the analyze cache
    key share. Only the compact serialization is admitted."""
    with pytest.raises(ValidationError, match="compact order-preserving form"):
        LLMRequest(**_kwargs(response_schema_json=non_compact))


def test_response_schema_json_admits_any_key_order_compact() -> None:
    """Key order is deliberately NOT normalized: the API emits object
    properties in the schema's defined order, so property order is part
    of the format identity (FUP-169 — a key-sorted schema forced prose
    fields to generate before the model's classification tokens existed).
    Two orders are two formats; both are admitted and carry two digests."""
    reasoning_order = '{"b":1,"a":2}'
    sorted_order = '{"a":2,"b":1}'
    req_reasoning = LLMRequest(**_kwargs(response_schema_json=reasoning_order))
    req_sorted = LLMRequest(**_kwargs(response_schema_json=sorted_order))
    assert req_reasoning.response_schema_json == reasoning_order
    assert req_reasoning.response_format_digest != req_sorted.response_format_digest


def test_phase_key_defaults_none_and_round_trips() -> None:
    """V1.5 phase attribution (`DECISIONS.md#064`): `phase_key` defaults to
    None (every existing constructor unaffected; sequential-era requests carry
    no key) and a worker-stamped key survives construction unchanged — the
    provider mirrors it verbatim onto `LLMCallEvent.phase_key`."""
    assert LLMRequest.model_fields["phase_key"].default is None
    request = LLMRequest(
        **_kwargs(
            node_id="analyze",
            context_summary=(_entry(),),
            phase_key="file:src/app.py#1",
        )
    )
    assert request.phase_key == "file:src/app.py#1"


def test_phase_key_rejected_outside_analyze() -> None:
    """Rule 3 (`DECISIONS.md#064`): phase attribution is analyze-fan-out-only.
    A triage/trace/synthesize request carrying a worker key would create
    ownership for a phase that cannot exist. Mirrored on `LLMCallEvent`."""
    with pytest.raises(ValidationError, match="phase_key is only valid"):
        LLMRequest(**_kwargs(node_id="triage", phase_key="file:src/app.py#0"))
