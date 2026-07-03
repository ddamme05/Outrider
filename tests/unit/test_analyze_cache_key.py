# Per specs/2026-06-11-file-hash-analyze-cache.md — cache-key recipe pins.
"""Analyze-cache key composition: determinism, per-component sensitivity,
and boundary-unambiguity. Every input that could change a per-file
analyze outcome must change the key; no pair of distinct input tuples
may collide by shifting bytes across a component boundary.
"""

from __future__ import annotations

import re

import pytest

from outrider.agent.nodes.analyze_observed import OBSERVED_PRODUCER_VERSION
from outrider.agent.nodes.analyze_parser import ANALYZE_PARSER_VERSION
from outrider.ast_facts.parameterized_calls import (
    ExecuteCallSite,
    ParameterizedCallScan,
    scan_digest,
)
from outrider.cache import ANALYZE_CACHE_KEY_VERSION, compute_analyze_cache_key
from outrider.llm.base import _canonical_prompt_hash
from outrider.policy.subsumption import SUBSUMES_DIGEST
from outrider.queries.registry import QUERY_REGISTRY_DIGEST, _registry_digest
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_FORMAT_DIGEST

_BASE_KWARGS = {
    "system_prompt": "system text",
    "user_prompt": "user text",
    "installation_id": 42,
    "repo_id": 7,
    "model": "claude-haiku-4-5",
    "prompt_template_version": "analyze-v4",
    "trivial_filter_version": "trivial-filter-v1",
    "query_registry_digest": "a" * 64,
    "active_policy_version": "policy-v1",
    "analyze_parser_version": ANALYZE_PARSER_VERSION,
    "response_format_digest": ANALYZE_RESPONSE_FORMAT_DIGEST,
    "parameterized_call_scan_digest": "d" * 64,
    "observed_producer_version": OBSERVED_PRODUCER_VERSION,
    "subsumes_digest": SUBSUMES_DIGEST,
    "from_import_map_digest": "9" * 64,
    # Host-identity triad (DECISIONS.md#056): the base case is QUALIFIED
    # (a Baseten-host, reasoning-off run) so the golden recipe exercises the
    # real fold; the unqualified all-None path has its own pin below.
    "profile_id": "baseten",
    "reasoning_enabled": False,
    "profile_contract_digest": "1" * 64,
}


def test_key_is_deterministic_64_hex() -> None:
    first = compute_analyze_cache_key(**_BASE_KWARGS)
    second = compute_analyze_cache_key(**_BASE_KWARGS)
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("system_prompt", "system text CHANGED"),
        ("user_prompt", "user text CHANGED"),
        ("installation_id", 43),
        ("repo_id", 8),
        ("model", "claude-sonnet-4-6"),
        ("prompt_template_version", "analyze-v5"),
        ("trivial_filter_version", "trivial-filter-v2"),
        ("query_registry_digest", "b" * 64),
        ("active_policy_version", "policy-v2"),
        # Derived probe: collision with the live version is impossible by
        # construction, so a future ANALYZE_PARSER_VERSION bump can never
        # silently turn this sensitivity row into a same-value comparison.
        ("analyze_parser_version", ANALYZE_PARSER_VERSION + "-probe"),
        ("response_format_digest", "c" * 64),
        ("parameterized_call_scan_digest", "e" * 64),
        # Derived probe (same rationale as analyze_parser_version): an
        # OBSERVED_PRODUCER_VERSION bump can never collide with this row.
        ("observed_producer_version", OBSERVED_PRODUCER_VERSION + "-probe"),
        # A SUBSUMES relation edit changes the admitted set (DECISIONS.md#055),
        # so its digest must change the key — "f"*64 is a distinct-from-live probe.
        ("subsumes_digest", "f" * 64),
        # A from-import change alters the corrected trace-candidate siblings
        # (#024 from-import amendment) without touching the rendered prompt,
        # so the map digest must change the key — "8"*64 is a probe.
        ("from_import_map_digest", "8" * 64),
        # Host-identity triad (DECISIONS.md#056): a different host, a flipped
        # reasoning state, or a different profile contract is a different output
        # population for identical prompt bytes, so each must change the key.
        ("profile_id", "fireworks"),
        ("reasoning_enabled", True),
        ("profile_contract_digest", "2" * 64),
    ],
)
def test_every_component_changes_the_key(field: str, changed: object) -> None:
    """Each of the eighteen inputs is load-bearing: changing any one of
    them alone produces a different key (the correct-by-construction
    invalidation property the spec pins). `observed_producer_version`
    (Cost Lever 3) pins the deterministic OBSERVED producer's admission
    logic, `subsumes_digest` (DECISIONS.md#055) pins the cross-type
    SUBSUMES relation, `from_import_map_digest` (#024 from-import
    amendment) pins candidate correction's per-file input, and the
    host-identity triad (DECISIONS.md#056) splits the cache by provider
    host, so each change invalidates entries."""
    base = compute_analyze_cache_key(**_BASE_KWARGS)
    varied = compute_analyze_cache_key(**{**_BASE_KWARGS, field: changed})
    assert varied != base, field


def test_prompt_boundary_shift_does_not_collide() -> None:
    """The attacker-relevant collision the canonical recipe exists for:
    (system='AB', user='C') must differ from (system='A', user='BC')."""
    a = compute_analyze_cache_key(**{**_BASE_KWARGS, "system_prompt": "AB", "user_prompt": "C"})
    b = compute_analyze_cache_key(**{**_BASE_KWARGS, "system_prompt": "A", "user_prompt": "BC"})
    assert a != b


def test_adjacent_scalar_boundary_shift_does_not_collide() -> None:
    """Adjacent integer components must not collide by digit-shifting:
    (installation 1, repo 23) vs (installation 12, repo 3)."""
    a = compute_analyze_cache_key(**{**_BASE_KWARGS, "installation_id": 1, "repo_id": 23})
    b = compute_analyze_cache_key(**{**_BASE_KWARGS, "installation_id": 12, "repo_id": 3})
    assert a != b


def test_golden_recipe_eighteen_framed_fields() -> None:
    """Golden pin of the FULL recipe, recomputed independently in the
    test: eighteen length-prefixed fields — `ANALYZE_CACHE_KEY_VERSION`
    (the recipe-structure version, DECISIONS.md#056) first, then
    `_canonical_prompt_hash` output (one recipe, two consumers; never forks
    from `LLMCallEvent.prompt_hash`), then the sixteen explicit
    scope/version/identity components in declaration order, each framed
    `{len(bytes)}:` on UTF-8 bytes. The host-identity triad
    (DECISIONS.md#056) is the last three of those sixteen: `profile_id`, then
    `reasoning_enabled` rendered `true`/`false`, then `profile_contract_digest`.
    Any change to the framing, the component order, or the prompt component's
    recipe fails this test — deliberately: that change is a cache-wide
    invalidation and must bump `ANALYZE_CACHE_KEY_VERSION` and be made here too."""
    import hashlib

    expected = hashlib.sha256()
    for component in (
        ANALYZE_CACHE_KEY_VERSION,
        _canonical_prompt_hash(
            system_prompt=_BASE_KWARGS["system_prompt"],
            user_prompt=_BASE_KWARGS["user_prompt"],
        ),
        str(_BASE_KWARGS["installation_id"]),
        str(_BASE_KWARGS["repo_id"]),
        _BASE_KWARGS["model"],
        _BASE_KWARGS["prompt_template_version"],
        _BASE_KWARGS["trivial_filter_version"],
        _BASE_KWARGS["query_registry_digest"],
        _BASE_KWARGS["active_policy_version"],
        _BASE_KWARGS["analyze_parser_version"],
        _BASE_KWARGS["response_format_digest"],
        _BASE_KWARGS["parameterized_call_scan_digest"],
        _BASE_KWARGS["observed_producer_version"],
        _BASE_KWARGS["subsumes_digest"],
        _BASE_KWARGS["from_import_map_digest"],
        _BASE_KWARGS["profile_id"],
        "true" if _BASE_KWARGS["reasoning_enabled"] else "false",
        _BASE_KWARGS["profile_contract_digest"],
    ):
        component_bytes = component.encode("utf-8")
        expected.update(f"{len(component_bytes)}:".encode())
        expected.update(component_bytes)

    assert compute_analyze_cache_key(**_BASE_KWARGS) == expected.hexdigest()


def test_unqualified_triad_is_stable_and_distinct_from_qualified() -> None:
    """An UNQUALIFIED (pre-#056) caller passes the whole triad as None. The
    three components fold as empty strings — deterministic and stable across
    calls — and the resulting key differs from the qualified base, so an
    unqualified row never collides with a real host's row. The empty fold is
    also distinct from a host whose `profile_id` happened to be empty, which
    cannot occur (`host_id` is non-empty), so no real host aliases here."""
    unqualified = {
        **_BASE_KWARGS,
        "profile_id": None,
        "reasoning_enabled": None,
        "profile_contract_digest": None,
    }
    first = compute_analyze_cache_key(**unqualified)
    second = compute_analyze_cache_key(**unqualified)
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert first != compute_analyze_cache_key(**_BASE_KWARGS)


@pytest.mark.parametrize(
    ("profile_id", "reasoning_enabled", "profile_contract_digest"),
    [
        ("baseten", None, None),
        (None, False, None),
        (None, None, "1" * 64),
        ("baseten", False, None),
        ("baseten", None, "1" * 64),
        (None, False, "1" * 64),
    ],
)
def test_partial_triad_raises(
    profile_id: str | None,
    reasoning_enabled: bool | None,
    profile_contract_digest: str | None,
) -> None:
    """The triad are peers (DECISIONS.md#056): all-present or all-None. A
    partial triad is incoherent — no valid #056 audit event can represent it,
    so the cache helper rejects it rather than minting a key no event matches.
    `build_graph` rejects partials upstream; this keeps the all-or-none
    invariant total at the exported cache boundary too (the gap Codex flagged)."""
    partial = {
        **_BASE_KWARGS,
        "profile_id": profile_id,
        "reasoning_enabled": reasoning_enabled,
        "profile_contract_digest": profile_contract_digest,
    }
    with pytest.raises(ValueError, match="all-present or all-None"):
        compute_analyze_cache_key(**partial)


def test_parameterized_call_scan_digest_closes_fup_171() -> None:
    """FUP-171 end-to-end: the veto's per-file outcome is now keyed. Two
    reviews with byte-identical prompts (and every other component equal)
    but a different parameterized-call scan — e.g. a syntax error in an
    out-of-scope region empties the scan and disables the veto, while a
    clean parse populates it — produce different scan digests and therefore
    different cache keys, so they can never share an entry."""
    veto_off = scan_digest(ParameterizedCallScan())  # empty: veto disabled
    veto_on = scan_digest(
        ParameterizedCallScan(
            safe_parameterized_calls=(ExecuteCallSite(line_start=10, line_end=12),),
            all_execute_like_calls=(ExecuteCallSite(line_start=10, line_end=12),),
        )
    )
    assert veto_off != veto_on
    key_off = compute_analyze_cache_key(
        **{**_BASE_KWARGS, "parameterized_call_scan_digest": veto_off}
    )
    key_on = compute_analyze_cache_key(
        **{**_BASE_KWARGS, "parameterized_call_scan_digest": veto_on}
    )
    assert key_off != key_on


# ---------------------------------------------------------------------------
# The two new version constants
# ---------------------------------------------------------------------------


def test_analyze_parser_version_pinned() -> None:
    """Bump rule: ANY change to the admitted-findings OR trace-candidate
    semantics bumps this (the spec's TRIVIAL_FILTER_VERSION precedent).
    v2: the FUP-162 parameterized-call veto joined the admission flow.
    v3: prefer-OBSERVED (DECISIONS.md#054) evicts a JUDGED proposal
    colliding with an OBSERVED finding. v4: cross-type subsumption
    (DECISIONS.md#055) drops an admitted OBSERVED finding under a
    same-span JUDGED subsumer — again changing what a cache row may
    serve. v5: from-import candidate correction rewrites a hallucinated
    module prefix on a trace candidate to the analyzed file's actual
    importing module — changing the trace_candidates a cache row stores.
    v6: multi-language dispatch — `collect_trace_candidates=False`
    suppresses collection for files whose language has no import
    resolver, changing what lands in `trace_candidates` for them.
    v7: relative-specifier admission — JS/TS files collect leading-dot
    specifier-form candidates (contained at admission), changing
    `trace_candidates` for them from always-empty to populated."""
    assert ANALYZE_PARSER_VERSION == "analyze-parser-v7"


def test_analyze_cache_key_version_pinned() -> None:
    """The recipe-structure version (DECISIONS.md#056), distinct from
    ANALYZE_PARSER_VERSION (admitted-findings semantics): bump on ANY change to
    the cache-key component set, order, or framing. v1 was the implicit original
    recipe (no constant); v2 is the host-identity re-key (#056 folded the triad
    in); v3 folds `from_import_map_digest` (#024 from-import amendment —
    corrected trace-candidate siblings depend on module-level imports the
    rendered prompt doesn't carry). It folds FIRST in
    `compute_analyze_cache_key`, so a bump re-keys the whole cache — the
    explicit, replay-durable marker #056 mandates."""
    assert ANALYZE_CACHE_KEY_VERSION == "analyze-cache-key-v3"


def test_query_registry_digest_is_stable_64_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", QUERY_REGISTRY_DIGEST)
    # Module-load pinned: recomputing over the same bodies + OBSERVED
    # metadata is identical (the second arg folds class/finding_type/title/
    # description per the Cost Lever 3 round-3 review).
    from outrider.queries.registry import (
        _OBSERVED_QUERIES,
        _QUERY_BODIES,
        VALUE_PREDICATES,
    )

    assert (
        _registry_digest(_QUERY_BODIES, _OBSERVED_QUERIES, VALUE_PREDICATES)
        == QUERY_REGISTRY_DIGEST
    )


def test_query_registry_digest_changes_with_body_semantics() -> None:
    """A pattern edit that keeps its id changes the digest — the
    FUP-166 property that makes cached OBSERVED findings safe. (Synthetic
    bodies with no OBSERVED metadata, so the second arg is empty.)"""
    base = _registry_digest({"python.x": "(call) @c"}, {}, {})
    edited = _registry_digest({"python.x": "(call function: (identifier)) @c"}, {}, {})
    renamed = _registry_digest({"python.y": "(call) @c"}, {}, {})
    assert base != edited
    assert base != renamed


def test_query_registry_digest_pair_boundaries_unambiguous() -> None:
    """Length-prefixing: ({'ab': 'c'}) must differ from ({'a': 'bc'})."""
    assert _registry_digest({"ab": "c"}, {}, {}) != _registry_digest({"a": "bc"}, {}, {})
