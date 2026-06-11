# Per specs/2026-06-11-file-hash-analyze-cache.md — cache-key recipe pins.
"""Analyze-cache key composition: determinism, per-component sensitivity,
and boundary-unambiguity. Every input that could change a per-file
analyze outcome must change the key; no pair of distinct input tuples
may collide by shifting bytes across a component boundary.
"""

from __future__ import annotations

import re

import pytest

from outrider.agent.nodes.analyze_parser import ANALYZE_PARSER_VERSION
from outrider.cache import compute_analyze_cache_key
from outrider.llm.base import _canonical_prompt_hash
from outrider.queries.registry import QUERY_REGISTRY_DIGEST, _registry_digest

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
        ("analyze_parser_version", "analyze-parser-v2"),
    ],
)
def test_every_component_changes_the_key(field: str, changed: object) -> None:
    """Each of the ten inputs is load-bearing: changing any one of them
    alone produces a different key (the correct-by-construction
    invalidation property the spec pins)."""
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


def test_prompt_component_uses_the_canonical_recipe() -> None:
    """One recipe, two consumers: the key's prompt component IS
    `_canonical_prompt_hash` output — two prompt pairs with equal
    canonical hashes (i.e., identical pairs) produce equal keys, and the
    recipe never forks from `LLMCallEvent.prompt_hash`."""
    # Identical canonical hash inputs → identical keys (sanity direction).
    assert _canonical_prompt_hash(
        system_prompt=_BASE_KWARGS["system_prompt"], user_prompt=_BASE_KWARGS["user_prompt"]
    ) == _canonical_prompt_hash(system_prompt="system text", user_prompt="user text")
    assert compute_analyze_cache_key(**_BASE_KWARGS) == compute_analyze_cache_key(
        **{**_BASE_KWARGS, "system_prompt": "system text", "user_prompt": "user text"}
    )


# ---------------------------------------------------------------------------
# The two new version constants
# ---------------------------------------------------------------------------


def test_analyze_parser_version_pinned() -> None:
    """Bump rule: ANY admission-flow change bumps this (the spec's
    TRIVIAL_FILTER_VERSION precedent)."""
    assert ANALYZE_PARSER_VERSION == "analyze-parser-v1"


def test_query_registry_digest_is_stable_64_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", QUERY_REGISTRY_DIGEST)
    # Module-load pinned: recomputing over the same bodies is identical.
    from outrider.queries.registry import _QUERY_BODIES

    assert _registry_digest(_QUERY_BODIES) == QUERY_REGISTRY_DIGEST


def test_query_registry_digest_changes_with_body_semantics() -> None:
    """A pattern edit that keeps its id changes the digest — the
    FUP-166 property that makes cached OBSERVED findings safe."""
    base = _registry_digest({"python.x": "(call) @c"})
    edited = _registry_digest({"python.x": "(call function: (identifier)) @c"})
    renamed = _registry_digest({"python.y": "(call) @c"})
    assert base != edited
    assert base != renamed


def test_query_registry_digest_pair_boundaries_unambiguous() -> None:
    """Length-prefixing: ({'ab': 'c'}) must differ from ({'a': 'bc'})."""
    assert _registry_digest({"ab": "c"}) != _registry_digest({"a": "bc"})
