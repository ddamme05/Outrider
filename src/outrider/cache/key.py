# Per specs/2026-06-11-file-hash-analyze-cache.md — the analyze-cache key.
"""Cache-key composition for the file-hash analyze cache (lever #8).

One pure function: every input that could change a per-file analyze
outcome becomes a length-prefixed component of one SHA-256. The rendered
prompt rides as `llm/base.py::_canonical_prompt_hash` output (one
recipe, two consumers — it must never fork from `LLMCallEvent.prompt_hash`);
the explicit components cover what prompt bytes can't see, and
`(installation_id, repo_id)` is the tenant boundary (installation is the
root — `repo_id` is only unique per installation).

Pure and import-light by design: callers pass the version values
(`TRIVIAL_FILTER_VERSION` loads tree-sitter; this module must not).
Length-prefixing every component makes boundaries unambiguous — adjacent
components can never collide by shifting bytes across a delimiter
(`("AB","C")` vs `("A","BC")`), which matters because several components
derive from attacker-influenced content.
"""

from __future__ import annotations

import hashlib

from outrider.llm.base import _canonical_prompt_hash


def compute_analyze_cache_key(
    *,
    system_prompt: str,
    user_prompt: str,
    installation_id: int,
    repo_id: int,
    model: str,
    prompt_template_version: str,
    trivial_filter_version: str,
    query_registry_digest: str,
    active_policy_version: str,
    analyze_parser_version: str,
) -> str:
    """The eight-component analyze-cache key, as one SHA-256 hex digest.

    Component order is part of the recipe — changing it is a cache-wide
    invalidation and must be deliberate. `active_policy_version` is the
    THREADED write-time value analyze stamps findings with (never the
    module constant read here); same for every version argument.
    """
    prompt_digest = _canonical_prompt_hash(system_prompt=system_prompt, user_prompt=user_prompt)
    h = hashlib.sha256()
    for component in (
        prompt_digest,
        str(installation_id),
        str(repo_id),
        model,
        prompt_template_version,
        trivial_filter_version,
        query_registry_digest,
        active_policy_version,
        analyze_parser_version,
    ):
        component_bytes = component.encode("utf-8")
        h.update(f"{len(component_bytes)}:".encode())
        h.update(component_bytes)
    return h.hexdigest()
