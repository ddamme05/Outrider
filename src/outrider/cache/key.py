# Per specs/2026-06-11-file-hash-analyze-cache.md — the analyze-cache key.
# Per DECISIONS.md#056: the host-identity triad (profile_id, reasoning_enabled,
# profile_contract_digest) joins the key, splitting the analyze cache by host.
"""Compose analyze-cache identity from the prompt, scope, rules, and admission facts.

Use the same canonical prompt hash as LLMCallEvent.prompt_hash. Explicit inputs
pin rule versions and admission context absent from prompt bytes; installation
and repository IDs partition the key by tenant and repository.

Callers supply version values so this module need not import tree-sitter-backed
version providers. Each component uses its UTF-8 byte length as a prefix, so
attacker-influenced strings cannot shift bytes across component boundaries.
"""

from __future__ import annotations

import hashlib
from typing import Final

from outrider.llm.base import _canonical_prompt_hash

# Recipe version: bump on ANY recipe change, including component additions,
# removals, reordering, or changes to length-prefix framing (DECISIONS.md#056).
# This explicit marker is separate from parser/producer admission versions.
# Keep it first in the recipe so a bump re-keys the cache.
#
# Earlier recipes added host identity (v2, #056), from-import correction inputs
# (v3, #024), and import-binding facts (v4). v5 added lexical visibility spans
# and widened the import digest to include the value-import marker.
# v6 adds module admission (#062): head-side added ranges, the bytes they cover,
# and every parsed scope span. These inputs can live outside rendered scopes;
# identical prompts alone cannot authorize reuse of their admitted findings.
ANALYZE_CACHE_KEY_VERSION: Final = "analyze-cache-key-v6"


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
    response_format_digest: str,
    parameterized_call_scan_digest: str,
    observed_producer_version: str,
    subsumes_digest: str,
    from_import_map_digest: str,
    import_bindings_digest: str,
    lexical_bindings_digest: str,
    module_admission_digest: str,
    profile_id: str | None,
    reasoning_enabled: bool | None,
    profile_contract_digest: str | None,
) -> str:
    """Hash the ordered recipe fields using UTF-8 byte-length prefixes.

    Version leads, then the canonical prompt hash, scope/rule inputs, and host
    identity. Preserve this order and framing; any recipe change requires an
    ANALYZE_CACHE_KEY_VERSION bump. Use caller-threaded write-time versions,
    including the policy version stamped on findings, rather than live constants.

    Admission inputs can change without changing the prompt:
    - parameterized_call_scan_digest pins whole-file SQL-veto facts, including
      the empty scan produced by any syntax error;
    - from_import_map_digest pins corrected trace-candidate siblings;
    - import_bindings_digest pins all imports' module, value marker, and names;
    - lexical_bindings_digest pins local names and their visibility spans;
    - module_admission_digest pins added ranges, their bytes, and all scope spans.
    Producer/parser versions pin admission rules; subsumes_digest pins the
    cross-type relation that drops OBSERVED findings under same-span JUDGED ones.

    response_format_digest separates constrained and free-form output populations.
    Analyze passes the real digest; a fixed sentinel such as "none" is valid only
    for callers with no format concept. Host, reasoning mode, and profile contract
    likewise split output populations. Supply all three host fields or all None;
    None folds to empty components, while reasoning False/True folds to false/true.
    """
    # Match the graph/event all-or-none rule at this direct-call boundary too (#056).
    host_identity = (profile_id, reasoning_enabled, profile_contract_digest)
    if any(identity_field is not None for identity_field in host_identity) and not all(
        identity_field is not None for identity_field in host_identity
    ):
        raise ValueError(
            "host-identity triad (DECISIONS.md#056) is peers — all-present or all-None; "
            f"got a partial set: profile_id={profile_id!r}, "
            f"reasoning_enabled={reasoning_enabled!r}, "
            f"profile_contract_digest={'<set>' if profile_contract_digest is not None else None}"
        )
    prompt_digest = _canonical_prompt_hash(system_prompt=system_prompt, user_prompt=user_prompt)
    # Preserve legacy None separately from an explicit reasoning False or True.
    reasoning_component = (
        "" if reasoning_enabled is None else ("true" if reasoning_enabled else "false")
    )
    key_hasher = hashlib.sha256()
    for component in (
        ANALYZE_CACHE_KEY_VERSION,
        prompt_digest,
        str(installation_id),
        str(repo_id),
        model,
        prompt_template_version,
        trivial_filter_version,
        query_registry_digest,
        active_policy_version,
        analyze_parser_version,
        response_format_digest,
        parameterized_call_scan_digest,
        observed_producer_version,
        subsumes_digest,
        from_import_map_digest,
        import_bindings_digest,
        lexical_bindings_digest,
        module_admission_digest,
        profile_id if profile_id is not None else "",
        reasoning_component,
        profile_contract_digest if profile_contract_digest is not None else "",
    ):
        component_bytes = component.encode("utf-8")
        key_hasher.update(f"{len(component_bytes)}:".encode())
        key_hasher.update(component_bytes)
    return key_hasher.hexdigest()
