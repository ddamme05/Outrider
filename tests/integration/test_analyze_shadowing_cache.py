# The shadowing guard's cross-subsystem gate (DECISIONS.md#060).
"""Shadowing guard end-to-end against real Postgres: ast_facts extraction →
OBSERVED producer/registry admission → cache-key composition → the real
`AnalyzeCacheStore`.

The spec's integration promise, verified on live subsystems: the shadowed
variant of a file admits ZERO OBSERVED findings and its cache row is keyed
with the widened digests; the value-import variant admits the finding under
a DIFFERENT key — so the two variants can never cross-serve, and a lookup
under one key never returns the other's payload.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.agent.nodes.analyze_observed import (
    OBSERVED_PRODUCER_VERSION,
    import_bindings_digest,
    lexical_bindings_digest,
    module_admission_digest,
    run_observed_matches,
)
from outrider.agent.nodes.analyze_parser import ANALYZE_PARSER_VERSION, from_import_map_digest
from outrider.ast_facts.registry import parse_source
from outrider.cache import AnalyzeCacheStore, compute_analyze_cache_key
from outrider.policy.subsumption import SUBSUMES_DIGEST
from outrider.queries.registry import QUERY_REGISTRY_DIGEST

_INSTALLATION_ID = 6161

_SHADOWED = (
    'import { createHash } from "node:crypto";\n'
    "export function f(createHash) {\n"
    '  return createHash("md5");\n'
    "}\n"
)
_VALUE = (
    'import { createHash } from "node:crypto";\n'
    "export function f(secret) {\n"
    '  return createHash("md5").update(secret).digest("hex");\n'
    "}\n"
)


async def _seed_review(engine):
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, account_type, "
                " permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": _INSTALLATION_ID},
        )
        result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, is_eval, "
                "  retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, :sha, 'running', false, "
                "  NOW() + make_interval(days => 180)"
                ") RETURNING id"
            ),
            {"id": _INSTALLATION_ID, "sha": uuid4().hex},
        )
        return result.scalar_one()


def _matches_and_key(source: str) -> tuple[tuple, str]:
    """The live chain for one variant: parse (ast_facts) → admit (producer +
    registry) → compose the key (cache) from the SAME parse facts the
    producer consumed."""
    parsed = parse_source(source.encode("utf-8"), "src/token.mjs", MagicMock())
    assert parsed.parser_outcome == "clean"
    matches = run_observed_matches(
        file_path="src/token.mjs",
        head_content=source,
        included_scope_units=parsed.scope_units,
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
    )
    key = compute_analyze_cache_key(
        system_prompt="system",
        # Byte-identical prompt stand-in for BOTH variants — deliberately:
        # the digests, not the prompt, must split the keys.
        user_prompt="identical prompt bytes",
        installation_id=_INSTALLATION_ID,
        repo_id=100,
        model="claude-haiku-4-5",
        prompt_template_version="analyze-v4",
        trivial_filter_version="trivial-filter-v1",
        query_registry_digest=QUERY_REGISTRY_DIGEST,
        active_policy_version="policy-v1",
        analyze_parser_version=ANALYZE_PARSER_VERSION,
        response_format_digest="c" * 64,
        parameterized_call_scan_digest="d" * 64,
        observed_producer_version=OBSERVED_PRODUCER_VERSION,
        subsumes_digest=SUBSUMES_DIGEST,
        from_import_map_digest=from_import_map_digest(parsed.imports),
        import_bindings_digest=import_bindings_digest(parsed.imports),
        lexical_bindings_digest=lexical_bindings_digest(parsed.lexical_bindings),
        # This probe drives the SHADOWING split (function-scope variants, no
        # module-level change): empty added ranges keep the module arm inert,
        # composed from the same parse facts as the node would.
        module_admission_digest=module_admission_digest(
            (), parsed.scope_units, source.encode("utf-8")
        ),
        profile_id=None,
        reasoning_enabled=None,
        profile_contract_digest=None,
    )
    return matches, key


@pytest.mark.asyncio
async def test_shadowed_and_value_variants_never_cross_serve(migrated_db: str) -> None:
    """The shadowed variant admits nothing and the value variant admits the
    weak-crypto finding; with byte-identical prompts their keys STILL differ
    (the lexical-bindings digest is the only splitting input — the import
    sets match), and a real-store lookup under one key never returns the
    other's payload."""
    shadowed_matches, shadowed_key = _matches_and_key(_SHADOWED)
    value_matches, value_key = _matches_and_key(_VALUE)

    assert shadowed_matches == ()
    assert [m.query_match_id for m in value_matches] == ["javascript.weak_crypto_hash"]
    assert shadowed_key != value_key

    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None

        def _write_kwargs(cache_key: str, payload: dict) -> dict:
            return {
                "cache_key": cache_key,
                "scope": scope,
                "source_review_id": review_id,
                "file_path": "src/token.mjs",
                "payload": payload,
                "model": "claude-haiku-4-5",
                "prompt_template_version": "analyze-v4",
                "trivial_filter_version": "trivial-filter-v1",
                "query_registry_digest": QUERY_REGISTRY_DIGEST,
                "active_policy_version": "policy-v1",
                "analyze_parser_version": ANALYZE_PARSER_VERSION,
                "prompt_hash": "b" * 64,
            }

        await store.write(**_write_kwargs(shadowed_key, {"findings": [], "trace_candidates": []}))
        await store.write(
            **_write_kwargs(
                value_key,
                {
                    "findings": [{"query_match_id": "javascript.weak_crypto_hash"}],
                    "trace_candidates": [],
                },
            )
        )

        shadowed_row = await store.lookup(shadowed_key, is_eval=False)
        value_row = await store.lookup(value_key, is_eval=False)
        assert shadowed_row is not None and shadowed_row.payload["findings"] == []
        assert value_row is not None and value_row.payload["findings"] != []
        # The load-bearing negative: the shadowed key can never serve the
        # value variant's finding set (or vice versa) — distinct rows,
        # distinct payloads, no cross-serve under byte-identical prompts.
        assert shadowed_row.payload != value_row.payload
    finally:
        await engine.dispose()
