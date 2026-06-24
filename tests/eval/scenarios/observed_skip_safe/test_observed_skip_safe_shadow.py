"""observed_skip_safe eval scenario (Cost Lever 3, DECISIONS#049) — the promotion
proof in SHADOW mode.

A fixture promotes ONE OBSERVED query to `skip_safe` (test-local, restored by
monkeypatch), so the file's only changed line is fully covered → the analyze node
records a `would_skip` `ObservedSkipShadowEvent`. But V1 is shadow-only: the LLM
STILL RUNS, and the JUDGED finding it returns shares a `content_hash` with the
OBSERVED set, so prefer-OBSERVED (DECISIONS.md#054) evicts it in favor of the
OBSERVED match — it surfaces as OBSERVED, leaving NO uncovered JUDGED. That is the
no-JUDGED-loss proof that gates a real promotion.

Parametrized over two candidates (Step 3a): `command_injection_subprocess_shell`
and `unsafe_deserialization_yaml` (the 3a candidate, after its precision edit). Both
prove the SAME contract, and the collision is SEMANTIC — the scripted LLM's JUDGED
finding is the same file/line/finding_type as the OBSERVED match, so it collides
through the real prefer-OBSERVED path (DECISIONS.md#054), not because fields were
hand-shaped around the hash.

This is NOT proving production skipping; it proves the promotion CONTRACT while
still calling the model. A separate test asserts the production registry stays
zero-`skip_safe` (the promotion never leaks past this scenario), protecting
DECISIONS#049 from silently becoming "V1 has one skip-safe seed".

Cache control (Step 3a, FUP-183): the run is pinned to `analyze_cache_store=None`
+ `CacheMode.SHADOW` so the OBSERVED producer fires on every file. A cache SERVE
hit reconstructs findings from the cached payload WITHOUT running the producer or
`compute_observed_skip_shadow`, biasing the would-skip evidence base (FUP-183); the
no-store + SHADOW pin forecloses that. These ARE the `run_review_persisting`
defaults — pinned explicitly so a later default change cannot silently bias the proof.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import text

import outrider.queries.registry as query_registry
from outrider.agent import run_review_persisting
from outrider.agent.nodes.analyze_observed import produce_observed_findings, run_observed_matches
from outrider.agent.nodes.cache_config import CacheMode
from outrider.ast_facts import parse_python
from outrider.policy import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.queries.observed import QueryClass

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_FILE_PATH = "src/vuln.py"

_SUBPROCESS_HEAD = (
    "import subprocess\n\n\ndef run_it(cmd):\n    return cmd\n    subprocess.run(cmd, shell=True)\n"
)
_YAML_HEAD = "import yaml\n\n\ndef load_it(data):\n    return data\n    yaml.load(data)\n"

# (label, fixture, promote_id, head): each promotes ONE query to skip_safe and proves
# no uncovered JUDGED survives. The yaml case is the Step-3a candidate; both share
# `_FILE_PATH` and a single changed line (line 6) fully covered by the promoted match.
_PROMOTION_CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "subprocess_shell",
        "tests/eval/fixtures/mock_github/observed_skip_safe.json",
        "python.command_injection_subprocess_shell",
        _SUBPROCESS_HEAD,
    ),
    (
        "yaml_load",
        "tests/eval/fixtures/mock_github/observed_skip_safe_yaml.json",
        "python.unsafe_deserialization_yaml",
        _YAML_HEAD,
    ),
)


def _promote_to_skip_safe(monkeypatch: pytest.MonkeyPatch, promote_id: str) -> None:
    """Swap the module's OBSERVED_QUERIES attribute for a COPY with `promote_id`
    flipped to skip_safe. The real `_OBSERVED_QUERIES` dict and the import-time
    `QUERY_REGISTRY_DIGEST` constant are untouched; monkeypatch restores the
    attribute at teardown."""
    promoted = dict(query_registry.OBSERVED_QUERIES)
    promoted[promote_id] = promoted[promote_id].model_copy(
        update={"query_class": QueryClass.SKIP_SAFE}
    )
    monkeypatch.setattr(query_registry, "OBSERVED_QUERIES", MappingProxyType(promoted))


async def _shadow_payloads(
    session_factory: async_sessionmaker[AsyncSession], review_id: UUID
) -> list[dict[str, object]]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT payload FROM audit_events WHERE review_id = :rid "
                    "AND event_type = 'observed_skip_shadow'"
                ),
                {"rid": review_id},
            )
        ).all()
    return [r.payload for r in rows]


async def _count_events(
    session_factory: async_sessionmaker[AsyncSession],
    review_id: UUID,
    *,
    event_type: str,
    node_id: str | None = None,
) -> int:
    sql = "SELECT COUNT(*) FROM audit_events WHERE review_id = :rid AND event_type = :etype"
    params: dict[str, object] = {"rid": review_id, "etype": event_type}
    if node_id is not None:
        sql += " AND payload->>'node_id' = :node"
        params["node"] = node_id
    async with session_factory() as session:
        return (await session.execute(text(sql), params)).scalar_one()


def _observed_content_hashes(head: str) -> set[str]:
    """The content_hashes the deterministic OBSERVED producer emits for the fixture
    head — the set a skip must not lose anything outside of."""
    scopes = parse_python(head.encode(), _FILE_PATH, MagicMock()).scope_units
    matches = run_observed_matches(
        file_path=_FILE_PATH, head_content=head, included_scope_units=scopes
    )
    findings = produce_observed_findings(
        matches,
        file_path=_FILE_PATH,
        review_id=uuid4(),
        installation_id=0,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )
    return {f.content_hash for f in findings}


@pytest.mark.parametrize(
    ("fixture", "promote_id", "head"),
    [(fixture, promote_id, head) for _label, fixture, promote_id, head in _PROMOTION_CASES],
    ids=[case[0] for case in _PROMOTION_CASES],
)
async def test_observed_skip_safe_promotion_would_skip_with_no_judged_loss(
    fixture: str,
    promote_id: str,
    head: str,
    eval_db: str,
    eval_db_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promote one query to skip_safe (test-local): the file records `would_skip`,
    the LLM still runs (shadow), and the JUDGED finding it returns collides with the
    OBSERVED set — under prefer-OBSERVED (DECISIONS.md#054) it is evicted to OBSERVED,
    so NO uncovered JUDGED survives and a skip would have lost nothing. The promotion
    contract, proven in shadow mode for each candidate without activating any
    production skip."""
    digest_before = query_registry.QUERY_REGISTRY_DIGEST
    _promote_to_skip_safe(monkeypatch, promote_id)

    # Cache pinned OFF + SHADOW so the OBSERVED producer fires (FUP-183): a SERVE hit
    # would reconstruct findings from the cached payload without running the producer.
    result = await run_review_persisting(
        fixture, db_url=eval_db, analyze_cache_store=None, cache_mode=CacheMode.SHADOW
    )

    # The test-local promotion swaps only the OBSERVED_QUERIES attribute; the
    # import-time cache-key digest constant is untouched (no cache-identity drift).
    assert digest_before == query_registry.QUERY_REGISTRY_DIGEST

    # 1. The promoted skip_safe query fully covers the only changed line → would_skip.
    payloads = await _shadow_payloads(eval_db_session_factory, result.review_id)
    assert len(payloads) == 1, "exactly one shadow event for the one analyzed file"
    shadow = payloads[0]
    assert shadow["outcome"] == "would_skip"
    assert shadow["node_id"] == "analyze"
    assert shadow["file_path"] == _FILE_PATH
    assert shadow["blockers"] == []
    assert [c["query_match_id"] for c in shadow["covering_matches"]] == [promote_id]  # type: ignore[union-attr]

    # 2. Shadow-only: the LLM STILL RAN (>=1 analyze llm_call). V1 never skips.
    assert (
        await _count_events(
            eval_db_session_factory, result.review_id, event_type="llm_call", node_id="analyze"
        )
        >= 1
    )

    # 3. No JUDGED loss: the scripted LLM's JUDGED finding shares an OBSERVED
    #    content_hash, so prefer-OBSERVED (DECISIONS.md#054) EVICTS it for the
    #    OBSERVED match — it surfaces as OBSERVED. The safety property a would_skip
    #    needs: NO JUDGED finding survives (a surviving JUDGED would be UNcovered by
    #    OBSERVED — a real skip-loss). The covered finding is now in the OBSERVED set.
    #    PRECONDITION: this whole-review `not judged` form is valid only because the
    #    fixture's single changed line is fully covered, so EVERY admitted JUDGED
    #    collides and is evicted. If the fixture ever gains an uncovered changed line
    #    (or a JUDGED finding_type the producer doesn't emit), scope the assertion to
    #    the covered line(s) — an uncovered JUDGED legitimately survives and is NOT a
    #    skip-loss for the promoted query.
    observed_hashes = _observed_content_hashes(head)
    judged = [f for f in result.findings if f.evidence_tier == EvidenceTier.JUDGED]
    assert not judged, (
        "a surviving JUDGED finding is uncovered by OBSERVED = a real skip-loss: "
        f"{[(f.finding_type, f.line_start) for f in judged]}"
    )
    observed = [f for f in result.findings if f.evidence_tier == EvidenceTier.OBSERVED]
    assert observed, "the covered finding must surface as OBSERVED after the prefer-OBSERVED swap"
    assert all(f.content_hash in observed_hashes for f in observed), (
        "every surviving OBSERVED finding must be in the deterministic OBSERVED set"
    )


async def test_production_registry_stays_zero_skip_safe() -> None:
    """DECISIONS#049 guard: the real registry seeds ZERO skip_safe queries. The
    promotion in the scenario above is test-local + monkeypatch-restored, so no
    production skip-safe seed leaks (this test runs with the real registry)."""
    classes = [q.query_class for q in query_registry.OBSERVED_QUERIES.values()]
    assert classes, "the OBSERVED registry is non-empty"
    assert all(c == QueryClass.SIGNAL_ONLY for c in classes)
    assert sum(1 for c in classes if c == QueryClass.SKIP_SAFE) == 0
