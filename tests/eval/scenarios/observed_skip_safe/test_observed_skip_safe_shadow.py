"""observed_skip_safe eval scenario (Cost Lever 3, DECISIONS#049) — the promotion
proof in SHADOW mode.

A fixture promotes ONE OBSERVED query to `skip_safe` (test-local, restored by
monkeypatch), so the file's only changed line is fully covered → the analyze node
records a `would_skip` `ObservedSkipShadowEvent`. But V1 is shadow-only: the LLM
STILL RUNS, and the JUDGED finding it returns is accounted for by the OBSERVED set
(same `content_hash`) — the no-JUDGED-loss proof that gates a real promotion.

This is NOT proving production skipping; it proves the promotion CONTRACT while
still calling the model. A separate test asserts the production registry stays
zero-`skip_safe` (the promotion never leaks past this scenario), protecting
DECISIONS#049 from silently becoming "V1 has one skip-safe seed".
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy import text

import outrider.queries.registry as query_registry
from outrider.agent import run_review_persisting
from outrider.agent.nodes.analyze_observed import produce_observed_findings, run_observed_matches
from outrider.ast_facts import parse_python
from outrider.policy import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.queries.observed import QueryClass

if TYPE_CHECKING:
    from uuid import UUID

    import pytest
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_FIXTURE = "tests/eval/fixtures/mock_github/observed_skip_safe.json"
_PROMOTE_ID = "python.command_injection_subprocess_shell"
_HEAD = (
    "import subprocess\n\n\ndef run_it(cmd):\n    return cmd\n    subprocess.run(cmd, shell=True)\n"
)


def _promote_to_skip_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the module's OBSERVED_QUERIES attribute for a COPY with `_PROMOTE_ID`
    flipped to skip_safe. The real `_OBSERVED_QUERIES` dict and the import-time
    `QUERY_REGISTRY_DIGEST` constant are untouched; monkeypatch restores the
    attribute at teardown."""
    promoted = dict(query_registry.OBSERVED_QUERIES)
    promoted[_PROMOTE_ID] = promoted[_PROMOTE_ID].model_copy(
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


def _observed_content_hashes() -> set[str]:
    """The content_hashes the deterministic OBSERVED producer emits for the fixture
    head — the set a skip must not lose anything outside of."""
    scopes = parse_python(_HEAD.encode(), "src/vuln.py", MagicMock()).scope_units
    matches = run_observed_matches(
        file_path="src/vuln.py", head_content=_HEAD, included_scope_units=scopes
    )
    findings = produce_observed_findings(
        matches,
        file_path="src/vuln.py",
        review_id=uuid4(),
        installation_id=0,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )
    return {f.content_hash for f in findings}


async def test_observed_skip_safe_promotion_would_skip_with_no_judged_loss(
    eval_db: str,
    eval_db_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promote one query to skip_safe (test-local): the file records `would_skip`,
    the LLM still runs (shadow), and the JUDGED finding it returns is in the OBSERVED
    set — so a skip would have lost nothing. The promotion contract, proven in
    shadow mode without activating any production skip."""
    digest_before = query_registry.QUERY_REGISTRY_DIGEST
    _promote_to_skip_safe(monkeypatch)

    result = await run_review_persisting(_FIXTURE, db_url=eval_db)

    # The test-local promotion swaps only the OBSERVED_QUERIES attribute; the
    # import-time cache-key digest constant is untouched (no cache-identity drift).
    assert digest_before == query_registry.QUERY_REGISTRY_DIGEST

    # 1. The promoted skip_safe query fully covers the only changed line → would_skip.
    payloads = await _shadow_payloads(eval_db_session_factory, result.review_id)
    assert len(payloads) == 1, "exactly one shadow event for the one analyzed file"
    shadow = payloads[0]
    assert shadow["outcome"] == "would_skip"
    assert shadow["node_id"] == "analyze"
    assert shadow["file_path"] == "src/vuln.py"
    assert shadow["blockers"] == []
    assert [c["query_match_id"] for c in shadow["covering_matches"]] == [_PROMOTE_ID]  # type: ignore[union-attr]

    # 2. Shadow-only: the LLM STILL RAN (>=1 analyze llm_call). V1 never skips.
    assert (
        await _count_events(
            eval_db_session_factory, result.review_id, event_type="llm_call", node_id="analyze"
        )
        >= 1
    )

    # 3. No JUDGED loss: every JUDGED finding is accounted for by the OBSERVED set
    #    (same content_hash) — skipping the LLM would have lost nothing here.
    observed_hashes = _observed_content_hashes()
    judged = [f for f in result.findings if f.evidence_tier == EvidenceTier.JUDGED]
    assert judged, "the scripted LLM returned at least one JUDGED finding"
    for finding in judged:
        assert finding.content_hash in observed_hashes, (
            f"JUDGED {finding.finding_type} @ {finding.line_start} not in OBSERVED set"
        )


async def test_production_registry_stays_zero_skip_safe() -> None:
    """DECISIONS#049 guard: the real registry seeds ZERO skip_safe queries. The
    promotion in the scenario above is test-local + monkeypatch-restored, so no
    production skip-safe seed leaks (this test runs with the real registry)."""
    classes = [q.query_class for q in query_registry.OBSERVED_QUERIES.values()]
    assert classes, "the OBSERVED registry is non-empty"
    assert all(c == QueryClass.SIGNAL_ONLY for c in classes)
    assert sum(1 for c in classes if c == QueryClass.SKIP_SAFE) == 0
