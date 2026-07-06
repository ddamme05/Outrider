# Per DECISIONS.md#055 — cross-type subsumption at the analyze node.
"""Cross-type subsumption merge (node-level): a same-span JUDGED subsumer drops
an admitted OBSERVED finding of a broader finding_type, the survivor stays JUDGED,
the dropped query_match_id is retained in `AnalyzeCompletedEvent.subsumed_matches`,
and the proposal-accounting equation stays balanced with no new term.

The producer fires OBSERVED `weak_crypto` on a `DES.new(...)` line (the shipped
broken-cipher query); the scripted model emits JUDGED `weak_password_hash` on the
SAME span. `weak_password_hash ⊐ weak_crypto` is the seed SUBSUMES edge, so the
CRITICAL password-hash survives and the HIGH weak-crypto is suppressed. (DES is
not literally password hashing — this exercises the MERGE mechanism with the only
OBSERVED crypto query that ships today; the md5/sha1 OBSERVED query is the
follow-on.)"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from test_analyze_node import run_analyze_pass_kw

from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS
from outrider.llm.anthropic_provider import (
    _ANTHROPIC_CONTRACT_DIGEST,
    _ANTHROPIC_PROFILE_ID,
)
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.policy import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION, FindingSeverity, FindingType
from outrider.schemas import ChangedFile, PRContext, ReviewState
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

_REVIEW_ID = UUID("0fedcba9-8765-4321-0fed-cba987654321")
_WEAK_CRYPTO_QMID = "python.weak_crypto_broken_cipher"

# Line 5 is `    cipher = DES.new(key)` — the shipped broken-cipher query fires
# OBSERVED weak_crypto there; the scripted model JUDGES the same span.
_HEAD = (
    "from Crypto.Cipher import DES\n\n\n"
    "def enc(key, data):\n"
    "    cipher = DES.new(key)\n"
    "    return cipher.encrypt(data)\n"
)
_BASE = "from Crypto.Cipher import DES\n"
_PATCH = (
    "--- a/app/crypto.py\n+++ b/app/crypto.py\n"
    "@@ -1 +1,6 @@\n from Crypto.Cipher import DES\n+\n+\n+def enc(key, data):\n"
    "+    cipher = DES.new(key)\n"
    "+    return cipher.encrypt(data)\n"
)


def _wph_finding(tier: str = "judged") -> dict[str, Any]:
    return {
        "finding_type": "weak_password_hash",
        "evidence_tier": tier,
        "query_match_id": None,
        "trace_path": None,
        "title": "Weak password hashing on line 5",
        "description": "The key derivation on this line uses a broken primitive.",
        "evidence": "    cipher = DES.new(key)",
        "line_start": 5,
        "line_end": 5,
        "trace_candidates": [],
    }


def _wc_judged_finding() -> dict[str, Any]:
    return {
        "finding_type": "weak_crypto",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "Weak cipher on line 5",
        "description": "DES is a broken cipher.",
        "evidence": "    cipher = DES.new(key)",
        "line_start": 5,
        "line_end": 5,
        "trace_candidates": [],
    }


class _ScriptedProvider:
    def __init__(self, findings: list[dict[str, Any]]) -> None:
        self._text = json.dumps({"findings": findings})

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=self._text,
            model=request.model,
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=10,
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )


class _PhaseSink:
    async def emit_phase(self, event: Any) -> None:  # noqa: ARG002
        return None


class _FileExamSink:
    async def emit_file_examination(self, event: Any) -> None:  # noqa: ARG002
        return None


class _AnalyzeSink:
    def __init__(self) -> None:
        self.findings: list[Any] = []
        self.completed: list[Any] = []

    async def emit_finding(self, finding: Any, *, is_eval: bool) -> None:  # noqa: ARG002
        self.findings.append(finding)

    async def emit_finding_proposal_rejected(self, event: Any) -> None:
        raise AssertionError(f"unexpected proposal rejection: {event!r}")

    async def emit_analyze_response_rejected(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_analyze_completed(self, event: Any) -> None:
        self.completed.append(event)

    async def emit_scope_exclusion(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_cache_lookup(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_cache_serve(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_observed_skip_shadow(self, event: Any) -> None:  # noqa: ARG002
        return None


def _state(head: str = _HEAD, patch: str = _PATCH) -> ReviewState:
    cf = ChangedFile(
        path="app/crypto.py",
        status="modified",
        additions=5,
        deletions=0,
        patch=patch,
        content_base=_BASE,
        content_head=head,
        previous_path=None,
        language="python",
    )
    pr_context = PRContext(
        installation_id=1,
        owner="acme",
        repo="widget",
        pr_number=9,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="t",
        pr_body=None,
        author="someone",
        total_additions=5,
        total_deletions=0,
        changed_files=(cf,),
    )
    triage = TriageResult(
        file_tiers={cf.path: ReviewTier.DEEP},
        overall_risk=RiskLevel.HIGH,
        relevant_dimensions=(ReviewDimension.SECURITY,),
        reasoning="test",
    )
    return ReviewState(
        review_id=_REVIEW_ID,
        received_at=datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC),
        pr_context=pr_context,
        triage_result=triage,
        is_eval=True,
    )


async def _run(provider: _ScriptedProvider, state: ReviewState | None = None) -> _AnalyzeSink:
    sink = _AnalyzeSink()
    await run_analyze_pass_kw(
        state if state is not None else _state(),
        provider=provider,  # type: ignore[arg-type]
        analyze_model="claude-sonnet-4-6",
        standard_analyze_model="claude-sonnet-4-6",
        phase_event_sink=_PhaseSink(),
        file_examination_sink=_FileExamSink(),
        analyze_event_sink=sink,
        anomaly_sink=AsyncMock(),
        import_path_resolver=MagicMock(),
        active_policy_version=ACTIVE_POLICY_VERSION,
        total_review_budget_tokens=DEFAULT_REVIEW_BUDGET_TOKENS,
        trivial_scope_filter_enabled=False,
    )
    return sink


def _assert_equation_balances(ev: Any) -> None:
    assert (
        ev.n_proposals_seen
        == (ev.n_findings_emitted - ev.n_findings_served - ev.n_findings_observed)
        + ev.n_proposals_rejected
        + ev.n_proposals_superseded_by_observed
    )


@pytest.mark.asyncio
async def test_judged_subsumer_drops_observed_weak_crypto() -> None:
    """OBSERVED weak_crypto (producer) + JUDGED weak_password_hash (model) on the
    same span → only the CRITICAL JUDGED survives; the OBSERVED is dropped, its
    query_match_id retained in subsumed_matches; the equation balances."""
    sink = await _run(_ScriptedProvider([_wph_finding()]))

    assert len(sink.findings) == 1
    finding = sink.findings[0]
    assert finding.finding_type is FindingType.WEAK_PASSWORD_HASH
    assert finding.evidence_tier is EvidenceTier.JUDGED
    assert finding.query_match_id is None
    assert finding.severity is FindingSeverity.CRITICAL

    [completed] = sink.completed
    assert len(completed.subsumed_matches) == 1
    rec = completed.subsumed_matches[0]
    assert rec.query_match_id == _WEAK_CRYPTO_QMID
    assert rec.finding_type is FindingType.WEAK_CRYPTO
    assert rec.subsumed_by_finding_type is FindingType.WEAK_PASSWORD_HASH
    assert rec.file_path == "app/crypto.py"
    assert completed.n_findings_observed == 0
    assert completed.n_findings_emitted == 1
    assert completed.n_proposals_superseded_by_observed == 0
    _assert_equation_balances(completed)


@pytest.mark.asyncio
async def test_triple_collision_resolves_to_single_critical() -> None:
    """Model emits BOTH JUDGED weak_crypto and JUDGED weak_password_hash on the
    line, and the producer fires OBSERVED weak_crypto: #054 swaps the JUDGED
    weak_crypto to OBSERVED, then cross-type drops that OBSERVED under the JUDGED
    weak_password_hash → only the CRITICAL survives, equation balances."""
    sink = await _run(_ScriptedProvider([_wc_judged_finding(), _wph_finding()]))

    assert len(sink.findings) == 1
    finding = sink.findings[0]
    assert finding.finding_type is FindingType.WEAK_PASSWORD_HASH
    assert finding.evidence_tier is EvidenceTier.JUDGED

    [completed] = sink.completed
    assert len(completed.subsumed_matches) == 1
    assert completed.n_findings_observed == 0
    assert completed.n_proposals_superseded_by_observed == 1
    assert completed.n_proposals_seen == 2
    assert completed.n_findings_emitted == 1
    _assert_equation_balances(completed)


@pytest.mark.asyncio
async def test_observed_weak_crypto_with_no_subsumer_survives() -> None:
    """No subsuming admitted finding → the OBSERVED weak_crypto survives normally
    (subsumption does not fire); subsumed_matches is empty."""
    # Model emits an unrelated JUDGED finding elsewhere (line 6), so no same-span
    # subsumer exists for the OBSERVED weak_crypto on line 5.
    elsewhere = {
        "finding_type": "missing_error_handling",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "x",
        "description": "y",
        "evidence": "    return cipher.encrypt(data)",
        "line_start": 6,
        "line_end": 6,
        "trace_candidates": [],
    }
    sink = await _run(_ScriptedProvider([elsewhere]))

    observed = [f for f in sink.findings if f.evidence_tier is EvidenceTier.OBSERVED]
    assert len(observed) == 1
    assert observed[0].finding_type is FindingType.WEAK_CRYPTO
    assert observed[0].query_match_id == _WEAK_CRYPTO_QMID
    [completed] = sink.completed
    assert completed.subsumed_matches == ()
    assert completed.n_findings_observed == 1
    _assert_equation_balances(completed)


@pytest.mark.asyncio
async def test_model_cited_observed_is_not_subsumed_and_does_not_crash() -> None:
    """Regression (review HIGH): a MODEL-cited OBSERVED finding (the model claimed
    `observed` citing a fired STRUCTURAL query id, which the parser admits without
    binding finding_type to the query) rides the parser's n_findings_emitted, NOT
    n_findings_observed. It must NOT be subsumed — only producer-origin OBSERVED
    findings (query_match_id in OBSERVED_QUERY_IDS) are subsumable. Subsuming a
    model-cited OBSERVED would decrement n_findings_observed below zero and crash
    the pass via the Field(ge=0) validator."""
    # Model emits OBSERVED weak_crypto citing `python.function_definition` (a
    # structural query that fires on `def enc` — so it is in the admissible set),
    # plus a same-span JUDGED weak_password_hash subsumer. The producer ALSO fires
    # weak_crypto on line 5, so observed_findings is non-empty and the #055 block
    # runs; #054 keeps the model-cited OBSERVED incumbent over the producer dup.
    model_observed_wc = {
        "finding_type": "weak_crypto",
        "evidence_tier": "observed",
        "query_match_id": "python.function_definition",
        "trace_path": None,
        "title": "model-cited weak crypto",
        "description": "the model claimed observed citing a structural query",
        "evidence": "    cipher = DES.new(key)",
        "line_start": 5,
        "line_end": 5,
        "trace_candidates": [],
    }
    sink = await _run(_ScriptedProvider([model_observed_wc, _wph_finding()]))

    # No crash, and the model-cited OBSERVED survives (it is not producer-origin,
    # so subsumption leaves it alone) alongside the JUDGED.
    tiers = {(f.finding_type, f.evidence_tier) for f in sink.findings}
    assert (FindingType.WEAK_CRYPTO, EvidenceTier.OBSERVED) in tiers
    assert (FindingType.WEAK_PASSWORD_HASH, EvidenceTier.JUDGED) in tiers
    [completed] = sink.completed
    assert completed.subsumed_matches == ()
    assert completed.n_findings_observed >= 0
    _assert_equation_balances(completed)


@pytest.mark.asyncio
async def test_different_span_does_not_subsume() -> None:
    """Span-safety (DECISIONS.md#055): a weak_password_hash subsumer on a
    DIFFERENT line span than the OBSERVED weak_crypto does NOT absorb it — exact
    span only, never loose overlap. Both survive."""
    wph_line6 = _wph_finding()
    wph_line6["line_start"] = 6
    wph_line6["line_end"] = 6
    wph_line6["evidence"] = "    return cipher.encrypt(data)"
    sink = await _run(_ScriptedProvider([wph_line6]))

    tiers = {(f.finding_type, f.evidence_tier) for f in sink.findings}
    assert (FindingType.WEAK_CRYPTO, EvidenceTier.OBSERVED) in tiers
    assert (FindingType.WEAK_PASSWORD_HASH, EvidenceTier.JUDGED) in tiers
    [completed] = sink.completed
    assert completed.subsumed_matches == ()
    _assert_equation_balances(completed)
