"""Trace admission instrument for the openai host (openai-native-host spec).

Frozen pass predicate (spec "Gates before any production-shaped use"): the
expected candidate RANKED (first, by the trace model's own ranking), no
fabricated candidate id (the ranking-layer analog of FUP-236's guessed module
path — the production parser rejects fabrication/omission/duplication
outright), zero rejected responses. The grader IS the production parser
(`parse_trace_ranking`); resolution after ranking is deterministic probe code
covered by `tests/integration/test_trace_node_end_to_end.py`, not a model
surface. Candidate PROPOSAL quality (FUP-236's analyze-side guessed paths) is
the ANALYZE model's surface — covered by the scorecard + prompt-transfer
probes, not this instrument.

ONE canonical paid path (per the review clarification): the paid row lives in
the wire probe (`spikes/openai/probe.py`, row `gpt-5.6-luna:trace`). This
file never spends: it proves the grader can FAIL via scripted negative twins
(free, normal eval gate) and grades the captured probe fixture OFFLINE —
skipped until the capture exists, HARD-asserted once it does (miss => Terra
swap + rerun, never a softened gate).

The scenario is duplicated in the probe DELIBERATELY (spikes/ is not
importable from tests). Drift fails loud: a probe that renders different
candidates yields responses whose ids cannot satisfy this file's parser-backed
grader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from outrider.agent.nodes.trace_parser import (
    TraceRankingParsed,
    TraceRankingRejected,
    parse_trace_ranking,
)
from outrider.policy.canonical import compute_candidate_id
from outrider.prompts import trace as trace_prompt
from outrider.schemas.trace_candidate import TraceCandidate

# --- The deterministic admission scenario (duplicated in the probe) ---------
# The FUP-236 shape: one REAL cross-file candidate (the finding's tainted call
# flows through `run_query` imported from app.db) and one peripheral
# distractor. The trace model must rank the load-bearing candidate first.
# candidate_id is CONTENT-DERIVED (compute_candidate_id — the schema validator
# rejects arbitrary hex), so both sides of the probe duplication derive the
# same ids from the same payloads.
_REAL_IMPORT = "app.db"
_DISTRACTOR_IMPORT = "app.render_helpers"
_REAL_REASON = (
    "handlers.py builds the flagged query via run_query imported from app.db; "
    "the finding's tainted value flows directly into it"
)
_DISTRACTOR_REASON = (
    "render_error_page formats the error string shown when the request fails; "
    "cosmetic to the finding's data flow"
)
_REAL_ID = compute_candidate_id(
    source_proposal_hash="3" * 64, import_string=_REAL_IMPORT, reason=_REAL_REASON
)
_DISTRACTOR_ID = compute_candidate_id(
    source_proposal_hash="4" * 64, import_string=_DISTRACTOR_IMPORT, reason=_DISTRACTOR_REASON
)

_TRACE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "spikes"
    / "openai"
    / "fixtures"
    / "gpt-5.6-luna_trace.json"
)


def _candidates() -> tuple[TraceCandidate, TraceCandidate]:
    real = TraceCandidate(
        candidate_id=_REAL_ID,
        source_proposal_hash="3" * 64,
        reason=_REAL_REASON,
        import_string=_REAL_IMPORT,
    )
    distractor = TraceCandidate(
        candidate_id=_DISTRACTOR_ID,
        source_proposal_hash="4" * 64,
        reason=_DISTRACTOR_REASON,
        import_string=_DISTRACTOR_IMPORT,
    )
    return (real, distractor)


def _ranking_json(*ids: str) -> str:
    return json.dumps({"ranked_candidate_ids": list(ids)})


def _grade_ranking(text: str) -> tuple[str, str]:
    """(verdict, detail) — verdict 'pass' | 'rejected' | 'misranked'. Built on
    the PRODUCTION parser so 'zero rejected responses' and 'no fabricated id'
    are graded by the exact code the trace node runs."""
    outcome = parse_trace_ranking(response_text=text, candidates=_candidates())
    if isinstance(outcome, TraceRankingRejected):
        return ("rejected", outcome.reason)
    assert isinstance(outcome, TraceRankingParsed)
    first = outcome.ordered_candidates[0]
    if first.candidate_id != _REAL_ID:
        return ("misranked", first.import_string)
    return ("pass", first.import_string)


def test_scenario_renders_through_real_prompt() -> None:
    """The probe's paid row renders THIS scenario through the real
    `trace_prompt.render`; the render must carry both candidate ids (the
    exact strings the parser will demand back) or the capture grades a
    prompt the node never sends."""
    parts = trace_prompt.render(_candidates())
    assert _REAL_ID in parts.user_prompt
    assert _DISTRACTOR_ID in parts.user_prompt
    assert _REAL_IMPORT in parts.user_prompt
    assert parts.system_prompt.strip()


def test_grader_negative_twins() -> None:
    """The grader can FAIL — one twin per parser rejection reason plus the
    misranking case, each differing from the passing control in exactly the
    graded property. (`ranking_id_missing` has no reachable twin here: with
    a matching count, no duplicates, and no fabricated ids, a two-candidate
    set is necessarily complete — the parser's missing branch is defensive.)"""
    assert _grade_ranking(_ranking_json(_REAL_ID, _DISTRACTOR_ID)) == ("pass", _REAL_IMPORT)

    verdict, reason = _grade_ranking("I think the db module matters most.")
    assert (verdict, reason) == ("rejected", "raw_response_unparseable")

    verdict, reason = _grade_ranking(_ranking_json(_REAL_ID))
    assert (verdict, reason) == ("rejected", "ranking_count_mismatch")

    verdict, reason = _grade_ranking(_ranking_json(_REAL_ID, _REAL_ID))
    assert (verdict, reason) == ("rejected", "ranking_id_duplicated")

    # A fabricated candidate id — the ranking-layer analog of FUP-236's
    # guessed module path — is rejected by the production parser.
    verdict, reason = _grade_ranking(_ranking_json(_REAL_ID, "f" * 64))
    assert (verdict, reason) == ("rejected", "ranking_id_fabricated")

    verdict, detail = _grade_ranking(_ranking_json(_DISTRACTOR_ID, _REAL_ID))
    assert (verdict, detail) == ("misranked", _DISTRACTOR_IMPORT)


def test_captured_paid_fixture_passes_frozen_predicate() -> None:
    """Grade the probe's captured paid row OFFLINE against the frozen
    predicate. Skips until the capture exists; once it does, a miss FAILS —
    the spec's rule is a Terra swap + rerun, never a softened gate."""
    if not _TRACE_FIXTURE.exists():
        pytest.skip(
            "paid trace capture absent — run the wire probe first "
            "(op run --env-file=.env -- uv run python spikes/openai/probe.py)"
        )
    doc = json.loads(_TRACE_FIXTURE.read_text(encoding="utf-8"))
    message = doc["choices"][0]["message"]
    assert not message.get("refusal"), "trace row returned a refusal — not gradeable"
    verdict, detail = _grade_ranking(message.get("content") or "")
    assert verdict != "rejected", (
        f"trace ranking REJECTED ({detail}) — the zero-rejected-responses / "
        "no-fabricated-id predicate fails"
    )
    assert verdict == "pass", (
        f"trace model ranked {detail!r} above the load-bearing candidate — the "
        "expected-candidate-ranked predicate fails"
    )
    print(  # noqa: T201 — operator verdict line
        f"\n[trace admission: PASS — gpt-5.6-luna ranked {detail!r} first]"
    )
