"""Analyze parser scaffolding contract tests.

Pins the §6 parser module's public surface BEFORE the admission flow
lands. Subsequent implementation commits replace the
`NotImplementedError` body with the spec's 10-step flow; the dataclass
shapes + the `parse_analyze_response(...)` signature stay locked.

Coverage:
- Each frozen dataclass refuses positional unpacking and post-
  construction mutation (swap-impossibility per the triage M1 rationale).
- `ParserResult` carries the documented field set with the expected
  types (admitted findings + trace candidates + proposal rejections +
  optional response rejection + counters).
- `ParserCounters` carries the documented counter set.
- `ProposalRejection` and `ResponseRejection` carry the documented
  payload sets (audit-context fields like `review_id` / `event_id` /
  `timestamp` are deliberately ABSENT — added by the node body at
  lift time).
- `parse_analyze_response` has the documented signature: one
  positional `response_text` arg + the documented keyword-only set.
- Calling `parse_analyze_response` with valid kwargs raises
  `NotImplementedError` with the documented message.
- Module `__all__` exports the expected surfaces.
"""

import dataclasses
import inspect
from uuid import uuid4

import pytest

from outrider.agent.nodes.analyze_parser import (
    ParserCounters,
    ParserResult,
    ProposalRejection,
    ResponseRejection,
    parse_analyze_response,
)
from outrider.policy.findings import EvidenceTier

# ---------------------------------------------------------------------------
# Frozen + slots discipline
# ---------------------------------------------------------------------------


def test_proposal_rejection_rejects_positional_unpacking() -> None:
    """ProposalRejection is a dataclass, not a tuple — positional unpack
    raises `TypeError`. Same swap-impossibility discipline as
    `AnalyzePromptParts`."""
    rej = ProposalRejection(
        proposal_hash="a" * 64,
        file_path="src/x.py",
        claimed_finding_type_hash="b" * 16,
        claimed_finding_type_len=10,
        claimed_evidence_tier=EvidenceTier.OBSERVED,
        rejection_reason="query_match_id_not_in_registry",
        rejection_detail="",
    )
    with pytest.raises(TypeError):
        _, _ = rej  # type: ignore[misc]


def test_proposal_rejection_is_frozen() -> None:
    rej = ProposalRejection(
        proposal_hash="a" * 64,
        file_path="src/x.py",
        claimed_finding_type_hash="b" * 16,
        claimed_finding_type_len=10,
        claimed_evidence_tier=EvidenceTier.OBSERVED,
        rejection_reason="query_match_id_not_in_registry",
        rejection_detail="",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rej.proposal_hash = "tampered"  # type: ignore[misc]


def test_response_rejection_rejects_positional_unpacking() -> None:
    rej = ResponseRejection(
        file_path="src/x.py",
        response_hash="a" * 64,
        rejection_reason="raw_response_unparseable",
        rejection_detail="",
    )
    with pytest.raises(TypeError):
        _, _ = rej  # type: ignore[misc]


def test_response_rejection_is_frozen() -> None:
    rej = ResponseRejection(
        file_path="src/x.py",
        response_hash="a" * 64,
        rejection_reason="raw_response_unparseable",
        rejection_detail="",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rej.file_path = "tampered"  # type: ignore[misc]


def test_parser_counters_rejects_positional_unpacking() -> None:
    counters = ParserCounters(
        n_proposals_seen=0,
        n_findings_emitted=0,
        n_proposals_rejected=0,
        n_responses_rejected=0,
        n_trace_candidates_emitted=0,
    )
    with pytest.raises(TypeError):
        _, _ = counters  # type: ignore[misc]


def test_parser_counters_is_frozen() -> None:
    counters = ParserCounters(
        n_proposals_seen=0,
        n_findings_emitted=0,
        n_proposals_rejected=0,
        n_responses_rejected=0,
        n_trace_candidates_emitted=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        counters.n_proposals_seen = 999  # type: ignore[misc]


def test_parser_result_rejects_positional_unpacking() -> None:
    result = ParserResult(
        admitted_findings=(),
        trace_candidates=(),
        proposal_rejections=(),
        response_rejection=None,
        counters=ParserCounters(
            n_proposals_seen=0,
            n_findings_emitted=0,
            n_proposals_rejected=0,
            n_responses_rejected=0,
            n_trace_candidates_emitted=0,
        ),
    )
    with pytest.raises(TypeError):
        _, _ = result  # type: ignore[misc]


def test_parser_result_is_frozen() -> None:
    result = ParserResult(
        admitted_findings=(),
        trace_candidates=(),
        proposal_rejections=(),
        response_rejection=None,
        counters=ParserCounters(
            n_proposals_seen=0,
            n_findings_emitted=0,
            n_proposals_rejected=0,
            n_responses_rejected=0,
            n_trace_candidates_emitted=0,
        ),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.admitted_findings = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dataclass field sets — pin what the parser produces, NOT what the
# audit events carry. Audit-context fields (review_id, event_id,
# timestamp, sequence_number, is_eval, node_id, event_type) MUST be
# absent from the parser payloads — they're added by the node body
# when lifting to event objects.
# ---------------------------------------------------------------------------


def test_proposal_rejection_field_set() -> None:
    """The parser-owned subset of FindingProposalRejectedEvent fields.
    Audit-context fields are deliberately omitted — the node body
    adds them at lift time."""
    fields = {f.name for f in dataclasses.fields(ProposalRejection)}
    expected = {
        "proposal_hash",
        "file_path",
        "claimed_finding_type_hash",
        "claimed_finding_type_len",
        "claimed_evidence_tier",
        "rejection_reason",
        "rejection_detail",
    }
    assert fields == expected


def test_proposal_rejection_omits_audit_context_fields() -> None:
    """Defense-in-depth: an audit-context field landing on the parser
    payload (e.g., the parser sets `event_id` itself) breaks the
    "parser is pure" boundary. Explicit exclusion list catches a
    future commit that adds them silently."""
    fields = {f.name for f in dataclasses.fields(ProposalRejection)}
    audit_context = {
        "event_id",
        "review_id",
        "timestamp",
        "sequence_number",
        "is_eval",
        "node_id",
        "event_type",
    }
    leakage = fields & audit_context
    assert leakage == set(), (
        f"ProposalRejection must not carry audit-context fields "
        f"(those are added at node-body lift time). Found: {leakage}"
    )


def test_response_rejection_field_set() -> None:
    fields = {f.name for f in dataclasses.fields(ResponseRejection)}
    expected = {
        "file_path",
        "response_hash",
        "rejection_reason",
        "rejection_detail",
    }
    assert fields == expected


def test_response_rejection_omits_audit_context_fields() -> None:
    fields = {f.name for f in dataclasses.fields(ResponseRejection)}
    audit_context = {
        "event_id",
        "review_id",
        "timestamp",
        "sequence_number",
        "is_eval",
        "node_id",
        "event_type",
    }
    leakage = fields & audit_context
    assert leakage == set()


def test_parser_counters_field_set() -> None:
    fields = {f.name for f in dataclasses.fields(ParserCounters)}
    expected = {
        "n_proposals_seen",
        "n_findings_emitted",
        "n_proposals_rejected",
        "n_responses_rejected",
        "n_trace_candidates_emitted",
    }
    assert fields == expected


def test_parser_result_field_set() -> None:
    fields = {f.name for f in dataclasses.fields(ParserResult)}
    expected = {
        "admitted_findings",
        "trace_candidates",
        "proposal_rejections",
        "response_rejection",
        "counters",
    }
    assert fields == expected


# ---------------------------------------------------------------------------
# parse_analyze_response signature pin
# ---------------------------------------------------------------------------


def test_parse_analyze_response_signature() -> None:
    """The parser's public signature is locked at scaffolding time.
    Implementation commits land the body; a refactor that changes
    parameter names or kw-only-ness breaks call sites and is caught
    here at scaffolding-test time."""
    sig = inspect.signature(parse_analyze_response)
    params = sig.parameters

    # response_text is positional-or-keyword (the only positional input).
    assert "response_text" in params
    assert params["response_text"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    # Every other parameter is keyword-only (the * separator after
    # response_text). Same misuse-resistance pattern as
    # compute_cost_usd, coordinates.tree_sitter_to_github — same-typed
    # parameters can't get swapped at the call site.
    expected_kwonly = {
        "review_id",
        "installation_id",
        "file_path",
        "file_content",
        "file_byte_length",
        "included_scope_units",
        "query_match_id_set",
        "degraded_mode",
        "active_policy_version",
        "pass_index",
    }
    kwonly = {name for name, p in params.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    assert kwonly == expected_kwonly, (
        f"parse_analyze_response keyword-only parameter set drifted. "
        f"Expected: {expected_kwonly}. Got: {kwonly}."
    )


def test_parse_analyze_response_does_not_take_audit_persister() -> None:
    """The parser is a pure function — no IO. An `audit_persister` (or
    any sink/persister-like) parameter would re-introduce the IO that
    the boundary discipline keeps out of the parser. Explicit
    not-present check catches a future commit that adds it."""
    sig = inspect.signature(parse_analyze_response)
    forbidden = {"audit_persister", "persister", "audit_emitter", "sink"}
    found = set(sig.parameters.keys()) & forbidden
    assert found == set(), (
        f"parse_analyze_response must not take a persister/sink/emitter "
        f"parameter (parser is pure; node body owns persistence). "
        f"Found: {found}."
    )


def test_parse_analyze_response_returns_parser_result() -> None:
    """Return annotation pin — implementation commits must keep the
    return type stable. `from __future__ import annotations` makes
    the annotation a string; check the string identity directly.
    (Runtime resolution via `typing.get_type_hints` fails because
    the parser module keeps `ReviewFinding`/`TraceCandidate`/etc.
    inside a `TYPE_CHECKING` block to avoid runtime imports of pure-
    annotation deps.)"""
    sig = inspect.signature(parse_analyze_response)
    assert sig.return_annotation == "ParserResult"


def test_parse_analyze_response_raises_not_implemented_at_scaffolding_stage() -> None:
    """Until the admission flow lands, calling the parser raises
    `NotImplementedError` with a stable message. The first
    implementation commit deletes this test (or rewrites it to
    exercise a real input)."""
    with pytest.raises(NotImplementedError, match="parser admission flow not implemented"):
        parse_analyze_response(
            "irrelevant",
            review_id=uuid4(),
            installation_id=12345,
            file_path="src/x.py",
            file_content="",
            file_byte_length=0,
            included_scope_units=(),
            query_match_id_set=frozenset(),
            degraded_mode=False,
            active_policy_version="1.0.0",
            pass_index=0,
        )


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exports_all_documented_surfaces() -> None:
    from outrider.agent.nodes import analyze_parser

    expected = {
        "ParserCounters",
        "ParserResult",
        "ProposalRejection",
        "ResponseRejection",
        "parse_analyze_response",
    }
    assert set(analyze_parser.__all__) == expected
