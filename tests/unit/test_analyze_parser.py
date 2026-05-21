"""Analyze parser contract + step-by-step behavior tests.

Pins the §6 parser module: the public surface (frozen dataclasses +
signatures) AND the admission flow as each step lands. Sections marked
by spec §6 step number.

Earlier scaffolding sections (frozen+slots discipline, signature pin,
field sets, audit-context exclusion, module `__all__`) stay locked.
The "parser admission flow not implemented" `NotImplementedError`
guard from the scaffolding pass is replaced as steps 0+ land — current
state implements step 0 (response parse + response-level rejection)
and a NotImplementedError fence for non-empty findings until step 1
(proposal iteration) lands.
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


# ---------------------------------------------------------------------------
# Spec §6 step 0 — response parse + response-level rejection
# ---------------------------------------------------------------------------


def _call_parser(response_text: str, **overrides: object) -> ParserResult:
    """Convenience: invoke `parse_analyze_response` with sane defaults
    for the per-file kwargs that don't matter for step-0 tests. Overrides
    supplied per-test."""
    defaults: dict[str, object] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "file_path": "src/x.py",
        "file_content": "",
        "file_byte_length": 0,
        "included_scope_units": (),
        "query_match_id_set": frozenset(),
        "degraded_mode": False,
        "active_policy_version": "1.0.0",
        "pass_index": 0,
    }
    defaults.update(overrides)
    return parse_analyze_response(response_text, **defaults)  # type: ignore[arg-type]


def test_step0_malformed_json_returns_response_rejection() -> None:
    """Raw response that isn't valid JSON at all fails
    `AnalyzeResponseRaw.model_validate_json` with a JSON-decode error.
    Parser must record the response-level rejection without raising."""
    result = _call_parser("not even json {{{")
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"
    assert result.admitted_findings == ()
    assert result.trace_candidates == ()
    assert result.proposal_rejections == ()


def test_step0_valid_json_wrong_shape_returns_response_rejection() -> None:
    """Valid JSON but wrong shape (missing required `findings` key) also
    fails parsing and routes to the response-rejection path."""
    result = _call_parser('{"not_findings": []}')
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"


def test_step0_response_hash_is_full_text_not_truncated() -> None:
    """`compute_response_hash` hashes the FULL response text (no
    truncation cap) per the round-3 audit fix. Two responses that
    differ only past 8 KiB still produce distinct hashes."""
    import hashlib

    response_text = "x" * 10000  # > 8 KiB; clearly malformed JSON too
    result = _call_parser(response_text)
    assert result.response_rejection is not None
    expected = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    assert result.response_rejection.response_hash == expected


def test_step0_response_rejection_file_path_matches_input() -> None:
    """The rejection payload's `file_path` is the one analyze passed in
    (already canonicalized at intake). Parser doesn't re-canonicalize."""
    result = _call_parser("malformed", file_path="src/path/to/file.py")
    assert result.response_rejection is not None
    assert result.response_rejection.file_path == "src/path/to/file.py"


def test_step0_rejection_detail_does_not_contain_response_text() -> None:
    """Per `DECISIONS.md#014` point 1: audit rows must not carry user
    code or prompt/completion content. The `rejection_detail` formatter
    must NEVER include the response text. Send a response with a
    distinctive sentinel and verify the sentinel does not appear."""
    sentinel = "SENSITIVE_CONTENT_DO_NOT_LEAK_42"
    result = _call_parser(f'{{"findings": [{{"evidence": "{sentinel}"}}]}}')
    assert result.response_rejection is not None
    detail = result.response_rejection.rejection_detail
    assert sentinel not in detail, f"rejection_detail leaked response text. Detail: {detail!r}"


def test_step0_rejection_detail_is_within_field_max_length() -> None:
    """`AnalyzeResponseRejectedEvent.rejection_detail` is
    `Field(max_length=500)`. The formatter must respect this even
    when many errors fire — pathological case is a response with many
    findings each carrying many invalid fields. Truncate with `"..."`
    marker."""
    # Construct a response with many findings, each missing many fields.
    bogus_finding = (
        '{"finding_type": null, "evidence_tier": null, "title": null, '
        '"description": null, "evidence": null, "span": null}'
    )
    response = '{"findings": [' + ",".join([bogus_finding] * 50) + "]}"
    result = _call_parser(response)
    assert result.response_rejection is not None
    assert len(result.response_rejection.rejection_detail) <= 500


def test_step0_rejection_detail_uses_json_pointer_format() -> None:
    """Format pin per spec §3: `findings[0].finding_type x1, ...`. An
    index segment attaches to its parent with `[N]`; a field segment
    joins with `.`. Pin the exact shape so the dashboard renderer can
    parse it deterministically."""
    # A response where findings[0] has a missing required field — produces
    # an error with location ("findings", 0, "<field>").
    result = _call_parser('{"findings": [{}]}')
    assert result.response_rejection is not None
    detail = result.response_rejection.rejection_detail
    # At minimum, the format must show `findings[0].<something>` shape
    assert "findings[0]" in detail, (
        f"rejection_detail does not use JSON-pointer format. Got: {detail!r}"
    )


def test_step0_response_rejection_counters() -> None:
    """On response-level rejection: every counter is zero except
    `n_responses_rejected == 1`. The node body sums per-file counters
    into the per-pass `AnalyzeCompletedEvent`."""
    result = _call_parser("malformed")
    assert result.counters.n_proposals_seen == 0
    assert result.counters.n_findings_emitted == 0
    assert result.counters.n_proposals_rejected == 0
    assert result.counters.n_responses_rejected == 1
    assert result.counters.n_trace_candidates_emitted == 0


def test_step0_valid_empty_findings_returns_clean_zero_result() -> None:
    """Valid JSON with `findings: []` is the trivial happy path —
    parsing succeeded, no proposals to process, zero counters across
    the board, no rejections."""
    result = _call_parser('{"findings": []}')
    assert result.response_rejection is None
    assert result.admitted_findings == ()
    assert result.trace_candidates == ()
    assert result.proposal_rejections == ()
    assert result.counters.n_proposals_seen == 0
    assert result.counters.n_findings_emitted == 0
    assert result.counters.n_proposals_rejected == 0
    assert result.counters.n_responses_rejected == 0
    assert result.counters.n_trace_candidates_emitted == 0


# ---------------------------------------------------------------------------
# Spec §6 commit-2 scaffold — proposal iteration + rejection-payload helper
# (no admission decisions yet; commit 3 wires the admission checks)
# ---------------------------------------------------------------------------


def _build_raw_proposal(**overrides: object):  # type: ignore[no-untyped-def]
    """Build a minimal valid AnalyzeFindingProposalRaw for helper tests."""
    from outrider.ast_facts.models import Span
    from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw

    defaults: dict[str, object] = {
        "finding_type": "sql_injection",
        "evidence_tier": "JUDGED",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "span": Span(byte_start=0, byte_end=10),
    }
    defaults.update(overrides)
    return AnalyzeFindingProposalRaw(**defaults)  # type: ignore[arg-type]


def test_build_proposal_rejection_populates_identity_fields() -> None:
    """`_build_proposal_rejection` computes `proposal_hash` via the
    canonical wrapper and `claimed_finding_type_hash` + `_len` per
    `DECISIONS.md#014`. Caller supplies the branch-specific fields."""
    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal()
    rej = _build_proposal_rejection(
        raw,
        file_path="src/x.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="no_near_enum_match",
        claimed_evidence_tier=EvidenceTier.JUDGED,
    )
    assert len(rej.proposal_hash) == 64  # SHA-256 hex
    assert len(rej.claimed_finding_type_hash) == 16  # short prefix
    assert rej.claimed_finding_type_len == len("sql_injection")
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED
    assert rej.rejection_reason == "finding_type_not_in_enum"
    assert rej.rejection_detail == "no_near_enum_match"
    assert rej.file_path == "src/x.py"


def test_build_proposal_rejection_claimed_evidence_tier_can_be_none() -> None:
    """For `rejection_reason="evidence_tier_not_in_enum"`, no parsed
    enum exists, so `claimed_evidence_tier=None` is the valid shape.
    The lifted event's cross-field validator enforces this iff."""
    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal(evidence_tier="WRONG_TIER_VALUE")
    rej = _build_proposal_rejection(
        raw,
        file_path="src/x.py",
        rejection_reason="evidence_tier_not_in_enum",
        rejection_detail="no_near_enum_match",
        claimed_evidence_tier=None,
    )
    assert rej.claimed_evidence_tier is None


def test_build_proposal_rejection_hash_canonicalizes_path() -> None:
    """`compute_proposal_hash` wrapper canonicalizes `source_file_path`
    via `validate_diff_path` BEFORE folding. Alias paths (`src/foo.py`
    vs `./src/foo.py`) MUST produce identical `proposal_hash`."""
    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal()
    rej_canonical = _build_proposal_rejection(
        raw,
        file_path="src/foo.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="",
        claimed_evidence_tier=EvidenceTier.JUDGED,
    )
    rej_alias = _build_proposal_rejection(
        raw,
        file_path="./src/foo.py",  # same logical file
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="",
        claimed_evidence_tier=EvidenceTier.JUDGED,
    )
    assert rej_canonical.proposal_hash == rej_alias.proposal_hash


def test_build_proposal_rejection_hash_distinct_across_files() -> None:
    """Per `DECISIONS.md#022`: proposal identity is PR/file-scoped.
    Same proposal shape from two DIFFERENT files produces DISTINCT
    hashes."""
    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal()
    rej_a = _build_proposal_rejection(
        raw,
        file_path="src/foo.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="",
        claimed_evidence_tier=EvidenceTier.JUDGED,
    )
    rej_b = _build_proposal_rejection(
        raw,
        file_path="src/bar.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="",
        claimed_evidence_tier=EvidenceTier.JUDGED,
    )
    assert rej_a.proposal_hash != rej_b.proposal_hash


def test_build_proposal_rejection_claimed_finding_type_hash_matches_recipe() -> None:
    """The hash recipe is pinned: sha256(raw.finding_type.encode())[:16].
    Drift here would mean event-side and parser-side claims disagree."""
    import hashlib

    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal(finding_type="some_bogus_type")
    rej = _build_proposal_rejection(
        raw,
        file_path="src/x.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="",
        claimed_evidence_tier=None,
    )
    expected = hashlib.sha256(b"some_bogus_type").hexdigest()[:16]
    assert rej.claimed_finding_type_hash == expected


def test_step1_iteration_raises_not_implemented_with_index() -> None:
    """Until the admission checks land (commit 3), iteration over
    `raw.findings` raises `NotImplementedError` with the proposal
    index in the message. The next commit replaces the body with
    per-proposal admission decisions."""
    response = (
        '{"findings": [{"finding_type": "sql_injection", '
        '"evidence_tier": "JUDGED", "title": "t", "description": "d", '
        '"evidence": "e", "span": {"byte_start": 0, "byte_end": 1}}]}'
    )
    with pytest.raises(NotImplementedError, match=r"findings\[0\]"):
        _call_parser(response)


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
