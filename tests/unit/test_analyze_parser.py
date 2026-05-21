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
        "evidence_tier": "judged",
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


# ---------------------------------------------------------------------------
# Spec §6 commit-3 — producer-admission checks (finding_type / evidence_tier /
# query_match_id / trace stub)
# ---------------------------------------------------------------------------


def _build_response_json(*proposals: dict[str, object]) -> str:
    """Wrap proposal dicts into a JSON `AnalyzeResponseRaw` shape."""
    import json

    return json.dumps({"findings": list(proposals)})


def _minimal_proposal(**overrides: object) -> dict[str, object]:
    """Minimum-fields raw proposal for parser tests."""
    base: dict[str, object] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "span": {"byte_start": 0, "byte_end": 1},
    }
    base.update(overrides)
    return base


def _build_scope_unit(*, byte_start: int = 0, byte_end: int = 100):  # type: ignore[no-untyped-def]
    """Minimal valid `ScopeUnit` for parser tests; byte range chosen
    to contain the `_minimal_proposal` default span `(0, 1)`."""
    from outrider.ast_facts.models import ScopeUnit, compute_unit_id

    return ScopeUnit(
        unit_id=compute_unit_id("src/x.py", kind="function", qualified_name="some_function"),
        kind="function",
        name="some_function",
        qualified_name="some_function",
        file_path="src/x.py",
        line_start=1,
        line_end=10,
        byte_start=byte_start,
        byte_end=byte_end,
    )


def test_step3_rejects_unknown_evidence_tier() -> None:
    """Raw evidence_tier outside `EvidenceTier` enum → rejection with
    `evidence_tier_not_in_enum` and `claimed_evidence_tier=None` (the
    bidirectional cross-field rule)."""
    response = _build_response_json(_minimal_proposal(evidence_tier="bogus_tier"))
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "evidence_tier_not_in_enum"
    assert rej.claimed_evidence_tier is None
    assert result.counters.n_proposals_seen == 1
    assert result.counters.n_proposals_rejected == 1
    assert result.admitted_findings == ()


def test_step2_rejects_unknown_finding_type_with_parsed_tier() -> None:
    """Raw finding_type outside `FindingType` enum → rejection with
    `finding_type_not_in_enum` and `claimed_evidence_tier` carrying
    the parsed enum value (the implementation-order swap ensures
    evidence_tier was admitted first)."""
    response = _build_response_json(
        _minimal_proposal(finding_type="unknown_type", evidence_tier="judged")
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "finding_type_not_in_enum"
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED


def test_step4_observed_rejects_when_query_match_id_absent() -> None:
    """OBSERVED proposal without a `query_match_id` → producer-
    admission rejection with `query_match_id_not_in_registry`,
    `claimed_evidence_tier=OBSERVED`."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="observed", query_match_id=None)
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "query_match_id_not_in_registry"
    assert rej.claimed_evidence_tier == EvidenceTier.OBSERVED
    assert rej.rejection_detail == "<absent>"


def test_step4_observed_rejects_fabricated_query_match_id() -> None:
    """OBSERVED proposal claiming an id NOT in the pre-supplied
    registry set → rejection; the claimed id is the rejection_detail
    (safe because raw schema constrains the pattern + length)."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id="fabricated_id",
        )
    )
    result = _call_parser(response, query_match_id_set=frozenset({"real_id"}))
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "query_match_id_not_in_registry"
    assert rej.rejection_detail == "fabricated_id"


def test_step4_observed_passes_when_query_match_id_in_registry() -> None:
    """Happy path: OBSERVED proposal with a real registry id passes
    producer admission. Span admission is supplied with a real scope
    unit containing the span; the proposal then reaches the step-5+
    `NotImplementedError` (ReviewFinding construction), confirming
    producer admission passed."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id="real_id",
        )
    )
    with pytest.raises(NotImplementedError, match=r"ReviewFinding construction for findings\[0\]"):
        _call_parser(
            response,
            query_match_id_set=frozenset({"real_id"}),
            included_scope_units=(_build_scope_unit(),),
        )


def test_step4_inferred_always_rejects_in_v1_stub() -> None:
    """V1 stub per spec §6 step 4: until the trace-node spec lands
    the resolver, every INFERRED is `Unwalkable` and gets
    `trace_path_not_admissible`."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="inferred",
            trace_path=("some.symbol", "step.two"),
        )
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "trace_path_not_admissible"
    assert rej.claimed_evidence_tier == EvidenceTier.INFERRED
    assert "V1 stub" in rej.rejection_detail


def test_step4_judged_skips_producer_admission() -> None:
    """JUDGED is the only tier the model can claim unilaterally — no
    structural artifact required. Producer admission is skipped; with
    span admission supplied a containing scope unit, the proposal
    reaches the step-5+ `NotImplementedError`."""
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    with pytest.raises(NotImplementedError, match=r"ReviewFinding construction for findings\[0\]"):
        _call_parser(response, included_scope_units=(_build_scope_unit(),))


def test_all_rejected_returns_aggregate_result() -> None:
    """Multiple proposals all rejected — each on a different reason —
    aggregate counters reflect the per-proposal outcomes; result is
    well-formed (no admitted findings, no trace candidates yet)."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="bogus_tier"),  # rejected at §3
        _minimal_proposal(finding_type="unknown_type"),  # rejected at §2
        _minimal_proposal(evidence_tier="inferred"),  # V1 stub rejection at §4
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 3
    reasons = {rej.rejection_reason for rej in result.proposal_rejections}
    assert reasons == {
        "evidence_tier_not_in_enum",
        "finding_type_not_in_enum",
        "trace_path_not_admissible",
    }
    assert result.counters.n_proposals_seen == 3
    assert result.counters.n_proposals_rejected == 3
    assert result.counters.n_findings_emitted == 0
    assert result.admitted_findings == ()


def test_proposal_rejection_uses_canonical_proposal_hash() -> None:
    """Every rejection's `proposal_hash` is the canonical hash from
    `compute_proposal_hash` — same recipe whether the proposal is
    admitted or rejected, so the join with `TraceCandidate.source_proposal_hash`
    works across both outcomes."""
    from outrider.policy.canonical import compute_proposal_hash

    response = _build_response_json(
        _minimal_proposal(finding_type="unknown_type", evidence_tier="judged")
    )
    result = _call_parser(response, file_path="src/example.py")
    rej = result.proposal_rejections[0]
    expected = compute_proposal_hash(
        source_file_path="src/example.py",
        finding_type="unknown_type",
        evidence_tier="judged",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        byte_start=0,
        byte_end=1,
    )
    assert rej.proposal_hash == expected


# ---------------------------------------------------------------------------
# Spec §6 commit-4 — span admission (per-outcome branch)
# ---------------------------------------------------------------------------


def test_step5_clean_span_inside_scope_unit_passes_admission() -> None:
    """Clean-outcome happy path: the proposal's span lies inside one of
    the file's included scope units → span admission passes → the
    proposal reaches the step-6+ `NotImplementedError` (ReviewFinding
    construction)."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 10, "byte_end": 30}))
    scope = _build_scope_unit(byte_start=0, byte_end=100)  # contains (10, 30)
    with pytest.raises(NotImplementedError, match=r"ReviewFinding construction"):
        _call_parser(response, included_scope_units=(scope,))


def test_step5_clean_span_outside_all_scope_units_rejects() -> None:
    """Clean outcome: span doesn't land in any included scope unit →
    `span_outside_scope_unit` rejection with `claimed_evidence_tier`
    carrying the parsed enum value."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            span={"byte_start": 200, "byte_end": 220},
        )
    )
    scope = _build_scope_unit(byte_start=0, byte_end=100)  # excludes (200, 220)
    result = _call_parser(response, included_scope_units=(scope,))
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_scope_unit"
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED
    assert rej.rejection_detail == "(200,220)"


def test_step5_clean_multiple_scope_units_admits_if_any_contains() -> None:
    """Clean outcome: span landing in the second scope unit of a tuple
    still admits — the `any(...)` covers every included unit."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 150, "byte_end": 160}))
    scope_a = _build_scope_unit(byte_start=0, byte_end=100)  # excludes
    scope_b = _build_scope_unit(byte_start=120, byte_end=200)  # contains
    with pytest.raises(NotImplementedError, match=r"ReviewFinding construction"):
        _call_parser(response, included_scope_units=(scope_a, scope_b))


def test_step5_clean_empty_scope_units_rejects_every_proposal() -> None:
    """Edge case: a clean-outcome call with no included scope units
    rejects every proposal at `span_outside_scope_unit`. (The node
    body shouldn't make this call — outcome would be
    NO_CHANGED_SCOPE_UNITS in that case — but the parser handles it
    defensively.)"""
    response = _build_response_json(_minimal_proposal())
    result = _call_parser(response, included_scope_units=())
    assert len(result.proposal_rejections) == 1
    assert result.proposal_rejections[0].rejection_reason == "span_outside_scope_unit"


def test_step5_degraded_span_within_file_passes_admission() -> None:
    """Degraded-outcome happy path: span lies within file bounds →
    span admission passes → reaches step-6+ NIE. No scope units
    consulted in degraded mode."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 50, "byte_end": 80}))
    with pytest.raises(NotImplementedError, match=r"ReviewFinding construction"):
        _call_parser(response, degraded_mode=True, file_byte_length=100)


def test_step5_degraded_span_past_eof_rejects() -> None:
    """Degraded outcome: `span.byte_end > file_byte_length` →
    `span_outside_file` rejection with `claimed_evidence_tier`
    carrying the parsed enum."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 90, "byte_end": 150}))
    result = _call_parser(response, degraded_mode=True, file_byte_length=100)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_file"
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED
    assert rej.rejection_detail == "(90,150)"


def test_step5_degraded_does_not_consult_scope_units() -> None:
    """Degraded outcome ignores `included_scope_units` — the deterministic
    bound is `span_within_file`, not `span_within_scope_unit`. Even an
    empty scope-unit tuple doesn't change the outcome when the span is
    within file bounds."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 50, "byte_end": 80}))
    with pytest.raises(NotImplementedError, match=r"ReviewFinding construction"):
        _call_parser(
            response,
            degraded_mode=True,
            file_byte_length=100,
            included_scope_units=(),  # ignored in degraded mode
        )


def test_step5_degraded_rejects_independent_of_query_match_id() -> None:
    """Degraded outcome: even if the model claimed `observed` with an
    id, the producer-admission step rejects FIRST (no registry-fired
    set in degraded mode). The span admission step never runs for an
    OBSERVED claim in degraded mode. This pin documents the rejection
    ordering."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id="some_id",
            span={"byte_start": 50, "byte_end": 80},
        )
    )
    # Empty query_match_id_set (typical for degraded mode) → producer
    # admission rejects with query_match_id_not_in_registry, not
    # span_outside_file.
    result = _call_parser(
        response,
        degraded_mode=True,
        file_byte_length=100,
        query_match_id_set=frozenset(),
    )
    assert len(result.proposal_rejections) == 1
    assert result.proposal_rejections[0].rejection_reason == "query_match_id_not_in_registry"


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
