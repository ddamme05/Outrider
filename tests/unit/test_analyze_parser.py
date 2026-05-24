"""Analyze parser contract + step-by-step behavior tests.

Pins the §6 parser module: the public surface (frozen dataclasses +
signatures) AND the admission flow as each step lands. Sections marked
by spec §6 step number.

Sections (in spec §6 step order): scaffolding (frozen+slots discipline,
signature pin, field sets, audit-context exclusion, module `__all__`),
step 0 (response parse + response-level rejection), commit-2 helper
unit tests (proposal_hash + rejection-payload helper), commit-3 +
commit-4 admission tests (enum + producer + span), commit-5 finding
construction + trace-candidate collection, commit-6 audit-fold
regression pins.
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
        "n_trace_candidates_dropped_malformed",
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
    truncation cap). Two responses that differ only past 8 KiB still
    produce distinct hashes."""
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
# Fenced-JSON envelope tolerance (regression for the silent-coverage-hole
# bug: Sonnet wrapping its response in ```json...``` was producing a
# `raw_response_unparseable` ResponseRejection — analyze degraded
# gracefully but every fenced response lost ALL its findings silently)
# ---------------------------------------------------------------------------


def test_fenced_json_response_is_parsed_not_rejected() -> None:
    """Sonnet sometimes wraps the response in ```json...``` despite the
    "no markdown fences" prompt instruction. With `strip_outer_json_fence`
    applied before model_validate_json, the fenced response is parsed
    normally — `response_rejection` stays None, the file's findings
    (here: zero) flow through admission as usual. Previously this path
    produced a ResponseRejection and the file's true findings were lost."""
    inner = '{"findings": []}'
    fenced = f"```json\n{inner}\n```"
    result = _call_parser(fenced)
    assert result.response_rejection is None, (
        "fenced response was treated as raw_response_unparseable — the "
        "fence-stripping normalizer regressed; analyze silently loses "
        "every file whose response is fenced"
    )
    assert result.admitted_findings == ()
    assert result.counters.n_responses_rejected == 0


def test_bare_fenced_json_response_is_parsed_not_rejected() -> None:
    """The bare-fence shape (```\\n...\\n``` without `json` tag) is also
    tolerated."""
    inner = '{"findings": []}'
    fenced = f"```\n{inner}\n```"
    result = _call_parser(fenced)
    assert result.response_rejection is None
    assert result.counters.n_responses_rejected == 0


def test_malformed_fence_still_routes_to_response_rejection() -> None:
    """A wrapper missing its closing fence is malformed — the helper
    falls through unchanged, the existing try/except routes the failure
    to the response-rejection path. Pins the policy boundary so the
    fence helper stays narrow: it tolerates ONE well-formed wrapper,
    not arbitrary recovery."""
    inner = '{"findings": []}'
    malformed = f"```json\n{inner}"  # opener but no closer
    result = _call_parser(malformed)
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"


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


def test_build_proposal_rejection_preserves_caller_proposal_hash() -> None:
    """Per commit 5's refactor: `_build_proposal_rejection` no longer
    computes `proposal_hash` internally — the caller passes it (so the
    same hash feeds both the rejection payload AND
    `_collect_trace_candidates_for`'s `source_proposal_hash`). The
    helper preserves the supplied value verbatim."""
    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal()
    sentinel_hash = "a" * 64
    rej = _build_proposal_rejection(
        raw,
        proposal_hash=sentinel_hash,
        file_path="src/x.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="no_near_enum_match",
        claimed_evidence_tier=EvidenceTier.JUDGED,
    )
    assert rej.proposal_hash == sentinel_hash
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
        proposal_hash="a" * 64,
        file_path="src/x.py",
        rejection_reason="evidence_tier_not_in_enum",
        rejection_detail="no_near_enum_match",
        claimed_evidence_tier=None,
    )
    assert rej.claimed_evidence_tier is None


def test_build_proposal_rejection_claimed_finding_type_hash_matches_recipe() -> None:
    """The hash recipe is pinned: sha256(raw.finding_type.encode())[:16].
    Drift here would mean event-side and parser-side claims disagree."""
    import hashlib

    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal(finding_type="some_bogus_type")
    rej = _build_proposal_rejection(
        raw,
        proposal_hash="a" * 64,
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
    producer admission AND span admission (containing scope unit
    supplied), reaches `ReviewFinding` construction, and is admitted."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id="real_id",
        )
    )
    result = _call_parser(
        response,
        query_match_id_set=frozenset({"real_id"}),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,  # provides line context for span_to_line_range
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1
    finding = result.admitted_findings[0]
    assert finding.evidence_tier == EvidenceTier.OBSERVED
    assert finding.query_match_id == "real_id"


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
    a containing scope unit supplied for span admission, the proposal
    reaches `ReviewFinding` construction and is admitted."""
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1
    assert result.admitted_findings[0].evidence_tier == EvidenceTier.JUDGED


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
    proposal is admitted with `line_start`/`line_end` derived from
    `coordinates.span_to_line_range`."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 10, "byte_end": 30}))
    scope = _build_scope_unit(byte_start=0, byte_end=100)  # contains (10, 30)
    result = _call_parser(
        response,
        included_scope_units=(scope,),
        file_content="x" * 200,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


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
    result = _call_parser(
        response,
        included_scope_units=(scope_a, scope_b),
        file_content="x" * 300,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


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
    """Degraded-outcome happy path: span lies within file bounds → span
    admission passes → finding admitted. No scope units consulted in
    degraded mode."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 50, "byte_end": 80}))
    result = _call_parser(
        response,
        degraded_mode=True,
        file_byte_length=100,
        file_content="x" * 100,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


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
    empty scope-unit tuple still admits when the span is within file
    bounds."""
    response = _build_response_json(_minimal_proposal(span={"byte_start": 50, "byte_end": 80}))
    result = _call_parser(
        response,
        degraded_mode=True,
        file_byte_length=100,
        file_content="x" * 100,
        included_scope_units=(),  # ignored in degraded mode
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


def test_step5_clean_rejects_zero_width_span() -> None:
    """Clean outcome: a zero-width span (`byte_start == byte_end`) anchors
    to no bytes, so it cannot serve as proof for any finding tier. Parser
    rejects with `span_outside_scope_unit` and `rejection_detail` carries
    the `zero_width:` prefix so the audit row distinguishes it from an
    EOF-overflow rejection on the same reason. `Span` itself admits
    zero-width (`byte_end >= byte_start`); the parser enforces the
    prompt's stricter `byte_start < byte_end` rule.
    """
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            span={"byte_start": 100, "byte_end": 100},
        )
    )
    # Scope unit covers the zero-width span's byte position; admission
    # rejects on the zero-width predicate alone, not on containment.
    scope = _build_scope_unit(byte_start=0, byte_end=200)
    result = _call_parser(response, included_scope_units=(scope,))
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_scope_unit"
    assert rej.rejection_detail.startswith("zero_width:")


def test_step5_degraded_rejects_zero_width_span() -> None:
    """Degraded outcome: same zero-width rejection on the `span_outside_file`
    rejection reason with the same `zero_width:` detail prefix. Span is
    well within file bounds; rejection fires on the zero-width predicate
    alone.
    """
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            span={"byte_start": 50, "byte_end": 50},
        )
    )
    result = _call_parser(
        response,
        degraded_mode=True,
        file_byte_length=100,
    )
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_file"
    assert rej.rejection_detail.startswith("zero_width:")


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
# Spec §6 commit-5 — ReviewFinding + TraceCandidate construction + counters
# ---------------------------------------------------------------------------


def test_step8_admitted_finding_severity_from_policy_table() -> None:
    """`ReviewFinding.severity` is set from `SEVERITY_POLICY[finding_type]`
    per `severity-set-by-policy` — never proposed by the model."""
    from outrider.policy.severity import SEVERITY_POLICY, FindingType

    response = _build_response_json(
        _minimal_proposal(finding_type="sql_injection", evidence_tier="judged")
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    finding = result.admitted_findings[0]
    assert finding.severity == SEVERITY_POLICY[FindingType.SQL_INJECTION]


def test_step8_admitted_finding_dimension_from_mapping_table() -> None:
    """`ReviewFinding.dimension` is set from `FINDING_TYPE_TO_DIMENSION[finding_type]`
    per `evidence-tier-schema-enforced` — never proposed by the model."""
    from outrider.policy.dimensions import FINDING_TYPE_TO_DIMENSION
    from outrider.policy.severity import FindingType

    response = _build_response_json(
        _minimal_proposal(finding_type="sql_injection", evidence_tier="judged")
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    finding = result.admitted_findings[0]
    assert finding.dimension == FINDING_TYPE_TO_DIMENSION[FindingType.SQL_INJECTION]


def test_step8_admitted_finding_policy_version_from_closure() -> None:
    """`ReviewFinding.policy_version` is the value passed into the
    parser via the node-body closure (analyze passes
    `active_policy_version=ACTIVE_POLICY_VERSION` by default)."""
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        active_policy_version="1.2.3",
    )
    finding = result.admitted_findings[0]
    assert finding.policy_version == "1.2.3"


def test_step8_admitted_finding_review_and_installation_ids() -> None:
    """`review_id` + `installation_id` flow through from caller."""
    from uuid import uuid4

    review_id = uuid4()
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        review_id=review_id,
        installation_id=99999,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    finding = result.admitted_findings[0]
    assert finding.review_id == review_id
    assert finding.installation_id == 99999


def test_step8_admitted_finding_content_hash_matches_recipe() -> None:
    """`content_hash` is `compute_finding_content_hash(file_path,
    line_start, line_end, finding_type)`. Mirror of the
    `ReviewFinding._verify_content_hash` validator — the parser
    constructs this canonically so construction never fails on the
    content-hash mismatch."""
    from outrider.audit.events import compute_finding_content_hash
    from outrider.policy.severity import FindingType

    response = _build_response_json(
        _minimal_proposal(
            finding_type="sql_injection",
            evidence_tier="judged",
            span={"byte_start": 0, "byte_end": 5},
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="line1\nline2\nline3\n",
        file_path="src/x.py",
    )
    finding = result.admitted_findings[0]
    expected = compute_finding_content_hash(
        file_path="src/x.py",
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=FindingType.SQL_INJECTION,
    )
    assert finding.content_hash == expected


def test_step8_line_range_derived_from_span() -> None:
    """`line_start`/`line_end` come from `coordinates.span_to_line_range`,
    not from the raw proposal. The model proposes a byte span; the
    parser deterministically translates to 1-indexed lines via the
    coordinate-translator boundary. Pins the exact mapping so a shift
    in `span_to_line_range` fails here, not silently downstream.

    `file_content = "line1\\nline2\\nline3\\n"` byte layout:
      bytes 0-4 "line1", byte 5 "\\n", bytes 6-10 "line2", byte 11 "\\n",
      bytes 12-16 "line3", byte 17 "\\n".
    Span byte_start=6, byte_end=10 is half-open; the covered bytes are
    6,7,8,9 ("line") — all on 1-indexed line 2.
    """
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            span={"byte_start": 6, "byte_end": 10},
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="line1\nline2\nline3\n",
    )
    finding = result.admitted_findings[0]
    # Exact mapping: bytes 6..9 are entirely on line 2.
    assert finding.line_start == 2
    assert finding.line_end == 2


def test_step10_trace_candidates_collected_from_admitted_proposal() -> None:
    """Step 10: trace_candidates are collected from admitted proposals
    AND stamped with the parent's `proposal_hash` as
    `source_proposal_hash`. `candidate_id` is content-derived via
    `compute_candidate_id`."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                {"import_string_raw": "other", "reason": "calls helper"},
            ],
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    assert len(result.trace_candidates) == 1
    parent_hash = result.admitted_findings[0]
    cand = result.trace_candidates[0]
    assert cand.import_string == "other"
    assert cand.reason == "calls helper"
    # source_proposal_hash links back to the parent (same recipe)
    from outrider.policy.canonical import compute_proposal_hash

    expected_parent_hash = compute_proposal_hash(
        source_file_path="src/x.py",
        finding_type="sql_injection",
        evidence_tier="judged",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        byte_start=0,
        byte_end=1,
    )
    assert cand.source_proposal_hash == expected_parent_hash
    _ = parent_hash  # admitted_findings doesn't expose proposal_hash; verified via recipe


def test_step10_trace_candidates_collected_from_rejected_proposal() -> None:
    """Per spec §6 step 10: trace_candidates are collected from BOTH
    admitted and proposal-level-REJECTED raw proposals. A rejected
    proposal's child candidates still surface — rejected
    `JUDGED`-claim might still flag a legitimate cross-file signal."""
    response = _build_response_json(
        _minimal_proposal(
            finding_type="unknown_type",  # rejects at finding_type_not_in_enum
            evidence_tier="judged",
            trace_candidates=[
                {"import_string_raw": "related", "reason": "should still surface"},
            ],
        )
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    assert len(result.admitted_findings) == 0
    # Trace candidate from the rejected proposal still appears
    assert len(result.trace_candidates) == 1
    assert result.trace_candidates[0].import_string == "related"


def test_step10_response_level_rejection_collects_no_trace_candidates() -> None:
    """Per spec §6 step 0: response-level rejections (parser step 0)
    have no proposals to collect from. Best-effort salvage from
    malformed JSON is forbidden per the clean-counter contract."""
    result = _call_parser("not valid json {{{")
    assert result.response_rejection is not None
    assert result.trace_candidates == ()


def test_counters_reflect_admitted_and_rejected_mix() -> None:
    """Mixed result: some proposals admitted, some rejected. Counters
    accurately reflect the per-proposal outcomes."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged"),  # admitted
        _minimal_proposal(evidence_tier="bogus_tier"),  # rejected at §3
        _minimal_proposal(evidence_tier="judged"),  # admitted
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert result.counters.n_proposals_seen == 3
    assert result.counters.n_findings_emitted == 2
    assert result.counters.n_proposals_rejected == 1
    assert result.counters.n_responses_rejected == 0
    assert result.counters.n_trace_candidates_emitted == 0


def test_counter_accounting_equation_holds() -> None:
    """`n_proposals_seen == n_findings_emitted + n_proposals_rejected`
    must hold at the parser layer (same equation the lifted
    `AnalyzeCompletedEvent._enforce_proposal_accounting` enforces at
    construction). Mix of admitted, rejected, and candidate-producing
    proposals."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged"),  # admitted
        _minimal_proposal(evidence_tier="inferred"),  # rejected (V1 stub)
        _minimal_proposal(finding_type="unknown_type"),  # rejected
        _minimal_proposal(evidence_tier="judged"),  # admitted
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert result.counters.n_proposals_seen == (
        result.counters.n_findings_emitted + result.counters.n_proposals_rejected
    )


# ---------------------------------------------------------------------------
# Spec §6 commit-6 — audit-fold regression pins
# ---------------------------------------------------------------------------


def test_hostile_import_string_does_not_crash_parser() -> None:
    """Sharp-edges H1 + general-purpose §4: a single hostile
    `import_string_raw` (e.g. path-shaped, shell metacharacter, Python
    keyword part) used to crash `parse_analyze_response` mid-loop
    because `TraceCandidate(...)` construction was outside any
    try/except. Fix: per-candidate try/except in
    `_collect_trace_candidates_for` drops the bad candidate; the
    parent proposal still produces its admission/rejection outcome.
    Post-DECISIONS.md#024 rename: hostile values are import-string-
    malformed instead of path-traversal-shaped, but the
    crash-resistance contract is identical."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                # Path separator — rejected by is_valid_import_string
                {"import_string_raw": "../../etc/passwd", "reason": "hostile"},
            ],
        )
    )
    # Must NOT raise — that was the bug
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1  # parent admitted
    assert len(result.trace_candidates) == 0  # hostile candidate dropped
    # Sharp-edges F1 audit-fold: the drop is counted, not silent.
    assert result.counters.n_trace_candidates_dropped_malformed == 1


def test_admitted_finding_carries_proposal_hash_from_compute_recipe() -> None:
    """Per DECISIONS.md#025: analyze's admission path threads proposal_hash
    from compute_proposal_hash through to ReviewFinding construction.
    The same value the rejected branch (FindingProposalRejectedEvent.proposal_hash)
    would emit appears on admitted findings — closes the trace-join contract."""
    from outrider.policy.canonical import compute_proposal_hash

    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    admitted = result.admitted_findings[0]

    # The expected proposal_hash matches the canonical recipe over the
    # admitted proposal's payload (same recipe the rejected branch uses).
    # Reading _minimal_proposal defaults: source_file_path=src/x.py,
    # finding_type=sql_injection, evidence_tier=judged, no query_match_id /
    # trace_path, title/description/evidence are "t"/"d"/"e", span 0-1.
    expected_hash = compute_proposal_hash(
        source_file_path="src/x.py",
        finding_type="sql_injection",
        evidence_tier="judged",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        byte_start=0,
        byte_end=1,
    )
    assert admitted.proposal_hash == expected_hash


def test_dropped_malformed_counter_increments_per_dropped_candidate() -> None:
    """Sharp-edges F1 audit-fold: the n_trace_candidates_dropped_malformed
    counter increments ONCE per dropped raw candidate, not per affected
    proposal. Operators reading AnalyzeCompletedEvent.n_trace_candidates_
    dropped_malformed should see aggregate drift across pass-level
    candidate emissions, not a binary "any proposal had a drop" signal."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                # Three different rejection cases — all drop, all count
                {"import_string_raw": "../../etc/passwd", "reason": "path"},
                {"import_string_raw": "foo;rm -rf", "reason": "shell"},
                {"import_string_raw": "foo.class", "reason": "keyword"},
                # One valid one — admits, doesn't count toward drops
                {"import_string_raw": "valid.module", "reason": "ok"},
            ],
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.trace_candidates) == 1  # only the valid one
    assert result.counters.n_trace_candidates_emitted == 1
    assert result.counters.n_trace_candidates_dropped_malformed == 3


def test_admitted_finding_survives_hostile_sibling_candidate() -> None:
    """Multi-proposal regression: proposal[0] admits cleanly,
    proposal[1] has a hostile candidate. Pre-fix, the proposal[1]
    crash dropped proposal[0]'s admission too."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged", span={"byte_start": 5, "byte_end": 10}),
        _minimal_proposal(
            evidence_tier="judged",
            span={"byte_start": 15, "byte_end": 20},
            trace_candidates=[
                # Python keyword part — rejected by is_valid_import_string
                {"import_string_raw": "foo.class", "reason": "hostile"},
            ],
        ),
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(byte_start=0, byte_end=100),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 2  # both parents survive
    assert len(result.trace_candidates) == 0  # only the hostile one was filtered


def test_decomposed_unicode_import_string_uses_canonical_nfc_form() -> None:
    """Post-DECISIONS.md#024 rename: the analog of the original "alias
    canonicalize" test (`./src/foo.py` → `src/foo.py`) for import strings
    is NFC normalization. The parser canonicalizes `import_string_raw`
    via `is_valid_import_string` before computing `candidate_id`;
    without that step, the raw NFD bytes would hash differently from
    the NFC bytes the schema validator stores, and
    `_enforce_candidate_id_matches_payload` would crash.
    Per M3 / adversarial-modeler #1."""
    decomposed = "café.bar"  # NFD: e + combining acute
    precomposed = "café.bar"  # NFC: precomposed
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                {"import_string_raw": decomposed, "reason": "decomposed"},
            ],
        )
    )
    # Must NOT raise — that was the bug class
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    assert len(result.trace_candidates) == 1
    # Candidate's `import_string` is the canonical NFC form, not the raw NFD
    assert result.trace_candidates[0].import_string == precomposed


def test_query_match_id_rejection_detail_sanitizes_ansi_escape() -> None:
    """Adversarial HIGH-2: raw schema for `query_match_id` ships only
    `max_length=256` (no pattern), but spec §3 named
    `[A-Za-z0-9_./:-]+` as the safety class. Parser-side
    `_sanitize_query_match_id_for_detail` enforces the spec-promised
    class so attacker bytes can't land verbatim in `rejection_detail`."""
    hostile = "fake_id\x1b]8;;file:///etc/passwd\x07click\x1b]8;;\x07"
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id=hostile,
        )
    )
    result = _call_parser(response, query_match_id_set=frozenset({"real_id"}))
    assert len(result.proposal_rejections) == 1
    detail = result.proposal_rejections[0].rejection_detail
    # ANSI escape sequences MUST be sanitized
    assert "\x1b" not in detail
    assert "]8" not in detail
    # Safe-class chars survive
    assert "fake_id" in detail
    # Out-of-class chars are replaced with `?`
    assert "?" in detail


def test_query_match_id_rejection_detail_preserves_safe_chars() -> None:
    """Benign IDs (matching the spec pattern) pass through unchanged so
    operators see the structural form."""
    benign = "python.security.sql_injection:42"
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id=benign,
        )
    )
    result = _call_parser(response, query_match_id_set=frozenset({"real_id"}))
    assert result.proposal_rejections[0].rejection_detail == benign


def test_narrow_exception_handler_lets_unexpected_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharp-edges M4 + adversarial M2: the `except ValidationError`
    block used to catch every exception class (MemoryError,
    RecursionError, etc.) and fold them into
    `schema_construction_failed`. Narrowed so genuinely-unexpected
    failures propagate as bugs rather than as misleading rejections.

    To prove narrowing actually narrows: monkeypatch the
    `compute_finding_content_hash` call (which fires inside the
    try-block during `ReviewFinding` construction) to raise a
    `RuntimeError`. The parser MUST propagate it instead of folding
    to `schema_construction_failed`. A re-broadening to
    `except Exception` would catch the RuntimeError and produce a
    rejection event — this test would fail with `RuntimeError` not
    raised, naming the regression precisely.

    Companion smoke assertion preserves the happy path.
    """
    from outrider.agent.nodes import analyze_parser

    # Smoke: clean admission still works (sanity that the test setup
    # is otherwise valid).
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1

    # Force a non-narrowed exception inside the try-block. If the
    # parser re-broadens its except, this RuntimeError gets folded
    # to a rejection and `pytest.raises` would fail.
    def _explode(**_kwargs: object) -> str:
        msg = "synthetic RuntimeError from monkeypatched compute_finding_content_hash"
        raise RuntimeError(msg)

    monkeypatch.setattr(analyze_parser, "compute_finding_content_hash", _explode)
    with pytest.raises(RuntimeError, match="synthetic RuntimeError"):
        _call_parser(
            response,
            included_scope_units=(_build_scope_unit(),),
            file_content="x" * 200,
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
