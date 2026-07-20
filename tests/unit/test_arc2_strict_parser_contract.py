"""Arc 2 — shape validity NEVER implies admission.

This is the file that keeps the arc honest about its own scope. The strict
schema enforces SHAPE at the wire; it cannot establish proof AUTHENTICITY — that
a `query_match_id` names a query that actually fired, or that a `trace_path`
element was really walked. Authenticity stays where it lives, in
`agent/nodes/analyze_parser.py` (step 4, `query_match_id_not_in_registry`).

So a fabricated-but-well-shaped proof passing `jsonschema` and then being
REJECTED by the real parser is the EXPECTED NEGATIVE CONTROL — confirmation the
two layers do different jobs, not a failure. Only fabrication that SURVIVES the
parser would be an authenticity escape.

See `specs/2026-07-20-arc2-strict-schema-feasibility.md`.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from spikes.openai.arc2.strict_schema import derive_strict_analyze_schema

from outrider.agent.nodes.analyze_parser import ParserResult, parse_analyze_response
from outrider.ast_facts import ScopeUnit, compute_unit_id
from outrider.policy import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION

VALIDATOR = Draft202012Validator(derive_strict_analyze_schema())

_FILE_PATH = "app/api/reports.py"
_FILE_CONTENT = "def run_report(conn, user_id):\n    return conn.execute(user_id)\n"
# The parser rejects a proposal whose lines fall outside every INCLUDED scope
# unit (`span_outside_scope_unit`), so an admission test must pack the scope the
# finding sits in — exactly as the analyze node does.
_SCOPE = ScopeUnit(
    unit_id=compute_unit_id(_FILE_PATH, kind="function", qualified_name="run_report"),
    kind="function",
    name="run_report",
    qualified_name="run_report",
    file_path=_FILE_PATH,
    line_start=1,
    line_end=2,
    byte_start=0,
    byte_end=len(_FILE_CONTENT.encode("utf-8")),
)


def _derive_fired_ids() -> frozenset[str]:
    """The ids that ACTUALLY fire for this file, via production's own builder.

    This was a hand-written `AUTHENTIC_QUERY_ID = "py.sql.string_concat"` — not a
    registered id at all — injected into `query_match_id_set` and then called
    "authentic". That proved only that the parser trusts the set it is handed:
    the test supplied the very authority that made its own value real.

    The derived member is `python.function_definition` — a fired STRUCTURAL
    query. The signal-only SQL rule is deliberately NOT here: production's
    `_build_query_match_id_set` iterates structural ids only, so a SIGNAL_ONLY
    query never enters OBSERVED admission at all.
    """
    from outrider.agent.nodes.analyze import (
        _build_query_match_id_set,
        _filter_query_ids_to_scopes,
    )

    source_bytes = _FILE_CONTENT.encode("utf-8")
    fired = _build_query_match_id_set(source_bytes, file_path=_FILE_PATH)
    return _filter_query_ids_to_scopes(fired, source_bytes, (_SCOPE,), file_path=_FILE_PATH)


#: DERIVED. A member of this set is authentic BY CONSTRUCTION.
FIRED_IDS = _derive_fired_ids()
AUTHENTIC_QUERY_ID = sorted(FIRED_IDS)[0]
#: Independently fabricated — never derived, so the negative control stays honest.
FABRICATED_QUERY_ID = "python.definitely_not_registered"


def _finding(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "SQL built by concatenation",
        "description": "User input is concatenated into a SQL string.",
        "evidence": "conn.execute(user_id)",
        "line_start": 1,
        "line_end": 2,
        "trace_candidates": [],
    }
    base.update(overrides)
    return base


def _body(*findings: dict[str, Any]) -> str:
    return json.dumps({"findings": list(findings)})


def _parse(body: str, **overrides: object) -> ParserResult:
    defaults: dict[str, object] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "file_path": _FILE_PATH,
        "file_content": _FILE_CONTENT,
        "file_byte_length": len(_FILE_CONTENT.encode("utf-8")),
        "included_scope_units": (_SCOPE,),
        "query_match_id_set": FIRED_IDS,
        "degraded_mode": False,
        # Production's CURRENT version. A literal here drifts silently: severity
        # policy is part of grader identity, and 1.0.0 was superseded by 1.2.0.
        "active_policy_version": ACTIVE_POLICY_VERSION,
    }
    defaults.update(overrides)
    return parse_analyze_response(body, **defaults)  # type: ignore[arg-type]


def test_fabricated_proof_is_rejected_by_real_parser() -> None:
    """THE negative control, over a DERIVED fired set.

    A fabricated `query_match_id` is perfectly
    well-shaped — it passes the strict schema — and is still rejected by the
    parser with `query_match_id_not_in_registry`.

    If this ever stopped rejecting, the arc would have an authenticity escape,
    which is a `STOP-authenticity` verdict, not a schema fix.
    """
    fabricated = _finding(evidence_tier="observed", query_match_id=FABRICATED_QUERY_ID)
    VALIDATOR.validate({"findings": [fabricated]})  # shape-valid: the schema is happy

    result = _parse(_body(fabricated))

    assert result.admitted_findings == ()
    reasons = [r.rejection_reason for r in result.proposal_rejections]
    assert "query_match_id_not_in_registry" in reasons


def test_shape_validity_and_admission_are_independent_layers() -> None:
    """Both directions of the claim in one place, so the split cannot silently
    collapse: shape-valid-but-unauthentic is rejected, and shape-valid-with-real
    -proof is admitted. Same schema, opposite parser outcomes."""
    fabricated = _finding(evidence_tier="observed", query_match_id=FABRICATED_QUERY_ID)
    authentic = _finding(evidence_tier="observed", query_match_id=AUTHENTIC_QUERY_ID)

    for body in (fabricated, authentic):
        VALIDATOR.validate({"findings": [body]})

    assert _parse(_body(fabricated)).admitted_findings == ()
    assert len(_parse(_body(authentic)).admitted_findings) == 1


def test_strict_response_parses_through_production_parser() -> None:
    """A strict-shaped response admits through the REAL parser to the same
    finding as the equivalent `json_object` response — no admission drift from
    the shape change.

    The strict body differs from a canonical one only by carrying every
    property explicitly (canonical leaves the three optional ones omittable),
    so if the two admitted differently the strict encoding would be changing
    behavior rather than constraining it.
    """
    strict_body = _finding(evidence_tier="observed", query_match_id=AUTHENTIC_QUERY_ID)
    VALIDATOR.validate({"findings": [strict_body]})

    # The json_object equivalent: optional properties OMITTED rather than nulled.
    loose_body = {
        k: v for k, v in strict_body.items() if k not in {"trace_path", "trace_candidates"}
    }

    strict_result = _parse(_body(strict_body))
    loose_result = _parse(_body(loose_body))

    assert len(strict_result.admitted_findings) == 1
    assert len(loose_result.admitted_findings) == 1
    strict_finding = strict_result.admitted_findings[0]
    loose_finding = loose_result.admitted_findings[0]
    assert strict_finding.evidence_tier == loose_finding.evidence_tier
    assert strict_finding.query_match_id == loose_finding.query_match_id
    assert strict_finding.finding_type == loose_finding.finding_type
    assert (strict_finding.line_start, strict_finding.line_end) == (
        loose_finding.line_start,
        loose_finding.line_end,
    )


def test_inferred_requires_trace_path_at_both_layers() -> None:
    """The INFERRED proof rule holds at the wire AND at the parser.

    Belt-and-braces is the point: the schema makes `trace_path: []`
    unrepresentable, and `policy/findings.py::_trace_path_is_valid` independently
    rejects it. The arc must not weaken the second by adding the first.
    """
    from jsonschema.exceptions import ValidationError

    empty_trace = _finding(evidence_tier="inferred", trace_path=[])
    try:
        VALIDATOR.validate({"findings": [empty_trace]})
    except ValidationError:
        pass
    else:  # pragma: no cover - guards the assertion itself
        raise AssertionError("strict schema must reject an empty trace_path")

    # And the parser rejects it independently, without relying on the schema.
    result = _parse(_body(empty_trace), valid_trace_path_elements=frozenset())
    assert result.admitted_findings == ()


def test_evidence_tier_wire_values_match_the_canonical_enum() -> None:
    """The branch enums use the values the parser actually reads
    (`EvidenceTier(raw.evidence_tier)`), so a casing drift fails here rather
    than at the wire on a paid row."""
    schema = derive_strict_analyze_schema()
    pinned = {
        branch["properties"]["evidence_tier"]["enum"][0]
        for branch in schema["properties"]["findings"]["items"]["anyOf"]
    }
    assert pinned == {tier.value for tier in EvidenceTier}
    for value in pinned:
        assert EvidenceTier(value)
