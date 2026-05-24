# See specs/2026-05-19-analyze-foundation.md §1 and specs/2026-05-23-trace-node.md.
"""`TraceCandidate` schema tests.

Pins the §1 schema discipline: frozen + `extra="forbid"`, SHA-256 hex
`candidate_id` and `source_proposal_hash`, bounded string lengths on
`reason` and `import_string`. Per `DECISIONS.md#024` (Accepted 2026-05-24),
trace candidates are dotted Python import strings (V1; no file-path
fallback) — `import_string` replaced `candidate_path` in lockstep with
the rename of `compute_candidate_id`'s kwarg and the
`coordinates.is_valid_import_string` field validator.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.policy.canonical import compute_candidate_id, compute_identity_hash
from outrider.schemas import TraceCandidate


def _kwargs(**overrides: object) -> dict[str, object]:
    """Fixture kwargs with a canonical `candidate_id` derived from the
    candidate's payload. Tests that deliberately exercise drift override
    `candidate_id` directly; tests that exercise other-field validators
    (import-string normalization, length, hex-shape on `source_proposal_hash`,
    etc.) override those fields and the helper recomputes the
    `candidate_id` to match — otherwise a canonical-ID-mismatch fires
    first and the field the test is actually exercising is never
    reached.
    """
    source_proposal_hash = compute_identity_hash({"prop": 1})
    import_string = "middleware.auth"
    reason = "auth middleware referenced by the finding"
    base: dict[str, object] = {
        "source_proposal_hash": source_proposal_hash,
        "reason": reason,
        "import_string": import_string,
    }
    base.update(overrides)
    if "candidate_id" not in overrides:
        source_value = base["source_proposal_hash"]
        import_value = base["import_string"]
        reason_value = base["reason"]
        assert isinstance(source_value, str)
        assert isinstance(import_value, str)
        assert isinstance(reason_value, str)
        base["candidate_id"] = compute_candidate_id(
            source_proposal_hash=source_value,
            import_string=import_value,
            reason=reason_value,
        )
    return base


def test_trace_candidate_admits_well_formed() -> None:
    c = TraceCandidate(**_kwargs())  # type: ignore[arg-type]
    assert c.import_string == "middleware.auth"


def test_trace_candidate_candidate_id_rejects_non_hex() -> None:
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(candidate_id="not-a-hash"))  # type: ignore[arg-type]


def test_trace_candidate_source_proposal_hash_rejects_non_hex() -> None:
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(source_proposal_hash="bad"))  # type: ignore[arg-type]


def test_trace_candidate_reason_max_length() -> None:
    """500-char cap defends against unbounded model output."""
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(reason="x" * 501))  # type: ignore[arg-type]


def test_trace_candidate_import_string_max_length() -> None:
    """1024-char cap defends against unbounded model output. Construct a
    dotted form that exceeds the cap by chaining valid identifier parts."""
    with pytest.raises(ValidationError):
        # Long but still identifier-valid: 'a' * 1100 is a valid identifier
        # (passes isidentifier()); the length cap fires before the validator.
        TraceCandidate(**_kwargs(import_string="a" * 1100))  # type: ignore[arg-type]


def test_trace_candidate_import_string_rejects_path_shape() -> None:
    """Per DECISIONS.md#024 + M3, the schema validator rejects path-shaped
    input: forward slash, backslash, leading/trailing/interior empty parts."""
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(import_string="src/middleware/auth.py"))  # type: ignore[arg-type]


def test_trace_candidate_import_string_rejects_python_keyword_part() -> None:
    """`class` is a Python keyword — rejected by `is_valid_import_string`."""
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(import_string="foo.class"))  # type: ignore[arg-type]


def test_trace_candidate_import_string_nfc_normalizes() -> None:
    """Per M3: schema field validator NFC-normalizes input. Decomposed
    Unicode (`café` with combining acute) admits and the stored value
    is the NFC-composed form.

    Constructed via `unicodedata.normalize` (not visually-identical
    string literals — those compile to the SAME bytes in a UTF-8 source
    file, defeating the test by making decomposed == precomposed at
    the literal level). The pre-assert below pins that the two forms
    actually DIFFER byte-wise before exercising the validator."""
    import unicodedata

    precomposed = "café.bar"
    decomposed = unicodedata.normalize("NFD", precomposed)
    # Sanity: the two forms must differ byte-wise; otherwise the test
    # is vacuous (a literal-equality assertion masquerading as
    # validator-behavior).
    assert decomposed != precomposed
    # The schema's field validator NFC-normalizes import_string BEFORE
    # the cross-field `_enforce_candidate_id_matches_payload` validator
    # runs, so candidate_id must be computed from the PRECOMPOSED form
    # (what the validator will store). `_kwargs` precomputes from the
    # supplied import_string, so we override candidate_id explicitly
    # here.
    source_proposal_hash = compute_identity_hash({"prop": 1})
    reason = "auth middleware referenced by the finding"
    canonical_id = compute_candidate_id(
        source_proposal_hash=source_proposal_hash,
        import_string=precomposed,
        reason=reason,
    )
    c = TraceCandidate(
        source_proposal_hash=source_proposal_hash,
        reason=reason,
        import_string=decomposed,
        candidate_id=canonical_id,
    )
    assert c.import_string == precomposed


def test_trace_candidate_frozen_rejects_mutation() -> None:
    c = TraceCandidate(**_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        c.import_string = "other.module"  # type: ignore[misc]


def test_trace_candidate_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(unexpected="bad"))  # type: ignore[arg-type]


def test_trace_candidate_source_proposal_hash_distinct_from_candidate_id() -> None:
    """Sanity: the two hash fields are distinct.

    Construct via the canonical recipe (`candidate_id` is bound to
    payload). The two hex strings come from independent derivations —
    `source_proposal_hash` from the raw proposal payload, `candidate_id`
    from `compute_candidate_id` — and are not coincidentally equal.
    """
    prop = compute_identity_hash({"b": 2})
    import_string = "whatever"
    reason = "r"
    cand = compute_candidate_id(
        source_proposal_hash=prop,
        import_string=import_string,
        reason=reason,
    )
    c = TraceCandidate(
        candidate_id=cand,
        source_proposal_hash=prop,
        reason=reason,
        import_string=import_string,
    )
    assert c.candidate_id == cand
    assert c.source_proposal_hash == prop
    assert c.candidate_id != c.source_proposal_hash
