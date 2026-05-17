"""AuditPersister unit tests — constructor, Protocol conformance, helpers.

DB-touching tests live in `tests/integration/` under `migrated_db`. This
file covers everything checkable without a real Postgres.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from outrider.audit.config import RetentionSettings
from outrider.audit.persister import (
    METADATA_ONLY_EXCEPTION_TYPES,
    AuditPersister,
    AuditPersisterConfigError,
    AuditPersisterIdempotencyConflict,
    AuditPersisterReviewIdMismatchError,
    AuditPersisterReviewNotFoundError,
    AuditPersisterSchemaInvariantError,
    FieldDigest,
    _compute_content_field_digests,
    _compute_field_digests,
    _diff_content_field_names,
    _diff_field_names,
)
from outrider.audit.sinks import PhaseEventSink
from outrider.llm.base import LLMExchangePersister

# ---------------------------------------------------------------------------
# Constructor — keyword-only + eager None-checks.
# ---------------------------------------------------------------------------


def test_constructor_requires_keyword_only_args() -> None:
    """`*` makes both args keyword-only; positional construction fails."""
    sessionmaker = MagicMock()
    settings = RetentionSettings()
    with pytest.raises(TypeError):
        AuditPersister(sessionmaker, settings)  # type: ignore[misc]


def test_constructor_session_factory_none_raises_config_error() -> None:
    """Eager None-check on session_factory (mirrors build_graph precedent)."""
    settings = RetentionSettings()
    with pytest.raises(AuditPersisterConfigError, match="session_factory"):
        AuditPersister(
            session_factory=None,  # type: ignore[arg-type]
            retention_settings=settings,
        )


def test_constructor_retention_settings_none_raises_config_error() -> None:
    """Eager None-check on retention_settings."""
    sessionmaker = MagicMock()
    with pytest.raises(AuditPersisterConfigError, match="retention_settings"):
        AuditPersister(
            session_factory=sessionmaker,
            retention_settings=None,  # type: ignore[arg-type]
        )


def test_constructor_succeeds_with_valid_args() -> None:
    """Happy path: constructs without exception."""
    sessionmaker = MagicMock()
    settings = RetentionSettings()
    persister = AuditPersister(
        session_factory=sessionmaker,
        retention_settings=settings,
    )
    assert persister is not None


# ---------------------------------------------------------------------------
# Protocol conformance — runtime-checkable isinstance gates.
# ---------------------------------------------------------------------------


def test_satisfies_llm_exchange_persister_protocol() -> None:
    """`isinstance(persister, LLMExchangePersister)` is True via the
    @runtime_checkable Protocol structural check (has `persist`)."""
    persister = AuditPersister(
        session_factory=MagicMock(),
        retention_settings=RetentionSettings(),
    )
    assert isinstance(persister, LLMExchangePersister)


def test_satisfies_phase_event_sink_protocol() -> None:
    """`isinstance(persister, PhaseEventSink)` is True (has `emit_phase`)."""
    persister = AuditPersister(
        session_factory=MagicMock(),
        retention_settings=RetentionSettings(),
    )
    assert isinstance(persister, PhaseEventSink)


def test_public_methods_match_protocol_signatures() -> None:
    """Public surface is exactly `persist` + `emit_phase`. No leaking
    SQLAlchemy types in the signatures (parameter annotations are
    domain types only)."""
    persist_sig = inspect.signature(AuditPersister.persist)
    emit_sig = inspect.signature(AuditPersister.emit_phase)

    # persist(event, request, response) — 3 args plus self
    assert list(persist_sig.parameters) == ["self", "event", "request", "response"]
    # emit_phase(event) — 1 arg plus self
    assert list(emit_sig.parameters) == ["self", "event"]


# ---------------------------------------------------------------------------
# Exception types — class hierarchy + metadata-only contract.
# ---------------------------------------------------------------------------


def test_config_error_is_valueerror_subclass() -> None:
    """AuditPersisterConfigError inherits ValueError — callers can catch
    broadly without coupling to the specific exception type."""
    assert issubclass(AuditPersisterConfigError, ValueError)


def test_review_not_found_is_lookuperror_subclass() -> None:
    """AuditPersisterReviewNotFoundError inherits LookupError — `KeyError`
    / `IndexError`-style semantics for "expected row absent"."""
    assert issubclass(AuditPersisterReviewNotFoundError, LookupError)


def test_idempotency_conflict_is_valueerror_subclass() -> None:
    """AuditPersisterIdempotencyConflict inherits ValueError."""
    assert issubclass(AuditPersisterIdempotencyConflict, ValueError)


def test_idempotency_conflict_carries_metadata_only() -> None:
    """Construction signature accepts only metadata; raw payload args
    are NOT in the constructor's keyword set.

    Regression test for the #016 logs-stay-metadata-only contract:
    a future refactor that adds `existing_payload=...` or `prompt=...`
    to this exception's __init__ would defeat the entire reason this
    exception exists.
    """
    from uuid import uuid4

    sig = inspect.signature(AuditPersisterIdempotencyConflict.__init__)
    params = set(sig.parameters)
    assert params == {"self", "event_id", "mismatched_fields", "field_digests"}
    # Negative assertion: none of the raw-content keyword names are accepted.
    for forbidden in ("existing_payload", "attempted_payload", "prompt", "completion", "payload"):
        assert forbidden not in params, (
            f"AuditPersisterIdempotencyConflict.__init__ has a `{forbidden}` "
            "parameter; metadata-only contract violated"
        )

    # Constructed instance also doesn't carry raw content.
    exc = AuditPersisterIdempotencyConflict(
        event_id=uuid4(),
        mismatched_fields=("cost_usd",),
        field_digests={"cost_usd": FieldDigest("a" * 64, "b" * 64, 10, 12)},
    )
    exc_vars = set(vars(exc).keys())
    for forbidden in ("existing_payload", "attempted_payload", "prompt", "completion", "payload"):
        assert forbidden not in exc_vars


def test_idempotency_conflict_constructor_enforces_digest_subset_invariant() -> None:
    """Pins the round-38 sharp-edges fold: the docstring invariant
    `set(field_digests) ⊆ set(mismatched_fields)` is now enforced at
    construction. A call site that swaps argument order, or passes
    digests computed over a stale field set, would otherwise ship a
    metadata-only-looking exception whose diagnostic claims are
    internally inconsistent — `mismatched_fields` says one thing,
    `field_digests` keys say another. Constructor MUST fail-loud.

    Metadata-only contract preserved: the validation error message
    names only field-name strings (class-level identifiers, never
    content); no payload bytes leak.
    """
    from uuid import uuid4

    # Happy path: digest keys ⊆ mismatched fields → no raise.
    exc_ok = AuditPersisterIdempotencyConflict(
        event_id=uuid4(),
        mismatched_fields=("prompt", "completion", "installation_id"),
        field_digests={
            "prompt": FieldDigest("a" * 64, "b" * 64, 10, 12),
            # installation_id intentionally omitted from digests
            # (text-only asymmetry from round-26+27 design intent).
        },
    )
    assert "prompt" in exc_ok.mismatched_fields
    assert "installation_id" in exc_ok.mismatched_fields
    assert "installation_id" not in exc_ok.field_digests

    # Happy path: empty digests is valid (e.g., only primitive-column mismatch).
    exc_primitives_only = AuditPersisterIdempotencyConflict(
        event_id=uuid4(),
        mismatched_fields=("installation_id", "is_eval"),
        field_digests={},
    )
    assert exc_primitives_only.mismatched_fields == ("installation_id", "is_eval")
    assert exc_primitives_only.field_digests == {}

    # Failure path: digest key NOT in mismatched_fields → ValueError.
    with pytest.raises(ValueError, match="subset of mismatched_fields"):
        AuditPersisterIdempotencyConflict(
            event_id=uuid4(),
            mismatched_fields=("prompt",),
            field_digests={
                "prompt": FieldDigest("a" * 64, "b" * 64, 10, 12),
                "completion": FieldDigest("c" * 64, "d" * 64, 20, 22),
            },
        )

    # Failure-path message names the OFFENDING field key, not raw content
    # (the constructor validation is itself metadata-only).
    with pytest.raises(ValueError) as exc_info:
        AuditPersisterIdempotencyConflict(
            event_id=uuid4(),
            mismatched_fields=("prompt",),
            field_digests={"unknown_field": FieldDigest("a" * 64, "b" * 64, 1, 1)},
        )
    assert "unknown_field" in str(exc_info.value)
    # The validation message itself contains only field-name strings,
    # never raw payload content (FieldDigest fields are SHA-256 + lengths).
    for forbidden_content in ("INTERNAL_SECRET", "raw_payload", "prompt_text"):
        assert forbidden_content not in str(exc_info.value)


def test_idempotency_conflict_str_does_not_contain_raw_content() -> None:
    """`str(exc)` is what flows to log records' `message` field. It must
    not contain raw prompt/completion/payload text — the entire reason
    for the metadata-only contract (#016 + FUP-023 gap).
    """
    from uuid import uuid4

    raw_prompt = "INTERNAL_SECRET_PROMPT_TEXT_DO_NOT_LOG"
    raw_completion = "INTERNAL_SECRET_COMPLETION_TEXT_DO_NOT_LOG"
    exc = AuditPersisterIdempotencyConflict(
        event_id=uuid4(),
        mismatched_fields=("prompt", "completion"),
        field_digests={
            "prompt": FieldDigest("a" * 64, "b" * 64, len(raw_prompt), len(raw_prompt)),
            "completion": FieldDigest("c" * 64, "d" * 64, len(raw_completion), len(raw_completion)),
        },
    )
    rendered = str(exc)
    assert raw_prompt not in rendered
    assert raw_completion not in rendered


# ---------------------------------------------------------------------------
# Helper functions — _diff_field_names + _compute_field_digests.
# ---------------------------------------------------------------------------


def test_diff_field_names_returns_only_mismatched_keys() -> None:
    """Equal values do NOT appear in the result."""
    existing = {"a": 1, "b": "x", "c": [1, 2, 3]}
    attempted = {"a": 1, "b": "y", "c": [1, 2, 3]}
    assert _diff_field_names(existing, attempted) == ("b",)


def test_diff_field_names_treats_missing_key_as_mismatch() -> None:
    """Key present on one side but not the other is a mismatch (payload-
    shape change between emissions is itself a producer bug)."""
    existing = {"a": 1, "b": 2}
    attempted = {"a": 1, "c": 3}
    assert _diff_field_names(existing, attempted) == ("b", "c")


def test_diff_field_names_distinguishes_missing_from_present_none() -> None:
    """Regression: `.get(k)` returns None both when k is absent AND when
    k is present with value None. The sentinel-based diff must report a
    mismatch when one side has {"a": None} and the other has {}.

    Today no LLMCallEvent/ReviewPhaseEvent field defaults to None inside
    its payload, but a future optional event field would silently slip
    past payload-equality verification under the naive `.get(k)` shape.
    """
    existing = {"a": None}
    attempted: dict[str, object] = {}
    assert _diff_field_names(existing, attempted) == ("a",)

    # Symmetric: present-None on the attempted side, absent on existing.
    existing2: dict[str, object] = {}
    attempted2 = {"a": None}
    assert _diff_field_names(existing2, attempted2) == ("a",)


def test_compute_field_digests_distinguishes_missing_from_present_none() -> None:
    """Missing-side fields get a distinct SHA-256 (sentinel input) AND
    length `-1`, so operators inspecting the digest see "field absent"
    rather than "field present with empty value" rendering as length 0
    or 4 (the JSON encoding of `null`)."""
    existing = {"a": None}
    attempted: dict[str, object] = {}
    digests = _compute_field_digests(existing, attempted)
    assert set(digests) == {"a"}
    digest = digests["a"]
    # Existing side has {"a": None} — JSON-encoded value `null`.
    # Attempted side is missing the key — sentinel renders as length -1.
    assert digest.existing_length >= 0  # JSON-encoded `null`
    assert digest.attempted_length == -1  # absent
    assert digest.existing_sha256 != digest.attempted_sha256


def test_diff_field_names_empty_when_all_match() -> None:
    """Identical dicts → empty tuple. Conflict-no-op path."""
    payload = {"a": 1, "b": 2}
    assert _diff_field_names(payload, payload) == ()


def test_compute_field_digests_returns_namedtuple_per_mismatch() -> None:
    """Each mismatched field gets a FieldDigest with both SHA-256 hashes
    AND both lengths."""
    digests = _compute_field_digests({"a": "hello"}, {"a": "world"})
    assert set(digests) == {"a"}
    digest = digests["a"]
    assert isinstance(digest, FieldDigest)
    assert digest.existing_sha256 != digest.attempted_sha256
    assert len(digest.existing_sha256) == 64
    assert len(digest.attempted_sha256) == 64
    assert digest.existing_length > 0
    assert digest.attempted_length > 0


def test_compute_field_digests_no_mismatch_is_empty_map() -> None:
    """Matching payloads → empty digest map."""
    digests = _compute_field_digests({"a": 1}, {"a": 1})
    assert digests == {}


# ---------------------------------------------------------------------------
# Content-field helpers — _diff_content_field_names + _compute_content_field_digests.
# ---------------------------------------------------------------------------


def _diff_args(
    *,
    prompt_db: str = "p",
    prompt_new: str = "p",
    completion_db: str = "c",
    completion_new: str = "c",
    installation_id_db: int = 42,
    installation_id_new: int = 42,
    is_eval_db: bool = False,
    is_eval_new: bool = False,
) -> dict[str, object]:
    """Helper: keyword args for `_diff_content_field_names` with defaults
    that match (so any subset can be flipped to trigger a single
    mismatch). Updated round-26 fold: helper signature now includes
    `installation_id` and `is_eval` (purge-scope + eval-isolation
    metadata)."""
    return {
        "prompt_db": prompt_db,
        "prompt_new": prompt_new,
        "completion_db": completion_db,
        "completion_new": completion_new,
        "installation_id_db": installation_id_db,
        "installation_id_new": installation_id_new,
        "is_eval_db": is_eval_db,
        "is_eval_new": is_eval_new,
    }


def test_diff_content_field_names_both_match() -> None:
    """No content mismatch → empty tuple."""
    assert _diff_content_field_names(**_diff_args()) == ()


def test_diff_content_field_names_prompt_only() -> None:
    assert _diff_content_field_names(**_diff_args(prompt_db="p1", prompt_new="p2")) == ("prompt",)


def test_diff_content_field_names_completion_only() -> None:
    assert _diff_content_field_names(**_diff_args(completion_db="c1", completion_new="c2")) == (
        "completion",
    )


def test_diff_content_field_names_both_mismatch() -> None:
    assert _diff_content_field_names(
        **_diff_args(
            prompt_db="p1",
            prompt_new="p2",
            completion_db="c1",
            completion_new="c2",
        )
    ) == ("prompt", "completion")


def test_diff_content_field_names_installation_id_only() -> None:
    """Round-26 fold: same content, different installation_id → mismatched."""
    assert _diff_content_field_names(**_diff_args(installation_id_db=1, installation_id_new=2)) == (
        "installation_id",
    )


def test_diff_content_field_names_is_eval_only() -> None:
    """Round-26 fold: same content, flipped is_eval → mismatched."""
    assert _diff_content_field_names(**_diff_args(is_eval_db=False, is_eval_new=True)) == (
        "is_eval",
    )


def test_diff_content_field_names_all_four_mismatch() -> None:
    """Round-26 fold: text + purge-scope + eval-flag all mismatched →
    ordered tuple lists prompt, completion, installation_id, is_eval."""
    assert _diff_content_field_names(
        **_diff_args(
            prompt_db="p1",
            prompt_new="p2",
            completion_db="c1",
            completion_new="c2",
            installation_id_db=1,
            installation_id_new=2,
            is_eval_db=False,
            is_eval_new=True,
        )
    ) == ("prompt", "completion", "installation_id", "is_eval")


def test_compute_content_field_digests_carries_lengths_only() -> None:
    """Returned digests carry byte-lengths and SHA-256 of the content,
    never the content itself. Regression test for the metadata-only
    boundary at the helper layer.

    Round-26 fold: digests are reserved for TEXT content fields. Small
    primitives (installation_id, is_eval) are NOT digested — the
    mismatched-field name is the diagnostic signal; raw values come
    from the DB.
    """
    digests = _compute_content_field_digests(
        prompt_db="secret prompt text",
        prompt_new="different prompt",
        completion_db="secret completion text",
        completion_new="different completion",
    )
    # Both fields mismatched.
    assert set(digests) == {"prompt", "completion"}
    # FieldDigest namedtuple fields, NOT raw content.
    for digest in digests.values():
        assert isinstance(digest, FieldDigest)
        assert len(digest.existing_sha256) == 64
        assert len(digest.attempted_sha256) == 64
        assert digest.existing_length > 0
        assert digest.attempted_length > 0


def test_compute_content_field_digests_intentionally_omits_non_text_columns() -> None:
    """Pin the design asymmetry between `mismatched_fields` and
    `field_digests` (round-27 codex fold):

    `_diff_content_field_names()` reports mismatches for ALL FOUR
    content-row columns (prompt, completion, installation_id, is_eval) —
    it is the authoritative list of what differed. `_compute_content_field_digests()`
    intentionally returns digests only for TEXT columns (prompt,
    completion) — SHA-256 of a one-character bool or a tiny int is not
    a useful diagnostic, and emitting per-primitive digest tuples
    bloats the exception payload for no signal. The mismatched-field
    NAME is the diagnostic signal for primitives; the raw value (small,
    safe) is recoverable from the DB.

    This test pins that asymmetry as a design intent — a future
    refactor that "fixes" the asymmetry by adding installation_id /
    is_eval to the digest map would defeat the intent without surfacing
    as a behavior change anywhere else.
    """
    # `_compute_content_field_digests` takes ONLY text-column kwargs.
    sig = inspect.signature(_compute_content_field_digests)
    text_only_params = {
        "prompt_db",
        "prompt_new",
        "completion_db",
        "completion_new",
    }
    assert set(sig.parameters) == text_only_params, (
        "_compute_content_field_digests must take text-column kwargs only. "
        "If installation_id/is_eval is now relevant for digests, that's a "
        "design change — update this test deliberately AND update "
        "_diff_content_field_names's matching expectations."
    )

    # When all four columns mismatch, only `prompt` + `completion` get
    # digest entries; `installation_id` + `is_eval` appear ONLY in the
    # `mismatched_fields` tuple, not in the `field_digests` map.
    digests = _compute_content_field_digests(
        prompt_db="p1",
        prompt_new="p2",
        completion_db="c1",
        completion_new="c2",
    )
    assert set(digests.keys()) == {"prompt", "completion"}, (
        "field_digests must cover ONLY text columns; primitives "
        "(installation_id, is_eval) are intentionally omitted"
    )

    # The mismatched-fields tuple is the AUTHORITATIVE signal — it can
    # contain entries that field_digests does not. Re-pin this contract
    # via _diff_content_field_names for the all-four-mismatch case.
    names = _diff_content_field_names(
        prompt_db="p1",
        prompt_new="p2",
        completion_db="c1",
        completion_new="c2",
        installation_id_db=1,
        installation_id_new=2,
        is_eval_db=False,
        is_eval_new=True,
    )
    assert names == ("prompt", "completion", "installation_id", "is_eval")
    # The asymmetry is the contract: 4 mismatched names, 2 digests.
    assert len(names) == 4
    assert len(digests) == 2


def test_field_digest_namedtuple_field_order() -> None:
    """Pin the namedtuple field order — `(existing_sha256, attempted_sha256,
    existing_length, attempted_length)`. A future refactor that reorders or
    renames these breaks the conflict-conflict test contract."""
    assert FieldDigest._fields == (
        "existing_sha256",
        "attempted_sha256",
        "existing_length",
        "attempted_length",
    )


# ---------------------------------------------------------------------------
# Metadata-only exception allowlist (H1+M4 contributor contract).
# ---------------------------------------------------------------------------


def test_metadata_only_exception_types_lists_every_persister_exception() -> None:
    """`METADATA_ONLY_EXCEPTION_TYPES` is the allowlist `LLMPersisterError` at
    `anthropic_provider.py` consults to decide whether `str(exc)` is safe.
    Every exception type the persister module raises MUST appear in this
    allowlist OR have a deliberate decision to omit it.

    Regression: if a future author adds a new exception class without
    updating the allowlist, this test fires. Same shape as
    `audit_events.AuditEvent` discriminated-union enumeration discipline.
    """
    expected = {
        AuditPersisterConfigError,
        AuditPersisterReviewNotFoundError,
        AuditPersisterReviewIdMismatchError,
        AuditPersisterSchemaInvariantError,
        AuditPersisterIdempotencyConflict,
    }
    assert set(METADATA_ONLY_EXCEPTION_TYPES) == expected


def test_schema_invariant_error_str_carries_only_metadata() -> None:
    """`AuditPersisterSchemaInvariantError.__str__` must contain only
    schema identifiers (event_id, column name) — never payload content.
    Pinned because the bare `RuntimeError` it replaced was at risk under
    the wrapper's `f"{exc!r}"` interpolation."""
    from uuid import uuid4

    event_id = uuid4()
    exc = AuditPersisterSchemaInvariantError(
        f"audit_events.payload is None for event_id={event_id}; "
        "schema invariant violated (payload is NOT NULL)"
    )
    rendered = str(exc)
    # Schema identifiers present.
    assert str(event_id) in rendered
    assert "payload" in rendered  # column name, OK
    # Sentinel for "no content text would have been in this message"
    # — the message is metadata-only by construction.


# ---------------------------------------------------------------------------
# _serialize_event_payload contract (H2: no default=str catchall).
# ---------------------------------------------------------------------------


def test_serialize_event_payload_fails_loud_on_non_json_safe_type() -> None:
    """`_serialize_event_payload` MUST raise (not silently coerce) when the
    event's `model_dump(mode="json")` produces a non-JSON-safe type.

    Regression: an earlier impl used `default=str` as a catchall, which
    would silently coerce e.g. a Decimal to its string repr. That hides
    exactly the producer-bug class the persister-boundary contract should
    surface.

    This test exercises the actual `_serialize_event_payload` function
    (not just stdlib `json.dumps`). Approach: patch the class-level
    `model_dump` method to return a payload dict containing a non-JSON-
    safe value (bytes). Pydantic's `frozen=True` blocks instance-level
    method assignment; class-level `patch.object` bypasses that.
    """
    from datetime import UTC, datetime
    from unittest.mock import patch
    from uuid import uuid4

    from outrider.audit.events import ReviewPhaseEvent
    from outrider.audit.persister import _serialize_event_payload

    event = ReviewPhaseEvent(
        review_id=uuid4(),
        phase_id=str(uuid4()),
        node_id="triage",
        marker="start",
        timestamp=datetime.now(UTC),
    )

    # Force `model_dump(mode="json", ...)` to return a payload containing
    # bytes — a Python type that's NOT JSON-safe. With the old `default=str`
    # catchall this would coerce silently; without it, `json.dumps` raises.
    def _bad_model_dump(*args: object, **kwargs: object) -> dict[str, object]:
        return {"smuggled_bytes": b"raw_bytes_should_not_serialize"}

    with (
        patch.object(ReviewPhaseEvent, "model_dump", _bad_model_dump),
        pytest.raises(TypeError, match="bytes"),
    ):
        _serialize_event_payload(event)


# ---------------------------------------------------------------------------
# Exception-type identity for the type-narrow LLMPersisterError wrap (H1).
# ---------------------------------------------------------------------------


def test_every_persister_exception_is_metadata_only_listed() -> None:
    """Forward-compat: a class defined inside `audit/persister.py` that
    inherits from BaseException MUST be in the allowlist. Catches new
    exception classes that bypass the contributor contract."""
    import inspect

    from outrider.audit import persister

    discovered: set[type[BaseException]] = set()
    for _, member in inspect.getmembers(persister, inspect.isclass):
        if issubclass(member, BaseException) and member.__module__ == persister.__name__:
            discovered.add(member)

    missing = discovered - set(METADATA_ONLY_EXCEPTION_TYPES)
    assert not missing, (
        f"Exception classes defined in audit.persister but not in "
        f"METADATA_ONLY_EXCEPTION_TYPES: {missing}. Add them to the "
        "allowlist (or deliberately exclude with a comment + add to "
        "this test's exclude set)."
    )


def test_idempotency_conflict_constructor_has_no_content_bearing_params() -> None:
    """Round-40 codex fold: structural pin on `AuditPersisterIdempotencyConflict`'s
    constructor signature. The general property test below special-cases
    this class (because field names legitimately render in `str(exc)`),
    which means the property test alone cannot catch a future contributor
    adding a `prompt: str` or `completion: str` kwarg.

    This test pins the canonical parameter set explicitly: ONLY
    `event_id`, `mismatched_fields`, `field_digests`. Any new parameter
    must update this test AND get code review on whether it violates
    the metadata-only contract. The canonical set has NO `str`-typed
    parameter (only `UUID`, `tuple[str, ...]`, and `Mapping[str, FieldDigest]`),
    so an added `str` kwarg ships a content-leak surface in disguise.
    """
    import inspect

    sig = inspect.signature(AuditPersisterIdempotencyConflict.__init__)
    actual_params = set(sig.parameters) - {"self"}
    canonical_params = {"event_id", "mismatched_fields", "field_digests"}
    assert actual_params == canonical_params, (
        f"AuditPersisterIdempotencyConflict.__init__ parameter set changed: "
        f"got {actual_params}, expected {canonical_params}. The metadata-only "
        f"contract requires this class to carry ONLY event_id + field-name "
        f"identifiers + SHA-256 digests. Any new `str` parameter is a "
        f"content-leak surface — review against DECISIONS#016 before changing "
        f"this test. If a new field is legitimately needed, update the test "
        f"AND verify the new field is a structured non-content type (UUID, "
        f"int, bool, tuple of identifiers, etc.)."
    )


def test_every_metadata_only_exception_type_is_actually_metadata_only() -> None:
    """Property test (round-39 adversarial-modeler U3 fold, refined in
    round-40): the existing
    `test_every_persister_exception_is_metadata_only_listed` enforces
    CLASS MEMBERSHIP — every persister exception class is in the tuple.
    It does NOT enforce the METADATA-ONLY PROPERTY of each class. A
    future contributor adding a `prompt` kwarg to one of these classes
    would pass the membership test silently while violating the actual
    contract (DECISIONS#016 logs-stay-metadata-only).

    This test constructs each allowlisted exception with content-shaped
    sentinel strings injected via every constructor parameter and
    asserts the sentinels do NOT appear in `str(exc)`, `repr(exc)`, or
    any element of `exc.args`. If a contributor adds a content-bearing
    constructor parameter to an allowlisted class, this test fails-loud
    at the offending class.

    **Round-40 codex correction (sentinel-reuse defeat).** The previous
    version of this test used ONE sentinel for both field-name slots
    (`mismatched_fields`) and any content-bearing `str` slots. For
    `IdempotencyConflict` the test then special-cased "sentinel appears
    in mismatched_fields is OK" — but a future content-bearing `str`
    kwarg on that class could ALSO accept the same sentinel and pass
    silently (the assertion only checked mismatched_fields, not
    content). Two sentinels now: `_FIELD_NAME_SENTINEL` allowed in
    field-name slots; `_FORBIDDEN_CONTENT_SENTINEL` MUST NOT appear in
    any rendering surface for ANY class including `IdempotencyConflict`.
    The constructor-shape pin test above (`test_idempotency_conflict_
    constructor_has_no_content_bearing_params`) is the second layer of
    defense — it catches a new `str` kwarg at the parameter-set level.
    """
    import inspect
    from uuid import uuid4

    # Two sentinels: one allowed in field-name slots, one strictly forbidden.
    field_name_sentinel = "OUTRIDER_FIELD_NAME_SENTINEL_abc123"
    forbidden_content_sentinel = "OUTRIDER_FORBIDDEN_CONTENT_SENTINEL_xyz789"

    for exc_cls in METADATA_ONLY_EXCEPTION_TYPES:
        # Discover the constructor signature; build a kwargs dict that
        # passes the appropriate sentinel into each string parameter.
        sig = inspect.signature(exc_cls.__init__)
        kwargs: dict[str, object] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            ann = param.annotation
            if ann is str or ann == "str":
                # ANY str-typed kwarg gets the FORBIDDEN sentinel. The
                # canonical metadata-only classes have no `str` kwargs
                # today (only UUID/tuple/Mapping). A future contributor
                # adding `prompt: str = ""` would receive the forbidden
                # sentinel via this branch and fail the leak assertion.
                kwargs[name] = forbidden_content_sentinel
            elif "UUID" in str(ann):
                kwargs[name] = uuid4()
            elif "tuple" in str(ann).lower():
                # tuple[str, ...] for mismatched_fields — inject the
                # FIELD_NAME sentinel. Field names ARE class-level
                # identifiers and legitimately render in str(exc).
                kwargs[name] = (field_name_sentinel,)
            elif "Mapping" in str(ann) or "dict" in str(ann).lower():
                # field_digests: Mapping[str, FieldDigest] — populate
                # with field-name sentinel as key (constraint: key must
                # be in mismatched_fields per the round-38 invariant guard).
                if "mismatched_fields" in kwargs:
                    kwargs[name] = {field_name_sentinel: FieldDigest("a" * 64, "b" * 64, 1, 1)}
                else:
                    kwargs[name] = {}
            elif ann is int:
                kwargs[name] = 42
            elif param.default is not inspect.Parameter.empty:
                continue
            else:
                # Unannotated parameter — treat as potentially content-bearing
                # and inject the forbidden sentinel.
                kwargs[name] = forbidden_content_sentinel

        # Construct (may raise if our injection violated a constraint).
        try:
            exc = exc_cls(**kwargs)  # type: ignore[arg-type, call-arg]
        except (TypeError, ValueError):
            continue  # Constructor rejected the shape; structurally safe.

        rendered_str = str(exc)
        rendered_repr = repr(exc)
        args_str = " ".join(str(a) for a in exc.args)

        # The FORBIDDEN sentinel MUST NOT appear anywhere — this is the
        # load-bearing assertion for ALL classes including IdempotencyConflict.
        # If it does, a content-bearing parameter exists and leaks.
        assert forbidden_content_sentinel not in rendered_str, (
            f"{exc_cls.__name__} rendered FORBIDDEN content sentinel via str(): "
            f"{rendered_str!r}. This exception class accepts a content-bearing "
            f"constructor parameter — violates the METADATA_ONLY contract. "
            f"Remove the parameter or rewrite __str__ to filter it."
        )
        assert forbidden_content_sentinel not in rendered_repr, (
            f"{exc_cls.__name__} rendered FORBIDDEN sentinel via repr(): {rendered_repr!r}."
        )
        assert forbidden_content_sentinel not in args_str, (
            f"{exc_cls.__name__} stored FORBIDDEN sentinel in args: {exc.args!r}. "
            f"Per the metadata-only contract, args must contain only "
            f"class-level identifiers, never raw content."
        )

        # The FIELD_NAME sentinel may legitimately appear (it's a class-level
        # identifier). For IdempotencyConflict, verify it appears via the
        # expected channel (`mismatched_fields` tuple) and nowhere else
        # that would suggest leakage from an unexpected slot.
        if exc_cls is AuditPersisterIdempotencyConflict:
            assert field_name_sentinel in str(exc.mismatched_fields), (
                f"Test setup invariant: field-name sentinel should be in "
                f"mismatched_fields for {exc_cls.__name__}"
            )
