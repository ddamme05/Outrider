"""FUP-220: the checkpoint serde registers every Outrider state type.

`build_checkpoint_serde()` returns a `JsonPlusSerializer` with an explicit
msgpack allowlist. An explicit allowlist permits the listed types + the
serializer's built-in safe types and BLOCKS the rest — so the list must cover
every Outrider type that can land in a checkpoint, or resume/replay breaks
(a blocked type deserializes to raw data instead of the model, silently
corrupting rehydrated state; under strict-msgpack it is refused outright).

These tests pin four things:
1. The allowlist is exact `(module, name)` pairs — no module-prefix wildcard.
2. The allowlist matches the ReviewState type graph exactly (drift guard: a new
   state type added without registering it fails here — FUP-220's trigger).
3. Every registered enum and a findings-bearing state round-trip through the
   serde with no blocked types (empirical: proves the traversal in (2) predicts
   real serialization, since str-enum `==` is too loose to catch a block).
4. An UNregistered type IS blocked (proves the serde is restrictive, i.e. it was
   not accidentally constructed with `allowed_msgpack_modules=True`).

The full real-state guarantee (hitl/publish/trace channels populated) is the
hitl_resume eval + integration path exercised under `LANGGRAPH_STRICT_MSGPACK=true`.
"""

from __future__ import annotations

import enum
import importlib
import logging
from datetime import UTC, datetime
from typing import get_args, get_origin, get_type_hints
from uuid import uuid4

from pydantic import BaseModel

from outrider.agent.checkpoint_serde import (
    OUTRIDER_MSGPACK_ALLOWLIST,
    build_checkpoint_serde,
)
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingType
from outrider.policy.canonical import compute_round_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION, lookup_severity
from outrider.schemas import (
    AnalysisRound,
    ReviewDimension,
    ReviewFinding,
    ReviewTier,
    RiskLevel,
    TriageResult,
)
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState

_SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"


# --------------------------------------------------------------------------- #
# (2) drift guard: re-derive the reachable type set independently of the module
# --------------------------------------------------------------------------- #
def _reachable_outrider_types() -> set[tuple[str, str]]:
    """Walk ReviewState's type graph, collecting every Outrider model + enum.

    Mirrors the serializer's ext-encoding surface: channel-level Pydantic models
    are ext-encoded, enums are ext-encoded at any depth, and nested models are
    inlined by their parent's `model_dump()`. We over-collect (every reachable
    model, not only channel-level ones) because over-listing an allowlist is
    harmless — the checkpoint_serde module lists the same superset.
    """
    found: set[tuple[str, str]] = set()
    seen_models: set[type] = set()

    def record(cls: type) -> None:
        mod = getattr(cls, "__module__", "")
        name = getattr(cls, "__name__", None)
        if name and mod.startswith("outrider"):
            found.add((mod, name))

    def walk_type(tp: object) -> None:
        origin = get_origin(tp)
        if origin is not None:
            for arg in get_args(tp):
                walk_type(arg)
            return
        if isinstance(tp, type):
            if issubclass(tp, enum.Enum):
                record(tp)
            elif issubclass(tp, BaseModel):
                walk_model(tp)

    def walk_model(model: type[BaseModel]) -> None:
        if model in seen_models:
            return
        seen_models.add(model)
        record(model)
        try:
            hints = get_type_hints(model, include_extras=False)
        except Exception:
            hints = {n: f.annotation for n, f in model.model_fields.items()}
        for ann in hints.values():
            walk_type(ann)

    walk_model(ReviewState)
    return found


# --------------------------------------------------------------------------- #
# fixtures for the empirical round-trip
# --------------------------------------------------------------------------- #
def _finding() -> ReviewFinding:
    return ReviewFinding(
        review_id=uuid4(),
        installation_id=1,
        policy_version=ACTIVE_POLICY_VERSION,
        finding_type=FindingType.SQL_INJECTION,
        severity=lookup_severity(FindingType.SQL_INJECTION),
        dimension=ReviewDimension.SECURITY,
        evidence_tier=EvidenceTier.JUDGED,
        file_path="src/foo.py",
        line_start=10,
        line_end=12,
        title="SQL injection",
        description="raw concat",
        evidence="concat at src/foo.py:11",
        content_hash=compute_finding_content_hash(
            file_path="src/foo.py",
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash=uuid4().hex + uuid4().hex,
    )


def _populated_state() -> ReviewState:
    now = datetime.now(UTC)
    finding = _finding()
    analysis_round = AnalysisRound(
        round_id=compute_round_id(
            pass_index=0,
            files_examined=("src/foo.py",),
            files_skipped=(),
            finding_content_hashes=(finding.content_hash,),
        ),
        pass_index=0,
        findings=(finding,),
        files_examined=("src/foo.py",),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )
    triage = TriageResult(
        file_tiers={"src/foo.py": ReviewTier.DEEP},
        overall_risk=RiskLevel.MEDIUM,
        relevant_dimensions=[ReviewDimension.SECURITY],
        reasoning="auth changes warrant a deep review.",
    )
    pr_context = PRContext(
        installation_id=1,
        owner="acme",
        repo="widgets",
        pr_number=42,
        pr_title="Test PR",
        pr_body=None,
        base_sha="b" * 40,
        head_sha="h" * 40,
        author="alice",
        total_additions=5,
        total_deletions=2,
        changed_files=(
            ChangedFile(
                path="src/foo.py",
                status="modified",
                additions=5,
                deletions=2,
                patch="@@ -1 +1 @@\n-old\n+new\n",
                content_base="old\n",
                content_head="new\n",
                previous_path=None,
            ),
        ),
    )
    return ReviewState(
        review_id=uuid4(),
        pr_context=pr_context,
        received_at=now,
        is_eval=False,
        triage_result=triage,
        analysis_rounds=[analysis_round],
    )


def _blocked_warnings(records: list[logging.LogRecord]) -> list[str]:
    return [
        r.getMessage()
        for r in records
        if "Blocked deserialization" in r.getMessage()
        or "Deserializing unregistered type" in r.getMessage()
    ]


# --------------------------------------------------------------------------- #
# (1) shape
# --------------------------------------------------------------------------- #
def test_allowlist_entries_are_exact_outrider_pairs() -> None:
    assert OUTRIDER_MSGPACK_ALLOWLIST, "allowlist must not be empty"
    assert len(set(OUTRIDER_MSGPACK_ALLOWLIST)) == len(OUTRIDER_MSGPACK_ALLOWLIST), (
        "allowlist has duplicate entries"
    )
    for entry in OUTRIDER_MSGPACK_ALLOWLIST:
        assert isinstance(entry, tuple) and len(entry) == 2, f"not a (module, name) pair: {entry!r}"
        module, name = entry
        assert isinstance(module, str) and isinstance(name, str)
        assert module.startswith("outrider."), f"non-Outrider module in allowlist: {module}"
        # Exact symbols only — the serializer intentionally rejects prefix wildcards.
        assert "*" not in module and "*" not in name, f"wildcard in allowlist entry: {entry!r}"


def test_every_allowlisted_symbol_is_importable() -> None:
    """Each (module, name) resolves to a real class — a stale entry fails here."""
    for module, name in OUTRIDER_MSGPACK_ALLOWLIST:
        mod = importlib.import_module(module)
        assert hasattr(mod, name), f"{module}.{name} does not exist"


# --------------------------------------------------------------------------- #
# (2) drift guard
# --------------------------------------------------------------------------- #
def test_allowlist_matches_reviewstate_type_graph() -> None:
    """The allowlist equals the set of Outrider types reachable from ReviewState.

    If ReviewState gains a field whose type (or a nested type) is a new Outrider
    model or enum, this fails until the type is registered in
    `checkpoint_serde.OUTRIDER_MSGPACK_ALLOWLIST` — the FUP-220 "new state type"
    trigger. A stale over-registration (a type no longer reachable) also fails.
    """
    reachable = _reachable_outrider_types()
    registered = set(OUTRIDER_MSGPACK_ALLOWLIST)
    missing = reachable - registered
    extra = registered - reachable
    assert not missing, f"reachable-but-unregistered checkpoint types: {sorted(missing)}"
    assert not extra, f"registered-but-unreachable types (stale): {sorted(extra)}"


# --------------------------------------------------------------------------- #
# (3) empirical round-trip
# --------------------------------------------------------------------------- #
def test_every_registered_enum_roundtrips_without_block(caplog) -> None:
    """Each registered enum deserializes back to the enum type, not a raw value.

    str-enum members compare `==` to their string value, so a blocked enum
    (which returns the raw string) would pass an equality check — assert
    `isinstance(..., EnumCls)` instead.
    """
    serde = build_checkpoint_serde()
    for module, name in OUTRIDER_MSGPACK_ALLOWLIST:
        cls = getattr(importlib.import_module(module), name)
        if not (isinstance(cls, type) and issubclass(cls, enum.Enum)):
            continue
        member = next(iter(cls))
        with caplog.at_level(logging.WARNING, logger=_SERDE_LOGGER):
            caplog.clear()
            back = serde.loads_typed(serde.dumps_typed(member))
        assert isinstance(back, cls), f"{module}.{name} did not round-trip as its enum type"
        assert back == member
        assert not _blocked_warnings(caplog.records), f"{module}.{name} was blocked"


def test_populated_state_channels_roundtrip_without_block(caplog) -> None:
    """Each ReviewState channel value round-trips with no blocked type.

    Serializes per-channel (the way a checkpointer does — the top-level state is
    decomposed into channels), so channel-level models are genuinely ext-encoded
    rather than inlined into one big state dict.
    """
    serde = build_checkpoint_serde()
    state = _populated_state()
    with caplog.at_level(logging.WARNING, logger=_SERDE_LOGGER):
        for field_name in ReviewState.model_fields:
            value = getattr(state, field_name)
            caplog.clear()
            back = serde.loads_typed(serde.dumps_typed(value))
            assert back == value, f"channel {field_name!r} did not round-trip cleanly"
            assert not _blocked_warnings(caplog.records), (
                f"channel {field_name!r} hit a blocked type: {_blocked_warnings(caplog.records)}"
            )


# --------------------------------------------------------------------------- #
# (4) restrictiveness — the serde is NOT permissive-True
# --------------------------------------------------------------------------- #
class _UnregisteredModel(BaseModel):
    value: int


def test_unregistered_type_is_blocked(caplog) -> None:
    """A type absent from the allowlist is refused (proves not `allowed=True`).

    A blocked Pydantic model deserializes to its raw `model_dump()` dict (the
    serializer's documented fallback) instead of the model instance, and logs a
    'Blocked deserialization' warning. If the serde were permissive-True it would
    reconstruct the model and only warn 'unregistered'.
    """
    serde = build_checkpoint_serde()
    obj = _UnregisteredModel(value=7)
    with caplog.at_level(logging.WARNING, logger=_SERDE_LOGGER):
        back = serde.loads_typed(serde.dumps_typed(obj))
    assert not isinstance(back, _UnregisteredModel), "unregistered model was reconstructed"
    assert any("Blocked deserialization" in m for m in _blocked_warnings(caplog.records))
