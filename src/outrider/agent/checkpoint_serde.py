"""Checkpoint serde with an explicit msgpack allowlist for Outrider state types.

langgraph-checkpoint's default serializer deserializes unregistered types with a
deprecation warning ("Deserializing unregistered type ... will be blocked in a
future version"). When strict mode becomes the default — or an operator sets
``LANGGRAPH_STRICT_MSGPACK=true`` — unregistered types are BLOCKED on deserialize,
which breaks HITL resume and replay-equivalence (both round-trip ``ReviewState``
through checkpoint storage). ``build_checkpoint_serde()`` registers every Outrider
type that can land in a checkpoint so resume/replay keep working under strict mode.

This does NOT weaken ``state-is-pure-data`` (spec §9.3). That invariant prohibits
runtime dependencies (DB sessions, HTTP/SDK clients, context managers) in state —
not typed Pydantic data models, which ``ReviewState`` is designed to carry and
round-trip. Registering the exact types is the correct fix, not a stopgap.

Encoding note (``langgraph.checkpoint.serde.jsonplus``): a Pydantic v2 model is
ext-encoded as ``(module, name, model_dump(), "model_validate_json")``.
``model_dump()`` inlines nested models as plain dicts (revived by the parent's
``model_validate_json``), so nested models are NOT separately ext-encoded — but
Enum members stay as members at any nesting depth and ARE ext-encoded
individually. The allowlist therefore must cover every channel-level model AND
every enum reachable from ``ReviewState``. We register every Outrider model + enum
reachable from the ``ReviewState`` type graph; over-approximation is harmless for
an allowlist (it only permits our own first-party types, never blocks a needed
one). Completeness + drift are guarded by
``tests/unit/test_checkpoint_serde.py``, which re-derives the reachable set from
``ReviewState`` and fails if this tuple no longer matches — the signal that a new
state type was added without registering it.
"""

from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Every Outrider Pydantic model reachable from the ReviewState type graph. Channel
# top-level models (e.g. AnalysisRound as a list[AnalysisRound] element) are
# ext-encoded and MUST be listed; nested-only models (ReviewFinding, ReviewMetrics,
# ChangedFile, ...) are inlined by their parent's model_dump() and are listed
# defensively (harmless, and future-proof if a refactor promotes one to a channel).
_MODEL_TYPES: tuple[tuple[str, str], ...] = (
    ("outrider.schemas.analysis_round", "AnalysisRound"),
    ("outrider.schemas.analyze_worker", "AnalyzeWorkerOutcome"),
    ("outrider.schemas.hitl", "HITLDecision"),
    ("outrider.schemas.hitl", "HITLRequest"),
    ("outrider.schemas.hitl", "PerFindingDecision"),
    ("outrider.schemas.observed_subsumption", "ObservedSubsumedMatch"),
    ("outrider.schemas.pr_context", "ChangedFile"),
    ("outrider.schemas.pr_context", "PRContext"),
    ("outrider.schemas.publish", "PublishResult"),
    ("outrider.schemas.review_finding", "ReviewFinding"),
    ("outrider.schemas.review_report", "ReviewMetrics"),
    ("outrider.schemas.review_report", "ReviewReport"),
    ("outrider.schemas.review_state", "ReviewState"),
    ("outrider.schemas.trace_candidate", "TraceCandidate"),
    ("outrider.schemas.trace_decision", "TraceDecision"),
    ("outrider.schemas.trace_fetched_file", "TraceFetchedFile"),
    ("outrider.schemas.triage_result", "TriageResult"),
)

# Every Outrider enum reachable from the ReviewState type graph. Enums are
# ext-encoded at ANY nesting depth (they stay as members through model_dump()), so
# an enum nested deep inside a finding still needs registration.
_ENUM_TYPES: tuple[tuple[str, str], ...] = (
    ("outrider.ast_facts.models", "SkipReason"),
    ("outrider.policy.findings", "EvidenceTier"),
    ("outrider.policy.severity", "FindingSeverity"),
    ("outrider.policy.severity", "FindingType"),
    ("outrider.schemas.hitl", "PerFindingOutcome"),
    ("outrider.schemas.review_finding", "PublishDestination"),
    ("outrider.schemas.review_finding", "ReviewDimension"),
    ("outrider.schemas.triage_result", "ReviewTier"),
    ("outrider.schemas.triage_result", "RiskLevel"),
)

# The explicit (module, name) allowlist passed to the checkpoint serializer. Exact
# pairs only — no module-prefix wildcards (the serializer intentionally does not
# support prefix allowlists; each symbol is listed exactly).
OUTRIDER_MSGPACK_ALLOWLIST: tuple[tuple[str, str], ...] = _MODEL_TYPES + _ENUM_TYPES


class _OutriderCheckpointSerde(JsonPlusSerializer):
    """Marker subclass identifying a serde produced by ``build_checkpoint_serde()``.

    Behaviourally identical to its base (the allowlist is passed to the inherited
    constructor). It exists so the runtime checkpointer guard (``tests/conftest.py``)
    can recognise a correctly-wired serde via ``isinstance`` on THIS project's
    class, instead of reading a langgraph-private attribute (e.g.
    ``_allowed_msgpack_modules``). A routine dependency bump could rename that
    private attribute, which would make the guard raise for every correctly-wired
    checkpointer — the exact upgrade fragility this module exists to avoid. The
    guard depends only on our own public surface (``is_outrider_checkpoint_serde``).
    """


def build_checkpoint_serde() -> JsonPlusSerializer:
    """Return a checkpoint serializer that permits Outrider's state types.

    Pass the result as ``serde=`` to every checkpointer construction site
    (``AsyncPostgresSaver`` and ``InMemorySaver`` alike — InMemorySaver
    round-trips through serde too, so it is NOT exempt from strict mode). The
    returned serializer behaves identically whether or not
    ``LANGGRAPH_STRICT_MSGPACK`` is set: an explicit allowlist always permits
    the listed types + the serializer's built-in safe types and blocks the rest.
    """
    return _OutriderCheckpointSerde(allowed_msgpack_modules=OUTRIDER_MSGPACK_ALLOWLIST)


def is_outrider_checkpoint_serde(serde: object) -> bool:
    """True iff ``serde`` was produced by ``build_checkpoint_serde()``.

    Public identity check used by the runtime checkpointer guard so its
    enforcement does not couple to langgraph-private internals (see
    ``_OutriderCheckpointSerde``).
    """
    return isinstance(serde, _OutriderCheckpointSerde)


__all__ = [
    "OUTRIDER_MSGPACK_ALLOWLIST",
    "build_checkpoint_serde",
    "is_outrider_checkpoint_serde",
]
