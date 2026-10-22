"""Allowlist Outrider data types for checkpoint serialization.

HITL resume round-trips ReviewState through the serializer. Register its reachable
models and enums so strict msgpack decoding can restore them. This preserves
`state-is-pure-data`: typed data is allowed; live clients and sessions are not.

Channel-level models and enums need explicit entries. Nested model data is
inlined by its parent; listing those models also covers a future channel use.
tests/unit/test_checkpoint_serde.py checks the allowlist against the state type
graph so new state types cannot silently break resume."""

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
    """Identify this project's configured serializer without reading vendor internals.

    The inherited constructor receives the allowlist. The checkpointer guard uses
    is_outrider_checkpoint_serde instead of a private serializer attribute."""


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
    """Identify this project's serializer for the runtime checkpointer guard."""
    return isinstance(serde, _OutriderCheckpointSerde)


__all__ = [
    "OUTRIDER_MSGPACK_ALLOWLIST",
    "build_checkpoint_serde",
    "is_outrider_checkpoint_serde",
]
