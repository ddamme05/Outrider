"""Audit event hierarchy + discriminated union per docs/spec.md §7.2.1 + §8.2.

Re-exports the public symbols of the audit package. The emitter
(`audit/emitter.py`, separate spec) constructs concrete events; replay
(`audit/replay.py`, separate spec) reconstructs them via `AuditEvent`.
"""

from outrider.audit.events import (
    AgentTransitionEvent,
    AuditEvent,
    AuditEventAdapter,
    AuditEventBase,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    LLMCallEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
)
from outrider.audit.sinks import PhaseEventSink

__all__ = [
    "AgentTransitionEvent",
    "AuditEvent",
    "AuditEventAdapter",
    "AuditEventBase",
    "ContextManifestEntry",
    "FileExaminationEvent",
    "FindingEvent",
    "HITLDecisionEvent",
    "HITLRequestEvent",
    "LLMCallEvent",
    "PhaseEventSink",
    "PublishEvent",
    "PublishRoutingEvent",
    "ReviewPhaseEvent",
    "TraceDecisionEvent",
    "compute_finding_content_hash",
]
