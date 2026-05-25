"""Pydantic-validated fixture factories for the eval harness.

Each factory's `.create(**overrides)` classmethod produces a schema-valid
instance of its target type with `is_eval=True` set on every audit-event /
review surface that carries the column (loud-failure pattern: a factory
that omits the flag is a bug, caught by the `eval_db` fixture's teardown
integrity gate in `tests/eval/conftest.py` — query-then-drop ordering,
UNION ALL across all five `is_eval`-bearing tables per `docs/schema.md`
"Eval isolation"). `ReviewFinding` is a cross-boundary type with no
`is_eval` field; the flag lives on the corresponding row in `findings`,
not on the cross-boundary type — see `FindingFactory` below.

What each factory produces:

  - `ReviewFactory.create()` → `dict[str, Any]` matching the `reviews`
    table column shape. Caller inserts via SQLAlchemy ORM
    (`Review(**factory_dict)`) or Core. The Review FK to `installations`
    is the caller's responsibility — factories don't manage FK targets.
  - `FindingFactory.create()` → `ReviewFinding` Pydantic instance with
    `content_hash` computed via `compute_finding_content_hash()`.
    `ReviewFinding` is a cross-boundary type with no `is_eval` field;
    the eval-isolation flag lives on the corresponding row in `findings`,
    not on the cross-boundary type.
  - `FindingEventFactory.create()` → `FindingEvent` Pydantic instance
    with canonical hash + `is_eval=True`.
  - `TraceDecisionEventFactory.create()` → `TraceDecisionEvent` (frozen
    + extra=forbid) with the three-rule cross-field validator satisfied
    (resolution_status="resolved" + target_file equal to the single
    `resolved_candidate_paths` entry, per #024 amendment to #017;
    overrides can supply unresolved/ambiguous shapes).
  - `HITLRequestEventFactory.create()` → `HITLRequestEvent` with
    `is_eval=True`.
  - `HITLDecisionEventFactory.create()` → `HITLDecisionEvent` with
    `is_eval=True` and at least one `PerFindingDecision` in `decisions`.

`PRContextFactory` is deferred to the webhook-receiver spec per the
eval-harness spec's Input boundary held item.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from outrider.audit.events import (
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
)
from outrider.policy import (
    EvidenceTier,
    FindingType,
    lookup_severity,
)
from outrider.schemas import (
    PerFindingDecision,
    PerFindingOutcome,
    ReviewDimension,
    ReviewFinding,
)

# Synthetic installation_id outside any plausible GitHub installation range
# per `docs/schema.md` "Eval isolation" rule. GitHub installation IDs are
# positive integers; using a negative value makes eval rows clearly
# distinguishable in any installation-scoped query and prevents collision
# with real installations the schema can't otherwise distinguish (the FK
# is `bigint`, no SQL-expressible "real vs synthetic" gate).
_EVAL_SYNTHETIC_INSTALLATION_ID = -1


class ReviewFactory:
    """Factory for `db.models.Review`-shaped row dicts.

    Returns a `dict[str, Any]` rather than a SQLAlchemy ORM instance so
    callers can spread it (`Review(**dict)`) or insert via Core. The
    `installation_id` FK target is the caller's responsibility.
    """

    @classmethod
    def create(cls, **overrides: Any) -> dict[str, Any]:
        _reject_is_eval_false(overrides)
        now = datetime.now(UTC)
        defaults: dict[str, Any] = {
            "id": uuid4(),
            "installation_id": _EVAL_SYNTHETIC_INSTALLATION_ID,
            "repo_id": 67890,
            "pr_number": 1,
            # Unique-per-call to avoid colliding with the
            # uq_review_natural_key UNIQUE constraint on
            # (repo_id, pr_number, head_sha) when a test inserts
            # multiple default-shaped reviews into the same DB.
            # Real git SHA-1 is 40 hex chars; uuid4().hex is 32, so
            # double-and-truncate gives a SHA-shaped 40-char string.
            "head_sha": (uuid4().hex + uuid4().hex)[:40],
            "status": "completed",
            "hitl_request": None,
            "hitl_decision": None,
            "files_examined": 0,
            "files_traced_beyond_diff": 0,
            "llm_calls_made": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": Decimal("0"),
            "wall_clock_seconds": Decimal("0"),
            "is_eval": True,
            "retention_expires_at": now + timedelta(days=180),
        }
        return {**defaults, **overrides}


class FindingFactory:
    """Factory for `ReviewFinding` Pydantic instances.

    Computes `content_hash` via `compute_finding_content_hash()` to match
    the canonical SHA-256 contract per spec §8.5. Overrides that change
    file_path / line_start / line_end / finding_type recompute the hash
    automatically; explicit `content_hash=...` overrides skip the recompute.
    """

    @classmethod
    def create(cls, **overrides: Any) -> ReviewFinding:
        finding_type = _normalize_finding_type(overrides)
        file_path = overrides.get("file_path", "src/foo.py")
        line_start = overrides.get("line_start", 10)
        line_end = overrides.get("line_end", 12)

        if "content_hash" not in overrides:
            overrides["content_hash"] = compute_finding_content_hash(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                finding_type=finding_type,
            )

        # Severity comes from SEVERITY_POLICY[finding_type] per
        # `severity-set-by-policy`. Hard-coding a default would drift if
        # the policy table changes; deriving via lookup_severity tracks
        # the canonical mapping. Explicit `severity=...` override still
        # wins (tests of severity-override paths need this).
        if "severity" not in overrides:
            overrides["severity"] = lookup_severity(finding_type)

        defaults: dict[str, Any] = {
            "review_id": uuid4(),
            "installation_id": _EVAL_SYNTHETIC_INSTALLATION_ID,
            "policy_version": "1.0.0",
            "finding_type": finding_type,
            "dimension": ReviewDimension.SECURITY,
            "evidence_tier": EvidenceTier.JUDGED,
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "title": "Eval finding",
            "description": "Generated by FindingFactory for the eval harness.",
            "evidence": "...",
            "content_hash": "0" * 64,
            # Per DECISIONS.md#025: admitted findings carry proposal_hash.
            # Unique-by-default (`uuid4().hex + uuid4().hex` = 64 hex
            # chars matching the SHA-256 shape) so eval scenarios that
            # compose multiple factory findings into one AnalysisRound
            # don't accidentally trip the
            # `_enforce_findings_proposal_hash_unique` validator on
            # identical defaults. Per-finding overrides can still pin
            # specific hashes when a scenario needs exact reproducibility.
            # Per CodeRabbit round-9 N4 — cohort sibling of the round-6
            # `_build_finding` fix in `test_trace_node.py`.
            "proposal_hash": uuid4().hex + uuid4().hex,
        }
        return ReviewFinding(**{**defaults, **overrides})


class FindingEventFactory:
    """Factory for `FindingEvent` audit-event Pydantic instances.

    `is_eval=True` by default per harness discipline. `finding_content_hash`
    computed via the canonical helper; the FindingEvent validator verifies
    equality on construction.
    """

    @classmethod
    def create(cls, **overrides: Any) -> FindingEvent:
        _reject_is_eval_false(overrides)
        finding_type = _normalize_finding_type(overrides)
        file_path = overrides.get("file_path", "src/foo.py")
        line_start = overrides.get("line_start", 10)
        line_end = overrides.get("line_end", 12)

        if "finding_content_hash" not in overrides:
            overrides["finding_content_hash"] = compute_finding_content_hash(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                finding_type=finding_type,
            )

        # Severity from SEVERITY_POLICY[finding_type] per
        # `severity-set-by-policy`; explicit override still wins.
        if "severity" not in overrides:
            overrides["severity"] = lookup_severity(finding_type)

        defaults: dict[str, Any] = {
            "review_id": uuid4(),
            "is_eval": True,
            "finding_id": uuid4(),
            "finding_type": finding_type,
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "dimension": ReviewDimension.SECURITY,
            "evidence_tier": EvidenceTier.JUDGED,
            "policy_version": "1.0.0",
            # Per DECISIONS.md#025: audit-shadow mirror of ReviewFinding.proposal_hash.
            "proposal_hash": "a" * 64,
        }
        return FindingEvent(**{**defaults, **overrides})


class TraceDecisionEventFactory:
    """Factory for `TraceDecisionEvent` Pydantic instances.

    Defaults to `resolution_status="resolved"` with `target_file` equal
    to the single `resolved_candidate_paths` entry (satisfies the
    three-rule cross-field validator per `DECISIONS.md#017` × #024
    amendment — `candidates_considered` was renamed to the parallel
    `proposed_import_strings` + `resolved_candidate_paths` tuples).
    Overrides for unresolved/ambiguous outcomes must also set
    `target_file=None` and adjust the two tuples accordingly.
    """

    @classmethod
    def create(cls, **overrides: Any) -> TraceDecisionEvent:
        _reject_is_eval_false(overrides)
        defaults: dict[str, Any] = {
            "review_id": uuid4(),
            "is_eval": True,
            "source_finding_id": uuid4(),
            "target_file": "src/bar.py",
            "reason": "called from src/foo.py:10 via direct import",
            "resolution_status": "resolved",
            # Per DECISIONS.md#024 (Accepted 2026-05-24): trace candidates are
            # dotted Python import strings; resolver outputs are file paths.
            # Parallel tuples carry the LLM-proposed + resolver-output halves.
            "proposed_import_strings": ("bar", "baz"),
            "resolved_candidate_paths": ("src/bar.py",),
        }
        return TraceDecisionEvent(**{**defaults, **overrides})


class HITLRequestEventFactory:
    """Factory for `HITLRequestEvent` Pydantic instances. `is_eval=True` default."""

    @classmethod
    def create(cls, **overrides: Any) -> HITLRequestEvent:
        _reject_is_eval_false(overrides)
        now = datetime.now(UTC)
        defaults: dict[str, Any] = {
            "review_id": uuid4(),
            "is_eval": True,
            "findings_requiring_approval": (uuid4(),),
            "auto_post_findings": (),
            "expires_at": now + timedelta(minutes=30),
        }
        return HITLRequestEvent(**{**defaults, **overrides})


class HITLDecisionEventFactory:
    """Factory for `HITLDecisionEvent` Pydantic instances.

    Constructs a single APPROVE PerFindingDecision in `decisions` by default;
    overrides can supply alternative decision sets.
    """

    @classmethod
    def create(cls, **overrides: Any) -> HITLDecisionEvent:
        _reject_is_eval_false(overrides)
        if "decisions" not in overrides:
            overrides["decisions"] = (
                PerFindingDecision(
                    finding_id=uuid4(),
                    outcome=PerFindingOutcome.APPROVE,
                    reason="",
                ),
            )
        defaults: dict[str, Any] = {
            "review_id": uuid4(),
            "is_eval": True,
            "reviewer_id": "eval-reviewer@example.com",
            "decision_latency_seconds": 42.5,
        }
        return HITLDecisionEvent(**{**defaults, **overrides})


def _normalize_finding_type(overrides: dict[str, Any]) -> FindingType:
    """Coerce `finding_type` override to a `FindingType` enum, defaulting to SQL_INJECTION.

    Default is `FindingType.SQL_INJECTION` (when absent from overrides). Caller
    may pass either the enum (`FindingType.SQL_INJECTION`) or a valid str-enum
    value (`"sql_injection"`); both are accepted and normalized to the enum
    form. Anything else raises `ValueError` at the factory call site (loud-
    failure, naming the bad value).

    Without this normalization, the str-input path silently broke the
    factory's `compute_finding_content_hash()` and `lookup_severity()`
    derivations (both gated on `isinstance(..., FindingType)`), leaving
    `content_hash` at the placeholder `"0"*64` and `severity` unset —
    Pydantic would then either coerce the str to enum and raise a confusing
    "missing severity" ValidationError, or accept the placeholder hash and
    fail downstream at the audit-event equality verifier.

    Mutates `overrides` in place (sets the normalized enum value back) so
    the model construction below sees the canonical type.
    """
    finding_type = overrides.get("finding_type", FindingType.SQL_INJECTION)
    if not isinstance(finding_type, FindingType):
        try:
            finding_type = FindingType(finding_type)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Factory received finding_type={finding_type!r} which is "
                f"not a valid FindingType. Pass the enum "
                f"(e.g., FindingType.SQL_INJECTION) or a valid str-enum value."
            ) from exc
        overrides["finding_type"] = finding_type
    return finding_type


def _reject_is_eval_false(overrides: dict[str, Any]) -> None:
    """Reject any `is_eval` override that isn't exactly `True`.

    The `eval_db` teardown integrity gate catches violations after the fact
    (UNION ALL across 5 tables); this helper catches them at construction so
    the error names the factory + caller, not just the row id at teardown.
    Loud-failure pattern matches `PerFindingDecision.reason` no-default and
    `proposed_import_strings` no-default (formerly `candidates_considered`,
    renamed per #024) — fail where the bug is, not later.

    The check is `is not True` (not `is False`) because Pydantic V2 will
    coerce truthy/falsy values like `0`, `""`, `"false"` to `False` in
    lenient mode, which would slip past a strict-False check and land a
    non-eval record. Three legal states:
      - `is_eval` not in overrides → factory default `True` applies
      - `is_eval=True` explicit → permitted (no-op vs default)
      - anything else → rejected here
    """
    if "is_eval" in overrides and overrides["is_eval"] is not True:
        raise ValueError(
            f"Eval-harness factory cannot construct a record with "
            f"is_eval={overrides['is_eval']!r}. Per docs/testing.md, every "
            "factory output must carry is_eval=True. If you genuinely need "
            "a non-eval record in a test, construct the type directly (not "
            "via the factory)."
        )


__all__ = [
    "FindingEventFactory",
    "FindingFactory",
    "HITLDecisionEventFactory",
    "HITLRequestEventFactory",
    "ReviewFactory",
    "TraceDecisionEventFactory",
]
