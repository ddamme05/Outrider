# OBSERVED-tier query metadata per specs/2026-06-14-observed-query-library-v1.md.
"""`QueryClass` + `ObservedQuery`: the registry-side metadata for the
OBSERVED-tier security query library (Cost Lever 3).

The deterministic OBSERVED producer (the analyze node, a later increment)
maps a `QueryMatchSpan` from `queries.registry.match(id, ...)` to a
`ReviewFinding` with NO model text: `finding_type` drives the
`SEVERITY_POLICY` severity (never model-set), and the static
`title`/`description` text is held here, registry-side. `query_class` is
the default-deny routing tag (`#048` + the spec's promotion gate).

Lives in `queries/` (not `ast_facts/`) because it references
`policy.FindingType`; the parse layer (`ast_facts/`) must not depend on
policy. `queries/` is a higher layer and may import `policy/`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

# Runtime import (not TYPE_CHECKING): Pydantic resolves field annotations at
# model-build time, so `FindingType` must be in the runtime namespace.
from outrider.policy.severity import FindingType  # noqa: TC001


class QueryClass(StrEnum):
    """Routing class for an OBSERVED query.

    Default-deny per `DECISIONS.md#049`: every query is `SIGNAL_ONLY` (it
    emits OBSERVED evidence that AUGMENTS the LLM pass, never skips it)
    unless explicitly promoted to `SKIP_SAFE` by a shadow comparison
    proving the skip loses no JUDGED finding. V1 seeds ZERO `SKIP_SAFE`
    queries — the class exists so the skip-routing telemetry can record
    eligibility; a `SKIP_SAFE` query is exercised only by the
    `observed_skip_safe` eval scenario or a later evidence-gated promotion.
    """

    SIGNAL_ONLY = "signal_only"
    SKIP_SAFE = "skip_safe"


class ObservedQuery(BaseModel):
    """Registry-side metadata for one OBSERVED-tier security query.

    `title`/`description` are DETERMINISTIC static text (not generated, not
    interpolated with attacker-controlled source — the matched code rides
    in `ReviewFinding.evidence`, which is data, not a format string). The
    output/routing fields (`finding_type`, `query_class`, `title`,
    `description`) enter the analyze cache-key digest
    (`queries.registry._registry_digest`, derived from the model minus
    `_DIGEST_EXCLUDED_OBSERVED_FIELDS` per FUP-181), so a metadata edit
    invalidates stale cached analyze outcomes. `query_match_id` is the digest
    KEY (folded as the id, not as a field); `filename` is excluded — an impl
    detail, since the `.scm` BODY is folded, not its name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_match_id: str
    filename: str
    finding_type: FindingType
    query_class: QueryClass = QueryClass.SIGNAL_ONLY
    title: str
    description: str
