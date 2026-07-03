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
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

# Runtime import (not TYPE_CHECKING): Pydantic resolves field annotations at
# model-build time, so `FindingType` must be in the runtime namespace.
from outrider.policy.severity import FindingType  # noqa: TC001

# Catalog partition key (specs/2026-07-03-js-ts-observed-query-catalog.md):
# which language's source a query is written against. "javascript" covers the
# whole JS/TS family — one catalog, compiled per grammar dialect by the
# registry (`_GRAMMARS_BY_QUERY_LANGUAGE`). Distinct from the registry's
# grammar kind: a QueryLanguage selects the query SET for a file; the grammar
# selects the compiled variant + parser for its bytes.
QueryLanguage = Literal["python", "javascript"]


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


class BindingRule(BaseModel):
    """Deterministic import-binding admission for a name-anchored OBSERVED
    query — the producer-side proof that a matched NAME actually binds to
    the dangerous API, joined against the file's `ast_facts` imports
    (`ImportRef.names` carries LOCAL binding names for every JS/TS form:
    ESM named/default/namespace and CJS `require`, whole-module and
    destructured).

    `anchor_import`: the match's anchor identifier — the `@_recv` receiver
    capture when present, else the `@_fn` callee capture — must be bound by
    an import whose `module` is in `modules`. `module_presence`: the FILE
    must import at least one of `modules` (for sinks whose receiver is a
    derived variable — a DB pool — where per-receiver proof needs
    assignment-flow, not an import join).

    `modules` is a sorted tuple, not a set: the rule rides into
    `_registry_digest` via `model_dump()`, and set iteration order is
    hash-randomized across processes — a set field would make the digest
    (an analyze cache-key input) nondeterministic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["anchor_import", "module_presence"]
    modules: tuple[str, ...]

    @field_validator("modules")
    @classmethod
    def _sorted_nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("BindingRule.modules must be non-empty")
        return tuple(sorted(set(v)))


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
    detail, since the `.scm` BODY is folded, not its name. `language` selects
    which files the query runs against (and which grammars compile it); it
    folds into the digest like the other routing fields, as does `binding`
    (an admission-affecting rule — a binding edit changes which matches are
    admitted, so it must invalidate cached outcomes).

    `binding=None` means the match is admitted on structure alone — correct
    for globals (`eval`, `Function`) and for queries whose pattern is
    already self-proving (`process.env` receiver). Every python-catalog
    entry is `None`: the binding step is a JS/TS admission rule; Python
    OBSERVED behavior is byte-stable (its sibling gap is FUP-184 scope).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_match_id: str
    filename: str
    language: QueryLanguage
    finding_type: FindingType
    query_class: QueryClass = QueryClass.SIGNAL_ONLY
    title: str
    description: str
    binding: BindingRule | None = None
