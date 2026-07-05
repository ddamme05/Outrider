"""Pure analyze degradation decision — skip / degraded / clean.

The analyze node's outcome determination (does a changed file get skipped,
reviewed in degraded mode, or reviewed cleanly?) extracted as a pure, LLM-free
function so structural eval scenarios — which validate `ast_facts`/`coordinates`
without an LLM — can exercise it directly. `decide_degradation` returns a typed
`DegradationDecision`; the analyze node is the ONLY place that turns that decision
into node behavior (`_emit_skip`, `render_degraded`, `render`).

This module imports no LLM or prompt machinery (that is exactly why it is
separate from `analyze.py`, whose module-level `llm`/`prompts` imports would
otherwise be pulled into a structural test). It consumes a `ParseResult` and the
file's `PatchedFile` — both domain models — and touches no raw `tree_sitter.Node`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from outrider.ast_facts.models import SkipReason
from outrider.coordinates import (
    added_line_numbers,
    patched_file_has_added_lines,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
)

if TYPE_CHECKING:
    from unidiff import PatchedFile

    from outrider.ast_facts.models import ParseResult, ScopeUnit

# Bidirectionally coupled with `LLMRequest.degraded_mode` per
# `_enforce_degradation_provenance` (llm/base.py). A new value here must be added in
# LOCKSTEP to `LLMRequest.degradation_reason` (llm/base.py) and
# `LLMCallEvent.degradation_reason` (audit/events.py) — three independent literals.
# `"parse_failed"` is V1-unreachable (raw-bytes intake path, FUP-053).
# `"tree_has_error_no_scope"` is the no-scope syntax-error case per DECISIONS.md#033:
# a changed addable line intersects a tree error but no scope recovered there
# (distinct from `"tree_has_error_in_changed_regions"`, which needs a recovered scope).
# `"module_level_observed_match"` is the module-scope routing reason
# (specs/2026-07-04-module-scope-admission-arm.md) and the one reason that is
# NOT a parse defect: the parse is CLEAN, but a module-only diff carries an
# eligible OBSERVED match, so the file degrades (bounded-hunks JUDGED review +
# module-level OBSERVED emission) instead of skipping at NO_CHANGED_SCOPE_UNITS.
_DegradationReason = Literal[
    "parse_failed",
    "tree_has_error_in_changed_regions",
    "tree_has_error_no_scope",
    "module_level_observed_match",
]

# `FileExaminationEvent.parse_status` values for the analyze node.
# `"failed"` is V1-unreachable for the same reason as `"parse_failed"` above.
_ParseStatus = Literal["clean", "failed", "degraded", "skipped"]


@dataclass(frozen=True, slots=True)
class DegradationDecision:
    """Typed outcome of the analyze degradation decision.

    `mode` is the discriminator: `"skip"` carries a `skip_reason`; `"degraded"`
    carries a `degradation_reason`; `"clean"` carries neither. `degraded`/`clean`
    additionally carry the changed scope units + their clipped hunks (the prompt
    inputs); `skip` carries none. `parse_status` is the `FileExaminationEvent`
    value the node emits on the non-skip path. The node maps this decision to
    behavior — it is the single place that does so.
    """

    mode: Literal["skip", "degraded", "clean"]
    parse_status: _ParseStatus
    skip_reason: SkipReason | None = None
    degradation_reason: _DegradationReason | None = None
    included_scope_units: tuple[ScopeUnit, ...] = ()
    included_clipped_hunks: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        """The discriminator must agree with the reason fields (fail-loud)."""
        if self.mode == "skip" and self.skip_reason is None:
            raise ValueError("DegradationDecision: mode='skip' requires a skip_reason")
        if self.mode == "degraded" and self.degradation_reason is None:
            raise ValueError("DegradationDecision: mode='degraded' requires a degradation_reason")
        if self.mode != "skip" and self.skip_reason is not None:
            raise ValueError(f"DegradationDecision: mode={self.mode!r} must not carry skip_reason")
        if self.mode != "degraded" and self.degradation_reason is not None:
            raise ValueError(
                f"DegradationDecision: mode={self.mode!r} must not carry degradation_reason"
            )


def _intersect_changed_scope_units(
    scope_units: tuple[ScopeUnit, ...],
    patched_file: PatchedFile,
) -> tuple[tuple[ScopeUnit, ...], tuple[tuple[str, ...], ...]]:
    """Return `(included_units, clipped_hunks_per_unit)` for the intersection.

    A scope unit is "included" iff `coordinates.scope_unit_has_added_lines`
    returns True AND `coordinates.scope_unit_diff_hunks` returns non-empty. The
    two tuples share indices: `included_units[i]` has clipped hunks
    `clipped_hunks_per_unit[i]`. Empty inputs / no intersection returns `((), ())`.

    Composition of two coordinates surfaces — the orchestration lives here
    (the decision picks which units feed which prompt), the coordinate math lives
    there. Backs the `outcome="skipped+NO_CHANGED_SCOPE_UNITS"` discriminator and
    the `clean+full_llm` prompt's `diff_hunks` block.
    """
    included: list[ScopeUnit] = []
    hunks: list[tuple[str, ...]] = []
    for su in scope_units:
        if not scope_unit_has_added_lines(su, patched_file):
            continue
        clipped = scope_unit_diff_hunks(su, patched_file)
        if not clipped:
            continue
        included.append(su)
        hunks.append(clipped)
    return tuple(included), tuple(hunks)


def decide_degradation(
    parse_result: ParseResult,
    patched_file: PatchedFile | None,
    *,
    module_level_observed_candidate: bool = False,
) -> DegradationDecision:
    """Decide skip / degraded / clean for one changed file from its parse + patch.

    Pure mirror of the analyze node's outcome determination (no LLM, no audit, no
    side effects). The node calls this once per PARSED file, then turns the returned
    `DegradationDecision` into behavior. Outcomes:

    - `failed` (V1-unreachable) with no addable text → `skip` NO_REVIEWABLE_CONTEXT;
      with addable text → `degraded` (`parse_failed`), no scope context.
    - `clean` with no patch / no changed scope units → `skip` NO_CHANGED_SCOPE_UNITS,
      UNLESS `module_level_observed_candidate` is set (the node precomputed that an
      eligible OBSERVED query admits a module-level match on the added lines) →
      `degraded` (`module_level_observed_match`, parse_status stays `clean` — a
      routing choice, not a parse defect;
      specs/2026-07-04-module-scope-admission-arm.md). Parse-error precedence: the
      error-lines check runs FIRST, so a syntax-error file degrades as
      `tree_has_error_no_scope` and the candidate flag is never consulted.
    - `clean` with a changed scope unit that carries a tree error → `degraded`
      (`tree_has_error_in_changed_regions`).
    - `clean` otherwise → `clean`, carrying the changed scope units + hunks.

    Parser-stage skips (`parser_outcome == "skipped"`) are NOT handled here — the
    node returns those before it calls this, because the precondition `patched_file`
    (looked up from a possibly-malformed patch) can raise for a skipped file, and a
    skipped file must skip cleanly regardless of its patch. Passing a skipped result
    here is a caller-contract violation and raises.
    """
    if parse_result.parser_outcome == "skipped":
        raise RuntimeError(
            "decide_degradation called on a parser-skipped result; the node must "
            "handle parser-stage skips before calling this (a skipped file may carry "
            "a malformed patch that lookup_patched_file rejects)."
        )

    if parse_result.parser_outcome == "failed":
        if patched_file is None or not patched_file_has_added_lines(patched_file):
            return DegradationDecision(
                mode="skip",
                parse_status="failed",
                skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
            )
        # failed+degraded_llm: no scope context survives a failed parse.
        return DegradationDecision(
            mode="degraded", parse_status="failed", degradation_reason="parse_failed"
        )

    # parser_outcome == "clean".
    if patched_file is None:
        return DegradationDecision(
            mode="skip", parse_status="clean", skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS
        )
    included_scope_units, included_clipped_hunks = _intersect_changed_scope_units(
        tuple(parse_result.scope_units), patched_file
    )
    if not included_scope_units:
        # No changed scope unit. Before skipping, the no-scope degraded case
        # (DECISIONS.md#033): a tree error breaking a scope's header yields no scope,
        # so `has_error` can't see it. If an ADDABLE changed line intersects an
        # `error_lines` line, degrade (JUDGED-only review of the bounded hunks)
        # instead of silently skipping — honoring `parse-errors-degrade-to-judged`.
        # Addable-lines-only: a pure deletion over an error line carries no target
        # line, so it stays a skip (FUP-050 limitation, an explicit non-goal).
        if parse_result.error_lines & added_line_numbers(patched_file):
            return DegradationDecision(
                mode="degraded",
                parse_status="degraded",
                degradation_reason="tree_has_error_no_scope",
                # No scope recovered → no scope context; the degraded prompt uses the
                # bounded diff hunks, so included_scope_units/hunks stay ().
            )
        if module_level_observed_candidate:
            # Module-scope admission arm: an eligible OBSERVED query admits a
            # module-level match on the added lines, so the file gets a
            # degraded (bounded-hunks, JUDGED-only) LLM review plus the
            # module-level OBSERVED emission instead of a silent skip. The
            # parse IS clean — parse_status stays truthful; the reason rides
            # LLMCallEvent. Ordered AFTER the error-lines check (parse-error
            # precedence: the producer never runs on an error-recovered tree;
            # the node additionally gates the candidate on an error-free
            # parse, so this branch is unreachable for error-bearing files).
            return DegradationDecision(
                mode="degraded",
                parse_status="clean",
                degradation_reason="module_level_observed_match",
            )
        return DegradationDecision(
            mode="skip", parse_status="clean", skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS
        )
    if any(parse_result.has_error.get(su.unit_id, False) for su in included_scope_units):
        return DegradationDecision(
            mode="degraded",
            parse_status="degraded",
            degradation_reason="tree_has_error_in_changed_regions",
            included_scope_units=included_scope_units,
            included_clipped_hunks=included_clipped_hunks,
        )
    return DegradationDecision(
        mode="clean",
        parse_status="clean",
        included_scope_units=included_scope_units,
        included_clipped_hunks=included_clipped_hunks,
    )
