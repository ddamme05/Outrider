"""Choose skip, degraded or clean analysis from a file's parse and patch.

The decision carries changed scope units and their clipped hunks. analyze.py
turns it into examination events and prompt selection. This module has no
model calls or raw tree-sitter nodes, so structural scenarios can use it directly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from outrider.ast_facts.models import SkipReason
from outrider.audit.events import DegradationReason
from outrider.coordinates import (
    added_line_numbers,
    patched_file_has_added_lines,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
)

if TYPE_CHECKING:
    from unidiff import PatchedFile

    from outrider.ast_facts.models import ParseResult, ScopeUnit

# DegradationReason owns the reason vocabulary; the decision properties below
# map those reasons to analysis gates.

# FileExaminationEvent statuses. Intake admits decoded strings, so analysis's
# string-to-bytes round-trip cannot produce a failed decode in the current path.
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
    degradation_reason: DegradationReason | None = None
    included_scope_units: tuple[ScopeUnit, ...] = ()
    included_clipped_hunks: tuple[tuple[str, ...], ...] = ()

    @property
    def is_module_level_route(self) -> bool:
        # The clean-parse module-scope degraded route (DECISIONS.md#062).
        return self.degradation_reason is DegradationReason.MODULE_LEVEL_OBSERVED_MATCH

    @property
    def runs_observed_producer(self) -> bool:
        # Structural proof is allowed only for clean parses, including the
        # module-scope route. Error-caused degradation keeps the producer off.
        return self.mode == "clean" or self.is_module_level_route

    @property
    def emits_skip_shadow(self) -> bool:
        # #049 skip-shadow telemetry is clean-mode-only: degraded routes —
        # the module route included (its matches are excluded from skip
        # coverage, DECISIONS.md#062) — emit no shadow event.
        return self.mode == "clean"

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
    """Return changed scopes and clipped hunks in matching tuple positions.

    Include a scope only when it has added lines and non-empty clipped hunks.
    Coordinate calculations stay in coordinates; this function selects prompt input."""
    included: list[ScopeUnit] = []
    hunks: list[tuple[str, ...]] = []
    for scope_unit in scope_units:
        if not scope_unit_has_added_lines(scope_unit, patched_file):
            continue
        clipped = scope_unit_diff_hunks(scope_unit, patched_file)
        if not clipped:
            continue
        included.append(scope_unit)
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
      DECISIONS.md#062). Parse-error precedence: the
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
            mode="degraded",
            parse_status="failed",
            degradation_reason=DegradationReason.PARSE_FAILED,
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
                degradation_reason=DegradationReason.TREE_HAS_ERROR_NO_SCOPE,
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
            # precedence for the MODULE arm: it never anchors proof on a tree
            # carrying error nodes — the node gates the candidate on a fully
            # error-free parse, so this branch is unreachable for
            # error-bearing files; normal scope-contained production has its
            # own, per-included-scope error discipline).
            return DegradationDecision(
                mode="degraded",
                parse_status="clean",
                degradation_reason=DegradationReason.MODULE_LEVEL_OBSERVED_MATCH,
            )
        return DegradationDecision(
            mode="skip", parse_status="clean", skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS
        )
    if any(
        parse_result.has_error.get(scope_unit.unit_id, False) for scope_unit in included_scope_units
    ):
        return DegradationDecision(
            mode="degraded",
            parse_status="degraded",
            degradation_reason=DegradationReason.TREE_HAS_ERROR_IN_CHANGED_REGIONS,
            included_scope_units=included_scope_units,
            included_clipped_hunks=included_clipped_hunks,
        )
    return DegradationDecision(
        mode="clean",
        parse_status="clean",
        included_scope_units=included_scope_units,
        included_clipped_hunks=included_clipped_hunks,
    )
