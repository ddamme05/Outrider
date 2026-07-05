# AST-facts domain models per docs/spec.md §5.4 + specs/2026-04-30-ast-facts-module.md.
# AST firewall per docs/trust-boundaries.md §4.
"""Domain models for the AST-facts module.

This file is the **single source of truth for typed shapes** in the
ast_facts/ surface. All Pydantic models, Literal types, and the str-enum
live here; pure logic lives in `parser_outcome.py` and adapter code in
`python_adapter.py`. Audit code (`outrider/audit/events.py`) imports
`SkipReason` from this module per `DECISIONS.md#018` point 6 — this
module must therefore stay import-light (no `tree_sitter`, no adapter
code), and `outrider/ast_facts/__init__.py` lazy-loads `parse_python` so
that `from outrider.ast_facts.models import SkipReason` doesn't pull
the adapter into audit's module graph.

Field validators are added only on `ParseResult`, `ImportResolution`,
`QueryMatchSpan`/`QueryCaptureSpan`, and `ExclusionRule`, where
load-bearing for replay correctness or where canonical contradictions
would otherwise silently produce wrong results. The §5.4 canonical
types (`ScopeUnit`, `ImportRef`, `CallSite`, `AssignmentSite`,
`ChangedRegion`, `SymbolCandidate`) carry no validators beyond their
typing — per spec-fidelity discipline, a canonical type is shipped
verbatim unless `DECISIONS.md` amends it.
"""

import hashlib
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_unit_id(file_path: str, *, kind: str, qualified_name: str) -> str:
    """Stable hash of (file_path, kind, qualified_name) per Internal contracts.

    `kind` and `qualified_name` are keyword-only — they're adjacent
    same-typed `str` parameters, and a positional swap would silently
    produce a different hash, which IS the dedup/index key for
    `ParseResult.has_error` and the `enclosing_scope_id` ref target on
    `CallSite` / `AssignmentSite`. Same misuse-resistance pattern as
    `outrider.audit.events.compute_finding_content_hash` and
    `outrider.llm.pricing.compute_cost_usd`. `file_path` stays
    positional as the natural subject of the call.

    Used as the dict key for `ParseResult.has_error` and as the
    `enclosing_scope_id` reference target for `CallSite` / `AssignmentSite`.
    Determinism is load-bearing for replay correctness — same source bytes
    produce the same `unit_id` across adapter invocations.
    """
    payload = f"{file_path}\x00{kind}\x00{qualified_name}".encode()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Literal types and the SkipReason enum
# ---------------------------------------------------------------------------

ParserOutcome = Literal["clean", "failed", "skipped"]
"""File-level adapter verdict. Three values; deliberately distinct from the
canonical four-value `FileExaminationEvent.parse_status` (which adds
`degraded`, computed downstream by the consuming node)."""

ComputedParserOutcome = Literal["clean", "failed"]
"""Strictly narrower than `ParserOutcome` — the return type of the
`compute_parser_outcome` Protocol method, which is only invoked AFTER
`parse_python` has resolved skip-classification. Makes "compute_parser_outcome
never returns skipped" type-checkable, not just prose-checkable."""

ResolutionStatus = Literal["resolved", "ambiguous", "unresolved"]
"""Outcome of simple-direct-import resolution. Carried by `ImportResolution.status`
and consumed by the trace node per `DECISIONS.md#017`."""


class TrivialityReason(StrEnum):
    """Why the trivial-scope classifier ruled the way it did, per scope.

    Produced by `ast_facts.triviality.classify_scope_triviality`; carried
    on `ScopeExclusionEvent` entries (audit side imports this from the
    light-types surface — the classifier module itself loads tree-sitter
    and stays behind the lazy `__getattr__`). `ALL_LINES_ORDINARY_COMMENT`
    is the only trivial=True reason; every other value is a fail-closed
    veto. See specs/2026-06-10-trivial-scope-filter.md.
    """

    ALL_LINES_ORDINARY_COMMENT = "all_lines_ordinary_comment"
    NON_COMMENT_CONTENT = "non_comment_content"
    BLANK_OR_WHITESPACE_LINE = "blank_or_whitespace_line"
    DIRECTIVE_COMMENT = "directive_comment"
    PARSE_ERROR = "parse_error"
    MISSING_BASE_CONTENT = "missing_base_content"
    NO_CHANGED_LINES = "no_changed_lines"


class SkipReason(StrEnum):
    """Skip-reason taxonomy across parser AND analyze-node decisions.

    Two naming axes per `DECISIONS.md#018` Amended 2026-05-20 + 2026-05-21:
    - Parser-stage (rule rooted in file content OR intake decode gate):
      `OVERSIZED`, `VENDORED`, `GENERATED_FILENAME`, `MINIFIED`,
      `GENERATED_BANNER`, `BINARY`. See `parser_outcome.EXCLUSION_RULES`
      for the actual rule tuple (multiple rules may share a reason).
      `BINARY` is set by intake's `_classify_or_reserve_decode` for
      NUL-byte / UTF-8-decode-failure content; the other five are set
      by `should_skip` over file content + path.
    - Analyze-stage (rule rooted in analyze's decision rationale):
      `COST_BUDGET_EXHAUSTED`, `NO_REVIEWABLE_CONTEXT`,
      `NO_CHANGED_SCOPE_UNITS`, `UNSUPPORTED_LANGUAGE`,
      `ALL_SCOPES_TRIVIAL`, `PATCH_HEAD_MISALIGNED`. Set by the
      analyze node body when it skips a file mid-pass.
      `UNSUPPORTED_LANGUAGE` is capability-scoped: it fires for
      extensions with no registered ast_facts adapter (the registry
      covers Python + JS/TS/TSX); the value names "today's analyze
      implementation cannot review this," not "Outrider forever
      cannot."

    Imported by `outrider/audit/events.py` per `DECISIONS.md#018`.
    """

    OVERSIZED = "OVERSIZED"
    VENDORED = "VENDORED"
    GENERATED_FILENAME = "GENERATED_FILENAME"
    MINIFIED = "MINIFIED"
    GENERATED_BANNER = "GENERATED_BANNER"
    # Intake decode-gate skip cause per `DECISIONS.md#018` Amended 2026-05-21.
    BINARY = "BINARY"
    # Analyze-stage skip causes per `DECISIONS.md#018` Amended 2026-05-20 + 2026-05-21.
    COST_BUDGET_EXHAUSTED = "COST_BUDGET_EXHAUSTED"
    NO_REVIEWABLE_CONTEXT = "NO_REVIEWABLE_CONTEXT"
    NO_CHANGED_SCOPE_UNITS = "NO_CHANGED_SCOPE_UNITS"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    # Every admitted scope classified trivial (ordinary-comment-only) —
    # the file's LLM call is skipped when the trivial-scope filter is
    # enforcing. "Skipped" = not sent to the LLM, parse succeeded
    # (COST_BUDGET_EXHAUSTED precedent). Fires AFTER the baseline cost
    # gate: COST_BUDGET_EXHAUSTED wins the race. Per
    # `DECISIONS.md#018` Amended 2026-06-11.
    ALL_SCOPES_TRIVIAL = "ALL_SCOPES_TRIVIAL"
    # Detected patch/head-content misalignment (a patch target line lies
    # beyond the fetched source, e.g. a force-push racing intake's
    # files-list vs content fetches). Every coordinate anchor for the
    # file — degraded span veto, module-scope admission, publish line
    # mapping — is unsound against mismatched content, so the file skips
    # with the data-integrity cause audit-visible instead of aborting or
    # silently reviewing on wrong coordinates (FUP-217; `DECISIONS.md#018`
    # taxonomy). Best-effort detection: a misalignment whose line numbers
    # still fit the fetched source is not detectable here.
    PATCH_HEAD_MISALIGNED = "PATCH_HEAD_MISALIGNED"

    def stage(self) -> Literal["parser", "analyze"]:
        """Return which decision stage produced this skip reason.

        The enum mixes two semantic axes (parser rule rooted in file
        content vs analyze rule rooted in decision rationale). Downstream
        consumers need a type-level discriminator to render or filter
        without string-parsing the value name. Two-set membership; the
        module-load check below asserts every enum value lives in
        exactly one set, so a future addition that forgets the membership
        update fails loud at import.
        """
        if self in _PARSER_STAGE_SKIP_REASONS:
            return "parser"
        if self in _ANALYZE_STAGE_SKIP_REASONS:
            return "analyze"
        # Unreachable: the module-load assertion below proves every
        # enum value lives in exactly one set. `mypy` won't infer this,
        # so the explicit raise is also the type-narrowing.
        raise AssertionError(
            f"SkipReason {self!r} is not in the parser-stage or analyze-stage "
            f"sets — the lockstep guard below should have prevented import. "
            f"This is unreachable; if you see it, the import-time check was "
            f"bypassed."
        )


# Module-private — drives `SkipReason.stage()`. Keep in lockstep with
# the analyze-stage value additions per DECISIONS.md#018 Amended 2026-05-20 + 2026-05-21.
_PARSER_STAGE_SKIP_REASONS: frozenset[SkipReason] = frozenset(
    {
        SkipReason.OVERSIZED,
        SkipReason.VENDORED,
        SkipReason.GENERATED_FILENAME,
        SkipReason.MINIFIED,
        SkipReason.GENERATED_BANNER,
        SkipReason.BINARY,
    }
)
_ANALYZE_STAGE_SKIP_REASONS: frozenset[SkipReason] = frozenset(
    {
        SkipReason.COST_BUDGET_EXHAUSTED,
        SkipReason.NO_REVIEWABLE_CONTEXT,
        SkipReason.NO_CHANGED_SCOPE_UNITS,
        SkipReason.UNSUPPORTED_LANGUAGE,
        SkipReason.ALL_SCOPES_TRIVIAL,
        SkipReason.PATCH_HEAD_MISALIGNED,
    }
)

# Import-time totality + disjointness check. A new SkipReason value
# added without an entry in one of the two sets fails this assertion
# at module load, BEFORE any code path can call `stage()` and get a
# silent fallback misclassification.
if _PARSER_STAGE_SKIP_REASONS & _ANALYZE_STAGE_SKIP_REASONS:
    raise AssertionError(
        f"SkipReason stage sets must be disjoint; overlap: "
        f"{_PARSER_STAGE_SKIP_REASONS & _ANALYZE_STAGE_SKIP_REASONS}"
    )
_combined: frozenset[SkipReason] = _PARSER_STAGE_SKIP_REASONS | _ANALYZE_STAGE_SKIP_REASONS
_all_skip_reasons: frozenset[SkipReason] = frozenset(SkipReason)
if _combined != _all_skip_reasons:
    missing = _all_skip_reasons - _combined
    raise AssertionError(
        f"SkipReason values must each belong to exactly one stage set; "
        f"unmapped: {sorted(r.value for r in missing)}. Add to "
        f"_PARSER_STAGE_SKIP_REASONS or _ANALYZE_STAGE_SKIP_REASONS."
    )


# ---------------------------------------------------------------------------
# Canonical §5.4 domain models — verbatim per spec-fidelity discipline
# ---------------------------------------------------------------------------


class ScopeUnit(BaseModel):
    """A function, method, or class definition."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str
    kind: Literal["function", "method", "class"]
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    decorators: tuple[str, ...] = ()
    parent_scope_id: str | None = None

    def to_span(self) -> "Span":
        """Return a `Span` covering this scope unit's byte range.

        callers need to bridge from
        `ScopeUnit.byte_start/byte_end` (raw int) to `Span` (Pydantic
        model used by the analyze proposal schemas + coordinate
        helpers). Without this helper, every call site reinvents
        `Span(byte_start=su.byte_start, byte_end=su.byte_end)`,
        defeating the "single chokepoint for span shape" property.
        """
        return Span(byte_start=self.byte_start, byte_end=self.byte_end)


class LexicalBinding(BaseModel):
    """A local name declaration and the lexical byte range in which it
    shadows — the OBSERVED shadowing guard's domain fact. §5.4 amendment
    per `DECISIONS.md#060`.

    Shadowing is a SPAN question, not a scope-unit-membership question:
    `ScopeUnit` is function/method/class range data only, so block/catch
    shadows would be invisible (and calls outside a shadow range wrongly
    suppressed) if names were attached to scope units. The visibility
    span is computed per kind by the adapter — params/`var` hoist to the
    enclosing function's span, `let`/`const` get the enclosing block's
    span, `function`/`class` declarations the nearest enclosing
    block/function span, catch params the catch clause, module-level
    declarations the whole module. Over-approximation only ever WIDENS a
    span (over-denial degrades an OBSERVED claim to JUDGED — the safe
    direction). CJS `require` declarators and import/export statements
    emit no record: an import binding must not shadow itself.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: str
    name: str
    kind: Literal["param", "var", "let", "const", "function", "class", "catch"]
    line: int
    visibility_byte_start: int
    visibility_byte_end: int


class ImportRef(BaseModel):
    """A single import statement, parsed into its parts.

    `is_value_import` (§5.4 amendment per `DECISIONS.md#060`)
    distinguishes imports that bind (or load) runtime
    values from forms that cannot back a runtime call: `import type`
    statements, `export … from` re-exports (no local binding at all),
    and side-effect imports. Python imports are always value imports
    (the default). Type-only SPECIFIERS (`import { type Pool, Client }`)
    are additionally excluded from `names` at extraction.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: str
    line: int
    import_kind: Literal["direct", "from", "relative", "star"]
    module: str
    names: tuple[str, ...] = ()
    is_simple_direct: bool
    is_value_import: bool = True


class CallSite(BaseModel):
    """A function or method invocation inside an extracted ScopeUnit.

    Module-level calls are not extracted in V1 (per non-goal —
    `enclosing_scope_id: str` forbids None and amending the §5.4 kind enum
    to add a synthetic "module" scope is a `DECISIONS.md` change, not a
    silent fold).
    """

    model_config = ConfigDict(extra="forbid")

    file_path: str
    line: int
    callee_name: str
    enclosing_scope_id: str


class AssignmentSite(BaseModel):
    """A variable assignment within a scope."""

    model_config = ConfigDict(extra="forbid")

    file_path: str
    line: int
    target_name: str
    enclosing_scope_id: str


class SymbolCandidate(BaseModel):
    """A name that could refer to a local, parameter, or import.

    Defined here per canonical §5.4 and re-exported from `__init__.py`;
    only the producer (the trace-node symbol-enumeration walker) is
    deferred to the trace-node spec.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    candidates: tuple[Literal["local", "parameter", "import", "unknown"], ...]


class ChangedRegion(BaseModel):
    """A diff hunk mapped to its owning scope units."""

    model_config = ConfigDict(extra="forbid")

    file_path: str
    patch_line_start: int
    patch_line_end: int
    head_line_start: int
    head_line_end: int
    base_line_start: int | None = None
    base_line_end: int | None = None
    owning_scope_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# New types added by this spec
# ---------------------------------------------------------------------------


class Span(BaseModel):
    """Byte-range over a parsed source file.

    Matches the `byte_start` / `byte_end` pattern used by `ScopeUnit` and the
    query-span models, lifted to a shared type so analyze proposals and the
    coordinates span-containment helpers can pass the same shape end-to-end.
    Per §1 of `specs/2026-05-19-analyze-foundation.md`.

    Upper bound is the JS-safe-int ceiling (2^53 - 1) per the -crazy-
    audit DI-L5: source files >9 PB are exotic enough to make fail-loud the
    right default, and the bound also makes the boundary citable for JS
    dashboard consumers.

    Interval semantics: half-open `[byte_start, byte_end)` — `byte_end` is
    exclusive. A 4-byte span starting at 0 has `byte_start=0, byte_end=4`,
    covering bytes 0/1/2/3. `coordinates.span_within_degraded_context` (§4)
    relies on this convention for intersection math.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    byte_start: int = Field(ge=0, le=2**53 - 1)
    byte_end: int = Field(ge=0, le=2**53 - 1)

    @model_validator(mode="after")
    def _enforce_byte_range(self) -> Self:
        if self.byte_end < self.byte_start:
            raise ValueError(
                f"byte_end ({self.byte_end}) must be >= byte_start ({self.byte_start})"
            )
        return self


class QueryCaptureSpan(BaseModel):
    """One named capture from a tree-sitter query match.

    Field validators enforce well-formedness because malformed input
    would silently produce wrong replay results downstream.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    byte_start: int = Field(ge=0)
    byte_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _enforce_byte_range(self) -> Self:
        if self.byte_end < self.byte_start:
            raise ValueError(
                f"byte_end ({self.byte_end}) must be >= byte_start ({self.byte_start})"
            )
        return self


class QueryMatchSpan(BaseModel):
    """One match from `queries.match(...)`. byte_start/byte_end are the
    envelope of all captures (min capture start, max capture end) per
    Internal contracts. Constructing with an envelope inconsistent with
    captures, or with empty captures (envelope undefined), raises.
    """

    model_config = ConfigDict(extra="forbid")

    byte_start: int = Field(ge=0)
    byte_end: int = Field(ge=0)
    captures: tuple[QueryCaptureSpan, ...]

    @model_validator(mode="after")
    def _enforce_byte_range(self) -> Self:
        if self.byte_end < self.byte_start:
            raise ValueError(
                f"byte_end ({self.byte_end}) must be >= byte_start ({self.byte_start})"
            )
        return self

    @model_validator(mode="after")
    def _enforce_envelope_consistency(self) -> Self:
        if not self.captures:
            raise ValueError(
                "QueryMatchSpan.captures must be non-empty (envelope is "
                "undefined for empty captures); registered queries must "
                "produce at least one capture per match per Internal contracts"
            )
        expected_start = min(c.byte_start for c in self.captures)
        expected_end = max(c.byte_end for c in self.captures)
        if (self.byte_start, self.byte_end) != (expected_start, expected_end):
            raise ValueError(
                f"QueryMatchSpan envelope ({self.byte_start}, {self.byte_end}) "
                f"does not match the captures envelope "
                f"({expected_start}, {expected_end})"
            )
        return self


class ImportResolution(BaseModel):
    """Result of `resolve_simple_direct_import`. `target_path` is non-None
    iff `status == "resolved"`; the trace node copies `target_path` to
    `TraceDecision.target_file` per `DECISIONS.md#017`. Cross-field
    validator matches #017 point 3 (a) and (b)."""

    model_config = ConfigDict(extra="forbid")

    status: ResolutionStatus
    target_path: str | None = None

    @model_validator(mode="after")
    def _enforce_status_target_path(self) -> Self:
        if self.status == "resolved" and self.target_path is None:
            raise ValueError("ImportResolution: status='resolved' requires a non-None target_path")
        if self.status in ("ambiguous", "unresolved") and self.target_path is not None:
            raise ValueError(
                f"ImportResolution: status={self.status!r} requires "
                f"target_path is None (got {self.target_path!r})"
            )
        return self


class ParseResult(BaseModel):
    """The bundled return value of the `parse_*` entry points
    (`parse_python`, `parse_javascript`, `parse_typescript`).
    Empty-tuples shape on the failed and skipped paths; full population
    on the clean path. `skip_reason` non-None iff
    `parser_outcome == "skipped"`.
    """

    model_config = ConfigDict(extra="forbid")

    parser_outcome: ParserOutcome
    skip_reason: SkipReason | None = None
    scope_units: tuple[ScopeUnit, ...] = ()
    imports: tuple[ImportRef, ...] = ()
    call_sites: tuple[CallSite, ...] = ()
    assignment_sites: tuple[AssignmentSite, ...] = ()
    # OBSERVED shadowing guard's per-file input (JS/TS adapters extract;
    # Python returns empty per the shadowing-guard spec's non-goal).
    lexical_bindings: tuple[LexicalBinding, ...] = ()
    has_error: dict[str, bool] = Field(default_factory=dict)
    # 1-indexed source lines covered by tree-sitter ERROR/MISSING nodes,
    # scope-INDEPENDENT (unlike `has_error`, keyed by recovered scope unit_id). A
    # syntax error that breaks a scope's header yields no scope node, so it is
    # invisible to `has_error` but present here — the signal degrade-don't-skip uses
    # for the no-scope case. See DECISIONS.md#033.
    error_lines: frozenset[int] = frozenset()

    @model_validator(mode="after")
    def _enforce_skip_reason_outcome(self) -> Self:
        skipped = self.parser_outcome == "skipped"
        has_reason = self.skip_reason is not None
        if skipped and not has_reason:
            raise ValueError(
                "ParseResult: parser_outcome='skipped' requires a non-None skip_reason"
            )
        if has_reason and not skipped:
            raise ValueError(
                f"ParseResult: skip_reason={self.skip_reason!r} requires "
                f"parser_outcome='skipped' (got {self.parser_outcome!r})"
            )
        return self

    @model_validator(mode="after")
    def _enforce_empty_collections_on_non_clean(self) -> Self:
        """Failed and skipped paths emit empty tuples per the docstring.

        Category F sweep — the docstring promises "Empty-tuples shape on
        the failed and skipped paths." Without a validator, a producer
        bug (or stale audit row from a previous schema iteration) could
        present `parser_outcome="failed"` AND non-empty `scope_units` —
        the downstream consumer would believe the file parsed (because
        scope_units exists) AND failed (because parser_outcome says so),
        which is the kind of contradiction the schema layer exists to
        rule out. `clean` allows empty collections (an empty Python
        file is a valid clean parse with 0 scope_units).
        """
        if self.parser_outcome == "clean":
            return self
        non_empty = []
        if self.scope_units:
            non_empty.append(f"scope_units (len={len(self.scope_units)})")
        if self.imports:
            non_empty.append(f"imports (len={len(self.imports)})")
        if self.call_sites:
            non_empty.append(f"call_sites (len={len(self.call_sites)})")
        if self.assignment_sites:
            non_empty.append(f"assignment_sites (len={len(self.assignment_sites)})")
        if self.lexical_bindings:
            non_empty.append(f"lexical_bindings (len={len(self.lexical_bindings)})")
        if self.has_error:
            non_empty.append(f"has_error (keys={sorted(self.has_error)})")
        if self.error_lines:
            non_empty.append(f"error_lines ({sorted(self.error_lines)})")
        if non_empty:
            raise ValueError(
                f"ParseResult: parser_outcome={self.parser_outcome!r} requires empty "
                f"collections (no scope_units / imports / call_sites / assignment_sites "
                f"/ has_error keys / error_lines); got non-empty: {', '.join(non_empty)}"
            )
        return self


class ExclusionRule(BaseModel):
    """One rule in `EXCLUSION_RULES` (defined in `parser_outcome.py`).
    Cross-field validator enforces `kind`/`pattern` runtime-type agreement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: SkipReason
    kind: Literal["size", "path_prefix", "filename_suffix", "banner"]
    pattern: str | bytes | int

    @model_validator(mode="after")
    def _enforce_kind_pattern_type(self) -> Self:
        kind = self.kind
        pattern = self.pattern
        # bool is a subclass of int — reject explicitly to avoid surprise
        if kind == "size":
            if isinstance(pattern, bool) or not isinstance(pattern, int):
                raise ValueError(
                    f"ExclusionRule kind='size' requires pattern: int "
                    f"(got {type(pattern).__name__})"
                )
        elif kind == "banner":
            if not isinstance(pattern, bytes):
                raise ValueError(
                    f"ExclusionRule kind='banner' requires pattern: bytes "
                    f"(got {type(pattern).__name__})"
                )
        elif kind in ("path_prefix", "filename_suffix") and not isinstance(pattern, str):
            raise ValueError(
                f"ExclusionRule kind={kind!r} requires pattern: str (got {type(pattern).__name__})"
            )
        return self
