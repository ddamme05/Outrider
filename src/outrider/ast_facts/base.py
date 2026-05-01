# LanguageAdapter and ImportPathResolver Protocols per
# specs/2026-04-30-ast-facts-module.md Internal contracts.
"""Protocol definitions for the ast_facts/ surface.

Two Protocols, both consumed by graph nodes via closure injection from
`build_graph(...)` per `nodes-receive-deps-via-closure`:

  - `LanguageAdapter` — implemented by per-language adapters (V1: only
    `PythonAdapter`). Six methods covering scope, import, call-site,
    assignment extraction; simple-direct-import resolution; and
    parse-outcome computation.

  - `ImportPathResolver` — the path-validation contract that
    `resolve_simple_direct_import` calls into. Implementation is owned
    by `coordinates/` per trust-boundary #5; this module ships only
    the Protocol so `ast_facts/` doesn't pull `coordinates/` into its
    type surface.

This module is import-light: only `pathlib`, `typing`, and the local
domain models. No `tree_sitter`, no `coordinates/`, no `audit/`.
"""

from pathlib import Path
from typing import Protocol

from outrider.ast_facts.models import (
    AssignmentSite,
    CallSite,
    ComputedParserOutcome,
    ImportRef,
    ImportResolution,
    ScopeUnit,
)


class LanguageAdapter(Protocol):
    """Per-language extraction surface. The adapter holds the parser,
    queries, and the injected `ImportPathResolver`; methods consume
    `source: bytes` (and `scope_units` for downstream methods that
    need the scope set already extracted).
    """

    def extract_scopes(self, source: bytes, file_path: str) -> tuple[ScopeUnit, ...]: ...

    def extract_imports(self, source: bytes, file_path: str) -> tuple[ImportRef, ...]: ...

    def extract_call_sites(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[CallSite, ...]:
        """Module-level calls produce no records — `enclosing_scope_id`
        always references a real `unit_id` in `scope_units` per non-goal."""

    def extract_assignments(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[AssignmentSite, ...]: ...

    def resolve_simple_direct_import(
        self, import_ref: ImportRef, import_root: Path
    ) -> ImportResolution:
        """Returns `ImportResolution(status, target_path)`. `target_path`
        is `candidate.as_posix()` when resolved, None otherwise — the
        trace node copies it to `TraceDecision.target_file` per #017."""

    def compute_parser_outcome(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[ComputedParserOutcome, dict[str, bool]]:
        """Returns `(ComputedParserOutcome, has_error_map)`.
        `ComputedParserOutcome = Literal["clean", "failed"]` — the
        narrow type makes "never returns skipped" type-checkable
        (skip-classification is `parse_python`'s job, not this method's)."""


class ImportPathResolver(Protocol):
    """Path-string-to-validated-path construction contract.

    Returned `Path` objects are repo-relative (relative to `import_root`)
    and validated per trust-boundary #5: relative-only, no `..`
    traversal, no shell metacharacters, prefix-validated against
    `import_root`, AND no path component (final or any ancestor up to
    `import_root`) is a symlink. The implementation site is responsible
    for all those validations; `ast_facts/` consumes already-validated
    paths and joins them with `import_root` for the symlink-safe stat.

    The Protocol is shipped from `ast_facts/base.py`; the V1
    implementation lives elsewhere (eventually `coordinates/`) and is
    out of scope for this module.
    """

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]: ...
