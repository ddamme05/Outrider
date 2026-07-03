"""diff_line_to_scope + path validators + file-in-patch helper.

Per docs/spec.md §5.6 (diff_line_to_scope), §10.1 (path validation),
and §4.1.7 (publish routing — the file-in-patch helper backs the
publish-routes-through-coordinates invariant). The ImportPathResolver
Protocol implemented here is canonical at src/outrider/ast_facts/base.py.
"""

from __future__ import annotations

import keyword
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind

if TYPE_CHECKING:
    from collections.abc import Iterable

    from unidiff import PatchedFile

    from outrider.ast_facts.models import ScopeUnit


# Shell metacharacters per `docs/spec.md` §10.1 + `docs/trust-boundaries.md` §5.3.
# Conservative reject set: any character that has special meaning in POSIX shells,
# could be used for command injection if the path ever flows to a subprocess, or
# breaks GitHub API path semantics. Newline / NUL prevent header-injection and
# null-byte attacks; glob characters prevent unintended pattern expansion at
# downstream consumers.
_SHELL_METACHARS_RE: Final = re.compile(r"[;&|`$()<>\n\r\x00*?~\[\]{}'\"]")

# Unicode bidi-override and invisible-format characters per CVE-2021-42574
# ("Trojan Source"). A path containing U+202E (RLO) renders left-to-right
# in editors and audit logs differently from what the bytes say it is —
# e.g., `report‮xls.py` displays as `reportyp.slx`. Operators reading
# audit logs or PR dashboards would see a different filename than the one
# actually fetched from GitHub.
#
# Narrow reject set — only chars that enable the trojan-source disguise
# AND have no legitimate filename use:
#   - U+200B Zero Width Space (invisible, no legitimate filename use)
#   - U+200E, U+200F LTR/RTL Mark (bidi format)
#   - U+202A-U+202E LRE/RLE/PDF/LRO/RLO (bidi format, CVE-2021-42574 core)
#   - U+2066-U+2069 LRI/RLI/FSI/PDI (bidi isolate, CVE-2021-42574 core)
#   - U+FEFF Byte Order Mark / Zero Width No-Break Space (invisible)
#
# Deliberately EXCLUDED from reject (legitimate in real scripts):
#   - U+200C Zero Width Non-Joiner (Persian, Arabic word-joining)
#   - U+200D Zero Width Joiner (Hindi/Devanagari conjuncts, emoji ZWJ
#     sequences, ligature control)
# Including these would block legitimate non-Latin-script filename
# contributors.
_TROJAN_SOURCE_CHARS_RE: Final = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")

# Reject paths whose FIRST path component is `.git` (case-insensitive).
# `.git/config`, `.git/HEAD`, etc. are not legitimate PR-modifiable files
# but the validator otherwise admits them (they're relative, have no `..`,
# no shell metachars). Workflow files under `.github/workflows/` ARE
# legitimate (Outrider doesn't audit `.github/` content but the API call
# is permitted). Narrow the reject to `.git` exactly (not `.gitignore`,
# `.github`, etc.) via component-equality, not prefix-match.
_GIT_INTERNAL_FIRST_COMPONENT: Final = ".git"

# Windows drive-letter prefix (e.g., `C:/`, `D:\\`, even `C:foo` for
# drive-relative). `PurePosixPath("C:/Users/file.py").is_absolute()` returns
# False (POSIX considers absolute = leading `/`), so a drive-prefixed path
# slips through the standard `pp.is_absolute()` check. Reject it explicitly
# so absolute Windows paths can't reach the GitHub comment API surface.
_WINDOWS_DRIVE_PREFIX_RE: Final = re.compile(r"^[A-Za-z]:")

# Hard ceiling on validated diff paths, matching the `max_length=1024`
# every path-bearing schema/audit field enforces (`TraceDecision` /
# `TraceDecisionEvent` path tuples, `ReviewFinding.file_path`,
# `AnalysisRound.files_examined`, ...). Without this cap a CONSTRUCTED
# path — a probe candidate joining a near-cap importing path with a
# near-cap specifier, or a near-cap module string plus `/__init__.py` —
# passes string validation, probe-resolves against a hostile repo, and
# then aborts the whole trace pass with a Pydantic ValidationError at
# event construction. Rejecting here degrades the candidate to a probe
# negative (`unresolved`) instead.
_MAX_DIFF_PATH_LENGTH: Final = 1024


def diff_line_to_scope(
    file_path: str,
    diff_line: int,
    scope_units: list[ScopeUnit],
) -> ScopeUnit | None:
    """Find the scope unit containing a given diff line.

    Returns None for top-level changes (module-level code outside any function),
    for `diff_line` outside any scope's line range, and for scopes belonging to
    a different file. Innermost-scope rule: when nested scopes contain the same
    line, return the deepest match (smallest line span wins).

    Per docs/spec.md §5.6. The six edge cases enumerated in the Month 0 spike
    `spikes/tree_sitter/demos/demo_q6_diff_line_to_scope.py` form the unit-test
    surface; see DECISIONS.md#006-two-month-0-spikes-not-five for the
    "exhaustive unit tests" discipline this implementation honors.

    Raises `CoordinateError` for `diff_line < 1`. Source lines are 1-indexed
    per git-diff convention; a 0 or negative value signals caller
    kind-confusion (e.g., a tree-sitter `Node.start_point[0]` row, which is
    0-indexed, accidentally passed instead of being converted). Surfacing
    the error explicitly keeps the silent-`None` failure mode reserved for
    the legitimate "no enclosing scope" case.

    Multi-file safety: `scope_units` may contain scopes for multiple files;
    only scopes where `ScopeUnit.file_path == file_path` are eligible. A
    `diff_line` that would otherwise map to a scope in a different file
    returns None. Both `file_path` and each `ScopeUnit.file_path` are
    expected in canonical POSIX form (the form `validate_diff_path` returns);
    divergent surface forms compare unequal and silently miss matches.
    """
    if diff_line < 1:
        raise CoordinateError(
            f"diff_line {diff_line} is not a valid 1-indexed source line",
            kind=CoordinateErrorKind.INVALID_DIFF_LINE,
        )
    candidates = [
        unit
        for unit in scope_units
        if unit.file_path == file_path and unit.line_start <= diff_line <= unit.line_end
    ]
    if not candidates:
        return None
    # Innermost = smallest enclosing line span. Matches ast_facts'
    # `_innermost_scope_containing` pattern (byte-span there; line-span here
    # because the input is a diff line, not a byte offset).
    return min(candidates, key=lambda unit: unit.line_end - unit.line_start)


def is_valid_import_string(value: str) -> str:
    """Validate and NFC-normalize a dotted Python import string.

    Raises ValueError on invalid input; returns the NFC-normalized value on
    valid input. Single source of truth shared by `TraceCandidate.import_string`
    field validator (raises directly) AND `resolve_candidate_paths` (catches +
    returns []) per `DECISIONS.md#024` point 1 and `specs/2026-05-23-trace-node.md` M3.

    NFC normalization runs first so the schema-time validator + the resolver
    + the audit-event per-element `_enforce_canonical_proposed_import_strings`
    on `TraceDecisionEvent` / `TraceDecision` all see the same canonical
    form. Homoglyph identifiers (e.g., Cyrillic `а` U+0430 vs Latin `a`
    U+0061) pass through unchanged — NFC is composition normalization, not
    transliteration — but they pass through CONSISTENTLY, preventing hash
    divergence between the raw input fed to `compute_candidate_id` and the
    NFC'd value that lands in the audit log via the audit-event per-element
    validator. (`validate_diff_path` is the parallel surface for the
    path-shaped `target_file` / `resolved_candidate_paths` audit fields —
    different validator for path vs import-string shape.)

    Rejection cases (each raises ValueError with a discriminating message):

    - empty input
    - any backslash or forward-slash (Python imports use `.` as separator)
    - any shell metacharacter from the existing `_SHELL_METACHARS_RE` set
    - leading dot, trailing dot, or empty interior part (e.g., `foo..bar`)
    - any part that is not a valid Python identifier (e.g., `123abc`)
    - any part that is a reserved Python keyword (e.g., `class`, `for`)
    """
    if not value:
        raise ValueError("import_string must not be empty")
    normalized = unicodedata.normalize("NFC", value)
    if "\\" in normalized or "/" in normalized:
        raise ValueError("import_string must not contain path separators (use `.`)")
    if _SHELL_METACHARS_RE.search(normalized):
        raise ValueError("import_string contains shell metacharacters")
    # CVE-2021-42574 ("Trojan Source") — `_TROJAN_SOURCE_CHARS_RE`
    # rejects bidi-override + invisible-format characters EXCEPT
    # U+200C (ZWNJ) and U+200D (ZWJ), which are deliberately admitted
    # (see the regex's documented rationale: legitimate use in
    # Persian/Arabic word-joining + Hindi/Devanagari conjuncts + emoji
    # ZWJ sequences; `tests/unit/test_coordinates_paths_protocol.py::
    # test_zwj_and_zwnj_deliberately_admitted` pins both). This gate
    # blocks the broader Trojan-Source family — the exact set is the
    # regex above: U+200B (ZWSP), U+200E/U+200F (LRM/RLM),
    # U+202A–U+202E (LRE/RLE/PDF/LRO/RLO), U+2066–U+2069 (LRI/RLI/FSI/PDI),
    # and U+FEFF (ZWNBSP/BOM). It mirrors `validate_diff_path`'s
    # parallel rejection on the path surface. The two are sibling
    # defenses for the two audit-side surfaces: `proposed_import_strings`
    # runs THIS validator; `resolved_candidate_paths` / `target_file`
    # runs `validate_diff_path`. The narrow ZWJ/ZWNJ admission is a
    # documented trade-off: rejecting them would block legitimate
    # non-Latin-script identifier contributions, which the project
    # accepts instead of the marginal homoglyph-attack risk. The
    # hangul-filler family (U+3164, U+FFA0) is OUT of scope here —
    # extend the regex if a future homoglyph audit pulls them in.
    if _TROJAN_SOURCE_CHARS_RE.search(normalized):
        raise ValueError(
            "import_string contains bidi-override or invisible-format characters "
            "(CVE-2021-42574 Trojan Source defense)"
        )
    parts = normalized.split(".")
    if not all(parts):
        raise ValueError(
            "import_string has empty leading/trailing/interior part "
            "(e.g., '.foo', 'foo.', 'foo..bar')"
        )
    bad_parts = [p for p in parts if not p.isidentifier() or keyword.iskeyword(p)]
    if bad_parts:
        raise ValueError(
            f"import_string parts not valid Python identifiers "
            f"(or are reserved keywords): {bad_parts!r}"
        )
    return normalized


def resolve_candidate_paths(
    import_string: str,
    import_root: Path,
) -> list[Path]:
    """ImportPathResolver Protocol implementation per `src/outrider/ast_facts/base.py`.

    The **root-aware** surface of the two-surface path-validation rule in
    `docs/spec.md` §10.1 / `docs/trust-boundaries.md` §5.3 (the other being
    `validate_diff_path` for the GitHub API path).

    Translates a dotted Python import string (e.g., `"foo.bar"`) into the
    two repo-relative candidate paths Python's import machinery would
    consider: the module file (`foo/bar.py`) and the package init
    (`foo/bar/__init__.py`). Each candidate is validated against the full
    Protocol contract before being returned:

    - relative-only, no `..` traversal in any component
    - no shell metacharacters or backslashes / forward-slashes in the
      import string itself
    - prefix-validated against `import_root` (after resolving symlinks,
      the candidate must still lie under `import_root`)
    - no path component is a symlink — final, any ancestor, AND
      `import_root` itself. Per the Protocol's "any ancestor up to
      `import_root`" rule (read as inclusive), a symlinked root is
      rejected up-front because `python_adapter.resolve_simple_direct_import`
      calls `is_file(follow_symlinks=False)` which only guards the FINAL
      component — ancestor symlinks (including root) would otherwise be
      followed at stat time.

    Candidates that cannot be guaranteed symlink-free, fail prefix-validation,
    or hit any filesystem error during the safety walk are omitted from the
    returned list per the ast_facts spec contract — `ast_facts/` treats
    omitted paths as "did not exist." Returns an empty list for malformed
    import strings (empty, leading/trailing dot, empty interior part, any
    rejected character, or any part that is not a valid Python identifier —
    e.g., numeric prefix `123abc`, a Python keyword like `class`).

    Input-string validation delegates to `is_valid_import_string` (shared with
    `TraceCandidate.import_string` field validator); validation failures map
    to the empty-list return per this function's existing "treats as does-
    not-exist" contract.

    V1.5 swap caution (FUP-209): this surface is module-form only. The
    V1 trace resolver (`agent/nodes/trace.py::_resolve_via_probes`)
    additionally resolves symbol-form candidates (`svc.queries.run_query`,
    `app.views.UserView.get`) via a suffix-strip ladder with symbol
    verification. Replacing the probe resolver with this function
    without reproducing that ladder regresses symbol-form resolution
    to `unresolved`.
    """
    try:
        normalized = is_valid_import_string(import_string)
    except ValueError:
        return []

    parts = normalized.split(".")

    # Two candidates: foo/bar.py and foo/bar/__init__.py
    base = PurePosixPath(*parts)
    module_relative = Path(base.with_suffix(".py"))
    package_relative = Path(base / "__init__.py")

    return _filesystem_safe_candidates((module_relative, package_relative), import_root)


def _filesystem_safe_candidates(
    candidates: Iterable[Path],
    import_root: Path,
) -> list[Path]:
    """Root-aware safety walk shared by the two filesystem resolution surfaces.

    `resolve_candidate_paths` (module form) and
    `resolve_specifier_candidate_paths` (relative-specifier form) both run
    their already-constructed repo-relative candidates through this single
    core, so the trust-boundary #5 sub-rule 3b contract — `.resolve()` +
    prefix-validation against `import_root`, no symlink component anywhere
    including `import_root` itself — lives in exactly one place.

    Candidates that fail any check (or hit any filesystem error) are
    omitted; a symlinked `import_root` empties the result outright.
    """
    try:
        if import_root.is_symlink():
            return []
        root_resolved = import_root.resolve(strict=False)
    except (OSError, RuntimeError):
        return []

    safe_candidates: list[Path] = []
    for candidate in candidates:
        # Defensive — construction guarantees these, but check anyway.
        if candidate.is_absolute() or ".." in candidate.parts:
            continue

        absolute = import_root / candidate
        try:
            resolved = absolute.resolve(strict=False)
        except (OSError, RuntimeError):
            continue

        # Prefix-validation: resolved path must lie under import_root.
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue

        # Symlink-component check: walk from the candidate's full path up to
        # (but not including) `import_root`, checking is_symlink() at each
        # level. Per the ast_facts spec contract, ANY symlink component
        # disqualifies the candidate.
        if _has_symlink_component(absolute, import_root):
            continue

        safe_candidates.append(candidate)

    return safe_candidates


def _has_symlink_component(absolute: Path, root: Path) -> bool:
    """True if `absolute` or any ancestor up to (but not including) `root` is
    a symlink. Returns True on any filesystem error (treats unstat-able
    components as unsafe). Returns True if the walk reaches the filesystem
    root without ever hitting `root` (i.e., `absolute` is not actually under
    `root`).
    """
    current = absolute
    while current != root:
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
        if current.parent == current:
            # Walked off the top of the filesystem without hitting `root`.
            return True
        current = current.parent
    return False


def is_valid_relative_specifier(value: str) -> str:
    """Validate and NFC-normalize a JS/TS relative import specifier.

    The relative-specifier form of the two-form `TraceCandidate.import_string`
    contract per `DECISIONS.md#024` (Amended 2026-07-03). SHAPE-ONLY and
    context-free: repo-escape containment is deliberately NOT this validator's
    contract — it requires the importing file's directory, which the schema
    and audit call sites don't have. Containment is enforced where that
    context exists: at parser admission and again inside
    `relative_specifier_candidate_paths` (defense in depth).

    Raises ValueError on invalid input; returns the NFC-normalized value on
    valid input — same raise/return shape as `is_valid_import_string`, its
    module-form sibling, so the two-form dispatcher
    (`is_valid_trace_import_string`) composes them uniformly.

    Accepted shape: exactly `.` or `..`, or a POSIX path beginning with `./`
    or with a leading chain of `../` segments. After the leading dot run,
    segments must be non-empty and must not be `.` or `..` — interior `..`
    (e.g. `./a/../b`) is rejected outright rather than normalized; leading
    `../` chains are the only parent traversal admitted, bounded downstream
    by containment against the importing file's depth.

    Rejection cases (each raises ValueError with a discriminating message):

    - empty input
    - any backslash (POSIX separators only)
    - any shell metacharacter from the shared `_SHELL_METACHARS_RE` set
      (includes NUL, newline, glob characters)
    - bidi-override / invisible-format characters per the shared
      `_TROJAN_SOURCE_CHARS_RE` battery (CVE-2021-42574 Trojan Source —
      same admit/reject trade-offs as `is_valid_import_string`, including
      the deliberate ZWJ/ZWNJ admission documented on the regex)
    - not starting with `./` or `../` (and not exactly `.` / `..`) — bare
      specifiers (`express`) and absolute paths are out of scope by shape
    - empty path segment (`.//x`, `./x//y`, trailing `/`)
    - interior `.` or `..` segment after the leading dot run
    """
    if not value:
        raise ValueError("relative specifier must not be empty")
    normalized = unicodedata.normalize("NFC", value)
    if "\\" in normalized:
        raise ValueError("relative specifier must not contain backslashes (POSIX separators only)")
    if _SHELL_METACHARS_RE.search(normalized):
        raise ValueError("relative specifier contains shell metacharacters")
    if _TROJAN_SOURCE_CHARS_RE.search(normalized):
        raise ValueError(
            "relative specifier contains bidi-override or invisible-format "
            "characters (CVE-2021-42574 Trojan Source defense)"
        )
    if normalized not in (".", "..") and not normalized.startswith(("./", "../")):
        raise ValueError(
            "relative specifier must begin with './' or '../' (or be exactly '.' or '..')"
        )
    segments = normalized.split("/")
    # Leading dot run: a single `.` (`./x`) or a contiguous chain of `..`
    # (`../../x`). Everything after it must be plain, non-empty segments.
    tail_start = 1
    if segments[0] == "..":
        while tail_start < len(segments) and segments[tail_start] == "..":
            tail_start += 1
    for segment in segments[tail_start:]:
        if not segment:
            raise ValueError(
                "relative specifier has an empty path segment (e.g., './/x' or a trailing '/')"
            )
        if segment == ".":
            raise ValueError("relative specifier has an interior '.' segment")
        if segment == "..":
            raise ValueError(
                "relative specifier has an interior '..' segment "
                "(only a leading '../' chain is admitted)"
            )
    return normalized


def is_relative_specifier_form(value: str) -> bool:
    """THE two-form syntactic discriminator per `DECISIONS.md#024`
    (Amended 2026-07-03): a leading `.` selects the relative-specifier
    form; anything else is the module form. Exported so every dispatch
    site — the `is_valid_trace_import_string` validator and the trace
    node's probe-path construction — shares one partition rule; a future
    third form changes it here or nowhere.
    """
    return value.startswith(".")


def is_valid_trace_import_string(value: str) -> str:
    """Shared two-form validator for `TraceCandidate.import_string`.

    THE single dispatch point for the two-form contract per `DECISIONS.md#024`
    (Amended 2026-07-03): a leading `.` selects the relative-specifier form
    (`is_valid_relative_specifier`); anything else is the module form
    (`is_valid_import_string`). The two validators partition the value
    space — a dotted Python import string can never begin with `.` (the
    module form rejects leading dots) and a relative specifier always
    does — so no value is accepted by both forms and no fallback branch
    exists.

    The three shape-validation sites — `TraceCandidate.import_string`,
    `TraceDecision._enforce_canonical_proposed_import_strings`, and
    `TraceDecisionEvent`'s audit-shadow validator — all dispatch through
    this helper so the state ↔ audit canonical-bytes lockstep holds for
    both forms. Raises ValueError on invalid input; returns the
    NFC-normalized canonical value otherwise.
    """
    if is_relative_specifier_form(value):
        return is_valid_relative_specifier(value)
    return is_valid_import_string(value)


# Pragmatic-six extension fan-out for relative-specifier resolution, in
# a deterministic pinned order, per `DECISIONS.md#024` (Amended
# 2026-07-03): four file suffixes on the target stem plus two
# directory-index names — preceded by the LITERAL joined target when
# the specifier's final segment already carries a REGISTERED JS/TS
# extension (`_LITERAL_TARGET_SUFFIXES` below, per the amendment's
# extension-bearing addendum; 7-path worst case).
# The order fixes probe/budget sequencing ONLY —
# no consumer applies Node-style extension priority: two or more real
# paths resolve as `ambiguous` under the M8 single-target contract
# (both in trace probes and the adapter's filesystem twin). `.mjs` /
# `.cjs` are deliberately excluded from the FAN-OUT set only (rare as
# import targets for extensionless specifiers; widen on eval evidence,
# not speculation — FUP-212). They DO trigger the literal-target probe
# when named explicitly, like every registered extension.
_RELATIVE_SPECIFIER_SUFFIXES: Final = (".js", ".jsx", ".ts", ".tsx")
_RELATIVE_SPECIFIER_INDEX_NAMES: Final = ("index.js", "index.ts")

# Literal-target trigger set (#024 addendum, widened same day): every
# REGISTERED JS/TS extension, not just the fan-out four — Outrider
# analyzes `.mjs`/`.cjs`/`.mts`/`.cts` files, so a specifier naming one
# literally must be able to resolve it ("registered" is the principled
# line: trace fetches feed pass-1 analysis, which needs a registered
# adapter). Used ONLY for the literal-first probe; the fan-out set
# above is unchanged. Kept as a local tuple (coordinates does not
# import ast_facts at runtime); a cross-module test pins it equal to
# the registry's JS/TS extension groups so registry growth fails loud
# here instead of drifting.
_LITERAL_TARGET_SUFFIXES: Final = (
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".mts",
    ".cts",
    ".tsx",
)


def relative_specifier_candidate_paths(
    specifier: str,
    importing_file_path: str,
) -> tuple[str, ...]:
    """Construct GitHub-probe candidate paths for a JS/TS relative specifier.

    The string-level construction surface for relative-specifier trace
    candidates per `DECISIONS.md#024` (Amended 2026-07-03): `'../db'`
    imported from `src/routes/user.js` fans out to `('src/db.js',
    'src/db.jsx', 'src/db.ts', 'src/db.tsx', 'src/db/index.js',
    'src/db/index.ts')` — the pragmatic-six set in its pinned
    deterministic order (probe sequencing only; consumers apply the M8
    single-target contract, so 2+ real paths are `ambiguous` with no
    priority pick). An extension-bearing specifier (`'./db.js'`) names
    its file directly, so the literal joined target leads the tuple
    ahead of the six (#024 addendum; `'./db.js'` from
    `src/routes/user.js` → `'src/routes/db.js'` first).
    Does NOT consult the filesystem or the GitHub API; existence
    testing is the caller's probe step (verified-real + budget-bounded in
    `agent/nodes/trace.py`, per `trace-fetches-only-resolved-files`).

    Containment (defense in depth — the second enforcement site after
    parser admission): the leading `../` chain is applied against the
    importing file's directory depth; a specifier that would escape the
    repo root returns () — no candidate is constructed, so nothing can be
    probed. A target resolving to the repo root itself (e.g. `.` from a
    root-level file) fans out to the index forms only (an extension cannot
    attach to an empty stem).

    Returns () for invalid specifiers (per `is_valid_relative_specifier`),
    invalid importing paths (per `validate_diff_path`), and root escapes —
    mirroring `resolve_candidate_paths`' empty-return "treats as
    does-not-exist" contract. Every returned path has passed
    `validate_diff_path`, the string-level surface for paths heading to
    the GitHub contents API.
    """
    try:
        normalized = is_valid_relative_specifier(specifier)
    except ValueError:
        return ()
    try:
        importing = validate_diff_path(importing_file_path)
    except CoordinateError:
        return ()

    target_parts = list(PurePosixPath(importing).parts[:-1])
    for segment in normalized.split("/"):
        if segment == ".":
            continue
        if segment == "..":
            if not target_parts:
                # Repo-root escape — reject the entire candidate set.
                return ()
            target_parts.pop()
        else:
            target_parts.append(segment)

    target = "/".join(target_parts)
    if target:
        raw_candidates = []
        if target_parts[-1].endswith(_LITERAL_TARGET_SUFFIXES):
            # Extension-bearing specifier (#024 addendum 2026-07-03):
            # `'./db.js'` names its file directly — mandatory form in
            # Node ESM relative imports — so the literal joined target
            # is probed FIRST, ahead of the pragmatic-six. Any
            # registered JS/TS extension triggers this; the ext-swap
            # mapping (`./db.js` → `db.ts`) stays deferred (FUP-212).
            raw_candidates.append(target)
        raw_candidates += [f"{target}{suffix}" for suffix in _RELATIVE_SPECIFIER_SUFFIXES]
        raw_candidates += [f"{target}/{name}" for name in _RELATIVE_SPECIFIER_INDEX_NAMES]
    else:
        raw_candidates = list(_RELATIVE_SPECIFIER_INDEX_NAMES)

    validated: list[str] = []
    for candidate in raw_candidates:
        try:
            validated.append(validate_diff_path(candidate))
        except CoordinateError:
            continue
    return tuple(validated)


def resolve_specifier_candidate_paths(
    specifier: str,
    importing_file_path: str,
    import_root: Path,
) -> list[Path]:
    """Filesystem twin of `relative_specifier_candidate_paths` — root-aware.

    The relative-specifier sibling of `resolve_candidate_paths` (module
    form): the **root-aware** surface of the two-surface path-validation
    rule for JS/TS adapter use (`resolve_simple_direct_import`), per
    `docs/trust-boundaries.md` §5 sub-rule 3b. Candidate construction
    delegates to `relative_specifier_candidate_paths` (string-level
    validation + containment); each candidate then passes the same
    symlink-safe walk `resolve_candidate_paths` runs — `.resolve()` +
    prefix-validation against `import_root`, rejection of symlink
    components anywhere in the path including `import_root` itself.

    Returns repo-relative `Path` objects; candidates that cannot be
    guaranteed symlink-free, fail prefix-validation, or hit any filesystem
    error are omitted — `ast_facts/` treats omitted paths as "did not
    exist." Returns [] for invalid input, same contract as the module-form
    sibling.
    """
    candidates = relative_specifier_candidate_paths(specifier, importing_file_path)
    return _filesystem_safe_candidates(
        [Path(PurePosixPath(candidate)) for candidate in candidates], import_root
    )


def validate_diff_path(file_path: str) -> str:
    """Diff-side path validation surface — publisher-facing.

    The **string-level** surface of the two-surface path-validation rule
    in docs/spec.md §10.1 / docs/trust-boundaries.md §5.3 (the other being
    `resolve_candidate_paths` for filesystem use). Backs the
    `paths-validated-before-use` invariant (security-critical).

    Rejects, with `CoordinateError`:
    - empty strings
    - absolute paths (`is_absolute()` on a `PurePosixPath`)
    - Windows drive-letter prefixes (`C:/`, `C:\\`, `C:foo`) — `PurePosixPath`
      treats these as relative, so they need a separate rejection
    - `..` traversal in any path component
    - backslash characters (Windows separators; GitHub paths are POSIX)
    - shell metacharacters (`;`, `&`, `|`, `` ` ``, `$`, `(`, `)`, `<`, `>`,
      `\\n`, `\\r`, NUL, `*`, `?`, `~`, `[`, `]`, `{`, `}`, `'`, `"`)

    Returns the validated path in repo-relative POSIX form, normalized
    to Unicode NFC. The NFC normalization is the load-bearing step for
    identity-hash stability per spec §1: two filesystems / two clients
    that submit the same logical path under different normalization
    forms (NFC `café` vs NFD `cafe + combining-acute`) would otherwise
    produce different byte sequences and different content-derived
    hashes (round_id, candidate_id), defeating the dedup-by-key
    reducer's idempotency promise on replay.

    No `.resolve()` and no prefix-validation here — those apply to the
    root-aware surface (`resolve_candidate_paths`), per the amended
    canonical's two-surface split. The GitHub comment API consumes string
    paths, and there is no host filesystem to resolve against in this
    surface.
    """
    if not file_path:
        raise CoordinateError(
            "file_path is empty",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    # NFC normalization FIRST, before any other check, so all downstream
    # validators (Trojan-Source, metachars, traversal) see the same byte
    # sequence the hash recipes will see. spec §1 promised NFC; implementation initially
    # omitted it. Pure-ASCII paths are NFC-idempotent so the change
    # affects only multibyte paths — which is exactly the surface where
    # the hash drift would manifest.
    file_path = unicodedata.normalize("NFC", file_path)
    if len(file_path) > _MAX_DIFF_PATH_LENGTH:
        raise CoordinateError(
            f"file_path exceeds {_MAX_DIFF_PATH_LENGTH} characters "
            f"({len(file_path)}); no schema or audit field admits it",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if "\\" in file_path:
        raise CoordinateError(
            f"file_path {file_path!r} contains a backslash (POSIX separators only)",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if _SHELL_METACHARS_RE.search(file_path):
        raise CoordinateError(
            f"file_path {file_path!r} contains shell metacharacters",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if _TROJAN_SOURCE_CHARS_RE.search(file_path):
        # Per CVE-2021-42574. Path bytes that render as a different
        # filename in audit logs / dashboards break the audit story
        # (operators see a different path than the one fetched).
        raise CoordinateError(
            f"file_path {file_path!r} contains Unicode bidi-override or "
            "zero-width characters (CVE-2021-42574 / trojan source)",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if _WINDOWS_DRIVE_PREFIX_RE.match(file_path):
        raise CoordinateError(
            f"file_path {file_path!r} has a Windows drive-letter prefix; "
            "must be repo-relative POSIX",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    pp = PurePosixPath(file_path)
    if pp.is_absolute():
        raise CoordinateError(
            f"file_path {file_path!r} is absolute; must be repo-relative",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if ".." in pp.parts:
        raise CoordinateError(
            f"file_path {file_path!r} contains '..' traversal",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if pp.parts and pp.parts[0].lower() == _GIT_INTERNAL_FIRST_COMPONENT:
        # `.git/HEAD`, `.git/config`, etc. — not legitimate PR targets.
        # Component-equality (not prefix-match) so `.github/`, `.gitignore`,
        # and `.gitkeep` are unaffected.
        raise CoordinateError(
            f"file_path {file_path!r} targets the `.git` internal directory; "
            "not a legitimate PR-modifiable path",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    return pp.as_posix()


def _wrap_github_hunks_with_headers(patch: str, file_path: str) -> str:
    """Synthesize ``--- a/X`` / ``+++ b/X`` file headers around a hunks-only
    patch so `unidiff.PatchSet` can parse it.

    Wire-format reality: GitHub's ``/pulls/{number}/files`` API returns
    each file's ``patch`` field as hunks only — no preceding ``--- a/...``
    / ``+++ b/...`` headers, no ``diff --git`` line. `unidiff` requires
    those headers to attach hunks to a file; without them, the first
    ``@@`` line raises ``UnidiffParseError("Unexpected hunk found")``.
    This helper detects the hunks-only shape and synthesizes a minimal
    header pair using ``file_path`` for both source and target sides.

    **Detector policy.** Strictly narrow: the input is treated as
    hunks-only ONLY when its first non-blank line begins with ``@@``.
    Anything else (``---`` / ``+++`` already present, ``diff --git ...``
    prefix, ``Index: ...`` extended header, leading metadata lines) is
    passed through unchanged — `unidiff` knows how to parse those.
    A too-narrow detector (e.g. only checking ``startswith("--- ")``)
    can wrap an already-unified diff and create a malformed hybrid.

    **What the wrap loses.** Using the same ``file_path`` on both header
    lines is enough for the three current consumers (membership query in
    ``file_in_patch``, path-keyed lookup in ``lookup_patched_file``, and
    publisher-facing translation in ``_find_patched_file`` — the
    wire-format normalization extension that lets ``tree_sitter_to_github``
    accept GitHub's ``/pulls/{n}/files`` hunks-only shape) but deliberately
    discards:

    - The rename source path. ``PatchedFile.is_rename`` will report
      ``False`` even for actually-renamed files.
    - The ``is_added_file`` / ``is_removed_file`` semantics derived
      from ``/dev/null`` source/target sides.

    The three callers here don't consult those flags. If a future caller
    needs rename-aware parsing, this helper must NOT be used — that
    caller should accept ``(patch, current_path, previous_path)`` and
    construct the headers accordingly, OR the producer side (intake +
    ``ChangedFile`` schema) should normalize to full unified-diff form.
    Both options are deferred to the rules-layer pass; for now this
    helper stays module-private to discourage misuse.
    """
    # First non-blank line: strip leading whitespace/newlines AND the
    # UTF-8 BOM (U+FEFF) — `str.lstrip()` does not strip BOM, so a
    # patch beginning `﻿@@ ...` would otherwise survive the
    # detector unchanged, get passed unwrapped to `unidiff.PatchSet`,
    # and silently produce an empty PatchSet (no parse error). That
    # collapses to `lookup_patched_file → None → NO_REVIEWABLE_CONTEXT`
    # in analyze, i.e. a silent review-tier downgrade for any file
    # whose patch text echoes a BOM (a file authored with BOM that
    # GitHub forwards in the diff payload). Strip the BOM ONCE (the
    # spec allows at most one) so it never reaches unidiff.
    leading_stripped = patch.lstrip().removeprefix("﻿").lstrip()
    if leading_stripped.startswith("@@"):
        # Synthesize headers around the BOM-stripped body so `unidiff`
        # sees neither BOM nor leading whitespace.
        return f"--- a/{file_path}\n+++ b/{file_path}\n{leading_stripped}"
    return patch


def file_in_patch(file_path: str, patch: str) -> bool:
    """True if `file_path` matches any normalized `unidiff.PatchedFile.path` in `patch`.

    `PatchedFile.path` is the operation-dependent canonical path:
    additions, modifications, and renames return the **target** (head-side)
    path; deletions return the **source** path because the target is
    `/dev/null`. Both sides of the comparison are normalized: `file_path`
    runs through `validate_diff_path` (canonicalizing `./foo.py` → `foo.py`,
    `a//b.py` → `a/b.py`), and `unidiff.PatchedFile.path` runs through
    `PurePosixPath(...).as_posix()`. The rename commitment ("match
    `to_file` only, not `from_file`") follows from the same rule because
    `PatchedFile.path` returns `to_file` for renames.

    Boolean-helper policy: returns `False` for empty patches (`patch == ""`),
    for paths absent from the diff, AND for paths that fail
    `validate_diff_path` (caller passed `..`, an absolute path, shell
    metachars, etc.). The publisher's "in-patch vs not-in-patch" distinction
    routes a malformed caller path the same way it routes an absent file —
    to the `non_diffed_file` / `DASHBOARD_ONLY` tier — without coupling
    `file_in_patch` to a security gate it isn't responsible for.
    `tree_sitter_to_github` is the path-shape-validation gate (raises);
    `file_in_patch` is the membership query (returns bool).

    Raises `CoordinateError` on malformed patch input (any underlying
    `unidiff` parse exception is wrapped, never leaked) and on patches
    containing duplicate file entries (webhook-attacker input per trust
    boundary #5; a duplicate is ambiguous routing input that deterministic
    systems reject).

    Backs the `publish-routes-through-coordinates` invariant: the
    publisher uses this to distinguish `unchanged_region` (in-patch) from
    `non_diffed_file` (absent) routing reasons WITHOUT inlining patch-
    membership math, which would violate trust boundary #3.
    """
    if not patch:
        return False
    try:
        normalized_file_path = validate_diff_path(file_path)
    except CoordinateError:
        # Boolean-helper policy: malformed caller path → not-in-patch.
        # `tree_sitter_to_github` is the surface that raises on bad shape;
        # routing-membership queries return False.
        return False
    try:
        patchset = PatchSet(_wrap_github_hunks_with_headers(patch, normalized_file_path))
    except UnidiffParseError as e:
        raise CoordinateError(
            f"malformed patch input: {e}",
            kind=CoordinateErrorKind.MALFORMED_PATCH,
        ) from e

    matches = [pf for pf in patchset if PurePosixPath(pf.path).as_posix() == normalized_file_path]
    if len(matches) > 1:
        raise CoordinateError(
            f"patch contains {len(matches)} duplicate entries for {normalized_file_path!r}",
            kind=CoordinateErrorKind.DUPLICATE_FILE_ENTRY,
        )
    return bool(matches)


class _CoordinatesImportPathResolver:
    """`ImportPathResolver` Protocol implementation wrapping the
    module-level `resolve_candidate_paths` function.

    The standalone function is the actual implementation (and the
    public surface readers reach for); this class is the
    Protocol-satisfying bridge so `build_graph`'s `isinstance(...)`
    gate accepts a stateless singleton resolver. Stateless by design —
    one instance per process is sufficient; constructed at module
    import time as `COORDINATES_IMPORT_PATH_RESOLVER`.

    Per `docs/trust-boundaries.md §5.3`: `ast_facts/` consumes
    already-validated paths via this Protocol; the validation
    (relative-only, no `..` traversal, prefix-validation, no symlinks)
    lives in the function this class wraps. Two-surface path-validation
    rule preserved.
    """

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return resolve_candidate_paths(import_string, import_root)

    def resolve_specifier_candidate_paths(
        self, specifier: str, importing_file_path: str, import_root: Path
    ) -> list[Path]:
        return resolve_specifier_candidate_paths(specifier, importing_file_path, import_root)


COORDINATES_IMPORT_PATH_RESOLVER: Final = _CoordinatesImportPathResolver()
"""Process-wide singleton `ImportPathResolver` instance. Used by
`lifespan.py` to wire `build_graph(import_path_resolver=...)` against
the canonical coordinates implementation. Stateless; safe to share
across concurrent reviews."""


def lookup_patched_file(patch: str | None, file_path: str) -> PatchedFile | None:
    """Return the `PatchedFile` matching `file_path` in `patch`, or None if absent.

    Mirrors `file_in_patch`'s defensive shape but returns the matched
    `PatchedFile` (or None) rather than a bool. None covers three
    semantically distinct cases the caller may want to distinguish from
    "in-patch": empty/None patch input, path that fails
    `validate_diff_path`, and file absent from a well-formed patch.
    The boolean-helper policy mirrors `file_in_patch` so a malformed
    caller path doesn't surface as a routing exception — the consumer
    interprets None as "no addable-text view of this file."

    Distinguished from `_find_patched_file` (translator.py), which
    raises `CoordinateError` on missing. Used by the analyze node body
    for changed-region intersection, where the absent case is a
    legitimate `skipped+NO_CHANGED_SCOPE_UNITS` outcome rather than an
    error.

    Raises `CoordinateError` only on malformed patch input (any
    underlying `unidiff` parse exception is wrapped) and on patches
    containing duplicate entries for the same normalized path — same
    discipline as `file_in_patch`.
    """
    if not patch:
        return None
    try:
        normalized_file_path = validate_diff_path(file_path)
    except CoordinateError:
        return None
    try:
        patchset = PatchSet(_wrap_github_hunks_with_headers(patch, normalized_file_path))
    except UnidiffParseError as e:
        raise CoordinateError(
            f"malformed patch input: {e}",
            kind=CoordinateErrorKind.MALFORMED_PATCH,
        ) from e
    matches = [pf for pf in patchset if PurePosixPath(pf.path).as_posix() == normalized_file_path]
    if len(matches) > 1:
        raise CoordinateError(
            f"patch contains {len(matches)} duplicate entries for {normalized_file_path!r}",
            kind=CoordinateErrorKind.DUPLICATE_FILE_ENTRY,
        )
    return matches[0] if matches else None
