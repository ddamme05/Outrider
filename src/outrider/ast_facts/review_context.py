"""Select bounded source excerpts that explain changed code and supplied tests.

Imports and module bindings follow names used in changed functions. Caller matches
use direct names only, without establishing resolved call relationships. Parent
constructor references follow a single declared local base, without proving MRO.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from outrider.ast_facts.parser_outcome import should_skip
from outrider.ast_facts.registry import get_adapter_factory, parse_source
from outrider.coordinates import CoordinateError, line_range_to_span, validate_diff_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from outrider.ast_facts.base import ImportPathResolver
    from outrider.ast_facts.models import ParseResult, ScopeUnit

_MAX_CONTEXT_SOURCE_BYTES = 131_072
_MODULE_CHAR_CAP = 1600
_CALLER_CHAR_CAP = 1000
_CONSTRUCTOR_ATTRIBUTES = frozenset(
    {"__init__", "__bases__", "__class_getitem__", "__mro_entries__"}
)
_MAX_CALL_CONTEXT_BYTES = 400 * 1024
_MAX_CALL_CONTEXT_FILES = 32
_MAX_CALLED_HELPER_CHARS = 2000
MAX_MODULE_SOURCE_BYTES = 16 * 1024
MAX_COMPANION_FUNCTIONS = 3
MAX_COMPANION_FUNCTION_BYTES = 4096


@dataclass(frozen=True)
class SourceExcerpt:
    """Source text and its exact head-file range, without exposing AST nodes."""

    name: str
    line_start: int
    line_end: int
    text: str


@dataclass(frozen=True)
class ParentConstructorSelection:
    """Complete same-file references; declared inheritance is not MRO proof."""

    excerpts: tuple[SourceExcerpt, ...]
    omissions: tuple[str, ...]


@dataclass(frozen=True)
class CalledFunctionIndex:
    """One bounded captured-source index; names do not establish receiver types."""

    sources: dict[str, bytes]
    parsed: dict[str, ParseResult]
    definitions: dict[str, tuple[ScopeUnit, ...]]
    excluded_definition_paths: frozenset[str]


@dataclass(frozen=True)
class CalledFunctionExcerpt:
    file_path: str
    excerpt: SourceExcerpt


@dataclass(frozen=True)
class ModuleContextIndex:
    """Bounded parsed heads plus identities from validated captured Python paths.

    Module labels describe captured names, not runtime import resolution. No
    import string is turned into a filesystem path or probed on disk.
    """

    source_index: CalledFunctionIndex
    module_paths: dict[str, tuple[str, ...]]
    module_names: dict[str, str]
    omitted_counts: dict[str, int]


@dataclass(frozen=True)
class ModuleContextSelection:
    references: tuple[CalledFunctionExcerpt, ...]
    omissions: tuple[str, ...]


def select_companion_functions(
    *,
    file_path: str,
    source: str,
    selected_scope_units: tuple[ScopeUnit, ...],
    captured_heads: Mapping[str, str],
) -> ModuleContextSelection:
    """Match selected top-level functions to an exact captured .py/.pyi sibling.

    These complete definitions are references, not resolved runtime bindings.
    Rendering and funding belong to the caller; primary scopes stay unchanged.
    """
    try:
        canonical = validate_diff_path(file_path)
    except CoordinateError:
        return ModuleContextSelection((), ("invalid primary source path",))
    path = PurePosixPath(file_path)
    if canonical != file_path or path.suffix not in {".py", ".pyi"}:
        return ModuleContextSelection((), ("primary path is not canonical Python",))
    sibling_path = str(path.with_suffix(".py" if path.suffix == ".pyi" else ".pyi"))
    sibling_source = captured_heads.get(sibling_path)
    if sibling_source is None:
        return ModuleContextSelection((), ("exact companion HEAD is not captured",))
    primary = _parse_context_source(source, file_path)
    sibling = _parse_context_source(sibling_source, sibling_path)
    if primary is None or sibling is None:
        return ModuleContextSelection((), ("primary or companion source is not eligible Python",))

    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    selected = {
        scope.qualified_name: scope
        for scope in selected_scope_units
        if scope.file_path == file_path
        and scope.kind == "function"
        and "." not in scope.qualified_name
    }
    references: list[CalledFunctionExcerpt] = []
    omissions: list[str] = []
    for name in dict.fromkeys(node.name for node in primary.body if isinstance(node, functions)):
        scope = selected.get(name)
        if scope is None:
            continue
        originals = [
            node for node in primary.body if isinstance(node, functions) and node.name == name
        ]
        matches = [
            node for node in sibling.body if isinstance(node, functions) and node.name == name
        ]
        if len(originals) != 1 or len(matches) != 1:
            omissions.append(f"{name}: no unique top-level definition in both sources")
            continue
        original, match = originals[0], matches[0]
        if (
            not scope.line_start
            <= original.lineno
            <= (original.end_lineno or original.lineno)
            <= scope.line_end
        ):
            continue
        if _parent_binding_conflict(primary.body, name, original) or _parent_binding_conflict(
            sibling.body, name, match
        ):
            omissions.append(f"{name}: visible definition binding is ambiguous or modified")
            continue
        start = min([match.lineno, *(decorator.lineno for decorator in match.decorator_list)])
        end = match.end_lineno or match.lineno
        span = line_range_to_span(start, end, sibling_source)
        text = sibling_source.encode("utf-8")[span.byte_start : span.byte_end].decode("utf-8")
        if (
            len(references) == MAX_COMPANION_FUNCTIONS
            or len(text.encode("utf-8")) > MAX_COMPANION_FUNCTION_BYTES
        ):
            omissions.append(f"{name}: companion definition count or source allowance exceeded")
            continue
        references.append(
            CalledFunctionExcerpt(sibling_path, SourceExcerpt(name, start, end, text))
        )
    return ModuleContextSelection(tuple(references), tuple(omissions))


def build_module_context_index(
    sources: Mapping[str, str],
    *,
    import_path_resolver: ImportPathResolver,
    excluded_definition_paths: frozenset[str] = frozenset(),
) -> ModuleContextIndex:
    """Reuse registered parser facts within the existing 32-file/400-KiB index.

    All validated captured Python names participate in ambiguity checks, even
    when their source is too large or otherwise ineligible for a reference.
    Source roots are not inferred from suffix matches.
    """
    eligible_sources: dict[str, str] = {}
    module_names: dict[str, str] = {}
    module_paths: dict[str, list[str]] = {}
    omitted_counts: dict[str, int] = {}
    for path, source in sorted(sources.items()):
        try:
            validated = validate_diff_path(path)
        except CoordinateError:
            reason = "invalid captured path"
        else:
            parts = PurePosixPath(validated).with_suffix("").parts
            if PurePosixPath(validated).suffix != ".py":
                reason = "unsupported language (Python modules only)"
            else:
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts and all(part.isidentifier() for part in parts):
                    name = ".".join(parts)
                    # A normalized alias still participates in ambiguity; do
                    # not make another captured spelling appear uniquely safe.
                    module_paths.setdefault(name, []).append(path)
                    if validated != path:
                        reason = "noncanonical captured path"
                        omitted_counts[reason] = omitted_counts.get(reason, 0) + 1
                        continue
                    module_names[path] = name
                    eligible_sources[path] = source
                    continue
                reason = "captured path has no explicit Python module identity"
        omitted_counts[reason] = omitted_counts.get(reason, 0) + 1
    source_index = build_called_function_index(
        eligible_sources,
        import_path_resolver=import_path_resolver,
        excluded_definition_paths=excluded_definition_paths,
    )
    not_parsed = len(eligible_sources) - len(source_index.parsed)
    if not_parsed:
        omitted_counts["not indexed (parser, exclusion or index bounds)"] = not_parsed
    return ModuleContextIndex(
        source_index,
        {name: tuple(paths) for name, paths in module_paths.items()},
        module_names,
        omitted_counts,
    )


def _explicit_import_module(module: str, *, importer: str, index: ModuleContextIndex) -> str | None:
    """Compare module labels only; imported aliases never invent child modules."""
    if not module.startswith("."):
        return module if all(part.isidentifier() for part in module.split(".")) else None
    dots = len(module) - len(module.lstrip("."))
    tail = module[dots:]
    if not tail or not all(part.isidentifier() for part in tail.split(".")):
        return None
    own_name = index.module_names[importer]
    package = own_name.split(".")
    if PurePosixPath(importer).name != "__init__.py":
        package = package[:-1]
    # Without import-root metadata, a relative import cannot leave its captured
    # top-level package. In particular, do not guess a src/ package root.
    if not package or dots > len(package):
        return None
    return ".".join([*package[: len(package) - dots + 1], tail])


def complete_module_context(index: ModuleContextIndex, *, file_path: str) -> ModuleContextSelection:
    """Select a complete primary module and at most one explicit import neighbour.

    Direct imports are considered before importers; ties follow captured path
    order and source import order. Ambiguous module/package names create no edge.
    Imported symbols, stars, unknown roots and runtime receiver types are not resolved.
    """
    omissions: list[str] = []
    neighbours: list[str] = []
    imported_by: list[str] = []
    for importer, facts in index.source_index.parsed.items():
        for ref in facts.imports:
            if ref.import_kind == "star":
                if importer == file_path:
                    omissions.append(f"import line {ref.line}: star import omitted")
                continue
            label = _explicit_import_module(ref.module, importer=importer, index=index)
            targets = index.module_paths.get(label, ()) if label is not None else ()
            if len(targets) != 1:
                if importer == file_path:
                    reason = (
                        "ambiguous captured module"
                        if targets
                        else "missing module or unsupported root/relative form"
                    )
                    omissions.append(f"import line {ref.line}: {reason}")
                elif file_path in targets:
                    omissions.append(
                        f"{importer} import line {ref.line}: ambiguous captured module"
                    )
                continue
            target = targets[0]
            if importer == file_path and target != file_path and target not in neighbours:
                neighbours.append(target)
            elif target == file_path and importer != file_path and importer not in imported_by:
                imported_by.append(importer)
    references: list[CalledFunctionExcerpt] = []
    connected_added = False
    candidates = [file_path, *neighbours, *(path for path in imported_by if path not in neighbours)]
    for path in candidates:
        if path != file_path and connected_added:
            omissions.append("further connected module omitted (one-neighbour limit)")
            break
        source = index.source_index.sources.get(path)
        if path not in index.module_names:
            reason = "unsupported or invalid captured Python module"
        elif path in index.source_index.excluded_definition_paths:
            reason = "test module (bounded test excerpts retained instead)"
        elif source is None or path not in index.source_index.parsed:
            reason = "not indexed (parser, exclusion or index bounds)"
        elif len(source) > MAX_MODULE_SOURCE_BYTES:
            reason = "source exceeds 16-KiB module limit"
        elif not source:
            reason = "empty source"
        else:
            text = source.decode("utf-8")
            references.append(
                CalledFunctionExcerpt(
                    path,
                    SourceExcerpt("complete captured module", 1, len(text.splitlines()), text),
                )
            )
            connected_added = path != file_path
            continue
        omissions.append(f"{path}: {reason}")
    return ModuleContextSelection(tuple(references), tuple(omissions))


def removed_module_importers(
    index: ModuleContextIndex,
    *,
    removed_path: str,
    max_source_chars: int = 2000,
) -> ModuleContextSelection:
    """Find small captured HEAD modules explicitly importing a removed Python name.

    Module labels are syntactic hints. They do not prove runtime import roots,
    reachability or whether a missing dependency is optional.
    """
    path = PurePosixPath(validate_diff_path(removed_path))
    parts = path.with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if path.suffix != ".py" or not parts or not all(part.isidentifier() for part in parts):
        return ModuleContextSelection((), ("no explicit removed Python module identity",))
    removed_name = ".".join(parts)
    if removed_name in index.module_paths:
        return ModuleContextSelection((), ("a surviving captured path shares the module label",))
    references: list[CalledFunctionExcerpt] = []
    omissions = [f"{reason}: {count}" for reason, count in index.omitted_counts.items()]
    for importer, facts in sorted(index.source_index.parsed.items()):
        if not any(
            ref.is_value_import
            and _explicit_import_module(ref.module, importer=importer, index=index) == removed_name
            for ref in facts.imports
        ):
            continue
        source = index.source_index.sources[importer].decode("utf-8")
        if len(source) > max_source_chars:
            omissions.append(f"{importer}: complete importer exceeds source bound")
            continue
        if len(references) == 2:
            omissions.append("further captured importers omitted (two-reference limit)")
            break
        references.append(
            CalledFunctionExcerpt(
                importer,
                SourceExcerpt("possible surviving importer", 1, len(source.splitlines()), source),
            )
        )
    return ModuleContextSelection(tuple(references), tuple(omissions))


def build_called_function_index(
    sources: Mapping[str, str],
    *,
    import_path_resolver: ImportPathResolver,
    excluded_definition_paths: frozenset[str] = frozenset(),
) -> CalledFunctionIndex:
    """Parse eligible captured heads once, never fetching imports or repository files.

    The index covers at most 32 files and 400 KiB, with a 128 KiB per-file limit.
    Excluded definition paths still contribute names, so test overrides cannot
    make an ambiguous production method look unique.
    """
    indexed_sources: dict[str, bytes] = {}
    parsed: dict[str, ParseResult] = {}
    definitions: dict[str, list[ScopeUnit]] = {}
    indexed_bytes = 0
    for path, source in sources.items():
        if get_adapter_factory(PurePosixPath(path).suffix) is None:
            continue
        if len(source) > _MAX_CONTEXT_SOURCE_BYTES:
            continue
        encoded = source.encode("utf-8")
        if len(encoded) > _MAX_CONTEXT_SOURCE_BYTES:
            continue
        if should_skip(path, encoded) is not None:
            continue
        if (
            len(indexed_sources) >= _MAX_CALL_CONTEXT_FILES
            or indexed_bytes + len(encoded) > _MAX_CALL_CONTEXT_BYTES
        ):
            continue
        facts = parse_source(encoded, path, import_path_resolver)
        indexed_bytes += len(encoded)
        indexed_sources[path] = encoded
        if facts.parser_outcome != "clean" or facts.error_lines:
            continue
        parsed[path] = facts
        for scope in facts.scope_units:
            if scope.kind != "class" and not facts.has_error.get(scope.unit_id, False):
                definitions.setdefault(scope.name, []).append(scope)
    return CalledFunctionIndex(
        indexed_sources,
        parsed,
        {name: tuple(scopes) for name, scopes in definitions.items()},
        excluded_definition_paths,
    )


def called_function_context(
    index: CalledFunctionIndex, *, file_path: str, changed_lines: frozenset[int]
) -> tuple[CalledFunctionExcerpt, ...]:
    """Select at most two small, uniquely named callees of changed scopes.

    Matches use a call's final name, not receiver or import resolution. Ambiguous
    names, test implementations and helpers already in primary scopes add nothing.
    """
    facts = index.parsed.get(file_path)
    if facts is None or not changed_lines:
        return ()
    primary_scopes = tuple(
        scope
        for scope in facts.scope_units
        if any(scope.line_start <= line <= scope.line_end for line in changed_lines)
    )
    references = []
    seen_names = set()
    for call in facts.call_sites:
        if not any(scope.line_start <= call.line <= scope.line_end for scope in primary_scopes):
            continue
        name = call.callee_name.rsplit(".", 1)[-1]
        if name in seen_names:
            continue
        seen_names.add(name)
        definitions = index.definitions.get(name, ())
        if len(definitions) != 1:
            continue
        target = definitions[0]
        if target.file_path in index.excluded_definition_paths:
            continue
        if target.file_path == file_path and any(
            scope.line_start <= target.line_start <= target.line_end <= scope.line_end
            for scope in primary_scopes
        ):
            continue
        text = index.sources[target.file_path][target.byte_start : target.byte_end].decode("utf-8")
        if len(text) > _MAX_CALLED_HELPER_CHARS:
            continue
        suffix = f"@{target.line_start}:{target.line_end}"
        references.append(
            CalledFunctionExcerpt(
                target.file_path,
                SourceExcerpt(
                    target.qualified_name[: 1024 - len(suffix)] + suffix,
                    target.line_start,
                    target.line_end,
                    text,
                ),
            )
        )
        if len(references) == 2:
            break
    return tuple(references)


def _parse_context_source(source: str, file_path: str) -> ast.Module | None:
    if len(source) > _MAX_CONTEXT_SOURCE_BYTES:
        return None
    source_bytes = source.encode("utf-8")
    if len(source_bytes) > _MAX_CONTEXT_SOURCE_BYTES:
        return None
    try:
        if should_skip(file_path, source_bytes) is not None:
            return None
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


def _excerpt(node: ast.stmt, source: str, name: str) -> SourceExcerpt:
    end = node.end_lineno or node.lineno
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        suffix = f"@{node.lineno}:{end}"
        name = name[: 1024 - len(suffix)] + suffix
    span = line_range_to_span(node.lineno, end, source)
    text = source.encode("utf-8")[span.byte_start : span.byte_end].decode("utf-8")
    return SourceExcerpt(name, node.lineno, end, text)


def _module_declarations(
    module: ast.Module,
    source: str,
    used_names: set[str],
    *,
    include_class_declarations: bool = False,
    supplied_spans: tuple[tuple[int, int], ...] = (),
    omit_competing_bindings: bool = False,
) -> tuple[SourceExcerpt, ...]:
    """Select complete syntactically named declarations within 1,600 characters."""
    excerpts: list[SourceExcerpt] = []
    declaration_chars = 0
    for statement in module.body:
        names: set[str] = set()
        if isinstance(statement, ast.Import):
            names = {alias.asname or alias.name.split(".")[0] for alias in statement.names}
        elif isinstance(statement, ast.ImportFrom):
            names = {alias.asname or alias.name for alias in statement.names}
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            names = {
                node.id
                for target in targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
        elif include_class_declarations and isinstance(statement, ast.ClassDef):
            if statement.decorator_list:
                continue
            names = {statement.name}
        if not names.intersection(used_names):
            continue
        if omit_competing_bindings and any(
            _parent_binding_conflict(
                [other for other in module.body if other is not statement], name, None
            )
            for name in names.intersection(used_names)
        ):
            continue
        excerpt = _excerpt(statement, source, f"module declaration at line {statement.lineno}")
        if any(
            start <= excerpt.line_start <= excerpt.line_end <= end for start, end in supplied_spans
        ):
            continue
        if declaration_chars + len(excerpt.text) <= _MODULE_CHAR_CAP:
            excerpts.append(excerpt)
            declaration_chars += len(excerpt.text)
    return tuple(excerpts)


def python_reference_declarations(
    source: str,
    *,
    file_path: str,
    reference_spans: tuple[tuple[int, int], ...],
    supplied_spans: tuple[tuple[int, int], ...] = (),
) -> tuple[SourceExcerpt, ...]:
    """Follow lexical names in complete supplied definitions, without resolving them.

    Numeric ranges must exactly match a definition in this bounded source. The
    caller supplies only definitions the current context covers, gated on an
    actually supplied parent reference. Small undecorated class declarations are
    considered here without changing the ordinary base-context selection. A
    visible competing module binding withholds its entire declaration statement.
    """
    module = _parse_context_source(source, file_path) if reference_spans else None
    if module is None:
        return ()
    ranges = set(reference_spans)
    used_names = {
        node.id
        for definition in ast.walk(module)
        if isinstance(definition, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and (definition.lineno, definition.end_lineno or definition.lineno) in ranges
        for node in ast.walk(definition)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return _module_declarations(
        module,
        source,
        used_names,
        include_class_declarations=True,
        supplied_spans=supplied_spans,
        omit_competing_bindings=True,
    )


def python_review_context(
    source: str, changed_lines: frozenset[int], *, file_path: str
) -> tuple[SourceExcerpt, ...]:
    """Select bounded declarations and possible callers for changed functions.

    Source above 128 KiB, excluded paths and invalid Python add no excerpts.
    This source is context, never an additional finding location.
    """
    module = _parse_context_source(source, file_path) if changed_lines else None
    if module is None:
        return ()
    functions = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    changed_functions = [
        node
        for node in functions
        if any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in changed_lines)
    ]
    used_names = {
        node.id
        for function in changed_functions
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
    }
    excerpts = list(_module_declarations(module, source, used_names))
    excerpts.extend(
        _caller_excerpts(
            functions,
            source,
            names={function.name for function in changed_functions},
            excluded=changed_functions,
        )
    )
    return tuple(excerpts)


def _unshadowed_call_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef, names: set[str]
) -> set[str]:
    """Exclude names shadowed by function parameters, bindings or declarations."""
    arguments = function.args
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        parameters.append(arguments.vararg)
    if arguments.kwarg is not None:
        parameters.append(arguments.kwarg)
    return {
        name
        for name in names
        if all(parameter.arg != name for parameter in parameters)
        and not any(
            parameter.name == name
            for parameter in function.type_params
            if isinstance(parameter, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple))
        )
        and not _parent_binding_conflict(
            function.body,
            name,
            None,
            constructor_attributes=_CONSTRUCTOR_ATTRIBUTES | {"__new__"},
        )
        and not any(
            isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names
            for node in _own_body_nodes(function.body)
        )
    }


def _caller_excerpts(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    source: str,
    *,
    names: set[str],
    excluded: list[ast.FunctionDef | ast.AsyncFunctionDef],
    existing: tuple[SourceExcerpt, ...] = (),
    supplied_spans: tuple[tuple[int, int], ...] = (),
    own_body_only: bool = False,
) -> tuple[SourceExcerpt, ...]:
    """Share source ordering and the two-caller/1,000-character allowance."""
    excerpts: list[SourceExcerpt] = []
    caller_chars = sum(len(excerpt.text) for excerpt in existing)
    caller_count = len(existing)
    for function in functions:
        if function in excluded:
            continue
        matching_names = names
        nodes = ast.walk(function)
        if own_body_only:
            matching_names = _unshadowed_call_names(function, names)
            nodes = _own_body_nodes(function.body)
        if not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in matching_names
            for node in nodes
        ):
            continue
        excerpt = _excerpt(function, source, function.name)
        if any(
            start <= excerpt.line_start <= excerpt.line_end <= end for start, end in supplied_spans
        ):
            continue
        if caller_count < 2 and caller_chars + len(excerpt.text) <= _CALLER_CHAR_CAP:
            excerpts.append(excerpt)
            caller_chars += len(excerpt.text)
            caller_count += 1
    return tuple(excerpts)


def _own_body_nodes(statements: list[ast.stmt]) -> Iterator[ast.AST]:
    """Do not treat calls in nested lexical scopes as calls of their container."""
    pending: list[ast.AST] = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def python_constructor_caller_context(
    source: str,
    *,
    file_path: str,
    selected_scope_units: tuple[ScopeUnit, ...],
    supplied_names: frozenset[str],
    supplied_spans: tuple[tuple[int, int], ...],
) -> tuple[SourceExcerpt, ...]:
    """Find complete top-level callers of selected own initializers by direct name.

    Visible binding conflicts withhold the new reference. This heuristic does not
    establish runtime class identity, inherited allocation behavior or call reachability.
    Already supplied callers consume the same two-caller/1,000-character allowance.
    """
    if not selected_scope_units or PurePosixPath(file_path).suffix.lower() != ".py":
        return ()
    module = _parse_context_source(source, file_path)
    if module is None:
        return ()
    names: set[str] = set()
    for child in module.body:
        if not isinstance(child, ast.ClassDef):
            continue
        constructors = [
            node
            for node in child.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]
        if len(constructors) != 1:
            continue
        constructor = constructors[0]
        selected = [
            scope
            for scope in selected_scope_units
            if scope.file_path == file_path
            and child.lineno
            <= scope.line_start
            <= scope.line_end
            <= (child.end_lineno or child.lineno)
        ]
        constructor_selected = any(
            scope.kind == "method"
            and scope.qualified_name == f"{child.name}.__init__"
            and scope.line_start
            <= constructor.lineno
            <= (constructor.end_lineno or constructor.lineno)
            <= scope.line_end
            for scope in selected
        )
        whole_class_selected = any(
            scope.kind == "class" and scope.qualified_name == child.name for scope in selected
        ) and not any(scope.kind != "class" for scope in selected)
        if not (constructor_selected or whole_class_selected):
            continue
        if (
            child.decorator_list
            or child.keywords
            or constructor.decorator_list
            or _parent_binding_conflict(
                module.body,
                child.name,
                child,
                constructor_attributes=_CONSTRUCTOR_ATTRIBUTES | {"__new__"},
            )
            or _parent_binding_conflict(child.body, "__init__", constructor)
            or _parent_binding_conflict(child.body, "__new__", None)
        ):
            continue
        names.add(child.name)
    if not names:
        return ()
    functions = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    # A supplied caller fits 1,000 source characters, so its function name cannot
    # reach the 1,024-character label truncation. Avoid slicing every definition.
    existing = tuple(
        _excerpt(function, source, function.name)
        for function in functions
        if f"{function.name}@{function.lineno}:{function.end_lineno or function.lineno}"
        in supplied_names
    )
    return _caller_excerpts(
        [function for function in functions if function in module.body],
        source,
        names=names,
        excluded=[],
        existing=existing,
        supplied_spans=supplied_spans,
        own_body_only=True,
    )


def _parent_binding_conflict(
    statements: list[ast.stmt],
    name: str,
    definition: ast.AST | None,
    *,
    constructor_attributes: frozenset[str] = _CONSTRUCTOR_ATTRIBUTES,
) -> bool:
    """Reject visible competing bindings or direct constructor monkey-patches."""
    pending: list[ast.AST] = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name and node is not definition:
                return True
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                return True
            if any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names):
                return True
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == name
        ):
            return True
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == name:
            return True
        if isinstance(node, ast.MatchMapping) and node.rest == name:
            return True
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and isinstance(node.value, ast.Name)
            and node.value.id == name
            and node.attr in constructor_attributes
        ):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == name
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in constructor_attributes
        ):
            return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def _constructor_calls_builtin_super(constructor: ast.FunctionDef, child: ast.ClassDef) -> bool:
    """Match an own-body zero-argument super call, excluding obvious shadows."""
    arguments = constructor.args
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        parameters.append(arguments.vararg)
    if arguments.kwarg is not None:
        parameters.append(arguments.kwarg)
    shadowed = {"super", "__class__"}
    if any(argument.arg in shadowed for argument in parameters) or any(
        parameter.name in shadowed
        for parameter in (*child.type_params, *constructor.type_params)
        if isinstance(parameter, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple))
    ):
        return False
    if any(_parent_binding_conflict(constructor.body, name, None) for name in shadowed):
        return False
    pending: list[ast.AST] = list(constructor.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__init__"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
            and not node.func.value.args
            and not node.func.value.keywords
        ):
            return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def python_parent_constructor_context(
    source: str,
    *,
    file_path: str,
    selected_scope_units: tuple[ScopeUnit, ...],
    max_excerpt_bytes: int = 4096,
) -> ParentConstructorSelection:
    """Select at most two complete constructors of unambiguous local base classes.

    Only a selected constructor, or a constructor already inside a selected class,
    can request a parent. One plain or subscripted named base is supported; imported,
    nested, decorated, rebound and multiple-base relationships remain unresolved.
    This is a syntactic source reference, never a new primary finding location.
    """
    if not 0 < max_excerpt_bytes <= 4096:
        raise ValueError("max_excerpt_bytes must be between 1 and 4096")
    if not selected_scope_units or PurePosixPath(file_path).suffix.lower() != ".py":
        return ParentConstructorSelection((), ())
    module = _parse_context_source(source, file_path)
    if module is None:
        return ParentConstructorSelection((), ("parent source is not eligible parsed Python",))
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    excerpts: list[SourceExcerpt] = []
    omissions: list[str] = []
    seen: set[tuple[int, int]] = set()
    for child in classes:
        constructors = [
            node
            for node in child.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]
        if len(constructors) != 1:
            continue
        constructor = constructors[0]
        end = constructor.end_lineno or constructor.lineno
        relevant = any(
            scope.file_path == file_path
            and scope.line_start <= constructor.lineno <= end <= scope.line_end
            and (
                (scope.kind == "class" and scope.qualified_name == child.name)
                or (scope.kind == "method" and scope.qualified_name == f"{child.name}.__init__")
            )
            for scope in selected_scope_units
        )
        if not relevant or not _constructor_calls_builtin_super(constructor, child):
            continue
        label = f"{child.name}.__init__"
        if (
            child.decorator_list
            or child.keywords
            or constructor.decorator_list
            or _parent_binding_conflict(module.body, child.name, child)
            or _parent_binding_conflict(module.body, "super", None)
            or _parent_binding_conflict(child.body, "__init__", constructor)
        ):
            omissions.append(f"{label}: modified class, constructor or super binding")
            continue
        if len(child.bases) != 1:
            omissions.append(f"{label}: one explicit base class is required")
            continue
        base = child.bases[0]
        if isinstance(base, ast.Subscript):
            if any(
                not isinstance(node, (ast.Name, ast.Attribute, ast.Tuple, ast.Constant, ast.Load))
                for node in ast.walk(base.slice)
            ):
                omissions.append(f"{label}: dynamic generic base argument is unresolved")
                continue
            base = base.value
        if not isinstance(base, ast.Name):
            omissions.append(f"{label}: dynamic or qualified base is unresolved")
            continue
        if any(
            parameter.name == base.id
            for parameter in child.type_params
            if isinstance(parameter, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple))
        ):
            omissions.append(f"{label}: class type parameter shadows the base name")
            continue
        parents = [node for node in classes if node.name == base.id and node.lineno < child.lineno]
        if len(parents) != 1:
            omissions.append(f"{label}: no unique earlier same-file base class")
            continue
        parent = parents[0]
        if (
            parent.decorator_list
            or parent.keywords
            or _parent_binding_conflict(module.body, parent.name, parent)
        ):
            omissions.append(f"{label}: base class binding is modified")
            continue
        parent_constructors = [
            node
            for node in parent.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]
        if len(parent_constructors) != 1:
            omissions.append(f"{label}: base has no unique explicit constructor")
            continue
        target = parent_constructors[0]
        if target.decorator_list or _parent_binding_conflict(parent.body, "__init__", target):
            omissions.append(f"{label}: base constructor binding is modified")
            continue
        if isinstance(child.bases[0], ast.Subscript) and any(
            _parent_binding_conflict(parent.body, name, None)
            for name in ("__class_getitem__", "__mro_entries__")
        ):
            omissions.append(f"{label}: custom generic base construction is unresolved")
            continue
        excerpt = _excerpt(target, source, f"{parent.name}.__init__")
        identity = (excerpt.line_start, excerpt.line_end)
        if identity in seen or any(
            scope.file_path == file_path
            and scope.line_start <= excerpt.line_start <= excerpt.line_end <= scope.line_end
            for scope in selected_scope_units
        ):
            continue
        if len(excerpt.text.encode("utf-8")) > max_excerpt_bytes:
            omissions.append(f"{label}: complete parent constructor exceeds source bound")
            continue
        if len(excerpts) == 2:
            omissions.append(f"{label}: two-parent reference limit")
            continue
        excerpts.append(excerpt)
        seen.add(identity)
    return ParentConstructorSelection(tuple(excerpts), tuple(omissions))


def python_parent_class_context(
    source: str, *, file_path: str, selected_scope_units: tuple[ScopeUnit, ...]
) -> tuple[SourceExcerpt, ...]:
    """Return one complete local class owning an eligible local parent constructor.

    Reuse constructor eligibility rather than infer another binding relationship.
    Source remains bounded to 128 KiB; an entire class must fit 16 KiB.
    """
    constructors = python_parent_constructor_context(
        source, file_path=file_path, selected_scope_units=selected_scope_units
    ).excerpts
    if not constructors:
        return ()
    module = _parse_context_source(source, file_path)
    if module is None:
        return ()
    ranges = {(item.line_start, item.line_end) for item in constructors}
    for parent in module.body:
        if not isinstance(parent, ast.ClassDef) or not any(
            isinstance(method, ast.FunctionDef)
            and (method.lineno, method.end_lineno or method.lineno) in ranges
            for method in parent.body
        ):
            continue
        excerpt = _excerpt(parent, source, parent.name)
        if len(excerpt.text.encode("utf-8")) <= 16 * 1024:
            return (excerpt,)
    return ()


def python_changed_test_context(
    source: str, changed_lines: frozenset[int], *, file_path: str, max_chars: int
) -> tuple[SourceExcerpt, ...]:
    """Select complete changed tests, including decorators, within the source cap."""
    module = _parse_context_source(source, file_path) if changed_lines else None
    if module is None:
        return ()
    lines = source.splitlines()
    source_bytes = source.encode("utf-8")
    excerpts: list[SourceExcerpt] = []
    used_chars = 0
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
        end = node.end_lineno or node.lineno
        if not any(start <= line <= end for line in changed_lines):
            continue
        # A parenthesized decorator can start before its AST expression's line.
        if node.decorator_list and not lines[start - 1].lstrip().startswith("@"):
            continue
        span = line_range_to_span(start, end, source)
        text = source_bytes[span.byte_start : span.byte_end].decode("utf-8")
        suffix = f"@{start}:{end}"
        name = node.name[: 1024 - len(suffix)] + suffix
        excerpt = SourceExcerpt(name, start, end, text)
        if used_chars + len(excerpt.text) <= max_chars:
            excerpts.append(excerpt)
            used_chars += len(excerpt.text)
        if len(excerpts) == 2:
            break
    return tuple(excerpts)
