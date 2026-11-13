"""Select Python import names and exact source references without fetching files.

Only explicit relative imports used by selected changed functions become requests.
Coordinates constructs paths; the agent owns reads and the final prompt budget. These
facts show syntactic relevance; they do not establish runtime import resolution.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from outrider.ast_facts.parser_outcome import should_skip
from outrider.ast_facts.review_context import SourceExcerpt
from outrider.coordinates import line_range_to_span

_MAX_REFERENCE_SOURCE_BYTES = 128 * 1024
_MAX_REFERENCE_EXCERPTS = 4


@dataclass(frozen=True)
class ImportReferenceRequest:
    """An explicit import and original names; no repository paths are inferred."""

    module: str
    level: int
    imported_names: tuple[str, ...]
    importer_path: str
    line: int


@dataclass(frozen=True)
class ImportReferenceRequests:
    """Requested imported names and reasons other import forms were omitted."""

    requests: tuple[ImportReferenceRequest, ...]
    omissions: tuple[str, ...]


@dataclass(frozen=True)
class ImportReferenceSelection:
    """Complete source excerpts and their explicit relative-import dependencies."""

    excerpts: tuple[SourceExcerpt, ...]
    omissions: tuple[str, ...]
    import_requests: tuple[ImportReferenceRequest, ...]


def _parse_reference_source(source: str, file_path: str) -> tuple[ast.Module | None, str | None]:
    """Apply the source/parser exclusion boundary before traversing an AST."""
    if not file_path.lower().endswith(".py"):
        return None, "unsupported language (Python references only)"
    if len(source) > _MAX_REFERENCE_SOURCE_BYTES:
        return None, "source exceeds 128-KiB reference bound"
    try:
        encoded = source.encode("utf-8")
        if len(encoded) > _MAX_REFERENCE_SOURCE_BYTES:
            return None, "source exceeds 128-KiB reference bound"
        excluded = should_skip(file_path, encoded)
        if excluded is not None:
            return None, f"source excluded: {excluded.value}"
        return ast.parse(source), None
    except (SyntaxError, ValueError, RecursionError, UnicodeError):
        return None, "source is not eligible parsed Python"


def _statement_start(statement: ast.stmt) -> int:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return min([statement.lineno, *(item.lineno for item in statement.decorator_list)])
    return statement.lineno


def _pattern_bound_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(
        node,
        (
            ast.ExceptHandler,
            ast.MatchAs,
            ast.MatchStar,
            ast.TypeVar,
            ast.ParamSpec,
            ast.TypeVarTuple,
        ),
    ):
        return (node.name,) if node.name is not None else ()
    if isinstance(node, ast.MatchMapping):
        return (node.rest,) if node.rest is not None else ()
    return ()


def _loaded_names(statement: ast.stmt) -> set[str]:
    """Conservatively exclude parameters and stores, including nested scopes."""
    loaded: set[str] = set()
    bound: set[str] = set()
    for node in ast.walk(statement):
        bound.update(_pattern_bound_names(node))
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif node is not statement and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            bound.add(node.name)
    return loaded - bound


def _module_binding_owners(module: ast.Module) -> dict[str, set[int]]:
    """Notice conditional module writes without entering function/class bodies."""
    bindings: dict[str, set[int]] = {}
    pending: list[ast.AST] = list(module.body)
    while pending:
        node = pending.pop()
        for name in _pattern_bound_names(node):
            bindings.setdefault(name, set()).add(id(node))
        if isinstance(node, ast.stmt):
            names = _bound_names(node)
            for name in names:
                bindings.setdefault(name, set()).add(id(node))
            if names:
                continue
        elif isinstance(node, ast.Lambda):
            continue
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bindings.setdefault(node.id, set()).add(id(node))
        pending.extend(ast.iter_child_nodes(node))
    return bindings


def _used_import_requests(
    module: ast.Module,
    selected: tuple[ast.stmt, ...],
    *,
    file_path: str,
) -> ImportReferenceRequests:
    """Match loaded local aliases to original names in visible import statements."""
    imports: dict[int, ast.Import | ast.ImportFrom] = {}
    used_names_by_import: dict[int, set[str]] = {}
    binding_owners_by_import: dict[int, dict[str, set[int]]] = {}
    local_import_ids: set[int] = set()
    calls: list[ast.Call] = []
    scope_omissions: list[str] = []
    parents = {
        id(child): parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)
    }
    module_bindings = _module_binding_owners(module)
    module_imports = []
    for statement in module.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            module_imports.append(statement)
            imports[id(statement)] = statement
            used_names_by_import[id(statement)] = set()
            binding_owners_by_import[id(statement)] = module_bindings
    for statement in selected:
        used_names = _loaded_names(statement)
        enclosing_names: set[str] = set()
        parent = parents.get(id(statement))
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for parameter in parent.type_params:
                    enclosing_names.update(_pattern_bound_names(parameter))
            parent = parents.get(id(parent))
        if used_names & enclosing_names:
            scope_omissions.append(
                f"scope line {statement.lineno}: enclosing type parameter omitted"
            )
            used_names -= enclosing_names
        local_imports = []
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            # A selected module import may itself be a re-export binding.
            used_names.update(_bound_names(statement))
        else:
            local_imports = [
                node
                for node in ast.walk(statement)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
        local_bindings: dict[str, set[int]] = {}
        for local_import in local_imports:
            for name in _bound_names(local_import):
                local_bindings.setdefault(name, set()).add(id(local_import))
        # A function's imports bind its local aliases even when a module import
        # has the same name. Keep each selected function's uses independent.
        module_names = used_names - local_bindings.keys()
        for module_import in module_imports:
            used_names_by_import[id(module_import)].update(module_names)
        for local_import in local_imports:
            identity = id(local_import)
            imports[identity] = local_import
            used_names_by_import.setdefault(identity, set()).update(used_names)
            binding_owners_by_import[identity] = local_bindings
            local_import_ids.add(identity)
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                calls.append(node)

    requests: list[ImportReferenceRequest] = []
    omissions = scope_omissions
    dynamic_functions = {"__import__"}
    dynamic_modules: set[str] = set()
    for statement in sorted(imports.values(), key=lambda item: (item.lineno, item.col_offset)):
        used_names = used_names_by_import[id(statement)]
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".")[0]
                if alias.name == "importlib":
                    dynamic_modules.add(local_name)
            relevant = any(
                (alias.asname or alias.name.split(".")[0]) in used_names
                for alias in statement.names
            )
            if relevant:
                omissions.append(f"import line {statement.lineno}: absolute import omitted")
            continue
        if statement.module == "importlib" and statement.level == 0:
            for alias in statement.names:
                if alias.name == "import_module":
                    dynamic_functions.add(alias.asname or alias.name)
        if any(alias.name == "*" for alias in statement.names):
            omissions.append(f"import line {statement.lineno}: star import omitted")
            continue
        matching_names: list[str] = []
        for alias in statement.names:
            local_name = alias.asname or alias.name
            if local_name not in used_names:
                continue
            binding_owners = binding_owners_by_import[id(statement)]
            other_bindings = binding_owners.get(local_name, set()) - {id(statement)}
            if other_bindings:
                scope = "local import" if id(statement) in local_import_ids else "module"
                omissions.append(
                    f"import line {statement.lineno}: name {local_name} has another {scope} binding"
                )
                continue
            matching_names.append(alias.name)
        imported_names = tuple(dict.fromkeys(matching_names))
        if not imported_names:
            continue
        if statement.level == 0:
            omissions.append(f"import line {statement.lineno}: absolute import omitted")
        elif statement.module is None:
            omissions.append(
                f"import line {statement.lineno}: relative symbol or child-module form omitted"
            )
        else:
            requests.append(
                ImportReferenceRequest(
                    module=statement.module,
                    level=statement.level,
                    imported_names=imported_names,
                    importer_path=file_path,
                    line=statement.lineno,
                )
            )
    for call in sorted(calls, key=lambda item: (item.lineno, item.col_offset)):
        function = call.func
        dynamic = isinstance(function, ast.Name) and function.id in dynamic_functions
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id in dynamic_modules
        ):
            dynamic = True
        if dynamic:
            omissions.append(f"call line {call.lineno}: dynamic import omitted")
    return ImportReferenceRequests(tuple(requests), tuple(dict.fromkeys(omissions)))


def python_import_reference_requests(
    source: str,
    *,
    file_path: str,
    changed_lines: frozenset[int],
) -> ImportReferenceRequests:
    """Request explicit relative imports used in functions owning changed lines.

    The caller supplies its admitted changed-scope lines, including retained
    scopes with removals. Unchanged functions, absolute imports, star imports,
    dynamic imports and unknown child-module forms do not cause acquisition.
    """
    module, omission = _parse_reference_source(source, file_path)
    if module is None:
        return ImportReferenceRequests((), (omission,) if omission else ())
    selected: list[ast.stmt] = []
    parents = {
        id(child): parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _statement_start(node) <= line <= (node.end_lineno or node.lineno)
            for line in changed_lines
        ):
            # Its enclosing function owns the same changed lines and bindings.
            # Reviewing the nested function again could mistake closure names for globals.
            parent = parents.get(id(node))
            while parent is not None and not isinstance(
                parent, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                parent = parents.get(id(parent))
            if parent is None:
                selected.append(node)
    if not selected:
        return ImportReferenceRequests((), ("no selected changed Python functions",))
    return _used_import_requests(module, tuple(selected), file_path=file_path)


def _bound_names(statement: ast.stmt) -> tuple[str, ...]:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (statement.name,)
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    elif isinstance(statement, (ast.Import, ast.ImportFrom)):
        return tuple(alias.asname or alias.name.split(".")[0] for alias in statement.names)
    else:
        return ()
    return tuple(
        dict.fromkeys(
            node.id
            for target in targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
    )


def python_import_reference_excerpts(
    source: str,
    *,
    file_path: str,
    requested_names: tuple[str, ...],
    max_excerpt_chars: int = 4096,
) -> ImportReferenceSelection:
    """Select at most four exact top-level definitions or binding statements.

    Excerpts retain decorators, docstrings and guards. A definition that exceeds
    the character cap is omitted whole. Duplicate bindings and unavailable names
    stay unresolved. Imports used by admitted excerpts can support one separately
    bounded acquisition hop; these requests never initiate I/O themselves.
    """
    if max_excerpt_chars <= 0:
        raise ValueError("max_excerpt_chars must be positive")
    module, omission = _parse_reference_source(source, file_path)
    if module is None:
        return ImportReferenceSelection((), (omission,) if omission else (), ())
    bindings: dict[str, list[ast.stmt]] = {}
    for statement in module.body:
        for name in _bound_names(statement):
            bindings.setdefault(name, []).append(statement)
    module_bindings = _module_binding_owners(module)

    excerpts: list[SourceExcerpt] = []
    selected: list[ast.stmt] = []
    omissions: list[str] = []
    encoded = source.encode("utf-8")
    for name in dict.fromkeys(requested_names):
        definitions = bindings.get(name, [])
        if len(definitions) != 1 or len(module_bindings.get(name, set())) != 1:
            reason = "ambiguous top-level binding" if definitions else "no top-level binding"
            omissions.append(f"name {name}: {reason}")
            continue
        statement = definitions[0]
        if statement in selected:
            continue
        if len(excerpts) == _MAX_REFERENCE_EXCERPTS:
            omissions.append(f"name {name}: four-excerpt limit")
            continue
        first_line = _statement_start(statement)
        last_line = statement.end_lineno or statement.lineno
        span = line_range_to_span(first_line, last_line, source)
        text = encoded[span.byte_start : span.byte_end].decode("utf-8")
        if len(text) > max_excerpt_chars:
            omissions.append(f"name {name}: complete excerpt exceeds character bound")
            continue
        suffix = f"@{first_line}:{last_line}"
        excerpts.append(
            SourceExcerpt(
                name[: 1024 - len(suffix)] + suffix,
                first_line,
                last_line,
                text,
            )
        )
        selected.append(statement)
    imports = _used_import_requests(module, tuple(selected), file_path=file_path)
    return ImportReferenceSelection(
        tuple(excerpts),
        tuple([*omissions, *imports.omissions]),
        imports.requests,
    )
