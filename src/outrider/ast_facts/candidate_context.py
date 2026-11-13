"""Find captured defaults, attribute assignments and possible initializers.

Selection follows parsed source names, not model prose. Receiver identity and
execution order remain unknown; these excerpts are reference material only.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from outrider.ast_facts.review_context import CalledFunctionExcerpt, SourceExcerpt
from outrider.coordinates import extract_line_range_body

if TYPE_CHECKING:
    from collections.abc import Iterator

    from outrider.ast_facts.review_context import CalledFunctionIndex

_MAX_ATTRIBUTES = 8
_MAX_ASSIGNMENTS = 8
_MAX_REFERENCES = 4
DEFAULT_CALLABLE_REFERENCE_REASON = "possible default implementation; overrides unproven"
_CONTROL_BLOCKS = (
    ast.If,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Match,
)


@dataclass(frozen=True)
class CandidateContextSelection:
    references: tuple[CalledFunctionExcerpt, ...]
    omissions: tuple[str, ...]


@dataclass(frozen=True)
class _Assignment:
    file_path: str
    statement: ast.Assign | ast.AnnAssign | ast.AugAssign
    context: ast.stmt
    value: ast.expr | None


def _attribute_assignments(tree: ast.Module, path: str, names: set[str]) -> list[_Assignment]:
    assignments = []
    pending: list[tuple[ast.AST, tuple[ast.stmt, ...]]] = [(tree, ())]
    while pending:
        node, controls = pending.pop()
        if isinstance(node, _CONTROL_BLOCKS):
            controls = (*controls, node)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(
                isinstance(target, ast.Attribute) and target.attr in names for target in targets
            ):
                assignments.append(
                    _Assignment(path, node, controls[0] if controls else node, node.value)
                )
        pending.extend((child, controls) for child in reversed(list(ast.iter_child_nodes(node))))
    return assignments


def _reference(
    index: CalledFunctionIndex, path: str, start: int, end: int, purpose: str
) -> CalledFunctionExcerpt:
    text = extract_line_range_body(start, end, index.sources[path].decode()).removesuffix("\n")
    return CalledFunctionExcerpt(path, SourceExcerpt(purpose, start, end, text))


def _scope_nodes(statements: list[ast.stmt]) -> Iterator[ast.AST]:
    pending: list[ast.AST] = list(reversed(statements))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _scope_bindings(statements: list[ast.stmt]) -> dict[str, list[ast.AST]]:
    bindings: dict[str, list[ast.AST]] = {}
    for node in _scope_nodes(statements):
        names = []
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.append(node.name)
        for name in names:
            bindings.setdefault(name, []).append(node)
    return bindings


def _default_callable_context(
    index: CalledFunctionIndex,
    tree: ast.Module,
    file_path: str,
    line_start: int,
    line_end: int,
    max_source_bytes: int,
) -> CandidateContextSelection:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line_start <= line_end <= (node.end_lineno or node.lineno)
    ]
    if not functions:
        return CandidateContextSelection((), ())
    function = min(functions, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)
    nodes = list(_scope_nodes(function.body))
    called = {
        node.func.id
        for node in nodes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults = [
        *zip(
            positional[len(positional) - len(function.args.defaults) :],
            function.args.defaults,
            strict=True,
        ),
        *zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True),
    ]
    if not any(parameter.arg in called and default is not None for parameter, default in defaults):
        return CandidateContextSelection((), ())
    parents: list[ast.Module | ast.ClassDef] = [tree]
    parents.extend(node for node in tree.body if isinstance(node, ast.ClassDef))
    parent = next((scope for scope in parents if function in scope.body), None)
    if parent is None:
        return CandidateContextSelection((), ("callable defaults: nested function omitted",))
    rebound = _scope_bindings(function.body)
    module_bindings = _scope_bindings(tree.body)
    parent_bindings = _scope_bindings(parent.body) if isinstance(parent, ast.ClassDef) else {}
    references: list[CalledFunctionExcerpt] = []
    omissions: list[str] = []
    seen = set()
    for parameter, default in defaults:
        name = parameter.arg
        if name not in called or default is None:
            continue
        if name in rebound:
            omissions.append(f"default callable {name}: parameter is rebound")
            continue
        target = None
        if isinstance(default, ast.Name):
            bindings = module_bindings.get(default.id, [])
            if default.id not in parent_bindings and len(bindings) == 1:
                candidate = bindings[0]
                if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    target = candidate
        elif isinstance(default, ast.Attribute) and isinstance(default.value, ast.Name):
            class_name = default.value.id
            bindings = module_bindings.get(class_name, [])
            if class_name not in parent_bindings and len(bindings) == 1:
                owner = bindings[0]
                if isinstance(owner, ast.ClassDef):
                    methods = _scope_bindings(owner.body).get(default.attr, [])
                    overwritten = any(
                        isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, (ast.Store, ast.Del))
                        and isinstance(node.value, ast.Name)
                        and node.value.id == class_name
                        and node.attr == default.attr
                        for node in _scope_nodes(tree.body)
                    )
                    if len(methods) == 1 and not overwritten:
                        candidate = methods[0]
                        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            target = candidate
        if target is None:
            omissions.append(f"default callable {name}: no unambiguous same-file definition")
            continue
        if target.decorator_list:
            omissions.append(f"default callable {name}: decorated implementation omitted")
            continue
        span = (target.lineno, target.end_lineno or target.lineno)
        if span in seen:
            continue
        seen.add(span)
        reference = _reference(index, file_path, *span, DEFAULT_CALLABLE_REFERENCE_REASON)
        if len(reference.excerpt.text.encode()) > max_source_bytes:
            omissions.append(f"default callable {name}: complete implementation exceeds cap")
        elif len(references) == _MAX_REFERENCES:
            omissions.append("further default callable references omitted (reference limit)")
        else:
            references.append(reference)
    return CandidateContextSelection(tuple(references), tuple(omissions))


def candidate_dependency_context(
    index: CalledFunctionIndex,
    *,
    file_path: str,
    line_start: int,
    line_end: int,
    max_source_bytes: int = 2000,
) -> CandidateContextSelection:
    """Select at most four exact Python references within the caller's source cap.

    Called parameter defaults precede weaker attribute-name matches. Guarded
    assignments retain their control block; oversized bodies are omitted whole.
    A uniquely named captured constructor may follow a selected assignment.
    """
    if line_start < 1 or line_end < line_start or max_source_bytes < 1:
        raise ValueError("candidate context requires an ordered range and positive source cap")
    if PurePosixPath(file_path).suffix != ".py":
        return CandidateContextSelection((), ("attribute context currently supports Python only",))
    omissions = []
    trees = {}
    for path in index.parsed:
        if path in index.excluded_definition_paths:
            continue
        if PurePosixPath(path).suffix != ".py":
            continue
        try:
            trees[path] = ast.parse(index.sources[path].decode())
        except (SyntaxError, ValueError, RecursionError):
            omissions.append(f"{path}: attribute context could not be parsed")
    if file_path not in trees:
        return CandidateContextSelection((), ("attribute context unavailable for this source",))

    defaults = _default_callable_context(
        index, trees[file_path], file_path, line_start, line_end, max_source_bytes
    )
    omissions.extend(defaults.omissions)
    attributes = list(
        dict.fromkeys(
            node.attr
            for node in ast.walk(trees[file_path])
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.lineno <= line_end
            and (node.end_lineno or node.lineno) >= line_start
        )
    )
    if len(attributes) > _MAX_ATTRIBUTES:
        omissions.append(f"{len(attributes) - _MAX_ATTRIBUTES} further attribute names omitted")
    names = set(attributes[:_MAX_ATTRIBUTES])
    assignments = [
        assignment
        for path, tree in trees.items()
        for assignment in _attribute_assignments(tree, path, names)
    ]
    assignments.sort(
        key=lambda item: (
            not isinstance(item.value, ast.Call),
            item.file_path,
            item.statement.lineno,
        )
    )
    matched_names: set[str] = set()
    for assignment in assignments:
        statement = assignment.statement
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        matched_names.update(target.attr for target in targets if isinstance(target, ast.Attribute))
    for name in attributes[:_MAX_ATTRIBUTES]:
        if name not in matched_names:
            omissions.append(f"attribute {name[:80]}: no captured assignment found")
    if len(assignments) > _MAX_ASSIGNMENTS:
        omissions.append(
            f"{len(assignments) - _MAX_ASSIGNMENTS} further matching assignments omitted"
        )

    references = list(defaults.references)
    seen = {(item.file_path, item.excerpt.line_start, item.excerpt.line_end) for item in references}
    for assignment in assignments[:_MAX_ASSIGNMENTS]:
        context = assignment.context
        reference = _reference(
            index,
            assignment.file_path,
            context.lineno,
            context.end_lineno or context.lineno,
            "attribute assignment name match (receiver unproven)",
        )
        candidates = [reference]
        if len(reference.excerpt.text.encode()) > max_source_bytes:
            omissions.append(
                f"{assignment.file_path}:{context.lineno}: assignment context exceeds cap"
            )
            continue
        value = assignment.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            classes = [
                scope
                for path, facts in index.parsed.items()
                if path in trees
                for scope in facts.scope_units
                if scope.kind == "class" and scope.name == value.func.id
            ]
            if len(classes) == 1:
                parent = classes[0]
                constructor = next(
                    (
                        scope
                        for scope in index.parsed[parent.file_path].scope_units
                        if scope.parent_scope_id == parent.unit_id and scope.name == "__init__"
                    ),
                    None,
                )
                if constructor is not None:
                    candidates.append(
                        _reference(
                            index,
                            constructor.file_path,
                            constructor.line_start,
                            constructor.line_end,
                            "possible constructor by unique captured name",
                        )
                    )
                else:
                    omissions.append(f"constructor {value.func.id[:80]}: no captured __init__ body")
            elif len(classes) > 1:
                omissions.append(f"constructor {value.func.id[:80]}: ambiguous captured name")
            else:
                omissions.append(f"constructor {value.func.id[:80]}: no captured class definition")
        for candidate in candidates:
            key = (candidate.file_path, candidate.excerpt.line_start, candidate.excerpt.line_end)
            if key in seen:
                continue
            seen.add(key)
            if any(
                previous.file_path == candidate.file_path
                and previous.excerpt.line_start
                <= candidate.excerpt.line_start
                <= candidate.excerpt.line_end
                <= previous.excerpt.line_end
                for previous in references
            ):
                continue
            if len(candidate.excerpt.text.encode()) > max_source_bytes:
                omissions.append(
                    f"{candidate.file_path}:{candidate.excerpt.line_start}: constructor exceeds cap"
                )
            elif len(references) == _MAX_REFERENCES:
                omissions.append("further attribute references omitted (reference limit)")
            else:
                references.append(candidate)
    return CandidateContextSelection(tuple(references), tuple(dict.fromkeys(omissions)))
