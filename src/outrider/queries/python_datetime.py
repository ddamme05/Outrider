"""Prove the local Python shape behind a datetime-awareness query match."""

from __future__ import annotations

import ast
import symtable
from typing import TYPE_CHECKING

from outrider.coordinates import query_span_to_source_lines

if TYPE_CHECKING:
    from outrider.ast_facts.models import QueryMatchSpan


def _returns_input_text(handler: ast.ExceptHandler, parameters: set[str]) -> bool:
    match handler.body:
        case [
            ast.Return(
                value=ast.Call(func=ast.Name(id="str"), args=[ast.Name(id=value)], keywords=[])
            )
        ]:
            return value in parameters
        case _:
            return False


def _initial_branch(function: ast.FunctionDef, parameters: set[str]) -> tuple[str, ast.If] | None:
    statements = iter(
        function.body[1:] if ast.get_docstring(function, clean=False) is not None else function.body
    )
    match next(statements, None):
        case ast.Import(names=[ast.alias(name="datetime", asname=alias)]):
            module = alias or "datetime"
        case _:
            return None
    if module in parameters:
        return None

    statement = next(statements, None)
    # A copy of an input, as used when a later formatting call needs the original.
    match statement:
        case ast.Assign(targets=[ast.Name(id=target)], value=ast.Name(id=value)):
            if (
                target in parameters
                or target in {module, "isinstance", "str"}
                or value not in parameters
            ):
                return None
            statement = next(statements, None)

    if isinstance(statement, ast.Try):
        # Allow the raw-text fallback seen in Humanize. Other handlers can recover
        # the intended naive date, so their behavior needs a separate proof.
        if not statement.handlers or statement.orelse or statement.finalbody:
            return None
        if not all(_returns_input_text(handler, parameters) for handler in statement.handlers):
            return None
        statement = statement.body[0]
    if isinstance(statement, ast.If):
        return module, statement
    return None


def _guarded_parameter(test: ast.expr, module: str, parameters: set[str]) -> str | None:
    match test:
        case ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Call(
                    func=ast.Name(id="isinstance"),
                    args=[
                        ast.Name(id=value),
                        ast.Attribute(value=ast.Name(id=checked_module), attr="datetime"),
                    ],
                    keywords=[],
                ),
                ast.Compare(
                    left=ast.Attribute(value=ast.Name(id=guarded), attr="tzinfo"),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
            ],
        ) if value == guarded and value in parameters and checked_module == module:
            return value
        case _:
            return None


def _checks_current_date(statement: ast.stmt, module: str, value: str) -> bool:
    match statement:
        case ast.Assign(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(value=ast.Name(id=receiver), attr="datetime"),
                            attr="now",
                        ),
                        args=[ast.Attribute(value=ast.Name(id=argument), attr="tzinfo")],
                        keywords=[],
                    ),
                    attr="date",
                ),
                args=[],
                keywords=[],
            ),
        ):
            return receiver == module and argument == value
        case _:
            return False


def _allows_naive_input(annotation: ast.expr | None, module: str) -> bool:
    match annotation:
        case None:
            return True
        case ast.Attribute(value=ast.Name(id=name), attr="datetime" | "date"):
            return name == module
        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
            return _allows_naive_input(left, module) and _allows_naive_input(right, module)
        case _:
            return False


def datetime_now_without_offset_check(
    matches: list[QueryMatchSpan], source: bytes
) -> list[QueryMatchSpan]:
    """Keep guards that send a caller's tzinfo directly to datetime.now()."""
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text)
        symbols = symtable.symtable(text, "<reviewed source>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        return []
    names = {"isinstance", "str"}
    for name in names.intersection(symbols.get_identifiers()):
        builtin = symbols.lookup(name)
        if builtin.is_assigned() or builtin.is_imported():
            return []
    if any(
        (isinstance(node, ast.ImportFrom) and any(name.name == "*" for name in node.names))
        or (isinstance(node, ast.Global) and names.intersection(node.names))
        for node in ast.walk(tree)
    ):
        return []

    guards = {
        query_span_to_source_lines(
            byte_start=match.byte_start, byte_end=match.byte_end, head_content=text
        )[0]: match
        for match in matches
    }
    scopes = {(table.get_name(), table.get_lineno()): table for table in symbols.get_children()}
    proven = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.decorator_list:
            continue
        annotations = {
            arg.arg: arg.annotation
            for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        }
        parameters = set(annotations)
        initial = _initial_branch(fn, parameters)
        if initial is None:
            continue
        module, guard = initial
        if guard.test.lineno not in guards:
            continue
        # A local binding anywhere in this function changes what the guard calls.
        locals_ = scopes.get((fn.name, fn.lineno))
        if locals_ is None:
            continue
        if any(
            locals_.lookup(name).is_local()
            for name in names.intersection(locals_.get_identifiers())
        ):
            continue
        value = _guarded_parameter(guard.test, module, parameters)
        if (
            value is not None
            and _allows_naive_input(annotations[value], module)
            and _checks_current_date(guard.body[0], module, value)
        ):
            proven.append(guards[guard.test.lineno])
    return proven
