# Per specs/2026-06-12-sqli-parameterized-call-veto.md (FUP-162).
"""Deterministic detection of provably-parameterized DB-execute calls.

The structural half of the `sql_injection` parameterized-call veto: the
analyze parser rejects a JUDGED `sql_injection` proposal whose claimed
lines land on a call this module classifies safe. Safe has TWO parts:
(1) the SQL argument is a pure string LITERAL (no f-string
interpolation, no concatenation with non-literals) passed alongside a
SEPARATE params argument — structural proof that no user input reaches
the string AT THE CALL SITE; and (2) the receiver idiom conventionally
pins binding semantics (`_BINDING_RECEIVERS` / the Django manager
chain) — because whether the callee BINDS the second argument or
interpolates it into the string is the callee's contract, not the call
shape's. Bare `execute(...)` and arbitrary-receiver wrappers fail open
to HITL.

Triviality-filter precedent (`ast_facts/triviality.py`): this module
parses with tree-sitter directly and walks raw nodes INTERNALLY; only
line-range domain models cross the AST firewall. Deliberately NOT a
registry query — registry entries are claimable OBSERVED evidence
(`query_match_id`), and a veto's detection must never be citable as
evidence.

Versioning: any change to the matcher rules or the method-name set
below changes per-file analyze outcomes and MUST ride an
`ANALYZE_PARSER_VERSION` bump (the FUP-166 principle; the veto is part
of the admission flow that constant versions).

See DECISIONS.md#041 (Amended 2026-06-12): this detection is the
deterministic half of the over-flag remediation that the #041 fix
previously carried entirely in the analyze prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import tree_sitter_python
from pydantic import BaseModel, ConfigDict, Field
from tree_sitter import Language, Parser

if TYPE_CHECKING:
    from tree_sitter import Node

_PY_LANGUAGE: Final = Language(tree_sitter_python.language())
_PARSER: Final = Parser(_PY_LANGUAGE)

# Execute-like method names. Conservative on purpose: the three shapes
# DECISIONS.md#041's over-flag evidence actually exercised (DB-API
# cursor.execute / .executemany, Django ORM .raw). Widening the set is
# an ANALYZE_PARSER_VERSION bump.
_EXECUTE_LIKE_METHODS: Final = frozenset({"execute", "executemany", "raw"})

# Receiver idioms whose binding contract is conventionally pinned. The
# safety claim has TWO parts and only one is structural: a pure-literal
# SQL argument proves no user input reaches the STRING at the call site,
# but whether the second argument is BOUND (safe) or interpolated into
# the string by the callee (injectable) is the callee's contract. A bare
# `execute(...)` or an arbitrary `wrapper.execute(...)` can interpolate —
# a project wrapper doing `sql % args` internally looks identical at the
# call site — so only receivers whose idiom conventionally pins DB-API /
# ORM binding semantics qualify for the safe set. Everything else stays
# execute-like (span-conflict protection) and FAILS OPEN to HITL.
_BINDING_RECEIVERS: Final = frozenset({"cursor", "cur", "conn", "connection"})


class ExecuteCallSite(BaseModel):
    """1-indexed inclusive line range of one execute-like call node.

    Line ranges — not byte spans — cross the firewall: the veto
    comparison is line-space (`coordinates/spans.py`), because the
    model anchors findings by line and a whole-line byte span starts
    at column 0, before an indented call node's token-based start.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)


class ParameterizedCallScan(BaseModel):
    """The two detection sets the veto consumes.

    `safe_parameterized_calls` ⊆ `all_execute_like_calls` by
    construction. The veto requires a proposal's line range to be
    contained in a safe call AND to intersect no execute-like site
    outside the safe set — so a range spanning a safe and an unsafe
    call passes through to HITL untouched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    safe_parameterized_calls: tuple[ExecuteCallSite, ...] = ()
    all_execute_like_calls: tuple[ExecuteCallSite, ...] = ()


def _identifier_text(node: Node | None) -> str | None:
    if node is not None and node.type == "identifier" and node.text is not None:
        return node.text.decode("utf-8")
    return None


def _final_attribute_name(node: Node) -> str | None:
    """The last dotted component of a receiver expression, if nameable.

    `cursor` → "cursor"; `self.cursor` → "cursor"; `conn.cursor()` →
    "cursor" (the chained-call idiom); anything else → None.
    """
    if node.type == "identifier":
        return _identifier_text(node)
    if node.type == "attribute":
        return _identifier_text(node.child_by_field_name("attribute"))
    if node.type == "call":
        function = node.child_by_field_name("function")
        if function is not None and function.type == "attribute":
            return _identifier_text(function.child_by_field_name("attribute"))
        if function is not None and function.type == "identifier":
            return _identifier_text(function)
    return None


def _call_names(call: Node) -> tuple[str | None, str | None]:
    """(method_name, receiver_final_component) for a call node.

    Receiver is None for bare calls (`execute(...)`) — which are
    execute-like for span-conflict purposes but never safe: a bare name
    is most likely a project wrapper with unknown binding semantics.
    """
    function = call.child_by_field_name("function")
    if function is None:
        return (None, None)
    if function.type == "attribute":
        method = _identifier_text(function.child_by_field_name("attribute"))
        obj = function.child_by_field_name("object")
        receiver = _final_attribute_name(obj) if obj is not None else None
        return (method, receiver)
    return (_identifier_text(function), None)


def _binding_receiver_ok(method: str, receiver: str | None) -> bool:
    """True iff the receiver idiom conventionally pins binding semantics."""
    if receiver is None:
        return False
    if method in ("execute", "executemany"):
        return receiver in _BINDING_RECEIVERS
    if method == "raw":
        # The Django manager chain: `Model.objects.raw(...)`.
        return receiver == "objects"
    return False


def _unwrap_parentheses(node: Node) -> Node:
    while node.type == "parenthesized_expression":
        inner = next((c for c in node.named_children if c.type != "comment"), None)
        if inner is None:
            return node
        node = inner
    return node


def _is_pure_literal_string(node: Node) -> bool:
    """True iff `node` is a string literal with NO interpolation anywhere.

    The Python grammar represents f-strings as `string` nodes WITH
    `interpolation` children — literal-purity is a descendant walk, not
    a node-type check. Implicit concatenation (`"a" "b"`) is pure iff
    every part is.
    """
    node = _unwrap_parentheses(node)
    if node.type == "concatenated_string":
        return all(_is_pure_literal_string(part) for part in node.named_children)
    if node.type != "string":
        return False
    stack = list(node.children)
    while stack:
        child = stack.pop()
        if child.type == "interpolation":
            return False
        stack.extend(child.children)
    return True


def _is_safe_parameterized(call: Node) -> bool:
    """First argument is a pure literal string AND ≥1 further argument."""
    arguments = call.child_by_field_name("arguments")
    if arguments is None:
        return False
    args = [c for c in arguments.named_children if c.type != "comment"]
    if len(args) < 2:
        return False
    return _is_pure_literal_string(args[0])


def scan_parameterized_calls(source: bytes) -> ParameterizedCallScan:
    """Scan source for execute-like calls; classify provably-safe ones.

    Pure computation, fail-open by design: anything this scan cannot
    prove safe is simply absent from `safe_parameterized_calls`, and
    the veto then lets the model's proposal through to HITL. A tree
    carrying ANY syntax error returns the empty scan — error recovery
    could misshape a call node, and a veto must never rest on an
    untrustworthy parse (`parse-errors-degrade-to-judged`; degraded-mode
    callers additionally pass `None` to the parser instead of a scan).
    """
    tree = _PARSER.parse(source)
    if tree.root_node.has_error:
        return ParameterizedCallScan()
    safe: list[ExecuteCallSite] = []
    all_sites: list[ExecuteCallSite] = []
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call":
            method, receiver = _call_names(node)
            if method in _EXECUTE_LIKE_METHODS:
                site = ExecuteCallSite(
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                )
                all_sites.append(site)
                if _binding_receiver_ok(method, receiver) and _is_safe_parameterized(node):
                    safe.append(site)
        stack.extend(node.named_children)
    return ParameterizedCallScan(
        safe_parameterized_calls=tuple(safe),
        all_execute_like_calls=tuple(all_sites),
    )
