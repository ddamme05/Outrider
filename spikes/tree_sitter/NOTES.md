# Tree-sitter spike — notes

**Scope.** Per `DECISIONS.md#006-two-month-0-spikes-not-five`: parse a Python
file, extract function definitions with line ranges, map a diff line range to
the containing function, confirm S-expression queries behave as documented.
Throwaway code — no production paths depend on this directory.

**Versions (pinned in `requirements.txt`).**
- Python 3.13.13 (from `../../.python-version`)
- `tree-sitter==0.25.2`
- `tree-sitter-python==0.25.0`

**Method.** Docs-first pass (`aegis-docs::tree-sitter/python-source-analysis.md`
and `tree-sitter/using-parsers/queries/1-syntax.md`); demos only for questions
the docs don't resolve cleanly. Each demo asserts and exits nonzero on failure
so `run_all.py` is mechanical verification, not eyeball inspection.

**Status.** `python run_all.py` → **6/6 demos pass**.

---

## Q1 — grammar coverage

**Q1a (must-pass): eval-target parsing.** `pygoat_introduction_views.py` (first
200 lines of `adeyosemanputra/pygoat@master:introduction/views.py`, real Django
views including decorators, SQLi-lab target, and `@dataclass class TestUser`)
parses with zero ERROR or MISSING nodes.

**Q1b (document-limitations): 3.13 feature parsing.** `modern_python.py` —
PEP 695 type aliases (`type Vector[T] = list[T]`), PEP 695 generic functions
(`def first[T](items: list[T]) -> T | None`), PEP 695 generic class with bound
(`class Box[T: (int, str)]`), structural pattern matching, walrus, async,
f-string with nested quotes — **all parse clean**. No limitations to document
for V1.

**Conclusion.** Tree-sitter-python 0.25.0 is production-ready for V1's Python-
only language scope. No grammar-level workarounds needed.

**Demo.** `demos/demo_q1_grammar_coverage.py`.

---

## Q2 — scope-node identification with decorators

**Finding.** When a function has decorators, tree-sitter-python wraps the
`function_definition` in a `decorated_definition` node. The parent of a matched
`function_definition` is `decorated_definition`, **not** the enclosing
`class_definition.body` or `module`. Un-decorated top-level functions have
`module` as parent directly.

**Why this matters for `ast_facts/`.**
- `ScopeUnit.line_start` (spec §5.4) must come from `decorated_definition.start_point`
  when decorators exist, otherwise from `function_definition.start_point`. Using
  `function_definition.start_point` unconditionally causes findings on decorator
  lines to fall outside the scope — which violates the Q6 edge case
  "line on decorator → innermost scope is the decorated function."
- Qualified-name walker (Q3) must skip `decorated_definition` when collecting
  named ancestors.

**Demo.** `demos/demo_q2_decorated_definition_parent.py`.

---

## Q3 — qualified name derivation

**Finding.** A single tree-walk produces every `qualified_name` spec §5.4
requires. Pattern: walk `node.parent` upward; collect the `name` field from
every ancestor whose type is `function_definition` or `class_definition`;
skip `decorated_definition`, `block`, and `module`; join with `.`.

The fixture `nested_and_decorators.py` exercises 13 distinct qualified names
including `Outer.Inner.greet`, `Outer.Inner.greet._clean` (nested function
inside a method), and `retry.decorator.wrapper` (function returned by a
function returned by a function). All 13 derive correctly from the walker.

**Gotcha.** `decorated_definition` is skipped implicitly because it's neither
`function_definition` nor `class_definition` — the walker ignores it and
continues to the next ancestor. This only works because the walker uses
`node.parent`, not `node.child_by_field_name(...)`-style navigation.

**Demo.** `demos/demo_q3_qualified_name.py`.

---

## Q4 — byte/point duality on multi-byte UTF-8

**Finding.** On non-ASCII source (`non_ascii.py` — Greek/Cyrillic identifiers,
emoji docstrings), `source[node.start_byte:node.end_byte]` and a reconstruction
via `start_point`/`end_point` with `source.split(b"\n")` produce identical
bytes. `start_point`/`end_point` columns are byte offsets within the row, not
codepoint offsets. Byte offsets always land on UTF-8 character boundaries
(every tested slice decodes without error).

PEP 3131 identifiers (`def α`, `class Привет`) parse as regular
`function_definition` / `class_definition` nodes — no special handling.

**Why this matters for `coordinates/`.** Spec §5.6 says coordinate translation
uses bytes. That's safe: bytes and points agree, and bytes give O(1) slicing.
If a future coordinate check ever uses `point.column` it still works, because
`point.column` is a byte offset.

**Demo.** `demos/demo_q4_utf8_byte_point_duality.py`.

---

## Q5 — S-expression query mechanics

**Answered by docs.** See
`aegis-docs::tree-sitter/python-source-analysis.md` §"Complete Runnable Example"
and `tree-sitter/using-parsers/queries/1-syntax.md` for the full reference:
field names (`name:`, `body:`, `parameters:`), negated fields (`!return_type`),
wildcard `(_)`, `(ERROR)` and `(MISSING)` matching, supertype `(expression)`.

No Q5 demo — every other demo uses the query machinery and would fail if it
didn't behave as documented.

**One real finding that came out of Q2/Q7.** See the "captures API" gotcha
below.

---

## Q6 — diff-line → innermost scope

**Finding.** The algorithm from `python-source-analysis.md` ("pick the innermost
containing scope") works on every edge case we tried:
- Line on a decorator above a method → method scope (because `decorated_definition`
  range starts at the decorator; Q2).
- Line at the first/last of a nested function → nested function, not its outer.
- Line in a class body between methods → class scope.
- Line at module level (an `import` line) → `None`, i.e., `ChangedRegion.owning_scope_ids`
  is legally empty per spec §5.4.
- Line past EOF → `None`.

**Implementation note.** Two equivalent strategies work:
1. Collect all scopes with `(start, end)` ranges, sort by width ascending,
   linear-scan for the first containing scope. O(N) per lookup, trivial code.
2. `tree.root_node.descendant_for_point_range(start, end)` walks the tree once.

The spike uses strategy 1 because it matches the mental model of
`ChangedRegion.owning_scope_ids` and because batching all scopes once per file
is what `ast_facts/` will do anyway. For PRs with many hunks per file, this is
the right shape.

**Dedup is required.** A decorated function matches *twice* against the
combined query (once as `decorated_definition`, once as the inner
`function_definition`). Dedupe by the inner function node's identity; keep the
wider extent (decorator line included). The spike dedup logic is in
`demos/demo_q6_diff_line_to_scope.py::collect_scopes`.

**Demo.** `demos/demo_q6_diff_line_to_scope.py`.

---

## Q7 — parse-error localization

**Finding.** `node.has_error` is a reliable per-scope signal. When a syntax
error is inside one function body:
- `root_node.has_error == True`
- `broken_function.has_error == True`
- `sibling_functions.has_error == False`

When a syntax error is at the module level outside any function:
- `root_node.has_error == True`
- All `function_definition` nodes in the file have `has_error == False`

**Implication for the `parse-errors-degrade-to-judged` invariant (§5.5).**
The policy can be implemented directly:
1. Parse the file.
2. If `root_node.has_error == False` → clean parse, findings can be OBSERVED/INFERRED.
3. For each scope containing a changed region, check the scope's `has_error`.
   If True → that scope degrades to JUDGED. If False → the scope is reliable
   even when the file has errors elsewhere.
4. If the whole file can't be parsed at all (e.g., encoding error) →
   `parse_failed` event, skip structural analysis, JUDGED.

This is what spec §5.5's four-tier degradation policy relies on, and the
primitive `ast_facts/` needs is `node.has_error`.

**Demo.** `demos/demo_q7_parse_errors_localized.py`.

---

## Gotchas discovered during the spike

### Captures are `list[Node]`, not `Node`

The canonical docs example in `python-source-analysis.md` writes:

```python
for _pattern_index, captures in QueryCursor(FUNC_QUERY).matches(tree.root_node):
    defn = captures["function.def"]        # ← this treats the capture as a Node
    name = captures["function.name"]
```

That is incorrect on `tree-sitter==0.25.2`. The actual shape is:

```python
captures["function.def"]  # list[Node] — one element per match in an
                          # un-quantified capture; more for quantified (*, +).
```

Every demo in this spike uses `captures["x"][0]` for single captures. The
`ast_facts/python_adapter.py` build should do the same, and a small utility
`def single(caps, key) -> Node: return caps[key][0]` is worth having.

**Action for `ast_facts/`.** Do not copy the docs snippet verbatim. Either
treat captures as always-a-list, or index `[0]` at the call site. The test
that catches a regression here is trivial: any capture that's used as a Node
without `[0]` raises `AttributeError: 'list' object has no attribute
'start_byte'` immediately on first parse.

---

## Spike findings — recommended defaults for `ast_facts/python_adapter.py`

These are informed starting points, not locked-in decisions. The real build
can deviate with a `DECISIONS.md` entry if experience shows a better choice;
this table captures what the spike actually observed and what it suggests
trying first.

| Question | Recommended default |
|---|---|
| ScopeUnit.line_start source | `decorated_definition.start_point[0] + 1` if decorated, else `function_definition.start_point[0] + 1`. |
| Qualified name strategy | Walk `node.parent`; collect `name` from `function_definition`/`class_definition`; skip others. |
| Text extraction | `source[node.start_byte:node.end_byte].decode("utf-8")`. Safe across UTF-8 (see Q4; non-UTF-8 not covered, see below). |
| Diff-line → ScopeUnit mapping | Pre-collect scopes sorted by width; linear scan; dedup by inner node identity when decorated. |
| Parse-failure classification | `root_node.has_error` for file-level; `scope.has_error` for per-scope. The 4-tier §5.5 policy maps directly. |
| Query machinery | Per docs. Treat captures as `list[Node]`. |

## What the spike did NOT cover — deferred to the real build

- **Non-UTF-8 source encodings.** Q4 covered multi-byte UTF-8 but not Latin-1
  or Windows-1252. `source[...].decode("utf-8")` will raise `UnicodeDecodeError`
  on non-UTF-8 files. `ast_facts/` needs either an encoding-detection fallback
  (e.g., `chardet`) or a policy — the natural one being *decode error → emit
  `parse_failed` event, file degrades to JUDGED* per spec §5.5.
- **Performance.** No benchmark on 10K-line files or batch-parsing a real PR.
  `ast_facts/` can measure this in-situ; the parse API is fast enough on the
  fixtures here (sub-millisecond for 200-line files).
- **Cross-file import resolution.** Per `DECISIONS.md#006`, that folds into
  the real `ast_facts/` build with same-file-only as the fallback.
- **Coordinate translation.** Stays in `coordinates/` per the trust boundary.
  The spike only confirmed the byte/point primitives are consistent.
- **Incremental reparsing.** `tree.edit()` + `changed_ranges()` from
  `using-parsers/3-advanced-parsing.md` — not on the V1 hot path.
- **Structural tests in `tests/eval/scenarios/structural/`.** That's the real
  correctness tier per `docs/testing.md` and remains required before
  `ast_facts/` ships.

## Reproducing

```
cd spikes/tree_sitter
/home/spinbot/projects/outrider/.venv/bin/python run_all.py
```

Exits 0 iff every claim above reproduces.
