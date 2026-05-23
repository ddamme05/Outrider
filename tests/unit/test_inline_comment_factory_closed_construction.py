# Test for `InlineComment.from_finding` structural-routing per spec §V.5.
"""Pin the contract that `InlineComment.from_finding` is the only
production construction path for `InlineComment` inside `src/outrider/`.

Per the publish-node spec §V Input boundary sub-rule 5 (note: spec text
predates the 2026-05-22 side-threading fix; the runtime signature now
requires `side` from `coordinates`):

    Structural-routing assertion: all GitHub-comment-body fields route
    through sanitizer; `InlineComment.from_finding(*, finding, path,
    line, side, body)` is the only documented production construction
    path. Direct Pydantic construction is permitted by the schema
    (test fixtures need it) but an import-graph test forbids it inside
    `src/outrider/`.

This module is the import-graph test. Direct `InlineComment(...)`
construction in production code bypasses the sanitizer guarantee that
`from_finding` enforces by composing through `sanitize_display_string`
+ `apply_size_cap` at the publisher's call site. A future contributor
adding `InlineComment(body=raw_text, ...)` in production should fail
this test loudly so the sanitizer's defense doesn't silently regress.

The test scans every `.py` file under `src/outrider/` for the pattern
`InlineComment(` (constructor call, not import, not method reference).
Exemption: `schemas/publish.py` itself (defines the class) and any
file that ONLY references the class via `from_finding(...)`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _project_root() -> Path:
    """Walk up from this test file until we find pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate project root")


# Files allowed to construct `InlineComment(...)` directly.
# - `schemas/publish.py` — defines the class and the factory itself.
# Add new entries ONLY with explicit reviewer justification (e.g., a
# future test helper module). Production code should always use the
# factory.
_ALLOWED_DIRECT_CONSTRUCTION_PATHS: frozenset[str] = frozenset(
    {
        "src/outrider/schemas/publish.py",
    }
)


def _find_inline_comment_constructions(source: str) -> list[tuple[int, str]]:
    """Return (lineno, snippet) for every direct `InlineComment(...)`
    constructor call in `source`.

    Distinguishes constructor calls from:
      - Import statements (`from outrider.schemas import InlineComment`)
      - Attribute accesses (`InlineComment.from_finding(...)`)
      - Type annotations (`comments: tuple[InlineComment, ...]`)
      - Method-on-class (NOT a constructor call)

    Resolves import aliases so `from outrider.schemas import InlineComment
    as IC; IC(...)` is correctly flagged as a direct construction (the
    bare-name walker would otherwise miss aliased imports).

    Uses AST analysis rather than regex so the test doesn't false-
    positive on the patterns above.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # Silently returning `[]` would mask an unscanned production
        # file and let the structural-routing guard pass green. A
        # SyntaxError inside `src/outrider/` IS the bug; surface it.
        raise AssertionError(
            "failed to parse source while scanning for direct "
            "InlineComment(...) construction; the unscanned file would "
            "have silently bypassed the structural-routing guard."
        ) from exc

    # First pass: collect every local name + module-base path that
    # resolves to the canonical `outrider.schemas[.publish].InlineComment`.
    # Constrains alias detection to the canonical source modules so an
    # unrelated `InlineComment` from a hypothetical third-party module
    # is not treated as a false-positive target.
    schema_modules = {"outrider.schemas", "outrider.schemas.publish"}
    inline_comment_local_names: set[str] = set()
    # `schema_module_bases` holds the DOTTED-NAME forms that can appear
    # as the base of `<base>.InlineComment(...)` at a call site, after
    # resolving the import statement's actual binding semantics:
    #
    #   `import outrider.schemas`            → call uses
    #       `outrider.schemas.InlineComment(...)` (Python binds
    #       `outrider`; the dotted-chain attribute access works because
    #       the import ensures parent attribute population)
    #   `import outrider.schemas as schemas` → call uses
    #       `schemas.InlineComment(...)` (local binding is `schemas`)
    #   `import outrider.schemas.publish`    → call uses
    #       `outrider.schemas.publish.InlineComment(...)`
    #   `import outrider.schemas.publish as p` → call uses
    #       `p.InlineComment(...)`
    schema_module_bases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # `from outrider.schemas import InlineComment[ as IC]` —
            # local name (or alias) resolves to the canonical class.
            if node.module in schema_modules:
                for alias in node.names:
                    if alias.name == "InlineComment":
                        inline_comment_local_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in schema_modules:
                    # If aliased (`as <local>`), the call-site base is
                    # the alias — a single-component name. Without an
                    # alias, the call uses the full dotted form.
                    if alias.asname is not None:
                        schema_module_bases.add(alias.asname)
                    else:
                        schema_module_bases.add(alias.name)

    def _dotted_name(expr: ast.AST) -> str | None:
        """Render `ast.Name` / `ast.Attribute` chains as a dotted string.

        Walks an attribute chain (right-to-left) until reaching the
        root `ast.Name`, then joins. Returns None for non-name/attr
        expressions (function-call returns, subscripts, etc.) so the
        caller can skip those rather than mis-match.
        """
        parts: list[str] = []
        cur: ast.AST = expr
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

    constructions: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Direct `<local-name>(...)` where local-name resolves to InlineComment.
        if isinstance(func, ast.Name) and func.id in inline_comment_local_names:
            constructions.append((node.lineno, ast.unparse(node)[:80]))
            continue
        # `<base>.InlineComment(...)` where <base> matches a known
        # schema-module call-site base exactly OR is a sub-attribute of
        # one. Exact-match catches `schemas.InlineComment(...)` after
        # `import outrider.schemas as schemas`; sub-attribute match
        # catches the nested case `outrider.schemas.publish.InlineComment(...)`
        # under `import outrider.schemas` (where the call site walks
        # through `schemas.publish` as an attribute chain).
        if isinstance(func, ast.Attribute) and func.attr == "InlineComment":
            base = _dotted_name(func.value)
            if base is not None and any(
                base == known or base.startswith(f"{known}.") for known in schema_module_bases
            ):
                constructions.append((node.lineno, ast.unparse(node)[:80]))
    return constructions


def _collect_src_py_files() -> list[Path]:
    """List every .py file under src/outrider/, excluding __pycache__."""
    root = _project_root() / "src" / "outrider"
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _collect_src_py_files(), ids=lambda p: str(p))
def test_no_direct_inline_comment_construction_in_src(path: Path) -> None:
    """Every `src/outrider/**/*.py` file calls only `InlineComment.from_finding(...)`,
    not `InlineComment(...)` directly.

    Per spec §V Input boundary sub-rule 5: the factory is the single
    production construction path so the sanitizer's guarantees are
    enforced at the call site.
    """
    project_root = _project_root()
    relpath = str(path.relative_to(project_root))
    if relpath in _ALLOWED_DIRECT_CONSTRUCTION_PATHS:
        pytest.skip(f"{relpath} is explicitly allowed direct construction")
    source = path.read_text(encoding="utf-8")
    constructions = _find_inline_comment_constructions(source)
    if constructions:
        formatted = "\n".join(
            f"  {relpath}:{lineno} — {snippet}" for lineno, snippet in constructions
        )
        msg = (
            f"Direct `InlineComment(...)` construction found in production code. "
            f"Per spec §V Input boundary sub-rule 5, use `InlineComment.from_finding"
            f"(finding=..., path=..., line=..., side=..., body=sanitized_body)` "
            f"so the sanitizer pipeline runs at the call site:\n{formatted}"
        )
        raise AssertionError(msg)
