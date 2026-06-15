#!/usr/bin/env python
"""Trust-boundary import lint — the deterministic CI/pre-commit floor (FUP-005).

Enforces the HARD-STOP vendor-SDK + shell-exec boundaries from
`docs/trust-boundaries.md` (allowlists mirror the `check-trust-boundaries` skill):

  - `tree_sitter` / `tree_sitter_python` — only in `ast_facts/` + `queries/`
    (+ the two ast_facts test files). AST firewall (§4).
  - `anthropic` / `openai` — only in `llm/`; `langsmith` — only in `llm/tracing.py`
    (the tracing decorator's single home, DECISIONS.md#035). LLM provider boundary (§8).
  - `githubkit` — only in `github/` + `api/webhooks/`. GitHub SDK boundary (§5 + §8).
  - `slack_sdk` — only in `notify/`. Slack notification boundary (vendor-sdks-only-in-wrappers).
  - no `subprocess` import / `os.system` / `os.popen` anywhere in `src/outrider/`.
    Input boundary, shell (§5 sub-rule 1).

This is the deterministic floor (DECISIONS.md#038): the in-session skill catches
Claude-authored edits; this catches commits made OUTSIDE a Claude session (manual
commits, force-pushes). Imports inside `if TYPE_CHECKING:`
count the same as runtime imports (the AST firewall is a type-surface boundary).

OUT OF SCOPE (stay with the skill + reviewer — soft, with legitimate exceptions,
not reliably lint-detectable): coordinate math outside `coordinates/`, path ops,
vendor wire-format normalization, and vendor-SDK / `subprocess` use in tests.

Test-scan asymmetry is by design: the AST firewall scans `tests/` (tree_sitter has no
legit test use — fixtures use ast_facts models; the 2 allowlisted test files are the
only exceptions), but the vendor-SDK rules do NOT. Per the check-trust-boundaries skill,
direct anthropic/openai/githubkit imports in tests are SURFACE-tier, not hard-stop:
`tests/unit/test_llm_*.py` legitimately import SDK symbols (wrapper tests + wire-shape
fixtures), which an import-lint can't distinguish from illicit client construction — the
reviewer adjudicates. This mirrors the skill's hard-stop/surface split exactly.

Exit 0 if clean; exit 1 + a per-violation report (`file:line  rule — detail`).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class _VendorRule:
    name: str
    doc_ref: str
    modules: tuple[str, ...]  # restricted top-level module names
    scan_globs: tuple[str, ...]  # repo-relative globs to scan
    allowed_prefixes: tuple[str, ...]  # repo-relative dir-prefixes / exact files where allowed


# See DECISIONS.md#038 — this lint is the deterministic trust-boundary floor (FUP-005).
VENDOR_RULES: tuple[_VendorRule, ...] = (
    _VendorRule(
        name="AST firewall",
        doc_ref="docs/trust-boundaries.md §4",
        modules=("tree_sitter", "tree_sitter_python"),
        scan_globs=("src/outrider/**/*.py", "tests/**/*.py"),
        allowed_prefixes=(
            "src/outrider/ast_facts/",
            "src/outrider/queries/",
            "tests/unit/test_ast_facts_python.py",
            "tests/integration/test_ast_facts_query_registry.py",
        ),
    ),
    _VendorRule(
        name="LLM provider boundary",
        doc_ref="docs/trust-boundaries.md §8",
        modules=("anthropic", "openai"),
        scan_globs=("src/outrider/**/*.py",),
        allowed_prefixes=("src/outrider/llm/",),
    ),
    _VendorRule(
        # langsmith is pinned tighter than the folder: DECISIONS.md#035 places the
        # tracing decorator (its only legit importer, lazily) in llm/tracing.py.
        name="LLM tracing boundary",
        doc_ref="docs/trust-boundaries.md §8 + DECISIONS.md#035",
        modules=("langsmith",),
        scan_globs=("src/outrider/**/*.py",),
        allowed_prefixes=("src/outrider/llm/tracing.py",),
    ),
    _VendorRule(
        name="GitHub SDK boundary",
        doc_ref="docs/trust-boundaries.md §5 + §8",
        modules=("githubkit",),
        scan_globs=("src/outrider/**/*.py",),
        allowed_prefixes=("src/outrider/github/", "src/outrider/api/webhooks/"),
    ),
    _VendorRule(
        # Slack notifications: the slack_sdk AsyncWebClient is confined to the
        # notify/ wrapper (notify/slack.py) per the dashboard-in-Slack spec; the
        # general vendor-sdks-only-in-wrappers boundary, same shape as the others.
        name="Slack SDK boundary",
        doc_ref="docs/trust-boundaries.md §8 (vendor-sdks-only-in-wrappers)",
        modules=("slack_sdk",),
        scan_globs=("src/outrider/**/*.py",),
        allowed_prefixes=("src/outrider/notify/",),
    ),
)

# Shell-exec hard-stop: never in src/outrider/. (tests get the sys.executable
# isolation exception — surface-tier, handled by the skill, not here.)
# Defense-in-depth over ruff's `S` family, which already gates these on the same
# files (S404 subprocess import, S605 os.system/os.popen, S602/S603 subprocess
# calls — trust-boundaries.md §5 names ruff S as the shell net). Kept so the floor
# holds if the ruff config drifts; the vendor rules above are the part ruff can't
# express (per-module, per-folder allowlists), which is FUP-005's irreducible core.
_SHELL_SCAN_GLOBS: tuple[str, ...] = ("src/outrider/**/*.py",)
_SHELL_DOC_REF = "docs/trust-boundaries.md §5 (Input boundary, sub-rule 1)"
_SUBPROCESS_DANGEROUS = frozenset({"run", "Popen", "call", "check_output"})
_OS_DANGEROUS = frozenset({"system", "popen"})


@dataclass(frozen=True)
class Violation:
    rel_path: str
    line: int
    rule: str
    detail: str
    doc_ref: str

    def render(self) -> str:
        return f"{self.rel_path}:{self.line}  [{self.rule}] {self.detail}  — see {self.doc_ref}"


def _module_restricted(mod: str | None, restricted: Iterable[str]) -> bool:
    """True if `mod` is one of `restricted` or a submodule of one (e.g. `anthropic.types`)."""
    if mod is None:  # bare relative import (`from . import x`)
        return False
    return any(mod == r or mod.startswith(f"{r}.") for r in restricted)


def _is_allowed(rel_path: str, allowed_prefixes: tuple[str, ...]) -> bool:
    # Dir entries end in `/` and match by trailing-slash prefix (so `llm/` doesn't
    # match a sibling `llm_x/`); exact-file entries match by equality — a bare
    # `startswith` would admit a path-prefixed sibling like `<file>.py_helper.py`.
    return any(
        rel_path.startswith(p) if p.endswith("/") else rel_path == p for p in allowed_prefixes
    )


def _iter_files(root: Path, globs: tuple[str, ...]) -> Iterable[tuple[Path, str]]:
    seen: set[str] = set()
    for glob in globs:
        for path in sorted(root.glob(glob)):
            rel = path.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            yield path, rel


def _scan_vendor(root: Path, rule: _VendorRule) -> list[Violation]:
    out: list[Violation] = []
    for path, rel in _iter_files(root, rule.scan_globs):
        if _is_allowed(rel, rule.allowed_prefixes):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _module_restricted(alias.name, rule.modules):
                        out.append(
                            Violation(
                                rel, node.lineno, rule.name, f"import {alias.name}", rule.doc_ref
                            )
                        )
            elif isinstance(node, ast.ImportFrom) and _module_restricted(node.module, rule.modules):
                out.append(
                    Violation(
                        rel, node.lineno, rule.name, f"from {node.module} import ...", rule.doc_ref
                    )
                )
    return out


def _scan_shell(root: Path) -> list[Violation]:
    out: list[Violation] = []
    rule = "Input boundary (shell)"
    for path, rel in _iter_files(root, _SHELL_SCAN_GLOBS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "subprocess" or alias.name.startswith("subprocess."):
                        out.append(
                            Violation(rel, node.lineno, rule, "import subprocess", _SHELL_DOC_REF)
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "subprocess":
                    bad = sorted(a.name for a in node.names if a.name in _SUBPROCESS_DANGEROUS)
                    if bad:
                        out.append(
                            Violation(
                                rel,
                                node.lineno,
                                rule,
                                f"from subprocess import {', '.join(bad)}",
                                _SHELL_DOC_REF,
                            )
                        )
                elif node.module == "os":
                    bad = sorted(a.name for a in node.names if a.name in _OS_DANGEROUS)
                    if bad:
                        out.append(
                            Violation(
                                rel,
                                node.lineno,
                                rule,
                                f"from os import {', '.join(bad)}",
                                _SHELL_DOC_REF,
                            )
                        )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _OS_DANGEROUS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                out.append(
                    Violation(rel, node.lineno, rule, f"os.{node.func.attr}(...)", _SHELL_DOC_REF)
                )
    return out


def find_violations(root: Path) -> list[Violation]:
    """All trust-boundary violations under `root` (a repo root), sorted for stable output."""
    found: list[Violation] = []
    for rule in VENDOR_RULES:
        found.extend(_scan_vendor(root, rule))
    found.extend(_scan_shell(root))
    return sorted(found, key=lambda v: (v.rel_path, v.line, v.rule))


def main() -> int:
    violations = find_violations(REPO_ROOT)
    if not violations:
        return 0
    sys.stderr.write(
        f"Trust-boundary import lint: {len(violations)} violation(s) "
        f"(vendor SDK in the wrong wrapper folder, or shell-exec in src/outrider/):\n"
    )
    for v in violations:
        sys.stderr.write(f"  {v.render()}\n")
    sys.stderr.write(
        "\nMove the import behind its wrapper boundary, or — if the boundary itself "
        "should change — amend docs/trust-boundaries.md + the check-trust-boundaries "
        "skill allowlists + this script's rules together.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
