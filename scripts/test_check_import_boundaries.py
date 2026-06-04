#!/usr/bin/env python3
"""Smoke-test check_import_boundaries.py — the trust-boundary import lint (FUP-005).

Not a pytest suite (matches `scripts/test_extract_invariants.py`'s standalone shape):
constructs temp trees with known violations + allowed cases and asserts the checker
flags exactly the hard-stops, AND confirms the REAL repo tree is clean. Run via the
`boundary-lint-tests-pass` pre-commit hook + the CI `pre-commit-tracked-only` job.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Same-dir import: scripts/ is sys.path[0] when run as `python scripts/test_...py`.
from check_import_boundaries import REPO_ROOT, find_violations

_failures = 0


def _fail(msg: str) -> None:
    global _failures
    _failures += 1
    print(f"  FAIL: {msg}")


def _check(name: str, files: dict[str, str], expected: int, rule_substr: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, src in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(src, encoding="utf-8")
        violations = find_violations(root)
        rendered = [v.render() for v in violations]
        if len(violations) != expected:
            _fail(f"{name}: expected {expected} violation(s), got {len(violations)}: {rendered}")
            return
        if rule_substr and expected and not all(rule_substr in v.rule for v in violations):
            _fail(f"{name}: expected rule ~{rule_substr!r}, got {[v.rule for v in violations]}")
            return
        print(f"  ok: {name}")


# --- the real tree must be clean (load-bearing: a production violation fails CI) ---
_real = find_violations(REPO_ROOT)
if _real:
    _fail(f"REAL TREE has {len(_real)} boundary violation(s): {[v.render() for v in _real]}")
else:
    print("  ok: real repo tree is clean")

# --- AST firewall ---
_check(
    "tree_sitter outside ast_facts/queries -> flagged",
    {"src/outrider/agent/nodes/x.py": "import tree_sitter\n"},
    1,
    "AST firewall",
)
_check(
    "tree_sitter in ast_facts/ -> allowed",
    {"src/outrider/ast_facts/x.py": "import tree_sitter\n"},
    0,
)
_check(
    "tree_sitter in queries/ -> allowed",
    {"src/outrider/queries/x.py": "from tree_sitter import Node\n"},
    0,
)
_check(
    "tree_sitter in the allowed ast_facts test file -> allowed",
    {"tests/unit/test_ast_facts_python.py": "import tree_sitter\n"},
    0,
)
_check(
    "tree_sitter in a non-allowed test -> flagged",
    {"tests/unit/test_other.py": "import tree_sitter\n"},
    1,
    "AST firewall",
)
_check(
    "tree_sitter under TYPE_CHECKING outside ast_facts -> flagged (type-surface)",
    {
        "src/outrider/agent/x.py": (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import tree_sitter\n"
        )
    },
    1,
    "AST firewall",
)
_check(
    "exact-file allowlist sibling (path-prefixed .py) -> flagged (no startswith over-match)",
    {"tests/unit/test_ast_facts_python.py_helper.py": "import tree_sitter\n"},
    1,
    "AST firewall",
)
_check(
    "relative `from . import x` -> NOT flagged (module is None, no crash)",
    {"src/outrider/agent/x.py": "from . import helpers\n"},
    0,
)

# --- LLM provider boundary ---
_check(
    "anthropic outside llm/ -> flagged",
    {"src/outrider/agent/x.py": "import anthropic\n"},
    1,
    "provider",
)
_check("anthropic in llm/ -> allowed", {"src/outrider/llm/x.py": "import anthropic\n"}, 0)
_check(
    "anthropic submodule outside llm/ -> flagged",
    {"src/outrider/agent/x.py": "from anthropic.types import Message\n"},
    1,
    "provider",
)
_check(
    "openai outside llm/ -> flagged",
    {"src/outrider/agent/x.py": "import openai\n"},
    1,
    "provider",
)
_check(
    "langsmith from-import outside llm/ -> flagged (DECISIONS#035)",
    {"src/outrider/audit/x.py": "from langsmith import traceable\n"},
    1,
    "tracing",
)
_check(
    "langsmith in llm/tracing.py -> allowed (exact file, production from-import form, #035)",
    {"src/outrider/llm/tracing.py": "from langsmith import traceable\n"},
    0,
)
_check(
    "langsmith elsewhere in llm/ -> flagged (#035 pins it to tracing.py, not the folder)",
    {"src/outrider/llm/provider.py": "import langsmith\n"},
    1,
    "tracing",
)
_check(
    "anthropic in a TEST -> NOT flagged (surface, tests not scanned)",
    {"tests/unit/test_x.py": "import anthropic\n"},
    0,
)
_check(
    "aliased `import anthropic as a` outside llm/ -> flagged",
    {"src/outrider/agent/x.py": "import anthropic as a\n"},
    1,
    "LLM",
)
_check(
    "sibling folder llm_x/ -> flagged (trailing-slash prefix guard, not un-banned by llm/)",
    {"src/outrider/llm_x/x.py": "import anthropic\n"},
    1,
    "LLM",
)

# --- GitHub SDK boundary ---
_check(
    "githubkit outside github//webhooks -> flagged",
    {"src/outrider/agent/x.py": "import githubkit\n"},
    1,
    "GitHub",
)
_check(
    "githubkit in github/ -> allowed",
    {"src/outrider/github/x.py": "from githubkit import GitHub\n"},
    0,
)
_check(
    "githubkit in api/webhooks/ -> allowed",
    {"src/outrider/api/webhooks/x.py": "import githubkit\n"},
    0,
)

# --- Input boundary (shell) ---
# NOTE: the `os.system(...)` / `subprocess` strings below are NON-EXECUTED source
# fixtures — written to temp files purely so the AST checker can flag them. They are
# never run; the test only asserts the checker detects the shell-exec pattern.
_check(
    "import subprocess in src/outrider/ -> flagged",
    {"src/outrider/agent/x.py": "import subprocess\n"},
    1,
    "shell",
)
_check(
    "from subprocess import run -> flagged",
    {"src/outrider/agent/x.py": "from subprocess import run\n"},
    1,
    "shell",
)
_check(
    "os.system(...) call -> flagged",
    {"src/outrider/agent/x.py": "import os\nos.system('ls')\n"},
    1,
    "shell",
)
_check(
    "os.popen(...) call -> flagged",
    {"src/outrider/agent/x.py": "import os\nos.popen('ls')\n"},
    1,
    "shell",
)
_check(
    "plain import os (no dangerous call) -> NOT flagged",
    {"src/outrider/agent/x.py": "import os\nx = os.environ\n"},
    0,
)
_check(
    "subprocess in a TEST -> NOT flagged (surface, tests not scanned for shell)",
    {"tests/unit/test_x.py": "import subprocess\n"},
    0,
)

print("\ndone.")
sys.exit(1 if _failures else 0)
