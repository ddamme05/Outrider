# Trust-boundary #8 floor test for the publish node per spec §V LLM provider boundary.
"""Pin the contract that the publish node makes ZERO LLM calls.

Per the publish-node spec §V LLM provider boundary:

    Publish makes ZERO LLM calls. Verified by absence of `LLMCallEvent`
    emission and import-graph unit test.

This is that import-graph unit test. Three layers of verification:

  1. The `agent.nodes.publish` module does NOT import any symbol from
     `outrider.llm.*`. (Direct-import check.)
  2. Importing `agent.nodes.publish` does NOT transitively cause
     `anthropic` (or any other LLM vendor SDK) to land in `sys.modules`.
     (Transitive-import check — defends against a future helper that
     re-exports an LLM symbol from a non-llm path.)
  3. The publish node's source contains no `LLMCallEvent` reference.
     (Emission check — defends against future code paths that mint
     an LLMCallEvent and pass it to a non-LLM sink.)

Trust boundary #8 (LLM provider boundary): vendor SDK imports are
confined to `outrider.llm.*`. The publish node is not an LLM consumer;
this test is the structural floor that pins the property even when a
contributor adds a "just one quick" import that crosses the boundary.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate project root")


_PUBLISH_NODE_PATH = "src/outrider/agent/nodes/publish.py"


def test_publish_node_source_imports_nothing_from_outrider_llm() -> None:
    """`agent.nodes.publish` MUST NOT import any symbol from `outrider.llm.*`.

    Catches direct imports — the most common future-regression shape.
    """
    source = (_project_root() / _PUBLISH_NODE_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offending: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("outrider.llm")
        ):
            offending.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("outrider.llm"):
                    offending.append(f"line {node.lineno}: import {alias.name}")
    if offending:
        msg = (
            "Publish node MUST NOT import from outrider.llm.* — trust "
            "boundary #8 + spec §V LLM provider boundary:\n  " + "\n  ".join(offending)
        )
        raise AssertionError(msg)


def test_publish_node_source_does_not_reference_llm_call_event() -> None:
    """`agent.nodes.publish` MUST NOT reference `LLMCallEvent` in code.

    The publish node makes no LLM calls; emitting an LLMCallEvent
    would either misuse the audit-event taxonomy OR signal a future
    code path that crossed the LLM-provider boundary.

    Walks the AST instead of substring-matching the source text so the
    module docstring (which describes the ABSENCE of LLM calls) doesn't
    false-positive the test.
    """
    source = (_project_root() / _PUBLISH_NODE_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offending: list[str] = []
    for node in ast.walk(tree):
        # ImportFrom: `from outrider.audit.events import LLMCallEvent`
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "LLMCallEvent":
                    offending.append(f"line {node.lineno}: import LLMCallEvent")
        # Name reference: `LLMCallEvent(...)` or `x: LLMCallEvent` or
        # any other code-level use of the bare name.
        elif isinstance(node, ast.Name) and node.id == "LLMCallEvent":
            offending.append(f"line {node.lineno}: code reference to LLMCallEvent")
        # Attribute reference: `events.LLMCallEvent(...)`.
        elif isinstance(node, ast.Attribute) and node.attr == "LLMCallEvent":
            offending.append(f"line {node.lineno}: attribute reference LLMCallEvent")
    if offending:
        msg = (
            "publish.py contains code-level `LLMCallEvent` reference(s). Per spec §V "
            "LLM provider boundary, the publish node makes ZERO LLM calls and emits "
            "no LLMCallEvent:\n  " + "\n  ".join(offending)
        )
        raise AssertionError(msg)


def test_publish_node_transitive_imports_do_not_load_anthropic() -> None:
    """Importing `agent.nodes.publish` MUST NOT cause `anthropic` to load.

    Defends against a future helper module that re-exports an LLM
    symbol from a non-llm path. If `anthropic` ends up in `sys.modules`
    after this import, something on the import chain crossed the
    trust boundary.
    """
    # Snapshot sys.modules BEFORE we touch anything LLM-adjacent.
    pre_modules = set(sys.modules.keys())
    if "anthropic" in pre_modules:
        pytest.skip(
            "anthropic already imported by a prior test in this process; "
            "cannot isolate the transitive-import check here. Run this test "
            "in a fresh process or before tests that touch outrider.llm."
        )
    # Force re-import to be sure the transitive chain runs FRESH for this
    # check. `importlib.reload` requires the module to already be loaded,
    # so we use `importlib.import_module` which handles both cases.
    module_name = "outrider.agent.nodes.publish"
    if module_name in sys.modules:
        # Already loaded — the transitive chain already ran in this
        # process. Check sys.modules state directly.
        pass
    else:
        importlib.import_module(module_name)
    if "anthropic" in sys.modules:
        raise AssertionError(
            "Importing outrider.agent.nodes.publish transitively loaded "
            "`anthropic`. Per trust boundary #8 + spec §V, the publish "
            "node MUST NOT depend on vendor LLM SDKs. Audit the import "
            "chain for the offending bridge."
        )
