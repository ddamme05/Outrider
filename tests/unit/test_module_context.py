"""Complete captured-module references: connections, ambiguity and rendered caps."""

from unittest.mock import MagicMock

import pytest

from outrider.agent.nodes.analyze_budget import (
    FileBudgetRequest,
    plan_file_budgets,
    reserve_context_headroom,
)
from outrider.agent.nodes.analyze_context import (
    MAX_MODULE_CONTEXT_BYTES,
    _rendered_context_bytes,
    prepare_review_context,
)
from outrider.ast_facts.review_context import (
    MAX_MODULE_SOURCE_BYTES,
    build_module_context_index,
    complete_module_context,
)


def test_optional_context_keeps_baseline_funding_and_respects_class_pools_and_file_caps() -> None:
    baseline = plan_file_budgets(
        (
            FileBudgetRequest("auth.py", 70, True),
            FileBudgetRequest("ordinary.py", 70, False),
            FileBudgetRequest("unfunded.py", 90, False),
        ),
        general_pool_tokens=140,
        reserved_pool_tokens=100,
        per_file_cap_tokens=100,
    )
    extended = reserve_context_headroom(
        baseline,
        {"auth.py": 100, "ordinary.py": 100, "unfunded.py": 95},
        per_file_cap_tokens=100,
    )
    assert [a.allocation_tokens for a in extended.allocations] == [100, 100, 0]
    assert [a.funded for a in extended.allocations] == [a.funded for a in baseline.allocations]
    assert extended.reserved_remainder_tokens == 0
    assert extended.general_remainder_tokens == 40
    assert (
        reserve_context_headroom(
            baseline, {"auth.py": 101, "ordinary.py": 101}, per_file_cap_tokens=100
        )
        == baseline
    )

    general_exhausted = plan_file_budgets(
        (FileBudgetRequest("auth.py", 20, True), FileBudgetRequest("ordinary.py", 80, False)),
        general_pool_tokens=80,
        reserved_pool_tokens=100,
        per_file_cap_tokens=150,
    )
    reserve_only = reserve_context_headroom(
        general_exhausted, {"auth.py": 50, "ordinary.py": 100}, per_file_cap_tokens=150
    )
    assert [a.allocation_tokens for a in reserve_only.allocations] == [50, 80]
    assert reserve_only.reserved_remainder_tokens == 50
    assert reserve_only.general_remainder_tokens == 0


@pytest.mark.parametrize("relationship", ["import", "from", "relative", "importer"])
def test_complete_modules_follow_explicit_captured_imports_without_filesystem_resolution(
    relationship: str,
) -> None:
    primary = "def changed():\n    return 2\n"
    related = "def check():\n    return True\n"
    imports = {
        "import": "import pkg.policy\n",
        "from": "from pkg.policy import check as allowed\n",
        "relative": "from .policy import check\n",
    }
    if relationship == "importer":
        related = "from pkg.handler import changed\n" + related
    else:
        primary = imports[relationship] + primary
    resolver = MagicMock()
    index = build_module_context_index(
        {"pkg/handler.py": primary, "pkg/policy.py": related},
        import_path_resolver=resolver,
    )
    selected = complete_module_context(index, file_path="pkg/handler.py")

    assert [r.file_path for r in selected.references] == ["pkg/handler.py", "pkg/policy.py"]
    assert selected.references[0].excerpt.text == primary
    assert selected.references[1].excerpt.text == related
    resolver.resolve_candidate_paths.assert_not_called()
    resolver.resolve_specifier_candidate_paths.assert_not_called()


def test_ambiguity_counts_large_captured_module_before_reference_eligibility() -> None:
    index = build_module_context_index(
        {
            "pkg/handler.py": "import pkg.policy\ndef changed():\n    return 2\n",
            "pkg/policy.py": "#" + "x" * MAX_MODULE_SOURCE_BYTES,
            "pkg/policy/__init__.py": "def check():\n    return True\n",
        },
        import_path_resolver=MagicMock(),
    )
    selected = complete_module_context(index, file_path="pkg/handler.py")

    assert [r.file_path for r in selected.references] == ["pkg/handler.py"]
    assert any("ambiguous captured module" in note for note in selected.omissions)


@pytest.mark.parametrize("module", ["..policy", ".", "pkg"])
def test_unknown_relative_roots_and_imported_symbols_do_not_invent_module_targets(
    module: str,
) -> None:
    index = build_module_context_index(
        {
            "pkg/handler.py": f"from {module} import policy as renamed\n"
            "def changed():\n    return 2\n",
            "pkg/policy.py": "def check():\n    return True\n",
        },
        import_path_resolver=MagicMock(),
    )
    selected = complete_module_context(index, file_path="pkg/handler.py")

    assert [r.file_path for r in selected.references] == ["pkg/handler.py"]
    assert selected.omissions


def test_module_index_and_complete_source_bytes_are_bounded() -> None:
    sources = {f"pkg/item{i:02}.py": "value = 1\n" for i in range(40)}
    sources["pkg/item00.py"] = "value = '" + "é" * 8192 + "'\n"
    index = build_module_context_index(sources, import_path_resolver=MagicMock())
    selected = complete_module_context(index, file_path="pkg/item00.py")

    assert len(index.source_index.sources) <= 32
    assert sum(len(s) for s in index.source_index.sources.values()) <= 400 * 1024
    assert not selected.references
    assert any("source exceeds 16-KiB" in note for note in selected.omissions)
    assert index.omitted_counts["not indexed (parser, exclusion or index bounds)"] > 0


@pytest.mark.parametrize("body", ["#\n" * 8000, "#" + "`" * 15000 + "\n"])
def test_number_labels_and_dynamic_outer_fences_count_toward_entire_supplement_cap(
    body: str,
) -> None:
    assert len(body.encode()) <= MAX_MODULE_SOURCE_BYTES
    index = build_module_context_index({"pkg/handler.py": body}, import_path_resolver=MagicMock())
    context = prepare_review_context(
        file_path="pkg/handler.py",
        content=body,
        patch="",
        changed_files=(),
        import_path_resolver=MagicMock(),
        module_context_index=index,
    )

    assert _rendered_context_bytes(context.text) <= MAX_MODULE_CONTEXT_BYTES
    assert "entire rendered supplement would exceed 32 KiB" in context.text
    assert "statement does not describe this supplement" in context.text
    assert "any complete numbered head admitted below" in context.text
    assert not context.manifest


def test_module_references_report_language_test_and_selection_limits() -> None:
    index = build_module_context_index(
        {
            "pkg/handler.py": "import pkg.a\nimport pkg.b\ndef changed():\n    return 2\n",
            "pkg/a.py": "value = 1\n",
            "pkg/b.py": "value = 2\n",
            "web/client.ts": "export const value = 1;\n",
        },
        import_path_resolver=MagicMock(),
    )
    context = prepare_review_context(
        file_path="pkg/handler.py",
        content=None,
        patch=None,
        changed_files=(),
        import_path_resolver=MagicMock(),
        module_context_index=index,
    )

    assert len(context.manifest) == 2
    assert all(entry.inclusion_reason == "captured_reference" for entry in context.manifest)
    assert "unsupported language (Python modules only)" in context.text
    assert "one-neighbour limit" in context.text
    assert "1 | import pkg.a" in context.text
