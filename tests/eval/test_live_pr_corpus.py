"""Check the live-demo corpus's documented structural matches without executing it.

Use real parsers, queries, changed-scope selection and OBSERVED admission.
This does not run triage, budgets, models, the review graph or publication.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import UUID

import pytest
from unidiff import PatchSet

from outrider.agent.eval_adapters import _NoOpImportPathResolver
from outrider.agent.nodes.analyze_observed import (
    module_admission_inputs,
    produce_observed_findings,
    run_observed_matches,
)
from outrider.agent.nodes.degradation import decide_degradation
from outrider.ast_facts.registry import get_adapter_factory, parse_source
from outrider.coordinates import query_span_to_source_lines
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts.analyze import VERSION
from outrider.queries import registry

_ROOT = Path(__file__).resolve().parents[2]
_CORPUS = _ROOT / "scripts/demo_fixtures/live_pr_corpus"
_SHEET = _ROOT / "LIVE_RUN_EXPECTED_FINDINGS.md"
_BASE_ONLY = "app/db/queries.py"

# Query ID, finding type, and inclusive head-line range, read from the planted
# call sites. Empty lists explicitly protect the corpus's structural negatives.
_EXPECTED: dict[str, list[tuple[str, str, int, int]]] = {
    "app/api/reports.py": [],
    "app/db/queries.py": [],
    "app/db/user_repository.py": [
        ("python.sql_injection_string_concat", "sql_injection", 40, 43),
        ("python.sql_injection_string_concat", "sql_injection", 58, 61),
    ],
    "app/models/invoice.py": [],
    "app/tasks/report_runner.py": [
        ("python.command_injection_subprocess_shell", "command_injection", 47, 47),
        ("python.unsafe_deserialization_yaml", "unsafe_deserialization", 59, 59),
        ("python.tls_verify_disabled", "tls_verify_disabled", 68, 73),
        ("python.weak_crypto_broken_cipher", "weak_crypto", 84, 84),
        ("python.weak_crypto_ecb_mode", "weak_crypto", 84, 84),
    ],
    "app/utils/pagination.py": [],
    "components/ExpressionPreview.tsx": [
        ("javascript.command_injection_eval", "command_injection", 36, 36),
        ("javascript.weak_crypto_hash", "weak_crypto", 44, 44),
    ],
    "data/user_repository.ts": [
        ("javascript.sql_injection_string_concat", "sql_injection", 55, 57),
        ("javascript.sql_injection_string_concat", "sql_injection", 70, 74),
        ("javascript.tls_verify_disabled", "tls_verify_disabled", 90, 90),
        ("javascript.tls_env_verify_disabled", "tls_verify_disabled", 134, 134),
    ],
    "ops/deploy_helpers.js": [
        ("javascript.command_injection_child_process", "command_injection", 24, 28),
        ("javascript.weak_crypto_hash", "weak_crypto", 37, 37),
        ("javascript.weak_crypto_broken_cipher", "weak_crypto", 47, 47),
        ("javascript.weak_crypto_ecb_mode", "weak_crypto", 60, 60),
    ],
}
_SEVERITY = {
    "sql_injection": "critical",
    "command_injection": "critical",
    "unsafe_deserialization": "high",
    "tls_verify_disabled": "high",
    "weak_crypto": "high",
}


def test_every_corpus_file_has_an_explicit_role() -> None:
    files = {path.relative_to(_CORPUS).as_posix() for path in _CORPUS.rglob("*") if path.is_file()}
    assert files == set(_EXPECTED) | {"README.md"}
    supported = {path for path in files if get_adapter_factory(Path(path).suffix) is not None}
    assert supported == set(_EXPECTED)
    assert sum(map(len, _EXPECTED.values())) == 17


@pytest.mark.parametrize(("file_path", "expected"), _EXPECTED.items(), ids=_EXPECTED)
def test_documented_matches_preserve_signal_and_finding_roles(
    file_path: str, expected: list[tuple[str, str, int, int]]
) -> None:
    head = (_CORPUS / file_path).read_text(encoding="utf-8")
    source = head.encode("utf-8")
    parsed = parse_source(source, file_path, _NoOpImportPathResolver())
    assert parsed.parser_outcome == "clean"
    assert not parsed.error_lines and not any(parsed.has_error.values())

    # The live PR adds eight code files; the trace target already exists on base.
    patch = None
    if file_path != _BASE_ONLY:
        patch = PatchSet(
            f"--- /dev/null\n+++ b/{file_path}\n@@ -0,0 +1,{len(head.splitlines())} @@\n"
            + "".join("+" + line for line in head.splitlines(keepends=True))
        )[0]
    decision = decide_degradation(parsed, patch)
    assert decision.mode == ("skip" if file_path == _BASE_ONLY else "clean")
    scopes, added_ranges = module_admission_inputs(parsed, patch, head)

    language = registry.query_language_for_path(file_path)
    grammar = registry.grammar_for_path(file_path)
    assert language is not None and grammar is not None
    raw = Counter(
        (
            query_id,
            query.finding_type.value,
            *query_span_to_source_lines(
                byte_start=span.byte_start, byte_end=span.byte_end, head_content=head
            ),
        )
        for query_id, query in registry.observed_queries_for(language).items()
        for span in registry.match(query_id, source, grammar=grammar)
    )
    assert raw == Counter(expected), f"Catalog matches changed for {file_path}"
    if file_path == _BASE_ONLY:
        # Trace exposes query evidence to the model but does not run this
        # producer. Scan its structural potential without treating it as added.
        assert not decision.included_scope_units and not added_ranges
        return

    matches = run_observed_matches(
        file_path=file_path,
        head_content=head,
        included_scope_units=decision.included_scope_units,
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
        all_scope_units=scopes,
        added_line_ranges=added_ranges,
    )
    admitted = Counter(
        (match.query_match_id, match.finding_type.value, match.line_start, match.line_end)
        for match in matches
    )
    assert admitted == Counter(expected), f"OBSERVED admission changed for {file_path}"

    findings = produce_observed_findings(
        matches,
        file_path=file_path,
        review_id=UUID(int=244),
        installation_id=1,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )
    produced = Counter(
        (finding.query_match_id, finding.finding_type.value, finding.line_start, finding.line_end)
        for finding in findings
    )
    expected_findings = Counter(
        entry for entry in expected if not registry.OBSERVED_QUERIES[entry[0]].requires_context
    )
    assert produced == expected_findings, f"Produced findings changed for {file_path}"
    assert {m.query_match_id for m in matches if m.requires_context} == {
        entry[0] for entry in expected if registry.OBSERVED_QUERIES[entry[0]].requires_context
    }
    for finding in findings:
        assert finding.file_path == file_path
        assert finding.evidence_tier is EvidenceTier.OBSERVED
        assert finding.severity.value == _SEVERITY[finding.finding_type.value]


def test_live_checklist_records_current_prompt_and_catalog() -> None:
    sheet = _SHEET.read_text(encoding="utf-8")
    for label, identity in (("prompt", VERSION), ("catalog", registry.QUERY_REGISTRY_DIGEST)):
        assert f"**Validated {label}:** `{identity}`" in sheet, (
            f"The live checklist's {label} identity is stale. Run this free corpus check, "
            "review any match changes, then update the validation stamp. Do not change "
            "expected matches solely to make the check pass."
        )
