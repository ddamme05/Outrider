"""Deterministic OBSERVED producer — `produce_observed_findings` (Cost Lever 3).

Pins that a tree-sitter security match becomes a policy-set OBSERVED
`ReviewFinding` with no model text: correct finding_type/severity/dimension,
the registry's static title/description, an `evidence` slice of the matched
source, a span mapped through coordinates, and a passing proof boundary. Also
pins the scope gate (no finding outside an included scope) and clean-parse use.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from outrider.agent.nodes.analyze_observed import produce_observed_findings, run_observed_matches
from outrider.ast_facts import parse_python
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import (
    ACTIVE_POLICY_VERSION,
    FindingSeverity,
    FindingType,
)
from outrider.queries.observed import QueryClass
from outrider.schemas import ReviewDimension


def _scopes(source: str):
    """Real ScopeUnits for `source` (the producer's scope-gate input)."""
    return parse_python(source.encode(), "src/x.py", MagicMock()).scope_units


def _produce(source: str, scopes=None, file_path: str = "src/x.py"):
    matches = run_observed_matches(
        file_path=file_path,
        head_content=source,
        included_scope_units=scopes if scopes is not None else _scopes(source),
    )
    return produce_observed_findings(
        matches,
        file_path=file_path,
        review_id=uuid4(),
        installation_id=12345,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )


def test_subprocess_shell_true_produces_observed_command_injection() -> None:
    source = "import subprocess\n\n\ndef run_it(cmd):\n    subprocess.run(cmd, shell=True)\n"
    findings = _produce(source)
    assert len(findings) == 1
    f = findings[0]
    assert f.evidence_tier == EvidenceTier.OBSERVED
    assert f.finding_type == FindingType.COMMAND_INJECTION
    assert f.severity == FindingSeverity.CRITICAL  # policy-set (DECISIONS.md#048)
    assert f.dimension == ReviewDimension.SECURITY
    assert f.query_match_id == "python.command_injection_subprocess_shell"
    assert "shell=True" in f.evidence
    assert f.title and f.description  # registry static text, no model output
    assert f.policy_version == ACTIVE_POLICY_VERSION
    # The span anchors on the call, not the enclosing function.
    assert f.line_start == f.line_end == 5


def test_finding_has_valid_proof_and_content_hash() -> None:
    """Construction through ReviewFinding means the proof boundary + content
    hash validators passed (OBSERVED ⇒ non-empty query_match_id; hash matches)."""
    source = "import pickle\n\n\ndef load(b):\n    return pickle.loads(b)\n"
    (f,) = _produce(source)
    assert f.finding_type == FindingType.UNSAFE_DESERIALIZATION
    assert f.severity == FindingSeverity.HIGH
    assert f.query_match_id == "python.unsafe_deserialization_pickle"
    assert len(f.content_hash) == 64
    assert len(f.proposal_hash) == 64
    # confidence is computed from the tier, never assigned.
    assert f.confidence is not None


def test_clean_file_with_no_security_pattern_yields_nothing() -> None:
    source = "def add(a, b):\n    return a + b\n"
    assert _produce(source) == ()


def test_match_outside_included_scopes_is_not_produced() -> None:
    """A match is gated to included scopes — passing no scopes yields nothing
    even though the query fires on the file."""
    source = "import os\n\n\ndef danger():\n    os.system(cmd)\n"
    assert _produce(source, scopes=()) == ()
    # ...but with the function's scope included, it is produced.
    assert len(_produce(source)) == 1


def test_multiple_patterns_produce_multiple_findings() -> None:
    source = (
        "import subprocess\nimport requests\n\n\n"
        "def f(cmd, url):\n"
        "    subprocess.run(cmd, shell=True)\n"
        "    requests.get(url, verify=False)\n"
    )
    findings = _produce(source)
    types = {f.finding_type for f in findings}
    assert FindingType.COMMAND_INJECTION in types
    assert FindingType.TLS_VERIFY_DISABLED in types
    assert len(findings) == 2


def test_security_pattern_in_test_file_is_suppressed() -> None:
    """Per spec §11.2: a security pattern in test code is not a production
    finding. The producer suppresses OBSERVED findings in test files
    structurally (the deterministic counterpart of the LLM's test-context
    judgment)."""
    source = "def test_thing():\n    eval(payload)\n"
    for test_path in (
        "tests/test_evaluator.py",
        "src/pkg/tests/helpers.py",
        "src/pkg/conftest.py",
        "src/pkg/widget_test.py",
    ):
        assert _produce(source, file_path=test_path) == (), f"{test_path} should be suppressed"
    # The SAME pattern in a production path IS flagged.
    assert len(_produce(source, file_path="src/pkg/widget.py")) == 1


def test_producer_is_deterministic() -> None:
    """Same input → identical findings (content/proposal hashes stable)."""
    source = "import os\n\n\ndef f():\n    os.system(c)\n"
    a = _produce(source)
    b = _produce(source)
    assert [(x.content_hash, x.proposal_hash) for x in a] == [
        (y.content_hash, y.proposal_hash) for y in b
    ]


def test_run_observed_matches_surfaces_query_class_and_record_fields() -> None:
    """The shared query pass returns `ObservedMatch` records carrying
    `query_class` (which the skip-routing increment filters on) alongside the
    finding-construction fields — one definition of the OBSERVED query facts for
    the file. Every V1 query is `signal_only` (default-deny)."""
    source = "import subprocess\n\n\ndef run_it(cmd):\n    subprocess.run(cmd, shell=True)\n"
    matches = run_observed_matches(
        file_path="src/x.py", head_content=source, included_scope_units=_scopes(source)
    )
    assert len(matches) == 1
    m = matches[0]
    assert m.query_match_id == "python.command_injection_subprocess_shell"
    assert m.query_class == QueryClass.SIGNAL_ONLY  # default-deny: all V1 queries signal_only
    assert m.finding_type == FindingType.COMMAND_INJECTION
    assert "shell=True" in m.evidence
    assert m.line_start == m.line_end == 5
    assert m.title and m.description


def test_long_match_evidence_truncated_not_crash() -> None:
    """A match envelope spans the whole call, so a long construct (e.g. a big SQL
    f-string) produces evidence exceeding ReviewFinding.evidence's 2000-char cap.
    The producer truncates so a long but legitimate match yields a finding rather
    than a ValidationError that would crash analyze for the file (code-review fold)."""
    long_tail = "A" * 2500
    source = f'def q(c, v):\n    c.execute(f"SELECT {{v}} {long_tail}")\n'
    findings = _produce(source)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.finding_type == FindingType.SQL_INJECTION
    assert len(finding.evidence) == 2000  # truncated to the field cap, not a crash


# ---------------------------------------------------------------------------
# JS/TS catalog (specs/2026-07-03-js-ts-observed-query-catalog.md): the
# producer selects the query set + grammar per file language.
# ---------------------------------------------------------------------------

_JS_WEAK_HASH_SOURCE = (
    "function getToken(secret) {\n"
    '  const h = crypto.createHash("md5");\n'
    '  return h.update(secret).digest("hex");\n'
    "}\n"
)


def _scopes_for(source: str, file_path: str):
    """Real ScopeUnits via the language-generic parse dispatch."""
    from outrider.ast_facts.registry import parse_source

    return parse_source(source.encode(), file_path, MagicMock()).scope_units


def _produce_at(source: str, file_path: str):
    matches = run_observed_matches(
        file_path=file_path,
        head_content=source,
        included_scope_units=_scopes_for(source, file_path),
    )
    return produce_observed_findings(
        matches,
        file_path=file_path,
        review_id=uuid4(),
        installation_id=12345,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )


def test_js_weak_hash_produces_observed_finding() -> None:
    """A .js file whose scope contains a catalog match produces a policy-set
    OBSERVED finding with a `javascript.*` query id — the language-selected
    counterpart of the Python producer tests above."""
    (f,) = _produce_at(_JS_WEAK_HASH_SOURCE, "src/token.js")
    assert f.evidence_tier == EvidenceTier.OBSERVED
    assert f.query_match_id == "javascript.weak_crypto_hash"
    assert f.finding_type == FindingType.WEAK_CRYPTO
    assert f.dimension == ReviewDimension.SECURITY
    assert 'crypto.createHash("md5")' in f.evidence
    assert f.title and f.description  # registry static text, no model output
    assert f.line_start == f.line_end == 2


def test_ts_file_selects_typescript_grammar_same_catalog() -> None:
    """The SAME source in a .ts file runs the same javascript catalog under
    the typescript grammar — extension picks the grammar, not the query set."""
    (f,) = _produce_at(_JS_WEAK_HASH_SOURCE, "src/token.ts")
    assert f.query_match_id == "javascript.weak_crypto_hash"


def test_js_file_never_runs_python_queries() -> None:
    """Language partition: a .js file's matches all carry javascript ids —
    python queries never execute over JS bytes (and vice versa)."""
    matches = run_observed_matches(
        file_path="src/token.js",
        head_content=_JS_WEAK_HASH_SOURCE,
        included_scope_units=_scopes_for(_JS_WEAK_HASH_SOURCE, "src/token.js"),
    )
    assert matches, "fixture must fire the catalog"
    assert all(m.query_match_id.startswith("javascript.") for m in matches)


def test_js_test_file_conventions_are_suppressed() -> None:
    """The JS/TS test conventions (`*.test.*`, `*.spec.*`, `__tests__/`)
    suppress the producer exactly like the Python conventions (spec §11.2);
    the same content in a production path IS flagged."""
    for test_path in (
        "src/auth.test.js",
        "src/login.spec.ts",
        "src/__tests__/token.js",
        "src/components/Widget.test.tsx",
    ):
        assert _produce_at(_JS_WEAK_HASH_SOURCE, test_path) == (), (
            f"{test_path} should be suppressed"
        )
    assert len(_produce_at(_JS_WEAK_HASH_SOURCE, "src/token.js")) == 1


def test_unregistered_extension_is_inert() -> None:
    """A language with no catalog selects zero queries — the producer
    returns empty rather than raising or running another language's set."""
    matches = run_observed_matches(
        file_path="src/main.go",
        head_content="eval(payload)\n",
        included_scope_units=(),
    )
    assert matches == ()
