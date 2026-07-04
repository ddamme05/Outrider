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


def _parsed(source: str):
    """Real ParseResult for a python fixture (scopes + imports from ONE parse
    — the same pairing the analyze node threads into the producer)."""
    return parse_python(source.encode(), "src/x.py", MagicMock())


def _scopes(source: str):
    """Real ScopeUnits for `source` (the producer's scope-gate input)."""
    return _parsed(source).scope_units


def _produce(source: str, scopes=None, file_path: str = "src/x.py"):
    parsed = _parsed(source)
    matches = run_observed_matches(
        file_path=file_path,
        head_content=source,
        included_scope_units=scopes if scopes is not None else parsed.scope_units,
        import_refs=parsed.imports,
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
    parsed = _parsed(source)
    matches = run_observed_matches(
        file_path="src/x.py",
        head_content=source,
        included_scope_units=parsed.scope_units,
        import_refs=parsed.imports,
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
    'const crypto = require("node:crypto");\n'
    "function getToken(secret) {\n"
    '  const h = crypto.createHash("md5");\n'
    '  return h.update(secret).digest("hex");\n'
    "}\n"
)


def _parsed_at(source: str, file_path: str):
    """Real ParseResult via the language-generic parse dispatch (scopes +
    imports from one parse — what the analyze node threads in)."""
    from outrider.ast_facts.registry import parse_source

    return parse_source(source.encode(), file_path, MagicMock())


def _scopes_for(source: str, file_path: str):
    """Real ScopeUnits via the language-generic parse dispatch."""
    return _parsed_at(source, file_path).scope_units


def _produce_at(source: str, file_path: str):
    parsed = _parsed_at(source, file_path)
    matches = run_observed_matches(
        file_path=file_path,
        head_content=source,
        included_scope_units=parsed.scope_units,
        import_refs=parsed.imports,
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
    assert f.line_start == f.line_end == 3


def test_ts_file_selects_typescript_grammar_same_catalog() -> None:
    """The SAME source in a .ts file runs the same javascript catalog under
    the typescript grammar — extension picks the grammar, not the query set."""
    (f,) = _produce_at(_JS_WEAK_HASH_SOURCE, "src/token.ts")
    assert f.query_match_id == "javascript.weak_crypto_hash"


def test_js_file_never_runs_python_queries() -> None:
    """Language partition: a .js file's matches all carry javascript ids —
    python queries never execute over JS bytes (and vice versa)."""
    parsed = _parsed_at(_JS_WEAK_HASH_SOURCE, "src/token.js")
    matches = run_observed_matches(
        file_path="src/token.js",
        head_content=_JS_WEAK_HASH_SOURCE,
        included_scope_units=parsed.scope_units,
        import_refs=parsed.imports,
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


def test_js_test_conventions_do_not_suppress_other_languages() -> None:
    """`__tests__/` and the inner `.test.`/`.spec.` markers are JS/TS
    ecosystem conventions, language-scoped in `_is_test_file` — a dotted-name
    Python PRODUCTION file keeps producing OBSERVED findings. Under
    observed-producer-v2 these paths were silently suppressed for every
    language (no finding, no skip event, no audit trace)."""
    source = "import pickle\n\n\ndef load(b):\n    return pickle.loads(b)\n"
    for prod_path in (
        "src/report.spec.py",
        "src/settings.test.py",
        "pkg/__tests__/util.py",
    ):
        findings = _produce(source, file_path=prod_path)
        assert len(findings) == 1, f"{prod_path} is production Python, not a test file"
        assert findings[0].query_match_id == "python.unsafe_deserialization_pickle"


def test_unregistered_extension_is_inert() -> None:
    """A language with no catalog selects zero queries — the producer
    returns empty rather than raising or running another language's set."""
    matches = run_observed_matches(
        file_path="src/main.go",
        head_content="eval(payload)\n",
        included_scope_units=(),
        # No registered adapter for .go — no imports are extractable either.
        import_refs=(),
    )
    assert matches == ()


# ---------------------------------------------------------------------------
# Import-binding admission (observed-producer-v4): a name-anchored match is
# admitted only when its anchor identifier provably binds to the dangerous
# API via the file's imports. Negative fixtures are the Greptile PR-round
# repros (2026-07-03), pinned per-variant.
# ---------------------------------------------------------------------------


def test_unbound_createhash_is_not_admitted() -> None:
    """A `createHash` imported from a non-crypto module, defined locally, or
    called on an arbitrary object produces NO OBSERVED finding — the name
    alone is not proof of a crypto construction (Greptile P1 repro)."""
    source = (
        'import { createHash } from "./cache";\n'
        "const cacheApi = { createHash(name) { return name; } };\n"
        "export function demo() {\n"
        '  const a = createHash("md5");\n'
        '  const b = cacheApi.createHash("sha1");\n'
        "  return [a, b];\n"
        "}\n"
    )
    assert _produce_at(source, "src/example.js") == ()


def test_esm_destructured_createhash_is_admitted() -> None:
    """The bare form bound by a destructured ESM import from node:crypto —
    the dominant modern idiom the binding step must NOT lose."""
    source = (
        'import { createHash } from "node:crypto";\n'
        "export function token(secret) {\n"
        '  return createHash("md5").update(secret).digest("hex");\n'
        "}\n"
    )
    (f,) = _produce_at(source, "src/token.mjs")
    assert f.query_match_id == "javascript.weak_crypto_hash"
    assert f.line_start == 3


def test_cjs_whole_module_receiver_is_admitted() -> None:
    """The member form bound by a CJS whole-module require (the
    _JS_WEAK_HASH_SOURCE fixture) — covered by the language tests above;
    here the bare-require ALIAS receiver proves too."""
    source = (
        'const c = require("crypto");\n'
        "function digest(s) {\n"
        '  return c.createHash("sha1").update(s).digest("hex");\n'
        "}\n"
    )
    (f,) = _produce_at(source, "src/digest.js")
    assert f.query_match_id == "javascript.weak_crypto_hash"


def test_unbound_exec_helper_is_not_admitted() -> None:
    """`import { exec } from './jobs'` — an unrelated exec helper must not
    produce a CRITICAL OBSERVED command-injection finding (the sharpest
    Greptile repro)."""
    source = (
        'import { exec } from "./jobs";\n'
        "export function runJob(id) {\n"
        "  return exec('job-' + id);\n"
        "}\n"
    )
    assert _produce_at(source, "src/jobs_runner.js") == ()


def test_child_process_destructured_exec_is_admitted() -> None:
    source = (
        'const { exec } = require("child_process");\n'
        "function run(cmd) {\n"
        "  exec('ls ' + cmd);\n"
        "}\n"
    )
    (f,) = _produce_at(source, "src/run.js")
    assert f.query_match_id == "javascript.command_injection_child_process"


def test_aliased_namespace_exec_is_now_admitted() -> None:
    """`const cp = require('child_process'); cp.exec(...)` — previously a
    documented recall gap (the member arm demanded the literal
    `child_process` receiver name); the import join proves the alias."""
    source = (
        'const cp = require("node:child_process");\n'
        "function run(cmd) {\n"
        "  cp.execSync(`run ${cmd}`);\n"
        "}\n"
    )
    (f,) = _produce_at(source, "src/run.js")
    assert f.query_match_id == "javascript.command_injection_child_process"


def test_query_concat_without_db_driver_is_not_admitted() -> None:
    """`.query(concat)` on a non-database API — no DB driver imported in the
    file — is not SQL injection (Greptile P1 repro)."""
    source = (
        'import { search } from "./search";\n'
        "export function lookup(searchClient, tag) {\n"
        "  return searchClient.query('tag:' + tag);\n"
        "}\n"
    )
    assert _produce_at(source, "src/lookup.js") == ()


def test_query_concat_with_db_driver_present_is_admitted() -> None:
    source = (
        'const { Pool } = require("pg");\n'
        "function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n"
    )
    (f,) = _produce_at(source, "src/find.js")
    assert f.query_match_id == "javascript.sql_injection_string_concat"


def test_reject_unauthorized_without_tls_consumer_is_not_admitted() -> None:
    """A standalone options literal with no TLS-capable module in the file is
    not a TLS-disabled finding (Greptile P1 repro)."""
    source = "function policy() {\n  return { rejectUnauthorized: false };\n}\n"
    assert _produce_at(source, "src/benign_config.js") == ()


def test_reject_unauthorized_with_https_import_is_admitted() -> None:
    source = (
        'const https = require("https");\n'
        "function agent() {\n"
        "  return new https.Agent({ rejectUnauthorized: false });\n"
        "}\n"
    )
    (f,) = _produce_at(source, "src/agent.js")
    assert f.query_match_id == "javascript.tls_verify_disabled"


def test_process_env_kill_switch_needs_no_import() -> None:
    """The process-wide kill switch is self-proving (`process.env` receiver
    constrained in the query) — it fires with ZERO imports, in both the dot
    and bracket forms, under its own query id."""
    for line in (
        'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";',
        'process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0";',
    ):
        source = f"function disable() {{\n  {line}\n}}\n"
        (f,) = _produce_at(source, "src/danger.js")
        assert f.query_match_id == "javascript.tls_env_verify_disabled"


def test_non_process_env_receiver_is_not_matched() -> None:
    """`mockEnv[...] = "0"` / `settings.NODE_TLS_... = "0"` mutate a local
    object, not the process TLS switch (Greptile P1 repro) — the receiver
    constraint lives in the query itself."""
    source = (
        "function setup(mockEnv, settings) {\n"
        '  mockEnv["NODE_TLS_REJECT_UNAUTHORIZED"] = "0";\n'
        '  settings.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n"
    )
    assert _produce_at(source, "src/mock_env.js") == ()


# ---------------------------------------------------------------------------
# import_bindings_digest — the analyze-cache-key component pinning the
# binding step's per-file input (the from_import_map_digest sibling).
# ---------------------------------------------------------------------------


def _ref(
    module: str,
    names: tuple[str, ...],
    *,
    kind: str = "direct",
    line: int = 1,
    file_path: str = "src/x.js",
):
    from outrider.ast_facts.models import ImportRef

    return ImportRef(
        file_path=file_path,
        line=line,
        import_kind=kind,  # type: ignore[arg-type]
        module=module,
        names=names,
        is_simple_direct=kind == "relative",
    )


def test_import_bindings_digest_is_canonical_over_admission_input() -> None:
    """Deterministic, and insensitive to exactly what `_binding_admits`
    ignores: ref order, duplicates, name order within a ref, and the
    kind/line/file_path fields. Sensitive to what it consumes: the module
    and the bound-name set."""
    from outrider.agent.nodes.analyze_observed import import_bindings_digest

    a = _ref("node:crypto", ("createHash", "createCipheriv"))
    b = _ref("pg", ("Pool",))
    base = import_bindings_digest((a, b))
    assert base == import_bindings_digest((a, b))
    assert base == import_bindings_digest((b, a))  # order-insensitive
    assert base == import_bindings_digest((a, b, a))  # duplicate-insensitive
    assert base == import_bindings_digest(  # name order within a ref
        (_ref("node:crypto", ("createCipheriv", "createHash")), b)
    )
    assert base == import_bindings_digest(  # admission ignores kind/line/path
        (
            _ref("node:crypto", ("createHash", "createCipheriv"), kind="from", line=9),
            _ref("pg", ("Pool",), file_path="src/other.js"),
        )
    )
    assert base != import_bindings_digest((_ref("crypto", ("createHash", "createCipheriv")), b))
    assert base != import_bindings_digest((_ref("node:crypto", ("createHash",)), b))
    assert base != import_bindings_digest((a,))
    assert import_bindings_digest(()) != import_bindings_digest((a,))
    assert import_bindings_digest(()) == import_bindings_digest(())


def test_import_bindings_digest_boundaries_unambiguous() -> None:
    """Length-prefix framing: adjacent components can't collide by shifting
    bytes across a boundary — across names, and across the module/names line."""
    from outrider.agent.nodes.analyze_observed import import_bindings_digest

    assert import_bindings_digest((_ref("m", ("ab", "c")),)) != import_bindings_digest(
        (_ref("m", ("a", "bc")),)
    )
    assert import_bindings_digest((_ref("ab", ("c",)),)) != import_bindings_digest(
        (_ref("a", ("bc",)),)
    )
