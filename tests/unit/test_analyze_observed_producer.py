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

import pytest

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
        lexical_bindings=parsed.lexical_bindings,
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
        lexical_bindings=parsed.lexical_bindings,
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
        lexical_bindings=parsed.lexical_bindings,
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
        lexical_bindings=parsed.lexical_bindings,
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
        lexical_bindings=(),
    )
    assert matches == ()


# ---------------------------------------------------------------------------
# Import-binding admission: a name-anchored match is
# admitted only when its anchor identifier provably binds to the dangerous
# API via the file's imports. Negative fixtures are the Greptile PR-round
# repros (2026-07-03), pinned per-variant.
# ---------------------------------------------------------------------------


# One row per admission variant: (source, path, expected query id or None
# for "no OBSERVED finding"). Each row keeps its own id so a single-variant
# regression names itself; rationale rides as a per-row comment. Cases with
# extra assertions (line anchors, multi-form loops) stay as standalone tests
# below the table.
_BINDING_ADMISSION_CASES = (
    # --- lexical shadowing guard (DECISIONS.md#060) ---
    pytest.param(
        # Codex round-4 repro: the param shadows the import — the call
        # resolves to the local, the import proves nothing.
        'import { createHash } from "node:crypto";\n'
        "export function f(createHash) {\n"
        '  return createHash("md5");\n'
        "}\n",
        "src/shadowed_import.mjs",
        None,
        id="shadowed-import-param-denied",
    ),
    pytest.param(
        # Greptile PR-round-2 repro: a local `process` parameter is a
        # mock, not the global — the shadow_guard denies the kill switch.
        'function apply(process) {\n  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n}\n',
        "src/apply_env.js",
        None,
        id="shadowed-process-param-denied",
    ),
    pytest.param(
        # Guarded-global module-scope shadow, `const` variant (Codex spec
        # finding 1): legal JS, shadows every function below.
        "const process = mock;\n"
        "function f() {\n"
        '  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n",
        "src/module_const_process.js",
        None,
        id="module-scope-const-process-denied",
    ),
    pytest.param(
        # Same, hoisted `var` variant — pinned separately so a
        # hoisting-path regression can't hide behind the const case.
        "var process = mock;\n"
        "function f() {\n"
        '  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n",
        "src/module_var_process.js",
        None,
        id="module-scope-var-process-denied",
    ),
    pytest.param(
        # RANGE positive: the catch param shadows only the catch clause —
        # the kill switch AFTER the block is the real global. Proves the
        # guard is span-contained, not scope-unit-coarse.
        "function f() {\n"
        "  try { g(); } catch (process) { log(process); }\n"
        '  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n",
        "src/catch_then_real.js",
        "javascript.tls_env_verify_disabled",
        id="shadow-in-catch-call-outside-admitted",
    ),
    pytest.param(
        # RANGE positive: a shadow in one function does not suppress the
        # bound call in a SIBLING function.
        'import { createHash } from "node:crypto";\n'
        "export function fake(createHash) { return createHash; }\n"
        "export function real(secret) {\n"
        '  return createHash("md5").update(secret).digest("hex");\n'
        "}\n",
        "src/sibling_shadow.mjs",
        "javascript.weak_crypto_hash",
        id="shadow-in-sibling-function-admitted",
    ),
    pytest.param(
        # /code-review convergent find (angles B + altitude): eval/Function
        # are binding=None globals, so their lexical proof IS the shadow
        # guard — a local `eval` parameter resolves to the local, not the
        # global.
        "export function run(eval, x) {\n  return eval(x + suffix);\n}\n",
        "src/sandboxed_eval.mjs",
        None,
        id="shadowed-eval-param-denied",
    ),
    pytest.param(
        # Same guard, `Function` global, module-scope mock variant.
        "const Function = mock.Function;\n"
        "export function build(body) {\n"
        "  return new Function(body);\n"
        "}\n",
        "src/mock_function.mjs",
        None,
        id="shadowed-function-mock-denied",
    ),
    pytest.param(
        # Positive control for the guard: the UNSHADOWED global still fires.
        "export function run(x, suffix) {\n  return eval(x + suffix);\n}\n",
        "src/real_eval.mjs",
        "javascript.command_injection_eval",
        id="unshadowed-eval-admitted",
    ),
    pytest.param(
        # The guard is MATCH-participating (Codex implementation-audit
        # find): shadowing the OTHER guarded global must not deny a real
        # `eval` match — `Function` does not participate in this match.
        "export function run(Function, x) {\n  return eval(x);\n}\n",
        "src/unrelated_guard_shadow.mjs",
        "javascript.command_injection_eval",
        id="unrelated-guarded-global-shadow-still-admits-eval",
    ),
    pytest.param(
        # The inverse pairing: a shadowed `eval` must not deny a real
        # `new Function(...)` match.
        "export function build(eval, body) {\n  return new Function(body);\n}\n",
        "src/unrelated_guard_shadow_fn.mjs",
        "javascript.command_injection_eval",
        id="unrelated-eval-shadow-still-admits-function",
    ),
    pytest.param(
        # Guard participation is guard-POSITION-only (Codex 2nd-round find):
        # here `Function` is shadowed AND appears — but only as the ARGUMENT
        # to a real global `eval`, not at the callee position. The `eval`
        # match must still admit.
        "export function run(Function, x) {\n  return eval(Function);\n}\n",
        "src/guard_name_in_arg.mjs",
        "javascript.command_injection_eval",
        id="guarded-name-in-argument-position-still-admits",
    ),
    pytest.param(
        # Inverse: `eval` shadowed and passed as the `new Function` ARGUMENT
        # — the Function constructor match still admits.
        "export function build(eval) {\n  return new Function(eval);\n}\n",
        "src/guard_name_in_ctor_arg.mjs",
        "javascript.command_injection_eval",
        id="guarded-name-in-ctor-argument-still-admits",
    ),
    pytest.param(
        # /code-review find: a `let` in a switch case is block-scoped to the
        # switch body — the post-switch `eval` is the real global and must
        # fire (the visibility span must be the switch_body, not the whole
        # function).
        "export function f(x) {\n"
        "  switch (x) {\n"
        "    case 1: { let eval = mock; break; }\n"
        "  }\n"
        "  return eval(userInput);\n"
        "}\n",
        "src/switch_scoped.mjs",
        "javascript.command_injection_eval",
        id="switch-scoped-shadow-does-not-reach-post-switch-global",
    ),
    pytest.param(
        # /code-review find: a module-scope import/require rebind of a
        # guarded global shadows the whole file — `const process =
        # require("./mock")` is not the global.
        'const process = require("./mock");\n'
        "export function f() {\n"
        '  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n",
        "src/require_rebound_process.js",
        None,
        id="module-scope-require-rebind-of-global-denied",
    ),
    pytest.param(
        # A re-export creates NO runtime binding — `process` inside this
        # file is still the global, so the kill switch must fire (only
        # VALUE imports rebind a guarded global).
        'export { process } from "./shim";\n'
        "export function enable() {\n"
        '  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n",
        "src/reexported_process.js",
        "javascript.tls_env_verify_disabled",
        id="re-export-of-guarded-global-still-admits",
    ),
    pytest.param(
        # `import type` creates no runtime binding either — same rule,
        # statement-level type-only spelling.
        'import type { process } from "./shim";\n'
        "export function enable() {\n"
        '  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n",
        "src/type_imported_process.ts",
        "javascript.tls_env_verify_disabled",
        id="type-only-import-of-guarded-global-still-admits",
    ),
    pytest.param(
        # A for-of loop-head `const eval` is scoped to the LOOP statement
        # (for-of parses as `for_in_statement` in all three grammars) —
        # the post-loop `eval` is the real global and must fire.
        "export function run(xs, userInput) {\n"
        "  for (const eval of xs) {\n"
        "    log(eval);\n"
        "  }\n"
        "  return eval(userInput);\n"
        "}\n",
        "src/for_of_scoped.mjs",
        "javascript.command_injection_eval",
        id="for-of-head-shadow-does-not-reach-post-loop-global",
    ),
    pytest.param(
        # Inside the loop body the head binding IS in scope — denied.
        "export function run(xs) {\n  for (const eval of xs) {\n    eval(userInput);\n  }\n}\n",
        "src/for_of_shadowed_body.mjs",
        None,
        id="for-of-head-shadow-denies-inside-loop-body",
    ),
    pytest.param(
        # CJS twin of the all-type-specifier case (Codex implementation-
        # audit find): a require that binds NO surviving local name loads
        # the module but proves no runtime callability.
        'const {} = require("pg");\n'
        "export function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/empty_require.js",
        None,
        id="binding-less-require-module-presence-denied",
    ),
    pytest.param(
        # Type-only import proves nothing at runtime (Codex round-4
        # residual, now closed): module_presence needs a VALUE import.
        'import type { Pool } from "pg";\n'
        "export function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/typed_only.ts",
        None,
        id="type-only-import-module-presence-denied",
    ),
    pytest.param(
        # The SPECIFIER spelling of a pure type import must deny like the
        # statement spelling (/code-review angle-A find).
        'import { type Pool } from "pg";\n'
        "export function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/typed_specifiers.ts",
        None,
        id="all-type-specifier-import-module-presence-denied",
    ),
    pytest.param(
        # Side-effect import binds no name a runtime call resolves through.
        'import "mysql2";\n'
        "export function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/side_effect.mjs",
        None,
        id="side-effect-import-module-presence-denied",
    ),
    pytest.param(
        # Re-export binds no local name — the digest-collision twin of a
        # value import (`import { Q } from "mysql"`), denied here.
        'export { Q } from "mysql";\n'
        "export function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/reexport_only.mjs",
        None,
        id="reexport-module-presence-denied",
    ),
    # --- import-binding admission (pre-guard rounds) ---
    pytest.param(
        # Greptile P1 repro: a `createHash` imported from a non-crypto module,
        # defined locally, or called on an arbitrary object — the NAME alone
        # is not proof of a crypto construction.
        'import { createHash } from "./cache";\n'
        "const cacheApi = { createHash(name) { return name; } };\n"
        "export function demo() {\n"
        '  const a = createHash("md5");\n'
        '  const b = cacheApi.createHash("sha1");\n'
        "  return [a, b];\n"
        "}\n",
        "src/example.js",
        None,
        id="unbound-createhash-denied",
    ),
    pytest.param(
        # CJS whole-module require: the bare-require ALIAS receiver proves.
        'const c = require("crypto");\n'
        "function digest(s) {\n"
        '  return c.createHash("sha1").update(s).digest("hex");\n'
        "}\n",
        "src/digest.js",
        "javascript.weak_crypto_hash",
        id="cjs-whole-module-receiver-admitted",
    ),
    pytest.param(
        # Greptile P1 repro (the sharpest): an unrelated exec helper must not
        # produce a CRITICAL OBSERVED command-injection finding.
        'import { exec } from "./jobs";\n'
        "export function runJob(id) {\n"
        "  return exec('job-' + id);\n"
        "}\n",
        "src/jobs_runner.js",
        None,
        id="unbound-exec-helper-denied",
    ),
    pytest.param(
        'const { exec } = require("child_process");\n'
        "function run(cmd) {\n"
        "  exec('ls ' + cmd);\n"
        "}\n",
        "src/run.js",
        "javascript.command_injection_child_process",
        id="child-process-destructured-exec-admitted",
    ),
    pytest.param(
        # Previously a documented recall gap (the member arm demanded the
        # literal `child_process` receiver name); the import join proves the
        # aliased namespace.
        'const cp = require("node:child_process");\n'
        "function run(cmd) {\n"
        "  cp.execSync(`run ${cmd}`);\n"
        "}\n",
        "src/run.js",
        "javascript.command_injection_child_process",
        id="aliased-namespace-exec-admitted",
    ),
    pytest.param(
        # Greptile P1 repro: `.query(concat)` on a non-database API — no DB
        # driver imported in the file — is not SQL injection.
        'import { search } from "./search";\n'
        "export function lookup(searchClient, tag) {\n"
        "  return searchClient.query('tag:' + tag);\n"
        "}\n",
        "src/lookup.js",
        None,
        id="query-concat-without-db-driver-denied",
    ),
    pytest.param(
        'const { Pool } = require("pg");\n'
        "function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/find.js",
        "javascript.sql_injection_string_concat",
        id="query-concat-with-db-driver-admitted",
    ),
    pytest.param(
        # `require("mysql2/promise")` — the dominant modern mysql2 idiom —
        # proves package presence for a rule naming the package root; under
        # exact-string matching this real SQL injection dropped to JUDGED.
        'const mysql = require("mysql2/promise");\n'
        "async function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/find.js",
        "javascript.sql_injection_string_concat",
        id="subpath-specifier-admitted",
    ),
    pytest.param(
        # Package-root matching is `/`-delimited: `mysql2-mock` shares the
        # `mysql2` byte prefix but is a different package.
        'const mock = require("mysql2-mock");\n'
        "function find(pool, name) {\n"
        '  return pool.query("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "}\n",
        "src/find.js",
        None,
        id="lookalike-package-root-denied",
    ),
    pytest.param(
        # Greptile P1 repro: a standalone options literal with no TLS-capable
        # module in the file is not a TLS-disabled finding.
        "function policy() {\n  return { rejectUnauthorized: false };\n}\n",
        "src/benign_config.js",
        None,
        id="reject-unauthorized-without-consumer-denied",
    ),
    pytest.param(
        'const https = require("https");\n'
        "function agent() {\n"
        "  return new https.Agent({ rejectUnauthorized: false });\n"
        "}\n",
        "src/agent.js",
        "javascript.tls_verify_disabled",
        id="reject-unauthorized-with-https-admitted",
    ),
    pytest.param(
        # The canonical managed-Postgres MITM idiom: `rejectUnauthorized:
        # false` inside a pg `ssl:` option with only the DB driver imported —
        # the option-honoring client families are in the TLS set, not just
        # HTTP/TLS clients.
        'const { Pool } = require("pg");\n'
        "function connect(url) {\n"
        "  return new Pool({ connectionString: url, ssl: { rejectUnauthorized: false } });\n"
        "}\n",
        "src/db.js",
        "javascript.tls_verify_disabled",
        id="reject-unauthorized-with-ssl-option-client-admitted",
    ),
    pytest.param(
        # Greptile P1 repro: `mockEnv[...]` / `settings....` mutate a local
        # object, not the process TLS switch — the receiver constraint lives
        # in the query itself.
        "function setup(mockEnv, settings) {\n"
        '  mockEnv["NODE_TLS_REJECT_UNAUTHORIZED"] = "0";\n'
        '  settings.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
        "}\n",
        "src/mock_env.js",
        None,
        id="non-process-env-receiver-denied",
    ),
)


@pytest.mark.parametrize(("source", "file_path", "expected_query_id"), _BINDING_ADMISSION_CASES)
def test_binding_admission_per_variant(
    source: str, file_path: str, expected_query_id: str | None
) -> None:
    """Per-variant import-binding admission pins: `None` rows must produce
    ZERO OBSERVED findings (the unbound/lookalike/absent-module negatives),
    admitted rows exactly one finding under the expected query id."""
    findings = _produce_at(source, file_path)
    if expected_query_id is None:
        assert findings == ()
    else:
        (f,) = findings
        assert f.query_match_id == expected_query_id


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


def test_member_exec_with_bound_bare_sibling_is_not_admitted() -> None:
    """`pattern.exec(text + suffix)` in a file that ALSO destructures `exec`
    from child_process: the member match must anchor on its RECEIVER
    (`pattern`, unbound → dropped), never fall back to the bare-callee name
    (`exec`, which IS bound here). A refactor swapping the `_recv`-over-`_fn`
    anchor preference would ship a false CRITICAL on this file while every
    import-free negative stays green; the bare call in the same file is the
    positive control proving the import itself admits."""
    source = (
        'const { exec } = require("child_process");\n'
        "function run(pattern, text, suffix, cmd) {\n"
        "  pattern.exec(text + suffix);\n"
        "  exec('ls ' + cmd);\n"
        "}\n"
    )
    (f,) = _produce_at(source, "src/run.js")
    assert f.query_match_id == "javascript.command_injection_child_process"
    assert f.line_start == 4  # the bare bound call — NOT the regex receiver on line 3


def test_process_env_kill_switch_needs_no_import() -> None:
    """The process-wide kill switch's receiver is text-constrained in the
    query itself — it fires with ZERO imports, in both the dot and bracket
    forms, under its own query id. (The shadowed-`process` FP class is
    closed by the shadow_guard — the table above pins those negatives.)"""
    for line in (
        'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";',
        'process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0";',
    ):
        source = f"function disable() {{\n  {line}\n}}\n"
        (f,) = _produce_at(source, "src/danger.js")
        assert f.query_match_id == "javascript.tls_env_verify_disabled"


def test_module_arm_denies_when_module_inputs_not_threaded() -> None:
    """Deny-by-default: a caller that does not thread `all_scope_units` +
    `added_line_ranges` (both default empty) gets exactly the pre-v7
    behavior — the module-top-level kill switch stays dropped. The module
    arm's positive control is
    `test_module_level_kill_switch_admits_on_changed_lines`; the
    function-wrapped forms in `test_process_env_kill_switch_needs_no_import`
    pin the normal containment arm."""
    source = 'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\nconst x = 1;\n'
    assert _produce_at(source, "src/index.js") == ()


# ---------------------------------------------------------------------------
# Module-scope admission arm (specs/2026-07-04-module-scope-admission-arm.md):
# a `module_scope_eligible` query's match admits without an enclosing scope
# iff its envelope is DISJOINT from every parsed scope and fully inside a
# head-side added-line byte range. `_KILL_SWITCH` is detection-target
# documentation, parsed for structure and never executed.
# ---------------------------------------------------------------------------

_KILL_SWITCH = 'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'


def _produce_module_level(source: str, file_path: str, *, ranges=None):
    """Producer run with the module-arm inputs threaded (no included scopes,
    all parsed scopes for disjointness, whole-file added ranges by default)."""
    parsed = _parsed_at(source, file_path)
    matches = run_observed_matches(
        file_path=file_path,
        head_content=source,
        included_scope_units=(),
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
        all_scope_units=parsed.scope_units,
        added_line_ranges=ranges if ranges is not None else ((0, len(source.encode())),),
    )
    return produce_observed_findings(
        matches,
        file_path=file_path,
        review_id=uuid4(),
        installation_id=12345,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )


def test_module_level_kill_switch_admits_on_changed_lines() -> None:
    """The canonical real-world form — the kill switch at module top level,
    on a changed line — now produces the OBSERVED finding (the veto this
    arm exists to close)."""
    source = _KILL_SWITCH + "const x = 1;\n"
    (finding,) = _produce_module_level(source, "src/index.js")
    assert finding.query_match_id == "javascript.tls_env_verify_disabled"
    assert finding.evidence_tier is EvidenceTier.OBSERVED
    assert finding.line_start == 1


def test_module_level_kill_switch_denied_on_unchanged_lines() -> None:
    """Same file, but the added ranges cover only the OTHER line — a
    module-level match in unchanged code stays excluded (the diff anchors
    the proof)."""
    source = _KILL_SWITCH + "const x = 1;\n"
    unchanged_only = ((len(_KILL_SWITCH.encode()), len(source.encode())),)
    assert _produce_module_level(source, "src/index.js", ranges=unchanged_only) == ()


def test_module_level_denied_inside_parsed_but_excluded_scope() -> None:
    """Disjointness is proven against ALL parsed scopes, not the included
    set: a kill switch inside a parsed-but-not-included function is NOT a
    module-level match, even with whole-file added ranges."""
    source = f"function disable() {{\n  {_KILL_SWITCH}}}\n"
    assert _produce_module_level(source, "src/index.js") == ()


def test_module_level_shadowed_process_denied() -> None:
    """The shadow guard still applies at module level: a module-wide local
    `process` binding means the name is a mock, not the global."""
    source = "const process = mockProcess;\n" + _KILL_SWITCH
    assert _produce_module_level(source, "src/index.js") == ()


def test_ineligible_query_never_admits_at_module_level() -> None:
    """Eligibility is what gates the arm: a module-top-level `eval` sink
    (command_injection_eval, NOT module_scope_eligible) stays denied even
    with the module inputs threaded. The eval string is an inert parse
    fixture."""
    source = "eval(userInput);\n"
    assert _produce_module_level(source, "src/app.js") == ()


def test_module_level_straddle_and_enclosure_denied() -> None:
    """Disjointness, not non-containment: an envelope that OVERLAPS a scope
    boundary or ENCLOSES a scope is a straddle and stays denied; a genuinely
    disjoint envelope inside an added range admits (synthetic spans against
    the real eligible query)."""
    from outrider.agent.nodes.analyze_observed import _module_level_admits
    from outrider.ast_facts.models import QueryCaptureSpan, QueryMatchSpan
    from outrider.queries import registry

    eligible = registry.OBSERVED_QUERIES["javascript.tls_env_verify_disabled"]

    def span(start: int, end: int) -> QueryMatchSpan:
        return QueryMatchSpan(
            byte_start=start,
            byte_end=end,
            captures=(QueryCaptureSpan(name="_proc", byte_start=start, byte_end=end),),
        )

    scope = ((10, 50),)
    ranges = ((0, 100),)
    # Overlapping a scope boundary: straddle, denied.
    assert not _module_level_admits(
        eligible, span(40, 60), all_scope_ranges=scope, added_line_ranges=ranges
    )
    # Enclosing the scope: straddle, denied.
    assert not _module_level_admits(
        eligible, span(5, 60), all_scope_ranges=scope, added_line_ranges=ranges
    )
    # Disjoint and inside an added range: admits.
    assert _module_level_admits(
        eligible, span(55, 70), all_scope_ranges=scope, added_line_ranges=ranges
    )
    # Deny-by-default with no added ranges.
    assert not _module_level_admits(
        eligible, span(55, 70), all_scope_ranges=scope, added_line_ranges=()
    )


def test_has_module_level_eligible_match_pre_check() -> None:
    """The routing pre-check delegates to the producer's own admission chain:
    True exactly when a module-level match would be admitted."""
    from outrider.agent.nodes.analyze_observed import has_module_level_eligible_match

    def pre_check(source: str) -> bool:
        parsed = _parsed_at(source, "src/index.js")
        return has_module_level_eligible_match(
            file_path="src/index.js",
            head_content=source,
            all_scope_units=parsed.scope_units,
            added_line_ranges=((0, len(source.encode())),),
            import_refs=parsed.imports,
            lexical_bindings=parsed.lexical_bindings,
        )

    assert pre_check(_KILL_SWITCH + "const x = 1;\n")
    # Shadowed global: the full chain (not just a raw match) decides.
    assert not pre_check("const process = mockProcess;\n" + _KILL_SWITCH)
    # No eligible match at all.
    assert not pre_check("const x = 1;\n")


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
    value: bool = True,
):
    from outrider.ast_facts.models import ImportRef

    return ImportRef(
        file_path=file_path,
        line=line,
        import_kind=kind,  # type: ignore[arg-type]
        module=module,
        names=names,
        is_simple_direct=kind == "relative",
        is_value_import=value,
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
    # The value marker is admission input (shadowing-guard spec): flipping
    # it alone must move the digest.
    assert base != import_bindings_digest(
        (_ref("node:crypto", ("createHash", "createCipheriv"), value=False), b)
    )
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


def _lex(name: str, start: int, end: int, *, kind: str = "param", line: int = 1):
    from outrider.ast_facts.models import LexicalBinding

    return LexicalBinding(
        file_path="src/x.js",
        name=name,
        kind=kind,  # type: ignore[arg-type]
        line=line,
        visibility_byte_start=start,
        visibility_byte_end=end,
    )


def test_lexical_bindings_digest_is_canonical_over_guard_input() -> None:
    """The `import_bindings_digest` sibling for the shadow guard's input:
    deterministic; insensitive to what `_shadowed` ignores (record order,
    duplicates, kind/line); sensitive to what it consumes (name +
    visibility span); the empty tuple digests distinctly."""
    from outrider.agent.nodes.analyze_observed import lexical_bindings_digest

    a = _lex("process", 0, 90)
    b = _lex("crypto", 40, 80)
    base = lexical_bindings_digest((a, b))
    assert base == lexical_bindings_digest((b, a))  # order-insensitive
    assert base == lexical_bindings_digest((a, b, a))  # duplicate-insensitive
    assert base == lexical_bindings_digest(  # guard ignores kind/line
        (_lex("process", 0, 90, kind="var", line=7), b)
    )
    assert base != lexical_bindings_digest((_lex("process2", 0, 90), b))  # name
    assert base != lexical_bindings_digest((_lex("process", 0, 91), b))  # span
    assert base != lexical_bindings_digest((a,))
    assert lexical_bindings_digest(()) != lexical_bindings_digest((a,))
    assert lexical_bindings_digest(()) == lexical_bindings_digest(())
    # Framing: name/span boundary can't collide by shifting digits.
    assert lexical_bindings_digest((_lex("x1", 2, 3),)) != lexical_bindings_digest(
        (_lex("x", 12, 3),)
    )
