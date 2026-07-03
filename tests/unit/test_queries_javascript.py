"""Per-variant structural pins for the JS/TS OBSERVED query catalog
(specs/2026-07-03-js-ts-observed-query-catalog.md).

Every admitted syntactic form is pinned INDIVIDUALLY (the adapters-arc
lesson: a union positive over an alternation hides single-variant
regressions — revert-the-fold must fail per variant), and every
precision-guard negative is pinned individually too: a negative that
starts firing is a proof-boundary regression, not a recall improvement.
Each case runs under ALL THREE grammars of the javascript catalog
(javascript / typescript / tsx) — the catalog is one query set compiled
per dialect, and the pins prove behavior parity across them.

All fixture sources are deliberately-insecure PARSER INPUTS for the
security queries under test — nothing here executes them.
"""

from __future__ import annotations

import pytest

from outrider.queries import registry

_GRAMMARS = ("javascript", "typescript", "tsx")

# (query_id_suffix, should_fire, variant_label, source)
_CASES: tuple[tuple[str, bool, str, str], ...] = (
    # --- weak_crypto_hash: positives, one per admitted form ---
    ("weak_crypto_hash", True, "member-md5", 'crypto.createHash("md5");'),
    ("weak_crypto_hash", True, "bare-sha1", 'createHash("sha1");'),
    ("weak_crypto_hash", True, "uppercase-MD5", 'crypto.createHash("MD5");'),
    # --- weak_crypto_hash: precision negatives ---
    ("weak_crypto_hash", False, "sha256", 'crypto.createHash("sha256");'),
    ("weak_crypto_hash", False, "sha512-bare", 'createHash("sha512");'),
    ("weak_crypto_hash", False, "non-literal-algo", "crypto.createHash(algo);"),
    ("weak_crypto_hash", False, "other-fn-name", 'lookupHash("md5");'),
    ("weak_crypto_hash", False, "sha1-inside-longer-name", 'crypto.createHash("xsha1");'),
    # --- weak_crypto_broken_cipher: positives ---
    (
        "weak_crypto_broken_cipher",
        True,
        "member-des-cbc",
        'crypto.createCipheriv("des-cbc", key, iv);',
    ),
    (
        "weak_crypto_broken_cipher",
        True,
        "bare-des-ede3-cbc",
        'createCipheriv("des-ede3-cbc", key, iv);',
    ),
    ("weak_crypto_broken_cipher", True, "rc4", 'crypto.createCipheriv("rc4", key, "");'),
    (
        "weak_crypto_broken_cipher",
        True,
        "legacy-createCipher-bf",
        'crypto.createCipher("bf-cbc", pw);',
    ),
    (
        "weak_crypto_broken_cipher",
        True,
        "uppercase-DES",
        'crypto.createCipheriv("DES-EDE3", key, iv);',
    ),
    # --- weak_crypto_broken_cipher: precision negatives ---
    (
        "weak_crypto_broken_cipher",
        False,
        "aes-256-gcm",
        'crypto.createCipheriv("aes-256-gcm", key, iv);',
    ),
    (
        "weak_crypto_broken_cipher",
        False,
        "non-literal-algo",
        "crypto.createCipheriv(algo, key, iv);",
    ),
    (
        "weak_crypto_broken_cipher",
        False,
        "desx-prefix-not-des",
        'crypto.createCipheriv("desx-cbc", k, iv);',
    ),
    (
        "weak_crypto_broken_cipher",
        False,
        "camellia",
        'crypto.createCipheriv("camellia-128-cbc", k, iv);',
    ),
    # --- weak_crypto_ecb_mode ---
    ("weak_crypto_ecb_mode", True, "aes-128-ecb", 'crypto.createCipheriv("aes-128-ecb", k, null);'),
    ("weak_crypto_ecb_mode", True, "bare-des-ecb", 'createCipheriv("des-ecb", k, null);'),
    ("weak_crypto_ecb_mode", False, "aes-128-cbc", 'crypto.createCipheriv("aes-128-cbc", k, iv);'),
    ("weak_crypto_ecb_mode", False, "ecb-mid-name", 'crypto.createCipheriv("aes-ecb-x", k, iv);'),
    # --- command_injection_child_process: positives, one per admitted form ---
    ("command_injection_child_process", True, "bare-exec-left-concat", 'exec("ls " + userInput);'),
    ("command_injection_child_process", True, "bare-exec-right-concat", 'exec(prefix + " -la");'),
    (
        "command_injection_child_process",
        True,
        "member-execSync-template",
        "child_process.execSync(`ls ${dir}`);",
    ),
    ("command_injection_child_process", True, "bare-execSync-template", "execSync(`cat ${f}`);"),
    (
        "command_injection_child_process",
        True,
        "member-exec-concat-with-callback",
        'child_process.exec("ping " + host, cb);',
    ),
    # --- command_injection_child_process: precision negatives ---
    ("command_injection_child_process", False, "constant-command", 'exec("ls -la");'),
    ("command_injection_child_process", False, "template-no-substitution", "exec(`ls -la`);"),
    ("command_injection_child_process", False, "identifier-arg", "exec(cmd);"),
    (
        "command_injection_child_process",
        False,
        "aliased-namespace-recall-gap",
        'cp.exec("ls " + x);',
    ),
    (
        "command_injection_child_process",
        False,
        "regex-exec-member",
        "pattern.exec(text + suffix);",
    ),
    (
        "command_injection_child_process",
        False,
        "execFile-array-form",
        'child_process.execFile("ls", [dir]);',
    ),
    # --- command_injection_eval: positives, one per admitted form ---
    ("command_injection_eval", True, "eval-identifier", "eval(payload);"),
    ("command_injection_eval", True, "eval-concat", 'eval("f(" + arg + ")");'),
    ("command_injection_eval", True, "eval-template-substitution", "eval(`return ${expr}`);"),
    ("command_injection_eval", True, "eval-member-expression", "eval(req.body.code);"),
    ("command_injection_eval", True, "new-Function-dynamic-body", 'new Function("a", body);'),
    ("command_injection_eval", True, "new-Function-call-arg", "new Function(buildBody());"),
    # --- command_injection_eval: precision negatives ---
    ("command_injection_eval", False, "eval-string-literal", 'eval("2+2");'),
    ("command_injection_eval", False, "eval-template-no-substitution", "eval(`2+2`);"),
    ("command_injection_eval", False, "evaluate-fn-name", "evaluate(payload);"),
    ("command_injection_eval", False, "member-eval-third-party", "math.eval(expr);"),
    (
        "command_injection_eval",
        False,
        "new-Function-all-literals",
        'new Function("a", "return a");',
    ),
    ("command_injection_eval", False, "new-Function-zero-args", "new Function();"),
    ("command_injection_eval", False, "new-other-constructor", "new Parser(input);"),
    # --- sql_injection_string_concat: positives, one per admitted form ---
    (
        "sql_injection_string_concat",
        True,
        "query-left-concat",
        'db.query("SELECT * FROM t WHERE id = " + id);',
    ),
    (
        "sql_injection_string_concat",
        True,
        "query-template-substitution",
        "pool.query(`SELECT * FROM t WHERE id = ${id}`);",
    ),
    ("sql_injection_string_concat", True, "execute-left-concat", 'conn.execute("SELECT " + c);'),
    (
        "sql_injection_string_concat",
        True,
        "query-right-concat",
        'db.query(prefix + " ORDER BY name");',
    ),
    (
        "sql_injection_string_concat",
        True,
        "chained-concat",
        'db.query("SELECT * FROM t WHERE a = " + a + " AND b");',
    ),
    # --- sql_injection_string_concat: precision negatives ---
    (
        "sql_injection_string_concat",
        False,
        "parameterized-query",
        'db.query("SELECT * FROM t WHERE id = $1", [id]);',
    ),
    ("sql_injection_string_concat", False, "template-no-substitution", "db.query(`SELECT 1`);"),
    ("sql_injection_string_concat", False, "identifier-only", "db.query(sql);"),
    ("sql_injection_string_concat", False, "concat-without-string", "db.query(a + b);"),
    ("sql_injection_string_concat", False, "bare-query-helper", 'query("SELECT " + x);'),
    ("sql_injection_string_concat", False, "non-sql-method", 'db.find("name = " + n);'),
    # --- tls_verify_disabled: positives, one per admitted form ---
    (
        "tls_verify_disabled",
        True,
        "pair-identifier-key",
        "https.request({ rejectUnauthorized: false });",
    ),
    (
        "tls_verify_disabled",
        True,
        "pair-string-key",
        'const opts = { "rejectUnauthorized": false };',
    ),
    (
        "tls_verify_disabled",
        True,
        "env-dot-assignment",
        'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";',
    ),
    (
        "tls_verify_disabled",
        True,
        "env-bracket-assignment",
        'process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0";',
    ),
    # --- tls_verify_disabled: precision negatives ---
    (
        "tls_verify_disabled",
        False,
        "rejectUnauthorized-true",
        "https.request({ rejectUnauthorized: true });",
    ),
    (
        "tls_verify_disabled",
        False,
        "rejectUnauthorized-variable",
        "https.request({ rejectUnauthorized: flag });",
    ),
    ("tls_verify_disabled", False, "other-key-false", "const o = { secure: false };"),
    (
        "tls_verify_disabled",
        False,
        "env-assigned-1",
        'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "1";',
    ),
    ("tls_verify_disabled", False, "other-env-var", 'process.env.NODE_ENV = "0";'),
)


@pytest.mark.parametrize("grammar", _GRAMMARS)
@pytest.mark.parametrize(
    ("suffix", "should_fire", "label", "source"),
    _CASES,
    ids=[f"{c[0]}-{'fires' if c[1] else 'silent'}-{c[2]}" for c in _CASES],
)
def test_js_catalog_variant(
    suffix: str, should_fire: bool, label: str, source: str, grammar: str
) -> None:
    """Each admitted form fires (and each guard negative stays silent)
    under every grammar of the javascript catalog."""
    query_id = f"javascript.{suffix}"
    matches = registry.match(query_id, source.encode("utf-8"), grammar=grammar)  # type: ignore[arg-type]
    assert bool(matches) == should_fire, (
        f"{query_id} / {label} under {grammar}: fired={bool(matches)}, expected={should_fire}"
    )


def test_every_js_catalog_query_has_both_pin_directions() -> None:
    """Completeness guard for the table above: every registered javascript
    OBSERVED query appears with at least one positive AND one negative pin,
    so a future catalog addition cannot land unpinned."""
    pinned_positive = {c[0] for c in _CASES if c[1]}
    pinned_negative = {c[0] for c in _CASES if not c[1]}
    registered = {
        oq.query_match_id.removeprefix("javascript.")
        for oq in registry.OBSERVED_QUERIES.values()
        if oq.language == "javascript"
    }
    assert registered == pinned_positive == pinned_negative


def test_js_match_envelopes_are_nonempty_byte_spans() -> None:
    """Producer contract: a firing catalog query yields a non-empty byte
    envelope (`query_span_to_source_lines` requires byte_end > byte_start;
    the mandatory-capture registration rule makes this structural)."""
    for suffix, should_fire, _label, source in _CASES:
        if not should_fire:
            continue
        for span in registry.match(
            f"javascript.{suffix}", source.encode("utf-8"), grammar="javascript"
        ):
            assert span.byte_end > span.byte_start
