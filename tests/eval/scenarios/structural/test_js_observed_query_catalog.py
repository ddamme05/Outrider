"""Structural eval scenario: JS/TS OBSERVED query catalog
(specs/2026-07-03-js-ts-observed-query-catalog.md).

The JS/TS mirror of the Python OBSERVED precision scenario: each catalog
family MUST fire on a canonical positive fixture and must NOT fire on a
lookalike negative, under EVERY grammar the catalog compiles for
(javascript / typescript / tsx) — the catalog is one query set with three
dialect variants, and dialect parity is part of the contract. LLM-free —
runs `queries.registry.match` and the deterministic producer directly.

The producer case is the spec's graded slice: a real broken-cipher use in
a JS file becomes an OBSERVED `ReviewFinding` (real `javascript.*`
query_match_id, policy severity, registry static text) with zero model
involvement — the evidence-tier parity the arc exists to close. The
exhaustive per-variant pins live in tests/unit/test_queries_javascript.py.

All fixture sources are deliberately-insecure PARSER INPUTS for the
security queries under test — nothing here executes them.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from outrider.agent.nodes.analyze_observed import (
    produce_observed_findings,
    run_observed_matches,
)
from outrider.ast_facts.registry import parse_source
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION, FindingType, lookup_severity
from outrider.queries import registry

_GRAMMARS = ("javascript", "typescript", "tsx")

# (query_id, positive_src, negative_src): positive must match >=1, negative 0.
_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "javascript.weak_crypto_hash",
        'crypto.createHash("md5");\ncreateHash("sha1");\n',
        'crypto.createHash("sha256");\ncrypto.createHash(algo);\n',
    ),
    (
        "javascript.weak_crypto_broken_cipher",
        'crypto.createCipheriv("des-ede3-cbc", key, iv);\n',
        'crypto.createCipheriv("aes-256-gcm", key, iv);\ncrypto.createCipheriv(algo, k, iv);\n',
    ),
    (
        "javascript.weak_crypto_ecb_mode",
        'crypto.createCipheriv("aes-128-ecb", key, null);\n',
        'crypto.createCipheriv("aes-128-cbc", key, iv);\n',
    ),
    (
        "javascript.command_injection_child_process",
        'exec("ls " + userInput);\nchild_process.execSync(`cat ${f}`);\n',
        'exec("ls -la");\nexec(cmd);\nchild_process.execFile("ls", [dir]);\n',
    ),
    (
        "javascript.command_injection_eval",
        "eval(payload);\nnew Function(buildBody());\n",
        'eval("2+2");\nnew Function("a", "return a");\nmath.eval(expr);\n',
    ),
    (
        "javascript.sql_injection_string_concat",
        'db.query("SELECT * FROM t WHERE id = " + id);\n'
        "pool.query(`SELECT * FROM t WHERE id = ${id}`);\n",
        'db.query("SELECT * FROM t WHERE id = $1", [id]);\ndb.query(sql);\n',
    ),
    (
        "javascript.tls_verify_disabled",
        "https.request({ rejectUnauthorized: false });\n"
        'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n',
        "https.request({ rejectUnauthorized: true });\n"
        'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "1";\n',
    ),
)


@pytest.mark.parametrize("grammar", _GRAMMARS)
@pytest.mark.parametrize(("query_id", "positive", "negative"), _CASES, ids=[c[0] for c in _CASES])
def test_catalog_query_fires_on_positive_not_negative(
    query_id: str, positive: str, negative: str, grammar: str
) -> None:
    assert registry.match(query_id, positive.encode(), grammar=grammar), (  # type: ignore[arg-type]
        f"{query_id} must fire on its positive fixture under {grammar}"
    )
    assert not registry.match(query_id, negative.encode(), grammar=grammar), (  # type: ignore[arg-type]
        f"{query_id} must stay silent on its negative fixture under {grammar}"
    )


def test_catalog_covers_every_registered_javascript_query() -> None:
    """Completeness guard: the cases above cover the whole registered
    javascript catalog, so an addition cannot land without a precision pin."""
    registered = {
        oq.query_match_id
        for oq in registry.OBSERVED_QUERIES.values()
        if oq.language == "javascript"
    }
    assert registered == {c[0] for c in _CASES}


def test_js_broken_cipher_produces_deterministic_observed_finding() -> None:
    """The graded slice: a JS file with a real broken-cipher use yields an
    OBSERVED ReviewFinding through the deterministic producer — LLM-free,
    policy-set severity, replay-resolvable query id."""
    source = (
        "function encryptLegacy(key, iv, plaintext) {\n"
        '  const cipher = crypto.createCipheriv("des-ede3-cbc", key, iv);\n'
        '  return cipher.update(plaintext, "utf8", "hex") + cipher.final("hex");\n'
        "}\n"
    )
    file_path = "src/legacy/crypto.js"
    scope_units = parse_source(source.encode(), file_path, MagicMock()).scope_units
    matches = run_observed_matches(
        file_path=file_path, head_content=source, included_scope_units=scope_units
    )
    (finding,) = produce_observed_findings(
        matches,
        file_path=file_path,
        review_id=uuid4(),
        installation_id=1,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )
    assert finding.evidence_tier == EvidenceTier.OBSERVED
    assert finding.query_match_id == "javascript.weak_crypto_broken_cipher"
    assert finding.finding_type == FindingType.WEAK_CRYPTO
    assert finding.severity == lookup_severity(FindingType.WEAK_CRYPTO)
    assert 'createCipheriv("des-ede3-cbc"' in finding.evidence
    assert finding.line_start == finding.line_end == 2
    # Replay's registry-membership check resolves the id (proof boundary).
    assert registry.get_query_source(finding.query_match_id).strip()
