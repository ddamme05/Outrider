"""The OBSERVED-tier query catalog — static data, no lookup logic.

Separated from `registry.py` so a reader can see WHAT is registered here and follow
HOW one query is selected there. Nothing in this module executes a query, compiles a
grammar, or reads a file; it declares entries and the module-name families some of
them bind against.

Order is behavior. `_OBSERVED_QUERIES` preserves the tuple order, and
`observed_queries_for` preserves it for `run_observed_matches`. Do not
reorder entries in a behavior-preserving edit. `_registry_digest` sorts
by query id, so catalog reordering does not change the digest.
"""

from __future__ import annotations

from typing import Final

from outrider.policy.severity import FindingType
from outrider.queries.observed import BindingRule, ObservedQuery, QueryClass

__all__: tuple[str, ...] = ()

# ---------------------------------------------------------------------------
# OBSERVED-tier security query library (Cost Lever 3,
# specs/2026-06-14-observed-query-library-v1.md). These carry routing/output
# metadata (finding_type, class, title/description) the deterministic OBSERVED
# producer consumes; structural queries in registry.py do not. All are SIGNAL_ONLY in
# V1 (default-deny — they augment the LLM, never skip it). Their .scm bodies
# are loaded + compiled alongside the structural queries, so match() and
# get_query_source() resolve them; their metadata folds into the cache-key
# digest (DECISIONS.md#048 for the FindingTypes).
# ---------------------------------------------------------------------------

# Named binding-module families (the JS/TS catalog's BindingRule inputs).
# `_module_matches` is package-root aware, so subpath specifiers
# (`mysql2/promise`) satisfy their root entry; Node core modules must list
# BOTH spellings (`crypto` + `node:crypto`) — no scheme equivalence is
# computed (FUP-215(f)).
_NODE_CRYPTO_MODULES: Final[tuple[str, ...]] = ("crypto", "node:crypto")
_CHILD_PROCESS_MODULES: Final[tuple[str, ...]] = ("child_process", "node:child_process")
# Drivers whose query/execute surface is the SQL-injection sink family.
_SQL_DRIVER_MODULES: Final[tuple[str, ...]] = (
    "better-sqlite3",
    "knex",
    "mariadb",
    "mssql",
    "mysql",
    "mysql2",
    "oracledb",
    "pg",
    "pg-promise",
    "sequelize",
    "sqlite3",
    "typeorm",
)
# Two families that honor `rejectUnauthorized`: HTTP/TLS clients, and the
# DB/queue/mail clients that accept it inside `ssl:`/`tls:` connection
# options (`new Pool({ ssl: { rejectUnauthorized: false } })` is the
# canonical managed-PG MITM idiom).
_TLS_OPTION_CONSUMER_MODULES: Final[tuple[str, ...]] = (
    "amqplib",
    "axios",
    "got",
    "http2",
    "https",
    "ioredis",
    "knex",
    "mariadb",
    "mongodb",
    "mssql",
    "mysql",
    "mysql2",
    "node-fetch",
    "node:http2",
    "node:https",
    "node:tls",
    "nodemailer",
    "pg",
    "pg-promise",
    "redis",
    "request",
    "sequelize",
    "tls",
    "typeorm",
    "undici",
    "ws",
)

# Module-scope eligibility constraints (eligible => binding=None; eligible =>
# never SKIP_SAFE) are enforced at the SCHEMA floor by
# `ObservedQuery._enforce_module_scope_constraints` — the model validator fires
# for every construction path, including every entry of `_OBSERVED_QUERIES` at
# import, so no registry-load loop is needed.
_OBSERVED_QUERIES: Final[dict[str, ObservedQuery]] = {
    oq.query_match_id: oq
    for oq in (
        ObservedQuery(
            query_match_id="python.command_injection_subprocess_shell",
            filename="command_injection_subprocess_shell.scm",
            finding_type=FindingType.COMMAND_INJECTION,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="subprocess invoked with shell=True",
            description=(
                "A subprocess is run with shell=True; untrusted input in the "
                "command string enables shell command injection. Pass an "
                "argument list with shell=False, or sanitize the input."
            ),
        ),
        ObservedQuery(
            query_match_id="python.command_injection_os_system",
            filename="command_injection_os_system.scm",
            finding_type=FindingType.COMMAND_INJECTION,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="os.system / os.popen command execution",
            description=(
                "os.system and os.popen pass a string to the shell; untrusted "
                "input enables command injection. Prefer subprocess with an "
                "argument list."
            ),
        ),
        ObservedQuery(
            query_match_id="python.command_injection_eval_exec",
            filename="command_injection_eval_exec.scm",
            finding_type=FindingType.COMMAND_INJECTION,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="eval / exec on a dynamic expression",
            description=(
                "eval/exec runs a non-literal expression as code; untrusted "
                "input enables arbitrary code execution. Avoid dynamic eval/exec "
                "or constrain the input to a vetted set."
            ),
        ),
        ObservedQuery(
            query_match_id="python.unsafe_deserialization_pickle",
            filename="unsafe_deserialization_pickle.scm",
            finding_type=FindingType.UNSAFE_DESERIALIZATION,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="pickle deserialization of untrusted data",
            description=(
                "pickle.load/loads executes arbitrary code embedded in the "
                "payload; never unpickle attacker-controlled data. Use a safe "
                "format such as JSON."
            ),
        ),
        ObservedQuery(
            query_match_id="python.unsafe_deserialization_yaml",
            filename="unsafe_deserialization_yaml.scm",
            finding_type=FindingType.UNSAFE_DESERIALIZATION,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="yaml.load without a safe Loader",
            description=(
                "yaml.load without a safe Loader can construct arbitrary Python "
                "objects from the document; use yaml.safe_load or pass "
                "Loader=SafeLoader."
            ),
        ),
        ObservedQuery(
            query_match_id="python.sql_injection_string_concat",
            filename="sql_injection_string_concat.scm",
            finding_type=FindingType.SQL_INJECTION,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="SQL built by string formatting / concatenation",
            description=(
                "A SQL statement passed to execute is assembled with an "
                "f-string, concatenation, or .format(); untrusted input enables "
                "SQL injection. Use parameterized queries."
            ),
        ),
        ObservedQuery(
            query_match_id="python.tls_verify_disabled",
            filename="tls_verify_disabled.scm",
            finding_type=FindingType.TLS_VERIFY_DISABLED,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="TLS certificate verification disabled (verify=False)",
            description=(
                "verify=False disables certificate validation, exposing the "
                "request to man-in-the-middle attacks. Keep verification enabled "
                "against a proper CA."
            ),
        ),
        ObservedQuery(
            query_match_id="python.blocking_call_in_async",
            filename="blocking_call_in_async.scm",
            finding_type=FindingType.BLOCKING_CALL_IN_ASYNC,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Blocking call inside an async function",
            description=(
                "A blocking call (time.sleep, requests, open) inside async code "
                "stalls the event loop; use an async equivalent or run it in a "
                "thread executor."
            ),
        ),
        ObservedQuery(
            query_match_id="python.weak_crypto_broken_cipher",
            filename="weak_crypto_broken_cipher.scm",
            finding_type=FindingType.WEAK_CRYPTO,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Broken or legacy cipher construction",
            description=(
                "A construction of a broken or legacy cipher (DES, 3DES, RC2/ARC2, "
                "RC4/ARC4, Blowfish) is cryptographically weak. Use a modern "
                "authenticated cipher such as AES-GCM."
            ),
        ),
        ObservedQuery(
            query_match_id="python.weak_crypto_ecb_mode",
            filename="weak_crypto_ecb_mode.scm",
            finding_type=FindingType.WEAK_CRYPTO,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Cipher constructed in ECB mode",
            description=(
                "ECB mode encrypts identical plaintext blocks to identical "
                "ciphertext, leaking structure. Use an authenticated mode such "
                "as GCM, or CBC with a random IV and a MAC."
            ),
        ),
        ObservedQuery(
            query_match_id="python.weak_asymmetric_key_size",
            filename="weak_asymmetric_key_size.scm",
            finding_type=FindingType.WEAK_CRYPTO,
            language="python",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Weak asymmetric key size (RSA/DSA < 2048 bits)",
            description=(
                "An RSA or DSA key is generated with fewer than 2048 bits, which "
                "is below current guidance and factorable by well-resourced "
                "attackers. Use at least 2048 bits (3072+ for long-term keys), or "
                "an elliptic-curve key."
            ),
        ),
        # JS/TS catalog (specs/2026-07-03-js-ts-observed-query-catalog.md):
        # four families, existing FindingTypes reused (the vulnerability class
        # is language-independent — no SEVERITY_POLICY change), all
        # SIGNAL_ONLY per the same default-deny rule as the Python entries.
        # Name-anchored queries carry a BindingRule: the producer admits a
        # match only when the anchor identifier provably binds to the
        # dangerous API via the file's extracted imports (`_recv`/`_fn`
        # anchor-capture protocol) or, for derived-receiver sinks, when the
        # file imports a module from the documented set.
        ObservedQuery(
            query_match_id="javascript.weak_crypto_hash",
            filename="weak_crypto_hash.scm",
            finding_type=FindingType.WEAK_CRYPTO,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Weak hash algorithm (MD5/SHA-1) construction",
            description=(
                "crypto.createHash with md5 or sha1 constructs a hash that is "
                "unsafe for signatures, certificates, or integrity protection. "
                "Use SHA-256 or stronger."
            ),
            binding=BindingRule(mode="anchor_import", modules=_NODE_CRYPTO_MODULES),
        ),
        ObservedQuery(
            query_match_id="javascript.weak_crypto_broken_cipher",
            filename="weak_crypto_broken_cipher.scm",
            finding_type=FindingType.WEAK_CRYPTO,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Broken or legacy cipher construction",
            description=(
                "A cipher is constructed with a broken or legacy algorithm "
                "(DES, 3DES, RC2, RC4, Blowfish), which is cryptographically "
                "weak. Use a modern authenticated cipher such as AES-GCM."
            ),
            binding=BindingRule(mode="anchor_import", modules=_NODE_CRYPTO_MODULES),
        ),
        ObservedQuery(
            query_match_id="javascript.weak_crypto_ecb_mode",
            filename="weak_crypto_ecb_mode.scm",
            finding_type=FindingType.WEAK_CRYPTO,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Cipher constructed in ECB mode",
            description=(
                "ECB mode encrypts identical plaintext blocks to identical "
                "ciphertext, leaking structure. Use an authenticated mode such "
                "as GCM, or CBC with a random IV and a MAC."
            ),
            binding=BindingRule(mode="anchor_import", modules=_NODE_CRYPTO_MODULES),
        ),
        ObservedQuery(
            query_match_id="javascript.command_injection_child_process",
            filename="command_injection_child_process.scm",
            finding_type=FindingType.COMMAND_INJECTION,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Shell command built from dynamic strings",
            description=(
                "child_process.exec/execSync is invoked with a concatenated or "
                "template-interpolated command string; untrusted input enables "
                "shell command injection. Use execFile with an argument list, "
                "or validate the input against a strict allowlist."
            ),
            binding=BindingRule(mode="anchor_import", modules=_CHILD_PROCESS_MODULES),
        ),
        ObservedQuery(
            query_match_id="javascript.command_injection_eval",
            filename="command_injection_eval.scm",
            finding_type=FindingType.COMMAND_INJECTION,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="eval / new Function on a dynamic expression",
            description=(
                "eval or new Function runs a non-literal expression as code; "
                "untrusted input enables arbitrary code execution. Avoid "
                "dynamic code construction or constrain the input to a vetted "
                "set."
            ),
            # No binding (`eval`/`Function` are globals — no import exists to
            # bind), so the shadow guard carries the whole lexical proof: a
            # local `eval` parameter or `Function` mock resolves to the local
            # binding, not the global.
            shadow_guard=("Function", "eval"),
        ),
        ObservedQuery(
            query_match_id="javascript.sql_injection_string_concat",
            filename="sql_injection_string_concat.scm",
            finding_type=FindingType.SQL_INJECTION,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="SQL built by string concatenation / template interpolation",
            description=(
                "A SQL statement passed to query/execute is assembled with "
                "string concatenation or template-literal interpolation; "
                "untrusted input enables SQL injection. Use parameterized "
                "queries."
            ),
            # The receiver is a derived variable (a pool/connection), not the
            # import name — per-receiver proof needs assignment-flow. File-level
            # driver presence is the deterministic V1 gate.
            binding=BindingRule(mode="module_presence", modules=_SQL_DRIVER_MODULES),
        ),
        ObservedQuery(
            query_match_id="javascript.tls_verify_disabled",
            filename="tls_verify_disabled.scm",
            finding_type=FindingType.TLS_VERIFY_DISABLED,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="TLS certificate verification disabled",
            description=(
                "rejectUnauthorized: false disables certificate validation, "
                "exposing connections to man-in-the-middle attacks. Keep "
                "verification enabled against a proper CA."
            ),
            # The option object is usually built separately from the sink call;
            # file-level presence of a module that honors the option is the
            # deterministic V1 gate (families documented on the constant).
            binding=BindingRule(mode="module_presence", modules=_TLS_OPTION_CONSUMER_MODULES),
        ),
        ObservedQuery(
            query_match_id="javascript.tls_env_verify_disabled",
            filename="tls_env_verify_disabled.scm",
            finding_type=FindingType.TLS_VERIFY_DISABLED,
            language="javascript",
            query_class=QueryClass.SIGNAL_ONLY,
            title="Process-wide TLS verification disabled via environment",
            description=(
                'NODE_TLS_REJECT_UNAUTHORIZED="0" on process.env disables '
                "certificate validation for the entire process, exposing every "
                "connection to man-in-the-middle attacks. Remove the override "
                "and trust a proper CA."
            ),
            # No binding: the query itself constrains the receiver to the
            # `process.env` identifier chain, which needs no import. The
            # TEXT constraint is completed by the shadow guard: a local
            # `process` binding whose visibility span contains the match
            # is a mock/parameter, not the global — the producer denies it.
            shadow_guard=("process",),
            # The kill switch's canonical real-world form is module
            # top-level (index.js/config, before any connection) — the one
            # import-free, self-proving query the module-scope admission
            # arm exists for (DECISIONS.md#062).
            module_scope_eligible=True,
        ),
    )
}
