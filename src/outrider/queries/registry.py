# Tree-sitter query registry per
# specs/2026-04-30-ast-facts-module.md Internal contracts, as amended by
# DECISIONS.md#061 (the spec's "semantic change requires a new id" ledger
# trigger is superseded: ids name claims; the ledger fires on claim
# retirement, not on every semantic body edit).
"""Query-id registry and execution surface.

Owns:
  * The `query_match_id` → query-body mapping (file-stem decoupled
    per Internal contracts: renaming a `.scm` file does not churn ids).
  * The compiled `tree_sitter.Query` cache (built at module load),
    language-keyed since the JS/TS OBSERVED catalog
    (specs/2026-07-03-js-ts-observed-query-catalog.md): each query
    compiles under every grammar of its `QueryLanguage` — python
    queries under the python grammar; javascript-family queries under
    javascript, typescript, AND tsx (one catalog, three dialects).
  * The extension → (query language, grammar) selection maps, derived
    from the `ast_facts` registry's extension groups so query
    selection can never disagree with adapter dispatch. An extension
    absent from the maps means "registered language with no catalog":
    the selectors return None/empty and the OBSERVED producer stays
    inert — fail-SAFE by construction (no query set → no OBSERVED
    claim possible), which is why there is deliberately no totality
    assert here, unlike analyze's fence/trace-form tables where a
    missing entry would be a wrong-ADMISSION bug.
  * Public functions:
      - `get_query_source(id) -> str` for documentation / audit-trail use.
      - `match(id, source, grammar=...) -> tuple[QueryMatchSpan, ...]`
        for replay and analyze-node use; returns fully domain-modeled
        results so no `tree_sitter.Query`/`Node`/`QueryCursor` ever
        leaves `queries/` per `docs/trust-boundaries.md` §4 (AST
        firewall). The grammar picks the parser + compiled variant;
        a grammar the query was not compiled for raises (a python
        query can never run over JS bytes, and vice versa).
      - `query_language_for_path` / `grammar_for_path` /
        `observed_queries_for` / `structural_query_ids_for` — the
        per-file selection surface the analyze node consumes.

Mandatory-capture rejection runs at module-load time per Internal
contracts: a registered pattern with zero `@` captures, or with all
captures quantified as optional (`?`/`*`), has an undefined envelope
and raises `ValueError` at import, not at runtime. The check requires
at least one MANDATORY capture (quantifier `''` or `'+'`) per pattern.

Sort order per Internal contracts:
  * Within a match, captures are flattened sorted by
    `(byte_start, byte_end, name)` ascending.
  * Across matches, the returned tuple is sorted by
    `(byte_start, byte_end)` ascending, with a primitive-projection
    tiebreaker on the captures (Pydantic models lack `__lt__`).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Parser, Query, QueryCursor, Tree

from outrider.ast_facts.errors import UnknownQueryMatchId
from outrider.ast_facts.models import QueryCaptureSpan, QueryMatchSpan
from outrider.ast_facts.registry import (
    JAVASCRIPT_EXTENSIONS,
    PYTHON_EXTENSIONS,
    TYPESCRIPT_DIALECT_BY_EXTENSION,
)
from outrider.policy.severity import FindingType
from outrider.queries.observed import (
    ANCHOR_CAPTURE_PREFERENCE,
    GUARD_POSITION_CAPTURES,
    BindingRule,
    ObservedQuery,
    QueryClass,
    QueryLanguage,
)
from outrider.queries.value_predicates import VALUE_PREDICATES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from outrider.queries.value_predicates import ValuePredicate

# ---------------------------------------------------------------------------
# Compiled languages and parsers (module-level singletons, one per grammar).
# All grammars load eagerly at module import — deliberately NOT the ast_facts
# lazy-per-language discipline: the registry's Internal-contracts guarantee is
# that every registered query compiles + passes mandatory-capture validation
# at IMPORT, not at first runtime use, and per-language laziness would move
# that failure to mid-review. The grammar wheels are non-optional deps of the
# analyze path that imports this module, so there is no isolation to buy.
# ---------------------------------------------------------------------------

# The grammar that parses a file's bytes and keys a query's compiled variant.
# Finer than `QueryLanguage`: the "javascript" CATALOG runs under three
# grammars (javascript / typescript / tsx), one per dialect family.
GrammarKind = Literal["python", "javascript", "typescript", "tsx"]

_LANGUAGES: Final[dict[GrammarKind, Language]] = {
    "python": Language(tree_sitter_python.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language_typescript()),
    "tsx": Language(tree_sitter_typescript.language_tsx()),
}
_PARSERS: Final[dict[GrammarKind, Parser]] = {
    grammar: Parser(language) for grammar, language in _LANGUAGES.items()
}

# Which grammars compile each catalog language's queries. The JS/TS family is
# ONE catalog (`queries/javascript/`) compiled per dialect grammar — the
# probe-validated node shapes are identical across the three for every
# construct the catalog anchors on, and a future dialect-specific divergence
# surfaces as a compile failure at import, not a silent mismatch.
_GRAMMARS_BY_QUERY_LANGUAGE: Final[dict[QueryLanguage, tuple[GrammarKind, ...]]] = {
    "python": ("python",),
    "javascript": ("javascript", "typescript", "tsx"),
}

# Extension → catalog-language / grammar selection, derived from the
# ast_facts registry's extension groups (single source of truth for what
# each extension IS). Absence = registered-or-unknown language with NO
# catalog → selectors return None/empty → OBSERVED producer inert
# (fail-safe; see module docstring for why there is no totality assert).
_QUERY_LANGUAGE_BY_EXTENSION: Final[dict[str, QueryLanguage]] = {
    **dict.fromkeys(PYTHON_EXTENSIONS, "python"),
    **dict.fromkeys(JAVASCRIPT_EXTENSIONS, "javascript"),
    **dict.fromkeys(TYPESCRIPT_DIALECT_BY_EXTENSION, "javascript"),
}
_GRAMMAR_BY_EXTENSION: Final[dict[str, GrammarKind]] = {
    **dict.fromkeys(PYTHON_EXTENSIONS, "python"),
    **dict.fromkeys(JAVASCRIPT_EXTENSIONS, "javascript"),
    **TYPESCRIPT_DIALECT_BY_EXTENSION,
}

# Parse memo (FUP-182). `match()` -- and every OBSERVED / structural / trivial-scope
# sweep that calls it -- re-parsed byte-identical source per query: up to ~12
# full-file parses of one file per review. Memoize the parse so a clean file is
# parsed ONCE across all sweeps (the firewall keeps the parse inside `queries/`,
# so this registry-internal cache is the only place cross-sweep sharing can live).
# Keyed by (source, grammar) since the JS/TS catalog: the same bytes parsed
# under two grammars are two distinct trees. Bounded LRU so trees don't
# accumulate; sized for moderate concurrency (V1.5 parallel-analyze may tune it
# to its file fan-out). A tree-sitter `Tree` is immutable after parse, so reuse
# across QueryCursor runs -- including concurrent reads -- is safe;
# `lru_cache`'s own lock makes the memo itself thread-safe.
_PARSE_CACHE_SIZE: Final = 16


@lru_cache(maxsize=_PARSE_CACHE_SIZE)
def _parse_cached(source: bytes, grammar: GrammarKind) -> Tree:
    return _PARSERS[grammar].parse(source)


# ---------------------------------------------------------------------------
# Id → .scm filename mapping (file-stem decoupled per Internal contracts:
# the id is the authoritative name; filenames are implementation detail).
# One .scm tree per catalog language.
# ---------------------------------------------------------------------------

_QUERIES_DIR_BY_LANGUAGE: Final[dict[QueryLanguage, Path]] = {
    "python": Path(__file__).parent / "python",
    "javascript": Path(__file__).parent / "javascript",
}

# `capture_quantifier(p, c)` returns the quantifier as a string:
# `''` = mandatory (one), `'+'` = one-or-more (also mandatory),
# `'?'` = zero-or-one, `'*'` = zero-or-more.
_MANDATORY_QUANTIFIERS: Final[frozenset[str]] = frozenset({"", "+"})

# Structural queries (scope/import extraction citations), per catalog
# language. The javascript entry is deliberately EMPTY in this arc: the JS/TS
# catalog ships OBSERVED security queries only, so the structural
# LLM-citation admission set for a JS/TS file is the empty set — model
# OBSERVED claims on those files keep rejecting at admission exactly as the
# dispatch spec pinned, now by per-language registration instead of a
# hardcoded Python gate.
_STRUCTURAL_QUERY_FILES_BY_LANGUAGE: Final[dict[QueryLanguage, dict[str, str]]] = {
    "python": {
        "python.function_definition": "function_definition.scm",
        "python.class_definition": "class_definition.scm",
        "python.import_statement": "import_statement.scm",
        "python.import_from_statement": "import_from_statement.scm",
    },
    "javascript": {},
}

# Flat view (id → filename) retained for the public REGISTERED_QUERY_IDS
# union surface and the unknown-id error listing (`_all_known_ids`);
# load/compile iterates the per-language table directly.
_QUERY_ID_TO_FILENAME: Final[dict[str, str]] = {
    query_id: filename
    for files in _STRUCTURAL_QUERY_FILES_BY_LANGUAGE.values()
    for query_id, filename in files.items()
}

# Deprecated-id ledger (see DECISIONS.md#061): a query id names a CLAIM,
# not a byte-exact pattern. Claim-preserving precision edits evolve the
# `.scm` body IN PLACE under the stable id (`QUERY_REGISTRY_DIGEST` pins
# the body epoch; git history of the tracked `.scm` files is the body
# archive). This ledger is populated only on CLAIM RETIREMENT — an id
# whose claim stops being produced — so historical reviews' ids keep
# resolving for membership replay. A claim SPLIT mints a new id for the
# split-out claim and needs no ledger while the old id's narrowed claim
# is still produced (the tls_env precedent: both ids live, ledger empty).
# Future source-rematch replay (#031) needs a body-versioning surface
# before it can re-run old reviews under old bodies.
_DEPRECATED_QUERY_ID_TO_BODY: Final[dict[str, str]] = {}


# ---------------------------------------------------------------------------
# OBSERVED-tier security query library (Cost Lever 3,
# specs/2026-06-14-observed-query-library-v1.md). These carry routing/output
# metadata (finding_type, class, title/description) the deterministic OBSERVED
# producer consumes; structural queries above do not. All are SIGNAL_ONLY in
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
        ),
    )
}


# ---------------------------------------------------------------------------
# Module-load: read .scm files, compile queries, run mandatory-capture
# rejection per Internal contracts (every pattern must have at least one
# capture quantified `''` or `'+'`; optional-only `?`/`*` patterns reject).
# ---------------------------------------------------------------------------


def _language_for_query_id(query_id: str) -> QueryLanguage:
    """Catalog language named by `query_id`'s namespace prefix. The
    `<language>.<purpose>` id shape per Internal contracts makes the prefix
    the language; an id without a dot, without a purpose segment, or whose
    prefix names no known catalog language raises at import. This is the
    single decoding site for the id scheme — live registration compares
    against it, the deprecated ledger derives from it, so the two can
    never drift."""
    prefix, dot, purpose = query_id.partition(".")
    if not dot or not purpose or prefix not in _GRAMMARS_BY_QUERY_LANGUAGE:
        raise ValueError(
            f"query id {query_id!r} does not match '<language>.<purpose>' "
            f"with a known catalog language prefix "
            f"({sorted(_GRAMMARS_BY_QUERY_LANGUAGE)})."
        )
    return prefix


def _load_and_compile() -> tuple[dict[str, str], dict[str, dict[GrammarKind, Query]]]:
    bodies: dict[str, str] = {}
    compiled: dict[str, dict[GrammarKind, Query]] = {}

    def _register(query_id: str, language: QueryLanguage, filename: str) -> None:
        # Id-namespace guard: the language IS the id's namespace prefix
        # ("python.", "javascript."), decoded by `_language_for_query_id` —
        # a metadata/language mismatch, which would compile a query under
        # the wrong grammars and select it for the wrong files, fails loud
        # at import.
        if _language_for_query_id(query_id) != language:
            raise ValueError(
                f"query id {query_id!r} is registered with language "
                f"{language!r} but its namespace prefix disagrees; the id "
                f"prefix and the registered language must match."
            )
        body = (_QUERIES_DIR_BY_LANGUAGE[language] / filename).read_text(encoding="utf-8")
        bodies[query_id] = body
        compiled[query_id] = {
            grammar: _compile_and_validate(query_id, body, filename, grammar=grammar)
            for grammar in _GRAMMARS_BY_QUERY_LANGUAGE[language]
        }

    for language, files in _STRUCTURAL_QUERY_FILES_BY_LANGUAGE.items():
        for query_id, filename in files.items():
            _register(query_id, language, filename)
    # OBSERVED-tier security queries: same load + compile + mandatory-capture
    # validation. Their .scm bodies join _QUERY_BODIES/_COMPILED_QUERIES so
    # match()/get_query_source() resolve them like any other registered id.
    # anchor_import queries additionally validate the anchor-capture
    # protocol — the producer's admission depends on it (see
    # _validate_anchor_captures).
    for query_id, observed in _OBSERVED_QUERIES.items():
        _register(query_id, observed.language, observed.filename)
        if observed.binding is not None and observed.binding.mode == "anchor_import":
            for grammar, query in compiled[query_id].items():
                _validate_anchor_captures(query_id, query, grammar=grammar)
        # A `shadow_guard` global is only checked at guard-POSITION captures;
        # if the query pins the global under some other capture the guard is
        # silently inert, so require at least one guard-position capture at
        # load (mirrors _validate_anchor_captures — the producer's guard
        # depends on it).
        if observed.shadow_guard:
            for grammar, query in compiled[query_id].items():
                _validate_guard_position_captures(query_id, query, grammar=grammar)
    # Deprecated bodies also compile and validate, under the grammars of
    # the language named by the id's namespace prefix — the same
    # `_language_for_query_id` decode live registration uses, so the
    # ledger needs no separate language field and a malformed id (dot-less,
    # empty purpose, unknown prefix) fails as loud as a live one.
    for query_id, body in _DEPRECATED_QUERY_ID_TO_BODY.items():
        language = _language_for_query_id(query_id)
        bodies[query_id] = body
        compiled[query_id] = {
            grammar: _compile_and_validate(
                query_id, body, source="deprecated_ledger", grammar=grammar
            )
            for grammar in _GRAMMARS_BY_QUERY_LANGUAGE[language]
        }
    return bodies, compiled


def _capture_quantifier_or_none(query: Query, pattern_index: int, capture_index: int) -> str | None:
    """`capture_quantifier`, with the by-design negative collapsed to None:
    tree-sitter's binding raises one of `(IndexError, ValueError,
    SystemError)` when the capture isn't part of the pattern — the single
    place that exception set is encoded (both registry validators consume
    this), so a future binding-version surprise is a one-site fix. Anything
    outside that set propagates: legitimate registry bugs (memory errors)
    must not be swallowed.
    """
    try:
        return query.capture_quantifier(pattern_index, capture_index)
    except (IndexError, ValueError, SystemError):
        return None


def _validate_anchor_captures(query_id: str, query: Query, *, grammar: GrammarKind) -> None:
    """Enforce the anchor-capture protocol for `anchor_import` queries at
    import time. The producer anchors a match on a capture named in
    `ANCHOR_CAPTURE_PREFERENCE` (`@_recv` member receiver, else `@_fn` bare
    callee) and DEFAULT-DENIES a match with neither, so a `.scm` that typos
    the capture name (`@_fun`) — or a pattern shape that captures neither —
    would silently suppress 100% of that pattern's matches: no error, no
    telemetry, no failing test. Every pattern must reference an anchor
    capture; participation is probed per-pattern via
    `_capture_quantifier_or_none` (OPTIONAL participation counts, since
    alternation arms quantify `?`).

    Known limitation: the guarantee is per-PATTERN, not per-alternation-ARM
    — the query API doesn't expose arms. A pattern mixing anchored arms
    with an anchorless one passes here, while matches via the anchorless
    arm carry no anchor capture and are default-denied at admission
    (empirically confirmed). When adding an arm to an anchor_import query,
    ensure the arm itself captures `_fn` or `_recv`.
    """
    capture_count = cast("int", query.capture_count)
    anchor_indexes = tuple(
        c for c in range(capture_count) if query.capture_name(c) in ANCHOR_CAPTURE_PREFERENCE
    )
    for p in range(cast("int", query.pattern_count)):
        if not any(_capture_quantifier_or_none(query, p, c) is not None for c in anchor_indexes):
            raise ValueError(
                f"Query {query_id!r} ({grammar}) has binding mode 'anchor_import' "
                f"but pattern {p} captures neither '_fn' nor '_recv'; the "
                f"producer would default-deny every match of that pattern "
                f"silently. Fix the .scm capture names or the BindingRule mode."
            )


def _validate_guard_position_captures(query_id: str, query: Query, *, grammar: GrammarKind) -> None:
    """Enforce that a `shadow_guard` query captures a global at a
    guard-POSITION (`GUARD_POSITION_CAPTURES` — callee/receiver identifier).
    The producer's `_guarded_global_shadowed` only tests a guarded name
    against those captures, so a query that `#eq?`-pins its global under a
    different capture name (`@_target`) would have a silently-inert guard —
    the shadowing FP the guard exists to close reopens with no failing
    test. Every pattern must reference a guard-position capture;
    participation is probed per-pattern via `_capture_quantifier_or_none`,
    mirroring `_validate_anchor_captures` — a query-wide presence check
    would let one well-captured pattern mask a mis-captured sibling whose
    matches carry no guard-position capture (silent-partial failure).

    Known limitation: the guarantee is per-PATTERN, not per-alternation-ARM
    — the query API doesn't expose arms. When adding an arm to a
    shadow_guard query, ensure the arm itself captures a guard-position
    name.
    """
    capture_count = cast("int", query.capture_count)
    guard_indexes = tuple(
        c for c in range(capture_count) if query.capture_name(c) in GUARD_POSITION_CAPTURES
    )
    for p in range(cast("int", query.pattern_count)):
        if not any(_capture_quantifier_or_none(query, p, c) is not None for c in guard_indexes):
            raise ValueError(
                f"Query {query_id!r} ({grammar}) declares a shadow_guard but pattern "
                f"{p} captures no guard-position identifier "
                f"({sorted(GUARD_POSITION_CAPTURES)}); the producer's shadow guard "
                f"would be silently inert for that pattern's matches. Capture the "
                f"guarded global under one of those names, or drop the shadow_guard."
            )


def _compile_and_validate(
    query_id: str, body: str, source: str | None = None, *, grammar: GrammarKind = "python"
) -> Query:
    """Compile a query body and reject any pattern lacking a mandatory capture.

    Per Internal contracts: every registered pattern MUST produce at
    least one capture per match (envelope rule). A pattern with zero
    captures, or with all captures quantified as optional (`?`/`*`),
    has an undefined envelope and would crash `match(...)` at runtime
    when `min()` sees empty captures.

    Validation walks each pattern via tree-sitter's per-pattern
    introspection (`capture_quantifier(pattern_index, capture_index)`
    raises when the capture isn't part of that pattern). Multi-pattern
    files are permitted — the envelope rule applies per-pattern, not
    per-file. Single-pattern is the V1 convention but not enforced.
    """
    where = f" (loaded from {source})" if source else ""
    query = Query(_LANGUAGES[grammar], body)
    # tree-sitter's type stubs declare these as `Callable[[], int]` but
    # at runtime they're int attributes — cast for mypy.
    pattern_count = cast("int", query.pattern_count)
    capture_count = cast("int", query.capture_count)
    if pattern_count < 1:
        raise ValueError(
            f"Query {query_id!r}{where} has pattern_count=0; the body "
            f"must define at least one pattern."
        )
    # Per-pattern check: each pattern must have at least one MANDATORY
    # capture. Optional quantifiers (`'?'`/`'*'`) might fire zero times
    # at runtime, leaving an empty captures tuple, which crashes
    # `QueryMatchSpan`'s envelope `min`/`max` over empty captures.
    # Per Internal contracts' optional-captures residual edge: V1's
    # non-empty-match guarantee depends on mandatory captures. A pattern
    # whose captures are ALL optional fails registration here rather
    # than crashing at first match. Absent-from-pattern captures probe
    # None via `_capture_quantifier_or_none` (the shared by-design
    # negative; see its docstring for the exception-set rationale).
    for p in range(pattern_count):
        pattern_mandatory_count = 0
        for c in range(capture_count):
            quantifier = _capture_quantifier_or_none(query, p, c)
            if quantifier is not None and quantifier in _MANDATORY_QUANTIFIERS:
                pattern_mandatory_count += 1
        if pattern_mandatory_count < 1:
            raise ValueError(
                f"Query {query_id!r}{where} pattern {p} has no "
                f"mandatory captures (all captures are optional/star "
                f"quantified). The envelope rule per Internal contracts "
                f"(specs/2026-04-30-ast-facts-module.md) requires every "
                f"registered pattern to produce at least one capture "
                f"per match; optional-only patterns might fire with "
                f"empty captures at runtime."
            )
    return query


_QUERY_BODIES, _COMPILED_QUERIES = _load_and_compile()

# Every value-predicate MUST key an OBSERVED query id. A typo'd or stale key
# would silently no-op — the OBSERVED producer iterates only OBSERVED_QUERIES, so
# a predicate keyed to a structural or deprecated id (both in _COMPILED_QUERIES)
# would pass a registered-id check yet never run in the producer path, AND its
# contract_token would still fold into the digest under a non-OBSERVED key. Scope
# the guard to OBSERVED ids so it matches the producer + the test invariant
# (FUP-193 audit sweep + code-review fold).
_unknown_predicate_ids = set(VALUE_PREDICATES) - set(_OBSERVED_QUERIES)
if _unknown_predicate_ids:
    raise ValueError(
        f"value-predicate(s) keyed to non-OBSERVED query id(s): "
        f"{sorted(_unknown_predicate_ids)}. Every VALUE_PREDICATES key must be a "
        f"registered OBSERVED query id; queries/value_predicates.py and the "
        f"registry disagree."
    )


# Fields of `ObservedQuery` EXCLUDED from the digest fold (FUP-181). The digest
# derives the folded set from the model (not a hardcoded tuple), so a FUTURE
# output- or routing-affecting field auto-folds into the cache key. `query_match_id`
# is the key (already folded as the id); `filename` is an impl detail — the .scm
# BODY is folded, so renaming a .scm must NOT move the digest. A new field not in
# this set folds by default; `test_digest_excluded_fields_pinned` guards the set.
_DIGEST_EXCLUDED_OBSERVED_FIELDS: Final[frozenset[str]] = frozenset({"query_match_id", "filename"})


def _registry_digest(
    bodies: dict[str, str],
    observed: Mapping[str, ObservedQuery],
    value_predicates: Mapping[str, ValuePredicate],
) -> str:
    """Length-prefixed SHA-256 over the sorted (id, body) pairs PLUS the
    routing-and-output metadata of OBSERVED queries PLUS any value-predicate
    contract token.

    The analyze-cache key component that pins query SEMANTICS AND emitted
    output. A pattern edit that keeps its id changes this digest — AND so
    does a change to ANY OBSERVED output/routing field (today language /
    class / finding_type / title / description, but derived from the model minus
    `_DIGEST_EXCLUDED_OBSERVED_FIELDS` so a future field auto-folds, FUP-181),
    each of which alters routing or the emitted finding (and the cached
    payload) while living OUTSIDE the `.scm` body. Folding them
    here keeps cached OBSERVED findings from being served under metadata
    that no longer produced them (specs/2026-06-11-file-hash-analyze-cache.md
    + FUP-166; the Cost Lever 3 round-3 review; `DECISIONS.md#048`).

    A value-predicate's `contract_token` is folded the same way (DECISIONS.md#057):
    the token encodes the predicate identity and every verdict-changing *parameter*
    (e.g. the key-size threshold), so a threshold change invalidates cached analyze
    rows. NOTE the asymmetry vs the `.scm` bodies: the body bytes are hashed
    verbatim (any edit auto-moves the digest), but only the predicate's *token* is
    hashed, not its function source — a change to the predicate's evaluation LOGIC
    that is not encoded in a token parameter requires a manual
    `VALUE_PREDICATE_CONTRACT_VERSION` bump (the `SHAPER_CONTRACT_VERSION`
    discipline), not an automatic move.

    Length-prefixing makes the field boundaries unambiguous — the
    `llm/base.py::_canonical_prompt_hash` precedent. Covers deprecated
    ledger bodies too: strictly safer, and ledger changes are rare.
    """
    h = hashlib.sha256()
    for query_id in sorted(bodies):
        qid_bytes = query_id.encode("utf-8")
        body_bytes = bodies[query_id].encode("utf-8")
        h.update(f"{len(qid_bytes)}:".encode())
        h.update(qid_bytes)
        h.update(f"{len(body_bytes)}:".encode())
        h.update(body_bytes)
        # OBSERVED queries fold their routing-and-output metadata: a change to
        # any of these alters emitted findings / routing without touching the
        # .scm body, so it must move the digest (invalidate stale cache). Derived
        # from the model (NOT a hardcoded tuple) so a future field auto-folds
        # (FUP-181); excludes `_DIGEST_EXCLUDED_OBSERVED_FIELDS`. Sorted +
        # field-named + length-prefixed for an order-stable, unambiguous fold;
        # json.dumps serializes any field type (enum->value, str, future int/list)
        # deterministically.
        oq = observed.get(query_id)
        if oq is not None:
            dumped = oq.model_dump(mode="json")
            for field_name in sorted(dumped):
                if field_name in _DIGEST_EXCLUDED_OBSERVED_FIELDS:
                    continue
                field = f"{field_name}={json.dumps(dumped[field_name], sort_keys=True)}"
                field_bytes = field.encode("utf-8")
                h.update(f"{len(field_bytes)}:".encode())
                h.update(field_bytes)
        # A value-predicate alters which matches survive (and the cached
        # payload) without touching the .scm body — fold its contract token.
        vp = value_predicates.get(query_id)
        if vp is not None:
            token_bytes = vp.contract_token.encode("utf-8")
            h.update(f"{len(token_bytes)}:".encode())
            h.update(token_bytes)
    return h.hexdigest()


# Code-pinned at module load from the actual compiled sources — never
# injectable, so the recorded digest cannot drift from the queries that
# actually ran (the TRIVIAL_FILTER_VERSION adjacency precedent).
QUERY_REGISTRY_DIGEST: Final[str] = _registry_digest(
    _QUERY_BODIES, _OBSERVED_QUERIES, VALUE_PREDICATES
)


def _all_known_ids() -> set[str]:
    return set(_QUERY_ID_TO_FILENAME) | set(_OBSERVED_QUERIES) | set(_DEPRECATED_QUERY_ID_TO_BODY)


# Public surface: the set of `query_match_id` strings the analyze node
# fires for OBSERVED-tier admission — the ALL-LANGUAGES union; per-file
# selection goes through `structural_query_ids_for`. Deprecated ids are
# intentionally excluded — they exist for replay of historical reviews, NOT
# for live OBSERVED claims. Adding a new query to
# `_STRUCTURAL_QUERY_FILES_BY_LANGUAGE` extends this set automatically;
# deprecating an id moves it to `_DEPRECATED_QUERY_ID_TO_BODY` and removes
# it from this surface in the same commit, mirroring the registry's
# Internal-contracts split.
REGISTERED_QUERY_IDS: Final[frozenset[str]] = frozenset(_QUERY_ID_TO_FILENAME)

_STRUCTURAL_QUERY_IDS_BY_LANGUAGE: Final[dict[QueryLanguage, frozenset[str]]] = {
    language: frozenset(files) for language, files in _STRUCTURAL_QUERY_FILES_BY_LANGUAGE.items()
}


# Public surface: the OBSERVED-tier security queries + their routing/output
# metadata, consumed by the deterministic OBSERVED producer (analyze). This is
# a SEPARATE surface from `REGISTERED_QUERY_IDS` (the structural LLM-citation
# admission set per analyze `_build_query_match_id_set`) — the two query KINDS
# stay distinct. `match()`/`get_query_source()` still resolve OBSERVED ids
# (their bodies are in `_QUERY_BODIES`). `MappingProxyType` blocks runtime
# mutation, the same defense-in-depth as `SEVERITY_POLICY`. All-languages
# union — the producer selects per file via `observed_queries_for`.
OBSERVED_QUERIES: Final[Mapping[str, ObservedQuery]] = MappingProxyType(dict(_OBSERVED_QUERIES))
OBSERVED_QUERY_IDS: Final[frozenset[str]] = frozenset(_OBSERVED_QUERIES)

_EMPTY_OBSERVED: Final[Mapping[str, ObservedQuery]] = MappingProxyType({})


# ---------------------------------------------------------------------------
# Per-file selection surface (JS/TS OBSERVED catalog spec). All four helpers
# treat an unmapped extension as "no catalog": None / empty — the fail-safe
# direction (no query set → no OBSERVED claim possible for that file).
# ---------------------------------------------------------------------------


def query_language_for_path(path: str) -> QueryLanguage | None:
    """Catalog language for `path`'s extension, or None when no catalog
    covers it (unregistered or catalog-less language). Case-insensitive,
    matching the ast_facts registry's extension normalization."""
    return _QUERY_LANGUAGE_BY_EXTENSION.get(PurePosixPath(path).suffix.lower())


def grammar_for_path(path: str) -> GrammarKind | None:
    """Grammar that parses `path`'s bytes for query execution, or None when
    no catalog covers the extension. Finer than the catalog language: `.ts`
    selects the javascript CATALOG but the typescript GRAMMAR."""
    return _GRAMMAR_BY_EXTENSION.get(PurePosixPath(path).suffix.lower())


def observed_queries_for(language: QueryLanguage | None) -> Mapping[str, ObservedQuery]:
    """OBSERVED queries registered for `language`; empty mapping for None.

    Derived from the module's `OBSERVED_QUERIES` attribute AT CALL TIME, not
    an import-time snapshot: `OBSERVED_QUERIES` is the single authoritative
    observed surface, and the observed_skip_safe scenario's documented
    test-local promotion seam (monkeypatching that attribute) must reach the
    producer's per-language view too. The registry stays small (tens of
    entries) — the per-call filter is negligible next to the parse it
    precedes.
    """
    if language is None:
        return _EMPTY_OBSERVED
    return {qid: oq for qid, oq in OBSERVED_QUERIES.items() if oq.language == language}


def structural_query_ids_for(language: QueryLanguage | None) -> frozenset[str]:
    """Structural (LLM-citation admission) query ids for `language`; empty
    for None. Empty is a meaningful state — a language whose catalog ships
    no structural queries (javascript today) admits no model OBSERVED
    claim, preserving the dispatch-era rejection behavior by registration
    rather than by hardcoded gate. A language missing from the structural
    table entirely (a future catalog language whose entry hasn't landed)
    also selects empty — the same fail-safe direction as the extension
    maps (no query set → no OBSERVED claim), never a mid-review KeyError."""
    if language is None:
        return frozenset()
    return _STRUCTURAL_QUERY_IDS_BY_LANGUAGE.get(language, frozenset())


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def get_query_source(query_match_id: str) -> str:
    """Return the raw `.scm` body for a registered or deprecated id.

    Raises `UnknownQueryMatchId` if the id is not in either ledger.
    """
    if query_match_id not in _QUERY_BODIES:
        raise UnknownQueryMatchId(
            f"query_match_id {query_match_id!r} is not in the registry "
            f"(known ids: {sorted(_all_known_ids())})"
        )
    return _QUERY_BODIES[query_match_id]


def match(
    query_match_id: str, source: bytes, *, grammar: GrammarKind = "python"
) -> tuple[QueryMatchSpan, ...]:
    """Run the named query against `source` parsed under `grammar`; return
    domain-modeled spans.

    `grammar` selects the parser AND the query's compiled variant (default
    "python" — the sole grammar before the JS/TS catalog; callers for JS/TS
    files pass `grammar_for_path(path)`). Empty tuple = registered query,
    zero matches against this source. Raises `UnknownQueryMatchId` if
    `query_match_id` is unknown, and `ValueError` if the query's language
    was not compiled for `grammar` — a python query can never run over
    JS/TS bytes (or vice versa); a mismatch is a caller gate bug, never a
    silent empty result.
    """
    if query_match_id not in _COMPILED_QUERIES:
        raise UnknownQueryMatchId(
            f"query_match_id {query_match_id!r} is not in the registry "
            f"(known ids: {sorted(_all_known_ids())})"
        )
    variants = _COMPILED_QUERIES[query_match_id]
    if grammar not in variants:
        raise ValueError(
            f"query {query_match_id!r} has no compiled variant for grammar "
            f"{grammar!r} (compiled for: {sorted(variants)}). Running a "
            f"query over bytes of another language would produce "
            f"error-recovery garbage matches; the caller's language "
            f"selection is wrong."
        )
    query = variants[grammar]
    tree = _parse_cached(source, grammar)

    raw_matches: list[QueryMatchSpan] = []
    for _pattern_index, captures in QueryCursor(query).matches(tree.root_node):
        # captures: dict[str, list[Node]] per Month 0 spike findings
        # (canonical docs say bare Node; runtime returns list[Node]).
        flat: list[QueryCaptureSpan] = []
        for capture_name, nodes in captures.items():
            for node in nodes:
                flat.append(
                    QueryCaptureSpan(
                        name=capture_name,
                        byte_start=node.start_byte,
                        byte_end=node.end_byte,
                    )
                )
        # Per Internal contracts: sort captures by (byte_start, byte_end, name).
        flat.sort(key=lambda c: (c.byte_start, c.byte_end, c.name))
        capture_tuple = tuple(flat)
        # Envelope per Internal contracts.
        envelope_start = min(c.byte_start for c in capture_tuple)
        envelope_end = max(c.byte_end for c in capture_tuple)
        raw_matches.append(
            QueryMatchSpan(
                byte_start=envelope_start,
                byte_end=envelope_end,
                captures=capture_tuple,
            )
        )

    # Value-predicate filter: an OBSERVED query may carry a deterministic
    # post-structure filter (queries/value_predicates.py) that drops matches
    # whose captured literal fails a numeric test tree-sitter's native
    # predicates cannot express (e.g. RSA key size >= 2048). Most queries have
    # no predicate and pass through unchanged. The predicate reads only the
    # QueryMatchSpan + source bytes (no raw node), so the AST firewall is
    # unaffected; its contract_token rides into QUERY_REGISTRY_DIGEST so a
    # threshold (parameter) change invalidates cached analyze rows -- a
    # predicate-LOGIC change instead needs a manual VALUE_PREDICATE_CONTRACT_VERSION
    # bump (the token, not the function source, is hashed). See
    # DECISIONS.md#057 + docs/trust-boundaries.md §1.
    predicate = VALUE_PREDICATES.get(query_match_id)
    if predicate is not None:
        raw_matches = [m for m in raw_matches if predicate.evaluate(m, source)]

    # Sort matches by (byte_start, byte_end) with captures-projection tiebreaker
    # per Internal contracts (Pydantic models lack a default `__lt__`).
    def _sort_key(m: QueryMatchSpan) -> tuple[int, int, tuple[tuple[int, int, str], ...]]:
        cap_proj = tuple((c.byte_start, c.byte_end, c.name) for c in m.captures)
        return (m.byte_start, m.byte_end, cap_proj)

    raw_matches.sort(key=_sort_key)
    return tuple(raw_matches)
