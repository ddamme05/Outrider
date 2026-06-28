# Tree-sitter query registry per
# specs/2026-04-30-ast-facts-module.md Internal contracts.
"""Query-id registry and execution surface.

Owns:
  * The `query_match_id` → query-body mapping (file-stem decoupled
    per Internal contracts: renaming a `.scm` file does not churn ids).
  * The compiled `tree_sitter.Query` cache (built at module load).
  * Two public functions:
      - `get_query_source(id) -> str` for documentation / audit-trail use.
      - `match(id, source) -> tuple[QueryMatchSpan, ...]` for replay
        and analyze-node use; returns fully domain-modeled results so
        no `tree_sitter.Query`/`Node`/`QueryCursor` ever leaves
        `queries/` per `docs/trust-boundaries.md` §4 (AST firewall).

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
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

from outrider.ast_facts.errors import UnknownQueryMatchId
from outrider.ast_facts.models import QueryCaptureSpan, QueryMatchSpan
from outrider.policy.severity import FindingType
from outrider.queries.observed import ObservedQuery, QueryClass
from outrider.queries.value_predicates import VALUE_PREDICATES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from outrider.queries.value_predicates import ValuePredicate

# ---------------------------------------------------------------------------
# Compiled language and parser (module-level singletons)
# ---------------------------------------------------------------------------

_PY_LANGUAGE: Final = Language(tree_sitter_python.language())
_PARSER: Final = Parser(_PY_LANGUAGE)


# ---------------------------------------------------------------------------
# Id → .scm filename mapping (file-stem decoupled per Internal contracts:
# the id is the authoritative name; filenames are implementation detail).
# ---------------------------------------------------------------------------

_QUERIES_DIR: Final = Path(__file__).parent / "python"

# `capture_quantifier(p, c)` returns the quantifier as a string:
# `''` = mandatory (one), `'+'` = one-or-more (also mandatory),
# `'?'` = zero-or-one, `'*'` = zero-or-more.
_MANDATORY_QUANTIFIERS: Final[frozenset[str]] = frozenset({"", "+"})

_QUERY_ID_TO_FILENAME: Final[dict[str, str]] = {
    "python.function_definition": "function_definition.scm",
    "python.class_definition": "class_definition.scm",
    "python.import_statement": "import_statement.scm",
    "python.import_from_statement": "import_from_statement.scm",
}

# V1: empty. Populated when a query's semantics change and a new id
# alongside the old one is needed for replay of historical reviews
# per Internal contracts.
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
_OBSERVED_QUERIES: Final[dict[str, ObservedQuery]] = {
    oq.query_match_id: oq
    for oq in (
        ObservedQuery(
            query_match_id="python.command_injection_subprocess_shell",
            filename="command_injection_subprocess_shell.scm",
            finding_type=FindingType.COMMAND_INJECTION,
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
            query_class=QueryClass.SIGNAL_ONLY,
            title="Weak asymmetric key size (RSA/DSA < 2048 bits)",
            description=(
                "An RSA or DSA key is generated with fewer than 2048 bits, which "
                "is below current guidance and factorable by well-resourced "
                "attackers. Use at least 2048 bits (3072+ for long-term keys), or "
                "an elliptic-curve key."
            ),
        ),
    )
}


# ---------------------------------------------------------------------------
# Module-load: read .scm files, compile queries, run mandatory-capture
# rejection per Internal contracts (every pattern must have at least one
# capture quantified `''` or `'+'`; optional-only `?`/`*` patterns reject).
# ---------------------------------------------------------------------------


def _load_and_compile() -> tuple[dict[str, str], dict[str, Query]]:
    bodies: dict[str, str] = {}
    compiled: dict[str, Query] = {}
    for query_id, filename in _QUERY_ID_TO_FILENAME.items():
        body = (_QUERIES_DIR / filename).read_text(encoding="utf-8")
        bodies[query_id] = body
        compiled[query_id] = _compile_and_validate(query_id, body, filename)
    # OBSERVED-tier security queries: same load + compile + mandatory-capture
    # validation. Their .scm bodies join _QUERY_BODIES/_COMPILED_QUERIES so
    # match()/get_query_source() resolve them like any other registered id.
    for query_id, observed in _OBSERVED_QUERIES.items():
        body = (_QUERIES_DIR / observed.filename).read_text(encoding="utf-8")
        bodies[query_id] = body
        compiled[query_id] = _compile_and_validate(query_id, body, observed.filename)
    # Deprecated bodies also compile and validate.
    for query_id, body in _DEPRECATED_QUERY_ID_TO_BODY.items():
        bodies[query_id] = body
        compiled[query_id] = _compile_and_validate(query_id, body, source="deprecated_ledger")
    return bodies, compiled


def _compile_and_validate(query_id: str, body: str, source: str | None = None) -> Query:
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
    query = Query(_PY_LANGUAGE, body)
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
    # than crashing at first match. `capture_quantifier(p, c)` raises
    # one of `(IndexError, ValueError, SystemError)` when capture c
    # isn't part of pattern p — the by-design negative case the narrow
    # `except` below handles. Anything outside that set propagates so
    # legitimate registry bugs (memory errors, future binding-version
    # surprises) aren't swallowed.
    for p in range(pattern_count):
        pattern_mandatory_count = 0
        for c in range(capture_count):
            try:
                quantifier = query.capture_quantifier(p, c)
            except (IndexError, ValueError, SystemError):
                # Narrow set: tree-sitter's binding raises one of these
                # when capture `c` isn't part of pattern `p` — the
                # by-design negative case, not an error to log.
                # Catching `Exception` would swallow legitimate registry
                # bugs (memory errors, future binding-version surprises).
                continue
            if quantifier in _MANDATORY_QUANTIFIERS:
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
    does a change to ANY OBSERVED output/routing field (today class /
    finding_type / title / description, but derived from the model minus
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
# fires for OBSERVED-tier admission. Deprecated ids are intentionally
# excluded — they exist for replay of historical reviews, NOT for live
# OBSERVED claims. Adding a new query to `_QUERY_ID_TO_FILENAME` extends
# this set automatically; deprecating an id moves it to
# `_DEPRECATED_QUERY_ID_TO_BODY` and removes it from this surface in the
# same commit, mirroring the registry's Internal-contracts split.
REGISTERED_QUERY_IDS: Final[frozenset[str]] = frozenset(_QUERY_ID_TO_FILENAME)


# Public surface: the OBSERVED-tier security queries + their routing/output
# metadata, consumed by the deterministic OBSERVED producer (analyze). This is
# a SEPARATE surface from `REGISTERED_QUERY_IDS` (the structural LLM-citation
# admission set per analyze `_build_query_match_id_set`) — the two query KINDS
# stay distinct. `match()`/`get_query_source()` still resolve OBSERVED ids
# (their bodies are in `_QUERY_BODIES`). `MappingProxyType` blocks runtime
# mutation, the same defense-in-depth as `SEVERITY_POLICY`.
OBSERVED_QUERIES: Final[Mapping[str, ObservedQuery]] = MappingProxyType(dict(_OBSERVED_QUERIES))
OBSERVED_QUERY_IDS: Final[frozenset[str]] = frozenset(_OBSERVED_QUERIES)


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


def match(query_match_id: str, source: bytes) -> tuple[QueryMatchSpan, ...]:
    """Run the named query against `source`; return domain-modeled spans.

    Empty tuple = registered query, zero matches against this source.
    Raises `UnknownQueryMatchId` if `query_match_id` is unknown.
    """
    if query_match_id not in _COMPILED_QUERIES:
        raise UnknownQueryMatchId(
            f"query_match_id {query_match_id!r} is not in the registry "
            f"(known ids: {sorted(_all_known_ids())})"
        )
    query = _COMPILED_QUERIES[query_match_id]
    tree = _PARSER.parse(source)

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
