"""Apply numeric checks to structural-query matches.

These checks use source facts, never model judgment. Their parameters enter
the registry digest; changing predicate behavior also requires a version bump.
See DECISIONS.md#057 and FUP-193.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from outrider.ast_facts.models import QueryMatchSpan

# Bump on ANY change to a predicate's evaluation logic that is not already
# encoded in a per-predicate parameter folded into its `contract_token`. The
# parameters below (e.g. the bit threshold) ARE in the token, so a threshold
# change moves the digest without a version bump; this version covers logic
# shape changes (e.g. switching the parse base, or the comparison operator).
VALUE_PREDICATE_CONTRACT_VERSION: Final = "v1"

# RSA/DSA keys below this many bits are weak (factorable by well-resourced
# attackers; below current NIST/industry guidance). Strict `<`: 2048 is the
# floor and is NOT flagged.
_RSA_DSA_MIN_SECURE_BITS: Final = 2048

# The capture every value-predicate'd key-size query binds to the integer
# literal under test. A registered predicate whose query does not produce this
# capture is a wiring bug and fails loud at first match (below).
_KEY_SIZE_CAPTURE: Final = "_keysize"


@dataclass(frozen=True)
class ValuePredicate:
    """A named deterministic filter applied to a query's matches post-structure.

    `evaluate(match, source) -> bool` returns True to KEEP the match (the
    finding fires) and False to DROP it. `contract_token` is the stable string
    folded into `QUERY_REGISTRY_DIGEST`: it encodes the predicate identity AND
    every parameter that changes its verdict, so a parameter edit invalidates
    cached analyze rows produced under the old verdict.
    """

    evaluate: Callable[[QueryMatchSpan, bytes], bool]
    contract_token: str


def _capture_text(match: QueryMatchSpan, source: bytes, name: str) -> str:
    """Return the source text of the single capture named `name`.

    Raises `ValueError` if the capture is absent — a predicate registered for a
    query that does not bind the expected capture is a wiring bug, caught loud
    at first match rather than silently dropping every finding. Decodes
    `strict`: `source` is already-validated UTF-8 (the analyze node decodes file
    content upstream), so a decode error is impossible in practice and would
    signal a corrupted byte envelope — fail loud rather than mask it with U+FFFD.
    """
    for capture in match.captures:
        if capture.name == name:
            return source[capture.byte_start : capture.byte_end].decode("utf-8")
    raise ValueError(
        f"value-predicate expected a capture named {name!r} but the match "
        f"bound only {sorted(c.name for c in match.captures)!r}; the query and "
        f"its predicate disagree on the captured literal."
    )


def _evaluate_weak_asymmetric_key_size(match: QueryMatchSpan, source: bytes) -> bool:
    """Keep iff the captured key-size literal is below the secure threshold.

    The `.scm` guarantees `_keysize` is an `(integer)` node, so the text is a
    valid Python integer literal; `int(text, 0)` honors `0x`/`0o`/`0b` prefixes
    and underscores. A literal that does not parse (only reachable via a
    future grammar quirk) drops conservatively — an OBSERVED finding must prove
    its claim, so an unevaluable size is not flagged.
    """
    text = _capture_text(match, source, _KEY_SIZE_CAPTURE)
    try:
        bits = int(text, 0)
    except ValueError:
        return False
    return bits < _RSA_DSA_MIN_SECURE_BITS


# query_match_id -> ValuePredicate. Only queries listed here are value-filtered;
# every other query's matches pass through `registry.match()` unchanged.
# MappingProxyType (the OBSERVED_QUERIES precedent): the public table is read-only,
# so an in-process mutation cannot drift the live `match()` filter from the
# import-pinned QUERY_REGISTRY_DIGEST (FUP-193 review).
VALUE_PREDICATES: Final[Mapping[str, ValuePredicate]] = MappingProxyType(
    {
        "python.weak_asymmetric_key_size": ValuePredicate(
            evaluate=_evaluate_weak_asymmetric_key_size,
            contract_token=(
                f"weak_asymmetric_key_size:min_secure_bits={_RSA_DSA_MIN_SECURE_BITS}"
                f":{VALUE_PREDICATE_CONTRACT_VERSION}"
            ),
        ),
    }
)
