"""Remove a broader finding when a more specific one covers the same range.

The declared rules determine which types can replace others; removal must not
lower severity. These rules participate in the analysis-cache identity.
See DECISIONS.md#021, #055.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from outrider.policy.severity import SEVERITY_POLICY, FindingSeverity, FindingType

if TYPE_CHECKING:
    from collections.abc import Mapping

# Wrapped in `MappingProxyType` so runtime mutation raises TypeError — same
# defense-in-depth shape as `SEVERITY_POLICY` / `FINDING_TYPE_TO_DIMENSION`.
# KEY = the more-specific subsumer; VALUE = the broader types it subsumes on
# the same line span. Read `WEAK_PASSWORD_HASH: {WEAK_CRYPTO}` as
# "weak_password_hash ⊐ weak_crypto".
SUBSUMES: Final[Mapping[FindingType, frozenset[FindingType]]] = MappingProxyType(
    {
        FindingType.WEAK_PASSWORD_HASH: frozenset({FindingType.WEAK_CRYPTO}),
    }
)


def subsumes(subsumer: FindingType, subsumed: FindingType) -> bool:
    """True iff `subsumer` is declared to subsume `subsumed` on the same line.

    Directional: `subsumes(WEAK_PASSWORD_HASH, WEAK_CRYPTO)` is True, the
    reverse is False. The analyze merge calls this as
    `subsumes(Y.finding_type, X.finding_type)` — does admitted finding `Y`
    subsume producer finding `X`.
    """
    return subsumed in SUBSUMES.get(subsumer, frozenset())


# FindingSeverity is declared most-severe-first (CRITICAL ... INFO), so the
# enum's declaration index IS the rank: lower index = more severe. Deriving
# the rank from the enum (rather than a separate map) means it cannot drift
# from FindingSeverity. Used only by the monotonicity guard below.
def _severity_rank(severity: FindingSeverity) -> int:
    return list(FindingSeverity).index(severity)


def verify_subsumption_wellformed() -> None:
    """Assert the `SUBSUMES` relation is well-formed at import time.

    Fails loud at app startup / test collection (the deterministic floor that
    fires even when `git commit --no-verify` bypasses CI), parallel to
    `policy/dimensions.py::verify_lockstep`. Public (no leading underscore)
    because `outrider/__init__.py` imports + calls it as a load-bearing entry
    point. Four checks:

    1. Enum membership (NOT total enum coverage — the map is sparse): every
       key and every subsumed member is a real `FindingType`, so a renamed /
       removed enum value can't silently drift the map.
    2. Irreflexivity: no type subsumes itself.
    3. Single-hop only: a subsumed type must not itself be a subsumer — this
       forbids both 2-cycles (A⊐B, B⊐A) and chains (A⊐B, B⊐C). V1's single-pass
       merge applies only directly-declared edges, never the transitive closure,
       so a chain would silently under-apply.
    4. Severity-monotonicity: the subsumer's policy severity is >= each
       subsumed type's, so subsumption can never LOWER severity (which would
       let a CRITICAL be masked by a "more specific" MEDIUM).
    """
    valid = set(FindingType)
    keys = set(SUBSUMES)
    for subsumer, subsumed_set in SUBSUMES.items():
        if subsumer not in valid:
            raise AssertionError(f"SUBSUMES key {subsumer!r} is not a FindingType member.")
        for subsumed in subsumed_set:
            if subsumed not in valid:
                raise AssertionError(
                    f"SUBSUMES[{subsumer.value!r}] member {subsumed!r} is not a FindingType."
                )
            if subsumed == subsumer:
                raise AssertionError(
                    f"SUBSUMES irreflexivity violation: {subsumer.value!r} subsumes itself."
                )
            # Single-hop only: a subsumed type must not itself be a subsumer.
            # This forbids 2-cycles (A⊐B, B⊐A) AND chains (A⊐B, B⊐C) — the
            # single-pass merge applies only directly-declared edges, so a chain
            # would silently under-apply rather than take the transitive closure.
            if subsumed in keys:
                raise AssertionError(
                    f"SUBSUMES is single-hop: {subsumed.value!r} is subsumed (by "
                    f"{subsumer.value!r}) yet is itself a subsumer; chains and cycles "
                    "are forbidden in V1."
                )
            if _severity_rank(SEVERITY_POLICY[subsumer]) > _severity_rank(
                SEVERITY_POLICY[subsumed]
            ):
                raise AssertionError(
                    "SUBSUMES severity-monotonicity violation: subsumer "
                    f"{subsumer.value!r} ({SEVERITY_POLICY[subsumer].value}) is LESS severe "
                    f"than subsumed {subsumed.value!r} ({SEVERITY_POLICY[subsumed].value}); "
                    "a subsumption must never lower severity."
                )


verify_subsumption_wellformed()


# SHA-256 over the canonicalized relation (sorted child + sorted parents),
# folded into the analyze cache key so any edge edit auto-invalidates cached
# rows (DECISIONS.md#055). The QUERY_REGISTRY_DIGEST idiom: digest the table
# so cached admitted_findings can't outlive a semantics change.
def _subsumes_digest() -> str:
    canonical = sorted(
        (subsumer.value, sorted(s.value for s in subsumed_set))
        for subsumer, subsumed_set in SUBSUMES.items()
    )
    payload = json.dumps(canonical, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


SUBSUMES_DIGEST: Final[str] = _subsumes_digest()
