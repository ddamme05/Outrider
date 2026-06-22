"""Demo fixture for the live-Claude smoke (`--diff-file`): the OBSERVED-tier path.

A token cipher that builds DES in ECB mode — DES is a broken 64-bit-block cipher
and ECB leaks plaintext structure (identical blocks → identical ciphertext). The
point of this fixture is that the weakness is caught DETERMINISTICALLY by the
tree-sitter OBSERVED queries, not only by the model: `weak_crypto_broken_cipher`
fires on the `DES.new(...)` construction and `weak_crypto_ecb_mode` on `DES.MODE_ECB`.

So the finding carries `evidence_tier=OBSERVED` and a replay-verifiable
`query_match_id` — the structural proof a third party can re-check, not the model's
opinion. It maps to `FindingType.WEAK_CRYPTO` → HIGH. This is the fixture that
demos the OBSERVED tier and prefer-OBSERVED dedup (`DECISIONS.md#053` dual-mode
taxonomy → `#054` prefer-OBSERVED): if the model ALSO flags this line as a JUDGED
`weak_crypto`, the deterministic proof wins and the OBSERVED finding (with its
`query_match_id`) is the one that survives.

Note this fixture does NOT exercise cross-type subsumption (`#055`): that fires for
md5/password-style cases where a more-specific JUDGED type (`weak_password_hash`)
subsumes `weak_crypto`. DES/ECB token encryption is not password hashing, so the
only `SUBSUMES` pair doesn't apply here — see `scripts/demo_fixtures/` if a
subsumption demo is wanted later.

Deliberately a single, clean weak-crypto site — no SQLi/secrets/auth mixed in — so
the finding the demo surfaces is exactly one `weak_crypto`. This file is demo
input, not production code: it is intentionally flawed.
"""

from Crypto.Cipher import DES


class TokenCipher:
    """Encrypts short opaque session tokens before they leave the service.

    The symmetric key is injected so the class stays unit-testable; in production
    it would be fetched from a KMS. The cipher CHOICE is the planted weakness.
    """

    def __init__(self, key: bytes) -> None:
        self._key = key

    def encrypt_token(self, plaintext: bytes) -> bytes:
        """Encrypt a token payload.

        DES is a broken cipher (64-bit blocks, brute-forceable in hours) and ECB
        mode reveals structure across blocks. Replace with AES-256-GCM.
        """
        cipher = DES.new(self._key, DES.MODE_ECB)  # noqa: S304  (intentional: demo weak-crypto fixture)
        return cipher.encrypt(plaintext)
