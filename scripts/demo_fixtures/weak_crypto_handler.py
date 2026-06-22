"""Symmetric encryption for short opaque session tokens.

Provides `TokenCipher`, a thin wrapper that encrypts session-token payloads before
they leave the service boundary. The key is supplied by the caller (typically
resolved from the KMS at startup) so the handler itself holds no key material at
rest and stays straightforward to wire into the token-issuance path.

Tokens are short, fixed-shape byte payloads, so the handler operates on a single
block-cipher pass without padding negotiation. Callers are responsible for
base64-encoding the ciphertext for transport.
"""

from Crypto.Cipher import DES


class TokenCipher:
    """Encrypts short opaque session tokens before they leave the service.

    The symmetric key is injected at construction so callers control its lifecycle
    and source it from the KMS rather than embedding it here.
    """

    def __init__(self, key: bytes) -> None:
        self._key = key

    def encrypt_token(self, plaintext: bytes) -> bytes:
        """Encrypt a token payload and return the raw ciphertext bytes.

        The payload length must be a multiple of the cipher block size; token
        payloads are pre-sized by the caller, so no padding is applied here.
        """
        cipher = DES.new(self._key, DES.MODE_ECB)
        return cipher.encrypt(plaintext)
