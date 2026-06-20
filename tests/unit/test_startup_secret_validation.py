"""Startup secret-validation helpers (AUDIT M1/M2).

`require_truncation_secret` (output_sanitizer) and `validate_token_enc_key`
(token_crypto) are the eager, boot-time validators the lifespan calls so a missing
or placeholder secret fails LOUD at startup instead of lazily — mid-review for the
truncation HMAC, or at the first OAuth callback / decrypt for the enc key.
"""

from __future__ import annotations

import pytest

from outrider.notify.token_crypto import (
    TOKEN_ENC_KEY_ENV,
    TokenCryptoError,
    validate_token_enc_key,
)
from outrider.policy.output_sanitizer import (
    TRUNCATION_HMAC_SECRET_ENV,
    require_truncation_secret,
)


def test_require_truncation_secret_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRUNCATION_HMAC_SECRET_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TRUNCATION_HMAC_SECRET_ENV):
        require_truncation_secret()


def test_require_truncation_secret_passes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUNCATION_HMAC_SECRET_ENV, "a-real-deploy-secret")
    require_truncation_secret()  # no raise


def test_validate_token_enc_key_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENC_KEY_ENV, raising=False)
    with pytest.raises(TokenCryptoError, match=TOKEN_ENC_KEY_ENV):
        validate_token_enc_key()


@pytest.mark.parametrize("placeholder", ["replace-me", "REPLACE-ME", "change-me", "secret"])
def test_validate_token_enc_key_rejects_placeholder(
    monkeypatch: pytest.MonkeyPatch, placeholder: str
) -> None:
    """A present-but-placeholder enc key is named explicitly (not a generic Fernet
    error) — the M2 fix that stops an uncommented .env.example `replace-me` from
    looking configured."""
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, placeholder)
    with pytest.raises(TokenCryptoError, match="placeholder"):
        validate_token_enc_key()


def test_validate_token_enc_key_rejects_malformed_non_fernet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present-but-non-Fernet key (not in the placeholder set) still fails closed."""
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, "not-a-valid-fernet-key")
    with pytest.raises(TokenCryptoError):
        validate_token_enc_key()
