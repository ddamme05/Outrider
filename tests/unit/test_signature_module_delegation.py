"""Confirm `api/webhooks/signature.py::verify_signature` delegates to
`outrider.github.webhooks.verify_webhook_signature` with arguments
forwarded unchanged.

Closes the two-module path so neither half can silently drift from the
other.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from outrider.api.webhooks.signature import verify_signature


def test_delegates_to_github_wrapper_with_arguments_unchanged() -> None:
    """`verify_signature(secret, body, header)` calls the wrapper with
    the exact same three arguments in the same order."""
    secret = "test-secret"  # noqa: S105 — test fixture, not a credential
    body = b'{"action": "opened"}'
    header = "sha256=abcd1234"

    with patch(
        "outrider.api.webhooks.signature.verify_webhook_signature",
        return_value=True,
    ) as mock_verify:
        result = verify_signature(secret, body, header)

    assert result is True
    mock_verify.assert_called_once_with(secret, body, header)


def test_returns_false_from_wrapper() -> None:
    """When the wrapper returns False, `verify_signature` returns False."""
    with patch(
        "outrider.api.webhooks.signature.verify_webhook_signature",
        return_value=False,
    ):
        result = verify_signature("s", b"b", "sha256=xyz")  # noqa: S106 — test
    assert result is False


def test_propagates_wrapper_exceptions() -> None:
    """If the wrapper raises (malformed header, etc.), the exception
    propagates — caller treats it as 401 per the route discipline."""

    class _FakeWebhookError(Exception):
        pass

    def _raise(*args: Any, **kwargs: Any) -> bool:
        raise _FakeWebhookError("malformed")

    with patch(
        "outrider.api.webhooks.signature.verify_webhook_signature",
        side_effect=_raise,
    ):
        import pytest

        with pytest.raises(_FakeWebhookError):
            verify_signature("s", b"b", "sha256=invalid")  # noqa: S106 — test
