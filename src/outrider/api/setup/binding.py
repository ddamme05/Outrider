# See DECISIONS.md#070 — conversion-response binding verification (verify-then-activate).
"""Verify a manifest conversion response against the bound attempt (`DECISIONS.md#070`).

Before persisting the credentials from a `POST /app-manifests/{code}/conversions` response, the
callback verifies the returned App matches the contract bound to THIS setup attempt (recorded on
`setup_state` at Start, returned by `consume_callback` as a `SetupBinding`). Only the
**response-verifiable** fields are checked — the three GitHub returns:

- `owner.login` **case-normalized-equals** `expected_org_login` (case-insensitive),
- `permissions` **exactly equals** `expected_permissions` (nothing wider — this is what stops
  a stolen `state` paired with a substituted `code` for a broader-permission App),
- the `events` **set equals** `expected_events` (order-independent).

`public` and the manifest URLs are submission-only (the conversion response omits them, so they
cannot be response-verified); the single-use nonce in the signed `state` is what binds the callback
to this attempt, and `manifest_contract_digest` is a recorded audit artifact of what was submitted
(V1 does not re-verify against it). A mismatch on the three checked fields raises
`BindingMismatchError`; the caller rejects — never persists — and routes to `ORPHANED`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from outrider.api.setup.state_machine import SetupBinding

__all__ = ["BindingMismatchError", "verify_conversion_binding"]


class BindingMismatchError(RuntimeError):
    """The conversion response's owner / permissions / events do not match the attempt binding — the
    signature of a stolen `state` paired with a substituted `code` (a different owner or wider
    permissions). Fail closed: reject, do not persist, route to `ORPHANED`."""


def verify_conversion_binding(
    *,
    owner_login: str,
    permissions: Mapping[str, str],
    events: list[str],
    binding: SetupBinding,
) -> None:
    """Raise `BindingMismatchError` unless the conversion response matches the bound attempt on all
    three response-verifiable fields. A `None` expectation (should never occur post-`CONVERTING`) is
    treated as a mismatch — fail closed."""
    expected_org = binding.expected_org_login
    if expected_org is None or owner_login.casefold() != expected_org.casefold():
        raise BindingMismatchError(
            f"conversion owner.login {owner_login!r} does not match the bound org {expected_org!r}"
        )
    expected_perms = binding.expected_permissions
    if expected_perms is None or dict(permissions) != dict(expected_perms):
        raise BindingMismatchError(
            "conversion permissions do not match the bound contract "
            f"(got {dict(permissions)!r}, expected {binding.expected_permissions!r})"
        )
    if binding.expected_events is None or set(events) != set(binding.expected_events):
        raise BindingMismatchError(
            "conversion events do not match the bound contract "
            f"(got {sorted(events)!r}, expected {sorted(binding.expected_events or [])!r})"
        )
