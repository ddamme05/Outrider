"""Check that GitHub returned the App requested by this setup attempt.

verify_conversion_binding checks organization, permissions and subscribed
events before credentials can be saved. GitHub does not return all submitted
manifest fields, so those fields cannot be checked here. The callback separately
checks the attempt's signed state and manifest digest.

See DECISIONS.md#070.
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
