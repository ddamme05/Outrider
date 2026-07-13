# See DECISIONS.md#070 — App-Manifest onboarding (setup state machine + signed state + config).
"""App-Manifest onboarding (`DECISIONS.md#070`).

The `database`-mode self-service setup surface: the durable singleton **state machine**
(`state_machine`), the HMAC-signed single-use **state** carrier (`state_token`) + its **nonce**
(`nonce`), and the onboarding **config** (`config`). The HTTP router + manifest builder + conversion
call land in a later slice and compose these; this package is the boot-safe engine + primitives.
"""

from __future__ import annotations

from outrider.api.setup.config import SetupSettings, validate_setup_config
from outrider.api.setup.nonce import hash_nonce, new_nonce
from outrider.api.setup.state_machine import (
    NONCE_TTL_SECONDS,
    SetupBinding,
    SetupConflictError,
    SetupIntegrityError,
    SetupNonceError,
    SetupStateMachine,
    SetupTransitionError,
)
from outrider.api.setup.state_token import (
    SETUP_STATE_SECRET_ENV,
    SetupStateError,
    SetupStateToken,
    sign_state,
    validate_setup_state_secret,
    verify_state,
)

__all__ = [
    "NONCE_TTL_SECONDS",
    "SETUP_STATE_SECRET_ENV",
    "SetupBinding",
    "SetupConflictError",
    "SetupIntegrityError",
    "SetupNonceError",
    "SetupSettings",
    "SetupStateError",
    "SetupStateMachine",
    "SetupStateToken",
    "SetupTransitionError",
    "hash_nonce",
    "new_nonce",
    "sign_state",
    "validate_setup_config",
    "validate_setup_state_secret",
    "verify_state",
]
