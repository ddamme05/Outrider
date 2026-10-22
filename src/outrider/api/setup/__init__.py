# See DECISIONS.md#070 — App-Manifest onboarding (setup state machine + signed state + config).
"""App-Manifest onboarding primitives (DECISIONS.md#070).

router.py composes the manifest, conversion wrapper, and durable state machine.
state_token.py signs callback state; nonce.py supplies the hashed, single-use
nonce consumed by the machine. config.py validates database-mode setup settings.
mount.py wires the router before startup dependencies and the SPA catch-all.
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
