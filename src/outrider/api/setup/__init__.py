"""Create and connect a GitHub App for this deployment.

router.py handles setup requests; state_machine.py saves progress; state_token.py
and nonce.py protect the callback. mount.py installs the setup routes before
the dashboard fallback. See DECISIONS.md#070.
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
