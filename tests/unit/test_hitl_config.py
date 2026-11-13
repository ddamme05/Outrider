"""Check approval timeout settings and rejection of automatic posting on expiry.

Startup must reject auto_post and nonpositive timeouts so unapproved HIGH or
CRITICAL findings cannot post on expiry (``hitl-gates-high-severity``;
docs/trust-boundaries.md §6)."""

import pytest
from pydantic import ValidationError

from outrider.agent.nodes.hitl_config import HITLConfig, HITLTimeoutAction


def test_default_config_is_expire_only_30min() -> None:
    config = HITLConfig()
    assert config.timeout_action is HITLTimeoutAction.EXPIRE_ONLY
    assert config.timeout_minutes == 30


def test_expire_only_constructs() -> None:
    config = HITLConfig(timeout_action=HITLTimeoutAction.EXPIRE_ONLY, timeout_minutes=1)
    assert config.timeout_action is HITLTimeoutAction.EXPIRE_ONLY
    assert config.timeout_minutes == 1


def test_auto_post_is_rejected() -> None:
    """The absolute no-auto-post guarantee — AUTO_POST must fail construction (V1)."""
    with pytest.raises(ValidationError, match="not supported in V1"):
        HITLConfig(timeout_action=HITLTimeoutAction.AUTO_POST)


@pytest.mark.parametrize("bad_minutes", [0, -1, -30])
def test_non_positive_timeout_minutes_rejected(bad_minutes: int) -> None:
    """gt=0: zero/negative would expire every gate immediately under expire_only."""
    with pytest.raises(ValidationError):
        HITLConfig(timeout_minutes=bad_minutes)
