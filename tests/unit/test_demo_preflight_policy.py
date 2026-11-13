"""Keep the shell-only demo preflight aligned with runtime secret policy."""

import re
from pathlib import Path

from outrider.policy.secret_placeholders import SECRET_PLACEHOLDERS


def test_demo_preflight_rejects_runtime_secret_placeholders() -> None:
    preflight = Path(__file__).resolve().parents[2] / "deploy" / "up.sh"
    # The literal arm also rejects empty; deployment must stay usable without
    # importing the Python application on the host.
    rejection_arms = re.findall(r'^\s*""\|([a-z0-9|-]+)\)\s*$', preflight.read_text(), re.MULTILINE)
    assert len(rejection_arms) == 1, "expected one explicit placeholder-rejection arm"
    assert frozenset(rejection_arms[0].split("|")) == SECRET_PLACEHOLDERS
