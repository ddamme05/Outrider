"""Authored controls for incomplete fixes and missing necessary evidence.

These expose small code mechanisms. They do not measure model quality or stand
in for held-out pull requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from outrider.agent.candidate_checks import CandidateText


@dataclass(frozen=True)
class CheckerCase:
    id: str
    file_path: str
    base_source: str
    head_source: str
    claim: CandidateText
    expected_status: Literal["supports", "refutes", "uncertain"]
    oracle: str


_BASE = (
    "def divide(value):\n"
    '    """Reject non-positive inputs with ValueError."""\n'
    "    return 10 / value\n"
)
_CLAIM = CandidateText(
    title="Zero input violates the validation contract",
    description=(
        "A zero input can reach the division and raise ZeroDivisionError "
        "instead of the documented ValueError."
    ),
    evidence="    return 10 / value",
    finding_type="missing_input_validation",
    line_start=5,
    line_end=5,
)

CHECKER_CASES = (
    CheckerCase(
        id="incomplete-zero-guard",
        file_path="ratio.py",
        base_source=_BASE,
        head_source=(
            "def divide(value):\n"
            '    """Reject non-positive inputs with ValueError."""\n'
            "    if value < 0:\n"
            "        raise ValueError('value must be positive')\n"
            "    return 10 / value\n"
        ),
        claim=_CLAIM,
        expected_status="supports",
        oracle="HEAD rejects negatives but still divides by zero; improving BASE is incomplete.",
    ),
    CheckerCase(
        id="complete-zero-guard",
        file_path="ratio.py",
        base_source=_BASE,
        head_source=(
            "def divide(value):\n"
            '    """Reject non-positive inputs with ValueError."""\n'
            "    if value <= 0:\n"
            "        raise ValueError('value must be positive')\n"
            "    return 10 / value\n"
        ),
        claim=_CLAIM,
        expected_status="refutes",
        oracle="HEAD rejects zero before division, closing this named failure.",
    ),
    CheckerCase(
        id="unknown-zero-guard",
        file_path="ratio.py",
        base_source=_BASE,
        head_source=(
            "def divide(value):\n"
            '    """Reject non-positive inputs with ValueError."""\n'
            "    if not accepted(value):\n"
            "        raise ValueError('value is not accepted')\n"
            "    return 10 / value\n"
        ),
        claim=_CLAIM,
        expected_status="uncertain",
        oracle=(
            "The accepted helper is not supplied; implementations that permit or reject zero "
            "are both compatible with this source."
        ),
    ),
)
