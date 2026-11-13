"""Build shared display sections for findings without formatting a channel.

Labels and text remain raw here. GitHub and Slack renderers escape them for
their own formats; this package performs no I/O or severity-policy decisions.
"""

from outrider.presentation.finding_sections import (
    FindingSections,
    build_finding_sections,
)

__all__ = ["FindingSections", "build_finding_sections"]
