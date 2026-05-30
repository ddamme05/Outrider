"""Demo fixture for the live-Claude smoke (`scripts/live_claude_smoke.py --diff-file`).

A deliberately, unambiguously flawed function: a blocking `time.sleep()` inside an
`async def`, which stalls the event loop. The natural finding type is
`blocking_call_in_async`, which `SEVERITY_POLICY` maps to MEDIUM — sub-HIGH, so it
publishes without tripping the HITL gate (the gate is demoed separately, on
purpose). No secrets, URLs, or user-input handling here, so it won't read as a
CRITICAL/HIGH vuln by accident.

This file is demo input, not production code — it is intentionally wrong.
"""

import time


async def process_batch(items: list[int]) -> list[int]:
    results: list[int] = []
    for item in items:
        results.append(item * 2)
        time.sleep(0.5)  # blocking sleep inside an async function stalls the event loop
    return results
