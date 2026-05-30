"""Demo fixture for the live-Claude smoke (`--diff-file`).

A realistic async HTTP request handler with two deliberate, unambiguous flaws that
the analyze stage (Sonnet) should flag, both sub-HIGH so they publish without
tripping the HITL gate:

  1. a blocking `time.sleep()` inside an `async def` (stalls the event loop)
     -> finding type `blocking_call_in_async` -> MEDIUM
  2. request input used without validation -> `missing_input_validation` -> MEDIUM

What gets this past TRIAGE is the diff itself: triage reads each file's full
diff content (the patch hunks, code-fenced into the triage prompt) plus its
path and +/- counts, and tiers it DEEP/STANDARD/SKIM on what the code does.
A substantial, clearly security-relevant handler diff earns DEEP/STANDARD and
reaches analyze; a trivially small diff reads as low-risk and gets SKIM'd
(analyze never examines SKIM) regardless of filename. This fixture is sized
and written to look like real handler logic for that reason.

Deliberately NO raw-SQL string formatting, secrets, auth, or path traversal here —
those map to CRITICAL/HIGH and would interrupt at the HITL gate (demoed
separately). This file is demo input, not production code: it is intentionally
flawed.
"""

import time


async def handle_search_request(query: str, limit: str) -> dict[str, object]:
    # limit comes straight off the request and is used unvalidated below —
    # no bounds check, no int coercion guard, no allowlist on `query`.
    page_size = int(limit)

    # Blocking sleep inside an async handler: stalls the whole event loop for
    # every concurrent request, not just this one.
    time.sleep(0.2)

    results = [{"id": i, "match": query} for i in range(page_size)]
    return {"query": query, "count": len(results), "results": results}
