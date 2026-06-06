#!/usr/bin/env python3
"""Smoke tests for check_decision_refs.py.

Cases:
  (a) happy path — all references resolve
  (b) unresolved reference in prose — should fail
  (c) unresolved reference inside a fenced code block — should pass (block skipped)
  (d) Supersedes #NNN to a valid entry — should pass
  (e) Superseded by #NNN to a missing entry — should fail
  (f) format-example block with illustrative numbers not in the entry list — should pass
  (g) multiple errors reported together — should fail with correct count
  (h) no references at all — should pass
  (i) duplicate ## NNN. headers — should fail
  (j) illustrative header inside code block — should pass (regression test for
      the real DECISIONS.md scenario where the Supersession format-example
      block contains a "## 007. ..." header that would otherwise be flagged
      as duplicating the real #007 entry)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent / "check_decision_refs.py"

_failures = 0


def run_checker(content: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(tmp)],
        capture_output=True,
        text=True,
    )
    tmp.unlink()
    return result.returncode, result.stdout, result.stderr


def case(
    name: str,
    content: str,
    expect_code: int,
    expect_stderr_contains: str = "",
) -> None:
    global _failures
    code, stdout, stderr = run_checker(content)
    ok = code == expect_code and expect_stderr_contains in stderr
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        _failures += 1
        print(f"       expected exit {expect_code}, got {code}")
        if expect_stderr_contains:
            print(f"       expected stderr to contain: {expect_stderr_contains!r}")
        print(f"       stderr: {stderr.strip()[:400]}")
        print(f"       stdout: {stdout.strip()[:200]}")


# (a) happy path
case(
    "happy path: all references resolve",
    content="""## 001. First entry

Some prose (see #002).

## 002. Second entry

Status: Accepted. Supersedes #001.
""",
    expect_code=0,
)

# (b) unresolved reference in prose
case(
    "fail: prose reference to non-existent entry",
    content="""## 001. First entry

See #099 for context.
""",
    expect_code=1,
    expect_stderr_contains="#099",
)

# (c) unresolved reference inside code block — should pass
case(
    "pass: reference inside fenced code block is ignored",
    content="""## 001. First entry

Example format:

```
## 099. Celery for background execution

**Status:** Accepted. Supersedes #098.
```

Real prose with no references.
""",
    expect_code=0,
)

# (d) Supersedes to a valid entry
case(
    "pass: Supersedes reference to valid entry",
    content="""## 001. First decision

Real content.

## 002. Second decision

Status: Accepted, 2026-04-22. Supersedes #001.
""",
    expect_code=0,
)

# (e) Superseded by to a missing entry
case(
    "fail: Superseded by reference to missing entry",
    content="""## 001. First decision

Status: Superseded by #007, 2026-04-22.
""",
    expect_code=1,
    expect_stderr_contains="#007",
)

# (f) DECISIONS.md-style format block with illustrative numbers
case(
    "pass: format-example code block with fake numbers is skipped",
    content="""## 001. How entries work

See below for the supersession format.

```
## 007. Celery for durable background execution

**Status:** Accepted, 2026-08-15. Supersedes #003.
```

Prose after the block with no references.
""",
    expect_code=0,
)

# (g) multiple errors reported together
case(
    "fail: multiple unresolved references all reported",
    content="""## 001. First

See #050 and also #099.
""",
    expect_code=1,
    expect_stderr_contains="2 unresolved",
)

# (h) no references at all — should pass
case(
    "pass: file with no cross-references",
    content="""## 001. First

No references here. Just prose.

## 002. Second

Also no references.
""",
    expect_code=0,
)

# (i) duplicate header numbers — should fail
case(
    "fail: duplicate ## NNN. headers are rejected",
    content="""## 001. First decision

Content.

## 001. Second decision with same number

Content.
""",
    expect_code=1,
    expect_stderr_contains="duplicate",
)

# (j) illustrative header inside code block should NOT trigger duplicate check
# Regression test: the real DECISIONS.md contains a "## 007. Celery..." example
# in a fenced code block within the Supersession section. If headers are
# collected without skipping code blocks, that example registers as a duplicate
# of the real #007 entry and the checker fails on a correct file.
case(
    "pass: illustrative header inside code block is not a duplicate",
    content="""## 007. Real entry seven

Real content.

## Supersession

Format example:

```
## 007. Example illustrative entry

**Status:** Accepted. Supersedes #001.
```

More prose.

## 008. Real entry eight

Supersedes #007.
""",
    expect_code=0,
)

# --- Unclosed code fence fails loud (was: silent cross-reference skip) ---
case(
    "unclosed code fence at EOF fails loud",
    content="# Decisions\n\n## 001. Foo\n\nprose\n\n```\nfile ends inside this fence block\n",
    expect_code=1,
    expect_stderr_contains="Unclosed code fence",
)

print("\ndone.")
sys.exit(1 if _failures else 0)
