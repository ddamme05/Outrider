#!/usr/bin/env python3
"""Smoke-test extract_invariants.py's validation paths.

Not a real pytest suite — just confirms the validator rejects the inputs
we care about before we trust it on the real spec.

NOTE ON TEST SUITE MAINTENANCE:
The "bare N.N without § is NOT a violation" case below is load-bearing.
It proves SECTION_RE doesn't over-fire on legitimate decimals (0.9, 0.75,
0.5, 1.5, "Python 3.13", etc.). The confidence-is-computed-not-assigned
invariant in the real spec literally contains "0.9, 0.75, 0.5" in its
rule field. Without that guard test, a future refactor tightening
SECTION_RE could pass all other cases and still break extraction on the
real spec. Do not remove that case when pruning the test suite.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent / "extract_invariants.py"

# Track failures so the script can set an exit code. Without this,
# `python test_extractor.py` exits 0 even when cases fail, and the
# pre-commit hook that runs this file would rubber-stamp broken
# extractor changes. Plain module state is fine here; this is a flat
# script, not a pytest suite.
_failures = 0


def run_extractor(spec_text: str) -> tuple[int, str, str, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = Path(tmpdir) / "spec.md"
        out = Path(tmpdir) / "invariants.md"
        spec.write_text(spec_text)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--spec", str(spec), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        output = out.read_text() if out.exists() else ""
        return result.returncode, result.stdout, result.stderr, output


def case(
    name: str,
    spec: str,
    expect_code: int,
    expect_stderr_contains: str = "",
    expect_output_contains: str = "",
) -> None:
    global _failures
    code, stdout, stderr, output = run_extractor(spec)
    ok = (
        code == expect_code
        and expect_stderr_contains in stderr
        and expect_output_contains in output
    )
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        _failures += 1
        print(f"       expected exit {expect_code}, got {code}")
        if expect_stderr_contains:
            print(f"       expected stderr to contain: {expect_stderr_contains!r}")
        if expect_output_contains:
            print(f"       expected output to contain: {expect_output_contains!r}")
        print(f"       actual stderr: {stderr.strip()[:400]}")
        print(f"       actual stdout: {stdout.strip()[:200]}")
        print(f"       actual output: {output.strip()[:400]}")


# --- Happy path: one valid tag in one section ---
case(
    "happy path: single valid tag",
    spec="""# Title

## 1. Section one

Some prose.

## 2. Section two

<!-- invariant:id=test-invariant
     rule: Something must always be true.
     violation: Something is false.
-->

More prose.
""",
    expect_code=0,
)

# --- Section number embedded in tag (the big one) ---
case(
    "reject: section number in rule field",
    spec="""## 1. First

<!-- invariant:id=bad
     rule: As per \u00a77.4, do the thing.
     violation: Not doing the thing.
-->
""",
    expect_code=1,
    expect_stderr_contains="hardcoded section reference",
)

# --- LOAD-BEARING: proves SECTION_RE doesn't over-fire on decimals. ---
# See module docstring note. Do not remove even if this file gets pruned.
case(
    "reject: bare N.N without § is NOT a violation (legitimate decimal)",
    spec="""## 1. First

<!-- invariant:id=with-decimal
     rule: Confidence values are 0.9, 0.75, 0.5 — not set by model.
     violation: confidence hardcoded as 0.85.
-->
""",
    expect_code=0,
)

case(
    "reject: 'section X.Y' prose form is caught",
    spec="""## 1. First

<!-- invariant:id=bad
     rule: Always X.
     violation: Not X, see section 7.4 for context.
-->
""",
    expect_code=1,
    expect_stderr_contains="hardcoded section reference",
)

# --- Tag before any heading ---
case(
    "reject: tag in preamble",
    spec="""<!-- invariant:id=early
     rule: Always X.
     violation: Not X.
-->

## 1. First

Prose.
""",
    expect_code=1,
    expect_stderr_contains="not inside any numbered",
)

# --- Duplicate IDs ---
case(
    "reject: duplicate id",
    spec="""## 1. First

<!-- invariant:id=dup
     rule: A.
     violation: Not A.
-->

## 2. Second

<!-- invariant:id=dup
     rule: B.
     violation: Not B.
-->
""",
    expect_code=1,
    expect_stderr_contains="Duplicate invariant id",
)

# --- Missing required field ---
case(
    "reject: missing rule field",
    spec="""## 1. First

<!-- invariant:id=bad
     violation: Something.
-->
""",
    expect_code=1,
    expect_stderr_contains="missing required field",
)

# --- Tag inside code fence is skipped ---
case(
    "skip: tag inside code fence is ignored",
    spec="""## 1. First

Here's an example of the tag format:

```markdown
<!-- invariant:id=example-in-code-block
     rule: This is just documentation.
     violation: N/A.
-->
```

## 2. Second

<!-- invariant:id=real-invariant
     rule: Something real.
     violation: Not real.
-->
""",
    expect_code=0,
)

# --- Security-critical label applied ---
case(
    "accept: security:critical field produces label",
    spec="""## 1. First

<!-- invariant:id=sec-test
     rule: Always compare_digest.
     violation: Using == for HMAC.
     security: critical
-->
""",
    expect_code=0,
    expect_output_contains="[security-critical]",
)

# --- Check field passes through ---
case(
    "accept: check field is optional and preserved",
    spec="""## 1. First

<!-- invariant:id=with-check
     rule: No subprocess imports.
     violation: Having one.
     check: grep -r "import subprocess" src/ should be empty
-->
""",
    expect_code=0,
    expect_output_contains="**Check.**",
)

# --- H3 and H4 also count as containing headings ---
case(
    "accept: H3 and H4 both valid containers",
    spec="""## 1. First

### 1.1 Subsection

<!-- invariant:id=in-h3
     rule: A.
     violation: Not A.
-->

#### 1.1.1 Sub-subsection

<!-- invariant:id=in-h4
     rule: B.
     violation: Not B.
-->
""",
    expect_code=0,
)

# --- Unclosed code fence fails loud (was: silent content drop) ---
case(
    "unclosed code fence at EOF fails loud",
    spec="# Title\n\n## 1. Section\n\nprose\n\n```\nfile ends inside this fence block\n",
    expect_code=1,
    expect_stderr_contains="Unclosed code fence",
)

print("\ndone.")
sys.exit(1 if _failures else 0)
