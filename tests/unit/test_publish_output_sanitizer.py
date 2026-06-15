# Tests for policy/output_sanitizer.py per spec §V publish sanitizer.
"""Pin the V1 output sanitizer.

Covers the five concerns named in the module docstring:
  1. Codepoint stripping (bidi, ANSI, NUL).
  2. Fence neutralization for fenced content.
  3. Markdown-semantic neutralization for prose.
  4. Size cap in UTF-8 bytes (with non-ASCII coverage).
  5. HMAC-tagged truncation marker (incl. fake-marker stripping).
"""

from __future__ import annotations

import pytest

from outrider.policy.output_sanitizer import (
    GITHUB_COMMENT_BODY_MAX,
    GITHUB_REVIEW_BODY_MAX,
    TRUNCATION_HMAC_SECRET_ENV,
    apply_size_cap,
    compute_truncation_hmac,
    escape_markdown_prose,
    is_safe_suggestion_replacement,
    render_fenced_block,
    sanitize_display_string,
    strip_fake_truncation_markers,
)


@pytest.fixture(autouse=True)
def _set_truncation_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope `TRUNCATION_HMAC_SECRET_ENV` to test execution.

    Sibling discipline to `test_publish_idempotency.py` /
    `test_publish_routing.py` / `test_publish_node_end_to_end.py`:
    module-level `os.environ.setdefault(...)` leaks across test
    modules and can mask missing-env failures elsewhere. Tests that
    deliberately mutate the env in-test can still use their own
    `monkeypatch` to override this fixture's setting.
    """
    monkeypatch.setenv(TRUNCATION_HMAC_SECRET_ENV, "test-secret-for-unit-tests")


# ---------------------------------------------------------------------------
# 1. Codepoint stripping
# ---------------------------------------------------------------------------


def test_strip_trojan_source_bidi_override() -> None:
    """Bidi-override codepoints (CVE-2021-42574) stripped from prose."""
    text = 'if access\u202e)⁦{⁩⁦ // "admin"'
    cleaned = sanitize_display_string(text)
    # Each bidi codepoint must be gone.
    for bidi in "\u202e⁦⁩":
        assert bidi not in cleaned


def test_strip_ansi_csi_color_sequence() -> None:
    """ANSI CSI color escape stripped (terminal-renderer attack defense)."""
    text = "before\x1b[31mRED\x1b[0mafter"
    cleaned = sanitize_display_string(text)
    assert "\x1b" not in cleaned
    assert "before" in cleaned
    assert "RED" in cleaned
    assert "after" in cleaned


def test_strip_ansi_osc_hyperlink_sequence() -> None:
    """ANSI OSC hyperlink (`ESC ] 8 ; ... BEL ... ESC ] 8 ;; BEL`) stripped."""
    text = "click \x1b]8;;https://evil.example\x07here\x1b]8;;\x07 now"
    cleaned = sanitize_display_string(text)
    assert "\x1b" not in cleaned
    assert "\x07" not in cleaned
    assert "here" in cleaned


def test_strip_nul_byte() -> None:
    """NUL bytes stripped (some renderers truncate at NUL)."""
    text = "before\x00after"
    cleaned = sanitize_display_string(text)
    assert "\x00" not in cleaned
    assert "before" in cleaned
    assert "after" in cleaned


def test_strip_lone_surrogate() -> None:
    """Lone surrogates dropped so UTF-8 encoding always succeeds downstream."""
    text = "before\ud800after"  # Lone high surrogate
    cleaned = sanitize_display_string(text)
    # Must encode without raising.
    cleaned.encode("utf-8")
    assert "\ud800" not in cleaned


def test_strip_zero_width_joiner() -> None:
    """Zero-width-joiner stripped (homoglyph attack defense)."""
    text = "looks‍like‍this"
    cleaned = sanitize_display_string(text)
    assert "‍" not in cleaned


# ---------------------------------------------------------------------------
# 2. Fence neutralization for code-fenced content
# ---------------------------------------------------------------------------


def test_render_fenced_block_default_3_backticks_when_content_has_none() -> None:
    """Outer fence is 3 backticks (floor) when content has no backtick run."""
    block = render_fenced_block("x = 1\n", language="python")
    assert "```python" in block
    assert block.count("```") == 2  # opening + closing


def test_render_fenced_block_grows_fence_to_outrun_inner_backticks() -> None:
    """Fence count = max(longest backtick run in content, 2) + 1.

    Content with 4 backticks needs a 5-backtick outer fence.
    """
    content = "before ```` after"  # 4 consecutive backticks
    block = render_fenced_block(content)
    # Outer fence MUST be at least 5 backticks.
    assert "`````" in block
    # And it must not break out — content is preserved literally.
    assert "before ```` after" in block


def test_render_fenced_block_language_info_string_sanitized() -> None:
    """Language info-string restricted to alphanum + `-_+`.

    An attacker controlling the language string can't inject newlines
    or other characters to break out of the info-string position. The
    sanitizer strips `\\n` and backtick from the input, producing the
    safe info-string `pythonbash` directly after the opening fence
    marker.
    """
    block = render_fenced_block("x", language="python\n```bash")
    # No closing-fence breakout (the canonical injection vector).
    assert "\n```bash" not in block
    # Pin the EXACT shape of the opening fence: any leading newlines
    # the renderer adds for block separation, then the fence marker +
    # sanitized info-string with no breakout characters. Strip the
    # leading newline(s) before splitting so the first content line is
    # the fence-with-info-string, not a leading empty line.
    first_line = block.lstrip("\n").splitlines()[0]
    assert first_line.startswith("```"), f"opening fence shape regression: {first_line!r}"
    # `[3:]` cuts the fence marker; what remains is the sanitized
    # info-string. Equality (not substring) so a regression that
    # accidentally re-admits `\n` or backtick chars surfaces here.
    assert first_line[3:] == "pythonbash", (
        f"info-string sanitization regression: {first_line[3:]!r} "
        f"(expected 'pythonbash' — sanitizer should have stripped "
        f"`\\n` and backtick)"
    )


def test_render_fenced_block_strips_control_codes_inside() -> None:
    """Bidi / ANSI / NUL stripped from INSIDE the fence too.

    Code fences don't escape inner text, so the bidi attack vector
    survives a naïve fence-only defense. Strip control codes inside
    too.
    """
    block = render_fenced_block("x \u202e= 1\x1b[31m")
    assert "\u202e" not in block
    assert "\x1b" not in block


# ---------------------------------------------------------------------------
# 3. Markdown-semantic neutralization for prose
# ---------------------------------------------------------------------------


def test_escape_at_mention() -> None:
    """`@user` neutralized so GitHub doesn't ping the user.

    GitHub treats `\\@alice` differently from `@alice` in markdown
    rendering — the backslash-prefixed form does not generate a
    mention notification. Verify the backslash escape lands.
    """
    text = sanitize_display_string("ping @alice for review")
    assert "\\@alice" in text
    # The substring `@alice` still appears literally inside `\@alice`;
    # the markdown-render distinction is the escape, not the absence.


def test_escape_issue_ref() -> None:
    """`#123` neutralized so GitHub doesn't auto-link an issue."""
    text = sanitize_display_string("see issue #123 for context")
    assert "\\#123" in text


def test_escape_link_syntax() -> None:
    """`[text](url)` neutralized."""
    text = sanitize_display_string("here is a [phishing](https://evil) link")
    assert "\\[phishing" in text
    assert "\\(https://evil\\)" in text


def test_escape_leading_blockquote() -> None:
    """Leading `>` on a line escaped (defends against `> [!CAUTION]` block-quote alerts)."""
    text = sanitize_display_string("> [!CAUTION]\nThis is fake guidance.")
    # Leading > on first line should be escaped.
    first_line = text.splitlines()[0]
    assert first_line.startswith("\\>")


def test_escape_leading_list_marker() -> None:
    """Leading `-` / `*` / `+` / `1.` escaped per-line."""
    text = sanitize_display_string("- item one\n* item two\n1. numbered")
    lines = text.splitlines()
    assert lines[0].startswith("\\-")
    assert lines[1].startswith("\\*")
    assert lines[2].startswith("\\1.")


def test_escape_markdown_prose_does_not_wrap_with_backslashes() -> None:
    """Backslashes are escaped FIRST so later escapes don't get doubled."""
    text = escape_markdown_prose("path\\to\\file")
    # `\` should become `\\` exactly once per original `\`.
    assert text == "path\\\\to\\\\file"


# ---------------------------------------------------------------------------
# 4. Size cap in UTF-8 BYTES (incl. non-ASCII coverage)
# ---------------------------------------------------------------------------


def test_apply_size_cap_ascii_under_cap_returns_unchanged() -> None:
    """Body within cap is returned untouched."""
    body = "short ascii body"
    assert apply_size_cap(body) == body


def test_apply_size_cap_ascii_over_cap_truncates_and_appends_marker() -> None:
    """Body over cap is truncated; final encoded length <= cap."""
    body = "x" * (GITHUB_COMMENT_BODY_MAX + 10_000)
    result = apply_size_cap(body)
    assert len(result.encode("utf-8")) <= GITHUB_COMMENT_BODY_MAX
    assert "[truncated" in result


def test_apply_size_cap_non_ascii_caps_by_bytes_not_codepoints() -> None:
    """The cap is UTF-8 bytes. Each CJK char is 3 bytes — building a
    payload that's larger in BYTES than the cap but smaller in
    CODEPOINTS catches a codepoint-vs-byte regression.

    A naïve `len(text) > cap` check would let this pass without
    truncation; the byte-based check truncates correctly.
    """
    # 30,000 CJK chars × 3 bytes/char = 90,000 bytes — over the 65,536 cap.
    # But 30,000 codepoints — under any codepoint-based cap < 65,536.
    body = "中" * 30_000
    assert len(body) < GITHUB_COMMENT_BODY_MAX  # codepoint count is under
    assert len(body.encode("utf-8")) > GITHUB_COMMENT_BODY_MAX  # byte count over
    result = apply_size_cap(body)
    # Result MUST be truncated (byte check, not codepoint check).
    assert len(result.encode("utf-8")) <= GITHUB_COMMENT_BODY_MAX
    assert "[truncated" in result


def test_apply_size_cap_truncation_does_not_split_mid_codepoint() -> None:
    """Trailing partial UTF-8 codepoint dropped cleanly via errors='ignore'."""
    # Build content that's exactly cap+1 bytes when encoded, with the
    # boundary mid-codepoint. The decode-with-ignore on truncation should
    # drop the partial codepoint rather than emit a replacement char.
    body = "a" * (GITHUB_COMMENT_BODY_MAX - 1) + "中"  # 中 is 3 bytes
    result = apply_size_cap(body)
    # MUST still be valid UTF-8 (no UnicodeDecodeError on the result).
    result.encode("utf-8").decode("utf-8")


def test_apply_size_cap_reserve_bytes_reduces_effective_cap() -> None:
    """`reserve_bytes` holds back room for a block the caller appends AFTER the
    cap (the S1 agent-marker block): the capped body fits within
    GITHUB_COMMENT_BODY_MAX - reserve, so prose + a reserve-sized suffix still
    fits GitHub's hard limit and the suffix is never truncated."""
    reserve = 500
    suffix = "Z" * reserve  # exactly `reserve` ASCII bytes
    body = "x" * (GITHUB_COMMENT_BODY_MAX + 10_000)
    capped = apply_size_cap(body, reserve_bytes=reserve)
    assert len(capped.encode("utf-8")) <= GITHUB_COMMENT_BODY_MAX - reserve
    assert "[truncated" in capped
    # prose + the reserved-size suffix fits the real GitHub limit.
    assert len((capped + suffix).encode("utf-8")) <= GITHUB_COMMENT_BODY_MAX


def test_apply_size_cap_reserve_bytes_zero_matches_default() -> None:
    """reserve_bytes=0 (the default) caps identically to the no-arg form."""
    body = "x" * (GITHUB_COMMENT_BODY_MAX + 10_000)
    assert apply_size_cap(body, reserve_bytes=0) == apply_size_cap(body)


def test_apply_size_cap_default_max_bytes_matches_comment_cap() -> None:
    """Omitting max_bytes caps at GITHUB_COMMENT_BODY_MAX — every existing caller
    is byte-identical (DECISIONS.md#050 parameterization is default-preserving)."""
    body = "x" * (GITHUB_COMMENT_BODY_MAX + 5_000)
    assert apply_size_cap(body) == apply_size_cap(body, max_bytes=GITHUB_COMMENT_BODY_MAX)


def test_apply_size_cap_custom_max_bytes_caps_to_that_limit() -> None:
    """A smaller max_bytes truncates a body that would pass under the default cap —
    the review-body surface passes GITHUB_REVIEW_BODY_MAX this way."""
    body = "x" * 2_000  # well under GITHUB_COMMENT_BODY_MAX → unchanged by default
    assert apply_size_cap(body) == body
    capped = apply_size_cap(body, max_bytes=500)
    assert len(capped.encode("utf-8")) <= 500
    assert capped != body  # actually truncated under the smaller cap


def test_github_review_body_max_is_a_distinct_constant() -> None:
    """GITHUB_REVIEW_BODY_MAX is its own named cap for the PR review-body surface
    (DECISIONS.md#050) — separate name from the comment cap, even if equal today."""
    assert isinstance(GITHUB_REVIEW_BODY_MAX, int)
    assert GITHUB_REVIEW_BODY_MAX > 0


def test_apply_size_cap_nonpositive_max_bytes_raises() -> None:
    """A non-positive caller-provided max_bytes is a caller bug — fail loud directly
    with ValueError, not via the indirect marker-budget RuntimeError."""
    with pytest.raises(ValueError, match="max_bytes must be > 0"):
        apply_size_cap("body", max_bytes=0)
    with pytest.raises(ValueError, match="max_bytes must be > 0"):
        apply_size_cap("body", max_bytes=-5)


def test_apply_size_cap_negative_reserve_raises() -> None:
    """A negative reserve is a caller bug — fail loud."""
    with pytest.raises(ValueError, match="reserve_bytes must be >= 0"):
        apply_size_cap("body", reserve_bytes=-1)


def test_apply_size_cap_reserve_exceeding_cap_raises() -> None:
    """A reserve_bytes large enough that no room remains for content + the
    truncation marker is a caller bug (the appended block can't coexist with any
    prose) — fail loud rather than silently drop all content."""
    body = "x" * (GITHUB_COMMENT_BODY_MAX + 10_000)
    with pytest.raises(RuntimeError, match="smaller than the truncation marker budget"):
        apply_size_cap(body, reserve_bytes=GITHUB_COMMENT_BODY_MAX)


# ---------------------------------------------------------------------------
# 5. HMAC-tagged truncation marker
# ---------------------------------------------------------------------------


def test_truncation_marker_carries_hmac_tag() -> None:
    """Marker embeds an 8-char HMAC tag for verifiability."""
    body = "x" * (GITHUB_COMMENT_BODY_MAX + 1)
    result = apply_size_cap(body)
    expected_hmac = compute_truncation_hmac(original_byte_count=len(body.encode("utf-8")))
    assert f"· {expected_hmac}]" in result


def test_compute_truncation_hmac_is_deterministic() -> None:
    """Same byte count → same HMAC under same secret."""
    assert compute_truncation_hmac(1000) == compute_truncation_hmac(1000)


def test_compute_truncation_hmac_differs_per_byte_count() -> None:
    """Different byte counts produce different HMACs."""
    assert compute_truncation_hmac(1000) != compute_truncation_hmac(2000)


def test_strip_fake_truncation_markers_removes_attacker_prepended_fake() -> None:
    """Attacker-prepended fake marker is stripped before sanitization.

    Defends against a reader being tricked into believing Outrider
    did the eliding when in fact the attacker chose what to elide.
    """
    fake = "[truncated, original 999 bytes · deadbeef]"
    text = f"normal content {fake} more content"
    cleaned = strip_fake_truncation_markers(text)
    assert "[truncated" not in cleaned


def test_sanitize_display_string_strips_fake_marker_inline() -> None:
    """The full sanitize pipeline runs fake-marker stripping first."""
    fake = "[truncated, original 999 bytes · deadbeef]"
    text = f"prepended {fake}"
    cleaned = sanitize_display_string(text)
    assert "[truncated" not in cleaned


def test_compute_truncation_hmac_raises_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing/empty secret fails LOUDLY (no forgeable markers).

    The OWASP-style "fail securely" pattern — if the deploy env
    forgot to set the secret, the sanitizer shouldn't silently
    produce HMACs an attacker could trivially forge.
    """
    monkeypatch.delenv(TRUNCATION_HMAC_SECRET_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TRUNCATION_HMAC_SECRET_ENV):
        compute_truncation_hmac(1000)


# ---------------------------------------------------------------------------
# Body cap byte-vs-codepoint contract (DECISIONS.md #023 append-only)
# ---------------------------------------------------------------------------


def test_github_comment_body_max_is_explicitly_65536_bytes() -> None:
    """The cap value is pinned by DECISIONS.md #023 as 65_536 UTF-8 bytes.

    Per DECISIONS.md #023 "Append-only enum-value + hash-recipe
    contract": the cap is an Outrider policy cap, not a vendor-derived
    limit. Pinning it here makes a value-change require an explicit
    test edit (forcing the contributor to read the docstring's
    "policy not vendor" framing).
    """
    assert GITHUB_COMMENT_BODY_MAX == 65_536


# ---------------------------------------------------------------------------
# is_safe_suggestion_replacement — the shared suggested-patch safety gate
# (DECISIONS.md#040; one source of truth for parser + renderer)
# ---------------------------------------------------------------------------


def test_is_safe_suggestion_replacement_accepts_clean_code() -> None:
    """A normal single-line fix is accepted — INCLUDING legitimate `<`/`>` (the
    suggestion is applicable code, not prose, so angle brackets are NOT escaped)."""
    assert is_safe_suggestion_replacement("    return sanitize(user_input)") is True
    assert is_safe_suggestion_replacement("if (a < b && c > d) { return x; }") is True
    assert is_safe_suggestion_replacement("\tresult: list[int] = []") is True  # tab is fine


@pytest.mark.parametrize(
    ("replacement", "why"),
    [
        ("", "empty"),
        ("   ", "whitespace-only"),
        ("a\nb", "multi-line LF"),
        ("a\rb", "multi-line CR"),
        ("x = `y`", "backtick (fence breakout)"),
        ("@@ -1 +1 @@", "diff hunk header"),
        ("+ added", "diff add line"),
        ("- removed", "diff remove line"),
        ("return user_is_admin\u202e;", "bidi-override (Trojan Source)"),
        ("return\u200bx", "zero-width space"),
        ("return x\x00", "NUL"),
        ("return x\x1b[31m", "ANSI escape"),
        ("<!-- outrider:severity low -->", "forged HTML-comment marker"),
        ("x = 1 -->", "HTML-comment close delimiter"),
    ],
)
def test_is_safe_suggestion_replacement_rejects(replacement: str, why: str) -> None:
    assert is_safe_suggestion_replacement(replacement) is False, f"should reject: {why}"
