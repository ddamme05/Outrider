# Output sanitizer per docs/spec.md §4.1.7 + docs/trust-boundaries.md §5/§6.
"""Sanitize finding text before it reaches the GitHub comment-body surface.

Every string that lands in a `POST /repos/.../reviews` body field is
attacker-controlled at the source (PR title, file contents, evidence
spans quoted into prompts) and must pass through this module first. The
publisher node calls these helpers via `InlineComment.from_finding(...)`;
the trust-boundary checklist (boundary #6) names this as the single
production construction path.

Five concerns this module owns:

1. **Codepoint stripping** — bidi-override (Trojan Source), ZWJ, ANSI
   CSI sequences, NUL, lone surrogates. Each carries known display-layer
   or terminal-rendering attacks; we strip rather than escape because
   GitHub's markdown renderer doesn't honor an escape syntax for them.

2. **Fence neutralization for fenced content** (`evidence` and
   similar code-quoting fields) — the outer fence is ALWAYS backticks
   for deterministic shape; the fence count is
   `max(longest_backtick_run_in_content, 2) + 1` so the content never
   breaks out. Computed AFTER truncation so the marker can't push the
   inner content past the fence boundary. (`suggested_fix` is the
   exception — it renders into a FIXED GitHub ```suggestion fence that
   cannot be neutralized without killing the Apply button, so it is
   REJECTED-not-transformed via `is_safe_suggestion_replacement`.)

3. **Markdown-semantic neutralization for prose content** (`title`,
   `description` and similar narrative fields) — escape `@`, `#`, `!`,
   `[`, `]`, `(`, `)`, `<`, `>`, and leading list/block markers
   (`>`, `-`, `*`, `+`, `1.`, `2.`, etc.). Stops mention-pinging,
   issue-link-spawning, ref-spawning, HTML-injection, and block-quote
   exploitation.

4. **Size cap** — `GITHUB_COMMENT_BODY_MAX = 65_536` UTF-8 bytes,
   measured INCLUDING the truncation marker and any closing fences the
   marker re-injects. Outrider policy cap, not vendor-derived (see
   DECISIONS.md #023 append-only contract; the 4a sandbox observed
   acceptance through 70,000 chars under apiVersion 2026-03-10 but
   did not establish the actual vendor maximum). The cap is in
   BYTES not codepoints because the existing truncation marker uses
   bytes wording (`[truncated, original N bytes · <hmac8>]`) and
   non-ASCII content (CJK, emoji, RTL) is the edge case ASCII probes
   hide.

5. **HMAC-tagged truncation marker** — an 8-char HMAC keyed by the
   `OUTRIDER_TRUNCATION_HMAC_SECRET` env var, embedded in the marker
   text. An attacker who prepends a fake `[truncated, original N bytes
   · faked8x]` to their content can't predict the HMAC, so the
   sanitizer's `strip_fake_truncation_markers` (called BEFORE
   sanitization) reliably distinguishes attacker-prepended markers
   from sanitizer-produced ones. The HMAC is short (8 hex chars =
   32 bits) — collision-resistance is NOT the goal; preimage
   resistance against a key-less attacker is. 32 bits is plenty.
"""

from __future__ import annotations

import hmac
import os
import re
from hashlib import sha256
from typing import Final
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Outrider policy cap on the rendered GitHub comment body, in UTF-8 bytes.
#
# This is an Outrider POLICY cap, NOT a measured GitHub maximum. GitHub
# was empirically observed (sandbox PR-2 reviews, 2026-05-22) to accept
# inline-comment bodies at least up to 70,000 ASCII chars under
# apiVersion 2026-03-10; the actual vendor maximum above the probed
# range is unverified and the cap was deliberately not walked further
# (no shipping decision rides on the exact ceiling). The cap exists for
# bounded output, deterministic truncation, and sanitizer predictability
# — truncation fires here, not at an opaque API boundary, so the
# publish call always succeeds (user sees a truncation marker, never
# a 422 from the body field).
#
# Cap is enforced in UTF-8 BYTES including the truncation marker and
# fencing overhead — ASCII test data hides the codepoint-vs-byte gap;
# non-ASCII content (CJK identifiers, emoji, RTL text in evidence)
# is the edge case that motivates byte-measurement.
GITHUB_COMMENT_BODY_MAX: Final[int] = 65_536
# Distinct GitHub surface from inline comments (the PR review BODY), kept separate
# per DECISIONS.md#050 so the two caps can diverge without re-introducing the
# conflation the publish-review-body spec calls out — equal today, not aliased.
GITHUB_REVIEW_BODY_MAX: Final[int] = 65_536

# Env var name for the truncation-marker HMAC secret. Deploy-time
# configuration; the sanitizer reads it on first use and caches the
# secret bytes per-process. A missing or empty env var fails loudly at
# first truncation rather than silently producing forgeable markers
# (the OWASP-style "fail securely" pattern).
TRUNCATION_HMAC_SECRET_ENV: Final[str] = "OUTRIDER_TRUNCATION_HMAC_SECRET"  # noqa: S105  (env var NAME, not a secret value)

# Length of the HMAC tag in hex characters embedded in the truncation
# marker. 8 hex chars = 32 bits = ~4.3 billion possible values;
# preimage-resistant against a key-less attacker (which is the only
# threat model — the secret never leaves the deploy environment).
_TRUNCATION_HMAC_LEN: Final[int] = 8

# Truncation marker text. The leading `\n\n` ensures the marker
# appears on its own line in markdown rendering even when the prior
# content didn't end with a newline. `original N bytes` wording pairs
# with the BYTE measurement of the cap (NOT codepoints) so a reader
# computing "how much was lost" gets the correct answer for non-ASCII
# content. The HMAC tag (`<hmac8>`) is what distinguishes
# sanitizer-produced markers from attacker-prepended fakes.
_TRUNCATION_MARKER_TEMPLATE: Final[str] = "\n\n[truncated, original {n} bytes · {hmac}]"

# Regex to detect any truncation-marker-shaped string in input. Used
# by `strip_fake_truncation_markers` to remove attacker-prepended
# fakes BEFORE the sanitizer's own truncation runs. We strip ALL
# markers matching the shape (whether the HMAC is valid or not),
# because: (a) the only legitimate marker is the one this module
# produces AT truncation time, never before; (b) preserving an
# attacker-prepended marker (even a forged one) is a confused-deputy
# vector where a reader trusts "this content was truncated by Outrider"
# when in fact the attacker chose what to elide.
_TRUNCATION_MARKER_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\n*\[truncated,\s+original\s+\d+\s+bytes\s+·\s+[0-9a-f]+\]",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Codepoint stripping — the hard-stop characters that have no legitimate
# representation in a code-review comment body.
# ---------------------------------------------------------------------------

# Bidirectional-override codepoints. The Trojan Source family
# (CVE-2021-42574); these characters reorder visible text so the
# rendered glyphs disagree with the byte sequence. Strip on input.
# `LRO` `RLO` `LRE` `RLE` `PDF` `LRI` `RLI` `FSI` `PDI` `ALM` `RLM` `LRM`
_BIDI_OVERRIDE_CHARS: Final[frozenset[str]] = frozenset("‪‫‬‭‮⁦⁧⁨⁩؜‎‏")

# Zero-width joiner + non-joiner. Used for homoglyph attacks against
# identifiers rendered in markdown. Stripping is conservative — there
# is no legitimate use of ZWJ/ZWNJ in a security-review comment body.
_ZERO_WIDTH_CHARS: Final[frozenset[str]] = frozenset("​‌‍﻿")

# ANSI Control Sequence Introducer + OS-command sequences. Markdown
# itself doesn't render ANSI, but if the body is ever piped through a
# terminal viewer (gh CLI, GitHub mobile push notifications on terminal
# theming, log files) the sequences become live control codes. Strip
# the whole CSI/OSC sequence; replacing with a placeholder would
# preserve the attacker-chosen byte count and still confuse some
# renderers.
#
# Pattern matches:
#   ESC [  ... [@-~]    — CSI sequence (color, cursor movement)
#   ESC ]  ... (BEL | ESC \)  — OSC sequence (window title, hyperlinks)
#   ESC P  ... ESC \    — DCS sequence
#   ESC X  ... ESC \    — SOS sequence
#   ESC ^  ... ESC \    — PM sequence
#   ESC _  ... ESC \    — APC sequence
#   bare ESC            — naked escape (no legitimate use in comment body)
_ANSI_CONTROL_REGEX: Final[re.Pattern[str]] = re.compile(
    "\x1b\\[[0-?]*[ -/]*[@-~]"  # CSI
    "|\x1b\\][^\x07\x1b]*(\x07|\x1b\\\\)"  # OSC
    "|\x1b[PX^_][^\x1b]*\x1b\\\\"  # DCS / SOS / PM / APC
    "|\x1b"  # naked ESC
)

# NUL byte. Some markdown renderers truncate at NUL; some pass it
# through to terminals as a literal `\0`. Either way, no legitimate
# code-review comment contains NUL.
_NUL_CHAR: Final[str] = "\x00"

# ---------------------------------------------------------------------------
# Markdown-semantic escaping for prose content.
# ---------------------------------------------------------------------------

# Inline-character escape map. Each maps to its backslash-escaped form,
# which markdown renders as the literal character. The set deliberately
# omits `*` and `_` because mid-word emphasis-suppression would break
# legitimate prose (e.g., `my_variable`); leading-position list markers
# are handled separately (`_LEADING_BLOCK_MARKER_REGEX`).
_MARKDOWN_INLINE_ESCAPES: Final[dict[str, str]] = {
    "\\": "\\\\",  # MUST be first; later escapes inject backslashes
    "`": "\\`",
    "@": "\\@",  # @mention spawning
    "#": "\\#",  # #issue / #PR spawning + ATX heading at line start
    "!": "\\!",  # ![image-embed] spawning
    "[": "\\[",  # [link-text]( spawning
    "]": "\\]",
    "(": "\\(",  # paired with [ above
    ")": "\\)",
    "<": "\\<",  # raw-HTML or autolink spawning
    ">": "\\>",  # blockquote (handled separately for leading position too)
}

# Leading-position block markers — these only have markdown semantics
# when they appear at the start of a line (possibly after whitespace).
# Matched per-line and the leading marker is escaped; the rest of the
# line is left alone (or runs through inline escaping separately).
_LEADING_BLOCK_MARKER_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(\s*)([>\-*+]|\d+\.)\s",
    flags=re.MULTILINE,
)

# The agent-marker signature (`<!-- outrider:KEY VALUE -->` / `<!-- outrider-review-id:… -->`).
# Prose escapes `<`→`\<` and `is_safe_suggestion_replacement` rejects `<!--` so untrusted body text
# can't forge a raw-byte-grepped marker (FUP-154); `render_fenced_block` applies the same defense to
# fenced content (a code fence renders `<!--` verbatim), targeted at the `outrider` namespace so
# legitimate HTML/XML comments in a code snippet are left intact. Case-insensitive (like
# `_TRUNCATION_MARKER_REGEX`) — prose/suggestion escape `<`/`<!--` letter-blind, so a case variant
# (`<!-- OUTRIDER…`) must not slip past a case-insensitive marker grep here either.
_OUTRIDER_MARKER_OPEN_REGEX: Final[re.Pattern[str]] = re.compile(
    r"<!--(\s*outrider)", flags=re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Public sanitization API.
# ---------------------------------------------------------------------------


def strip_fake_truncation_markers(text: str) -> str:
    """Remove any attacker-prepended truncation markers from input.

    Called BEFORE other sanitization steps so an attacker who embeds
    `[truncated, original 999 bytes · deadbeef]` in their finding text
    can't trick a reader into believing Outrider did the eliding.

    The legitimate truncation marker is appended by `_apply_byte_cap`
    AT truncation time — never present in input. So stripping ALL
    markers shaped like the regex is correct: zero false positives in
    real finding text (nobody writes that pattern by accident), zero
    false negatives (attacker can't avoid the regex without losing the
    visual shape of the spoof).
    """
    return _TRUNCATION_MARKER_REGEX.sub("", text)


def sanitize_display_string(text: str) -> str:
    """Strip codepoints, escape markdown semantics, return prose-safe text.

    For PROSE fields (`title`, `description`). The code-quoting field
    `evidence` uses `render_fenced_block(text)` instead, which fence-wraps
    rather than escapes; `suggested_fix` uses `is_safe_suggestion_replacement`
    (reject-not-transform) + a fixed GitHub ```suggestion fence.

    Pipeline:
      1. Strip fake truncation markers (attacker-prepended).
      2. Strip bidi-override / zero-width / ANSI / NUL codepoints.
      3. Drop lone surrogates (Python str admits them; UTF-8 encode rejects).
      4. Escape markdown inline characters.
      5. Escape leading block markers per line.

    Does NOT truncate or fence — the publisher's `from_finding(...)`
    composes sanitized fields into the final body, then calls
    `apply_size_cap(body)` to truncate the assembled result.
    """
    text = strip_fake_truncation_markers(text)
    text = _strip_bidi_and_zero_width(text)
    text = _ANSI_CONTROL_REGEX.sub("", text)
    text = text.replace(_NUL_CHAR, "")
    text = _drop_lone_surrogates(text)
    text = escape_markdown_prose(text)
    return text


def escape_markdown_prose(text: str) -> str:
    """Escape inline + leading-position markdown semantics for prose.

    Public so a caller composing a body manually (e.g., a future
    `synthesize` review-body builder) can apply the same escaping
    without going through the full sanitizer pipeline.
    """
    # Inline escapes: process backslash FIRST so later escapes don't
    # get doubly-escaped.
    for char, escaped in _MARKDOWN_INLINE_ESCAPES.items():
        text = text.replace(char, escaped)
    # Leading-position block markers: escape per line.
    text = _LEADING_BLOCK_MARKER_REGEX.sub(r"\1\\\2 ", text)
    return text


def render_fenced_block(content: str, *, language: str = "") -> str:
    """Wrap `content` in a backtick fence safe against breakout.

    Fence count = max(longest backtick run in content, 2) + 1. The
    outer fence is ALWAYS backticks (never tildes) for deterministic
    shape; sanitizer consumers downstream pattern-match on the
    backtick form.

    Content INSIDE the fence is NOT escaped — that's the point of
    a code fence. But the bidi/ANSI/NUL stripping from
    `sanitize_display_string` is reapplied here because a code-fenced
    block can still contain a Trojan Source attack visible on
    GitHub's syntax-highlighting renderer. For the same reason the
    `<!-- outrider… -->` agent-marker signature is neutralized: a fence
    renders `<!--` verbatim, so a multiline snippet could otherwise plant
    a byte-perfect marker in the raw comment (FUP-154 defense parity with
    the prose / suggested_fix fields).

    `language` is an optional info-string (e.g., `python`); it lands
    after the opening fence per CommonMark. Sanitized to alphanum +
    `-_+` so attacker-controlled language strings can't break out.
    """
    # Inner content: strip control codes but keep the literal text shape.
    content = strip_fake_truncation_markers(content)
    content = _strip_bidi_and_zero_width(content)
    content = _ANSI_CONTROL_REGEX.sub("", content)
    content = content.replace(_NUL_CHAR, "")
    content = _drop_lone_surrogates(content)
    # Neutralize any agent-marker signature by breaking `<!` → `\<\!` (exactly what prose escaping
    # does via `<`→`\<` + `!`→`\!`): the raw `<!--` token no longer appears in the bytes, so a
    # marker grep — substring OR line-anchored — can't see a forged `<!-- outrider… -->`.
    content = _OUTRIDER_MARKER_OPEN_REGEX.sub(r"\\<\\!--\1", content)

    # Fence count: scan for the longest run of backticks; ours must be
    # at least one longer. Floor of 3 because single/double backticks
    # are inline-code in markdown, not block fences.
    longest_run = _longest_backtick_run(content)
    fence_count = max(longest_run, 2) + 1
    fence = "`" * fence_count

    # Language info-string sanitization: alphanum + `-_+` only.
    safe_language = re.sub(r"[^A-Za-z0-9_+-]", "", language)

    # Always end the content with a newline so the closing fence is on
    # its own line per CommonMark; same for the leading newline before
    # the opening fence so the block is visually separated from any
    # surrounding prose.
    if not content.endswith("\n"):
        content = content + "\n"
    return f"\n{fence}{safe_language}\n{content}{fence}\n"


def is_safe_suggestion_replacement(replacement: str) -> bool:
    """Whether `replacement` is safe to emit as a one-line GitHub ```suggestion block.

    Single source of truth for the suggested-patch safety gate (DECISIONS.md#040),
    called by BOTH the patch-generation parser (`_is_valid_replacement`, which adds the
    no-op `!= original_line` check on top) AND the publish renderer
    (`_render_suggestion_block`) — so the two layers are genuine defense in depth, not a
    copy-pasted reject list that drifts.

    A ```suggestion is committed VERBATIM by GitHub's Apply button and is grepped
    raw-byte by AI agents (the S1 marker contract). Unlike prose fields it CANNOT be
    `<`/`>`-escaped (that would corrupt the applied code), so unsafe content is
    REJECTED, not transformed. Rejects: empty/whitespace-only; multi-line (`\\n`/`\\r`
    — V1 is one line); backtick (would break out of the fence, killing the Apply
    button); diff markers (`@@`/`+ `/`- `); Trojan-Source control codepoints
    (bidi-override / zero-width / ANSI / NUL / lone surrogate — committed verbatim;
    sibling prose channels strip these via `sanitize_display_string`); and HTML-comment
    delimiters (`<!--`/`-->` — could forge a grep-parseable `<!-- outrider:KEY VALUE -->`
    agent marker in the raw comment bytes, which the suggestion can't `<`/`>`-escape).
    """
    if not replacement or not replacement.strip():
        return False
    if "\n" in replacement or "\r" in replacement:
        return False
    if "`" in replacement:
        return False
    if replacement.lstrip().startswith("@@") or replacement[:2] in ("+ ", "- "):
        return False
    if "<!--" in replacement or "-->" in replacement:
        return False
    # Trojan-Source codepoints — the same set `render_fenced_block` strips from prose;
    # here we REJECT (an applicable code fix can't be silently transformed).
    return not (
        _strip_bidi_and_zero_width(replacement) != replacement
        or _ANSI_CONTROL_REGEX.search(replacement) is not None
        or _NUL_CHAR in replacement
        or _drop_lone_surrogates(replacement) != replacement
    )


def is_safe_link_url(value: str) -> bool:
    """Whether `value` is safe to embed as a link URL in BOTH GitHub markdown
    (`[text](url)`) and Slack mrkdwn (`<url|text>`).

    Single source of truth for the deep-link URL-safety gate — called by the publish
    review-body renderer (`_review_deep_link`) AND the Slack notification deep-link
    builder (`notify/deeplink.py`). Requires an http(s) scheme (case-insensitive)
    with a non-empty parsed host (`urlparse().netloc`), and rejects whitespace,
    C0/C1 control chars + DEL, and the link-delimiter characters of BOTH formats:
    `()` (markdown link target), `<>` (HTML / Slack `<...>`), `[]` (markdown), and
    `|` (the Slack `<url|text>` separator). The host check rejects scheme-only /
    empty-host URLs (`https://`, `https:///foo`) whose trailing-slash strip would
    yield a malformed `https:/...`. The base URL is operator/per-install config, so
    the threat is misconfiguration, not attacker input; a malformed URL degrades to
    a no-link fallback at the call site.
    """
    try:
        parsed = urlparse(value)
    except ValueError:
        return False  # e.g. malformed IPv6 literal — degrade to no-link
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return False
    return not any(
        ch.isspace() or ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F or ch in "()<>[]|" for ch in value
    )


def apply_size_cap(
    body: str, *, reserve_bytes: int = 0, max_bytes: int = GITHUB_COMMENT_BODY_MAX
) -> str:
    """Truncate `body` to fit `max_bytes - reserve_bytes` UTF-8 bytes.

    `max_bytes` defaults to `GITHUB_COMMENT_BODY_MAX` (inline comments); the PR
    review-body surface passes `GITHUB_REVIEW_BODY_MAX` (DECISIONS.md#050). Every
    existing caller omits it, so behaviour is unchanged.

    `reserve_bytes` holds back room for content the caller appends AFTER the
    cap and that must NOT itself be truncated — the S1 agent-marker block is the
    motivating case (an agent must be able to parse every `<!-- outrider:... -->`
    line intact). The effective budget is `max_bytes - reserve_bytes`, so
    `prose + appended-block` still fits GitHub's hard byte limit. Default
    `reserve_bytes=0` preserves the original behaviour for every existing caller.

    The truncation marker is INCLUDED in the (reduced) cap — if the assembled
    body is already within `max_bytes - reserve_bytes`, it is returned unchanged;
    otherwise truncated such that
    `len((truncated + marker).encode("utf-8")) <= max_bytes - reserve_bytes`.

    Care taken NOT to truncate mid-codepoint for non-ASCII content:
    the loop trims bytes from the end and decodes with `errors="ignore"`
    so any trailing partial UTF-8 sequence drops cleanly.
    """
    if reserve_bytes < 0:
        raise ValueError(f"reserve_bytes must be >= 0; got {reserve_bytes}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be > 0; got {max_bytes}")
    effective_max = max_bytes - reserve_bytes

    encoded = body.encode("utf-8")
    if len(encoded) <= effective_max:
        return body

    # Reserve space for the marker + the byte-count digits + the HMAC.
    # Worst-case marker length: `\n\n[truncated, original 999999999 bytes · 12345678]`
    # = 51 chars. Reserve 64 for safety + future-proofing.
    marker_budget = 64
    available = effective_max - marker_budget
    if available <= 0:
        # `effective_max` below the marker budget means EITHER a deploy-time
        # shrink of GITHUB_COMMENT_BODY_MAX below 64, OR a caller passing a
        # `reserve_bytes` so large it leaves no room for content + marker.
        # Both are bugs (the latter a caller bug: the appended block is too
        # big to coexist with any prose). Total content loss is the only way
        # to fit; fail loud so the misconfiguration surfaces in monitoring
        # rather than silently dropping all finding content.
        raise RuntimeError(
            f"effective cap ({effective_max} = max_bytes="
            f"{max_bytes} - reserve_bytes={reserve_bytes}) is "
            f"smaller than the truncation marker budget ({marker_budget}); "
            f"cannot truncate without dropping ALL content. Increase the cap, "
            f"shrink the reserved/appended block, or revisit the marker shape."
        )

    # Trim to `available` bytes; ignore any trailing partial codepoint.
    truncated = encoded[:available].decode("utf-8", errors="ignore")

    marker = _build_truncation_marker(original_byte_count=len(encoded))
    result = truncated + marker

    # Verify the result actually fits (defense-in-depth — the marker
    # template length could change without updating the budget).
    if len(result.encode("utf-8")) > effective_max:
        # Truncate further and try again. One iteration suffices in
        # practice (marker is bounded); cap at 3 iterations to avoid
        # any pathological loop.
        for _ in range(3):
            available -= 64
            if available <= 0:
                # See the early-return rationale above: total content
                # loss is a deploy/caller bug, not a runtime case.
                raise RuntimeError(
                    f"effective cap ({effective_max} = max_bytes="
                    f"{max_bytes} - reserve_bytes={reserve_bytes}) "
                    f"too small to fit marker after defensive truncation loop. "
                    f"This is a deploy-time or caller misconfiguration."
                )
            truncated = encoded[:available].decode("utf-8", errors="ignore")
            result = truncated + marker
            if len(result.encode("utf-8")) <= effective_max:
                return result
        # Loop exhausted without fitting — pathological marker growth.
        # Loud failure beats silent total-content-loss for the same
        # reason the early-return raises above.
        raise RuntimeError(
            f"apply_size_cap could not fit the truncation marker within the "
            f"effective cap ({effective_max} = max_bytes="
            f"{max_bytes} - reserve_bytes={reserve_bytes}) after "
            f"3 defensive iterations. Marker template or budget needs revision."
        )
    return result


def compute_truncation_hmac(original_byte_count: int) -> str:
    """Compute the HMAC tag for a truncation marker.

    Public so tests can verify markers without re-implementing the
    recipe. The HMAC inputs:

      key: `OUTRIDER_TRUNCATION_HMAC_SECRET` env var bytes (UTF-8).
      msg: f"truncated:{original_byte_count}".encode("utf-8")

    Returned as the first 8 hex chars of the SHA-256 HMAC digest.
    """
    secret = _get_truncation_secret()
    digest = hmac.new(secret, f"truncated:{original_byte_count}".encode(), sha256)
    return digest.hexdigest()[:_TRUNCATION_HMAC_LEN]


def require_truncation_secret() -> None:
    """Assert ``OUTRIDER_TRUNCATION_HMAC_SECRET`` is set — for eager startup validation.

    The truncation marker's HMAC keys off this secret, read LAZILY (only when a body
    actually exceeds the size cap inside ``apply_size_cap``). That laziness means a
    deploy missing the secret boots clean and reviews short PRs fine, then fails the
    whole publish node with a bare ``RuntimeError`` the first time any finding body
    truncates (the per-finding routing loop has no recovery wrapper, so it aborts the
    review). The lifespan calls this once at startup so the misconfiguration fails
    loud at boot instead of mid-review. Raises ``RuntimeError`` (via
    ``_get_truncation_secret``) when the secret is unset/empty; returns None when set.
    """
    _get_truncation_secret()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _strip_bidi_and_zero_width(text: str) -> str:
    """Strip bidi-override + zero-width codepoints. Inline so callers
    using `render_fenced_block` get the same protection without
    re-importing the constant sets."""
    if not text:
        return text
    result_chars: list[str] = []
    for char in text:
        if char in _BIDI_OVERRIDE_CHARS or char in _ZERO_WIDTH_CHARS:
            continue
        result_chars.append(char)
    return "".join(result_chars)


def _drop_lone_surrogates(text: str) -> str:
    """Drop unpaired surrogate codepoints (U+D800..U+DFFF).

    Python `str` admits these; UTF-8 encoding rejects them with
    `UnicodeEncodeError`. Stripping at sanitizer time produces a
    valid UTF-8-encodable string downstream — defense against a
    file content that contains decoded-surrogate-half output from
    a buggy upstream decoder.
    """
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def _longest_backtick_run(text: str) -> int:
    """Length of the longest consecutive-backtick substring in `text`.

    Used by `render_fenced_block` to pick a safe outer fence count.
    """
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _build_truncation_marker(*, original_byte_count: int) -> str:
    """Build the truncation marker text with the keyed HMAC."""
    return _TRUNCATION_MARKER_TEMPLATE.format(
        n=original_byte_count,
        hmac=compute_truncation_hmac(original_byte_count),
    )


def _get_truncation_secret() -> bytes:
    """Read the HMAC secret from env. Fails loudly if absent.

    Read on every call (not cached) so a test fixture's monkeypatch
    of the env var takes effect immediately. Per-call hashlib.sha256
    is microsecond-cheap; no performance concern.
    """
    secret = os.environ.get(TRUNCATION_HMAC_SECRET_ENV, "")
    if not secret:
        raise RuntimeError(
            f"{TRUNCATION_HMAC_SECRET_ENV} must be set (non-empty) to compute "
            f"HMAC-tagged truncation markers. The sanitizer fails loudly here "
            f"rather than producing forgeable markers."
        )
    return secret.encode("utf-8")
