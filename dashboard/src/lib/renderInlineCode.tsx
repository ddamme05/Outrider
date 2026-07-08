import type { ReactNode } from "react";

// Render backtick-delimited inline code in model-authored text (a finding's title +
// description) as <code> spans. The model writes descriptions with Markdown inline code for
// identifiers and snippets (`status`, `sort_column`, `f"WHERE status = '{status}'"`); GitHub
// and Slack render their own Markdown, but the dashboard showed the raw backticks. This is a
// deliberately MINIMAL renderer — inline code only, no bold/links/HTML.
//
// XSS-safe by construction: the text is split into plain segments and code captures, each
// emitted as a React text node (auto-escaped). It never uses dangerouslySetInnerHTML, so
// attacker-influenced finding text (the model reads attacker-controlled PR content) cannot
// inject markup.
export function renderInlineCode(text: string): ReactNode {
  // Split on `…` spans. String.split with one capture group interleaves the captures at odd
  // indices: [plain, code, plain, code, …]. An unmatched trailing backtick stays plain text.
  const parts = text.split(/`([^`]+)`/g);
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <code key={i} className="f-code">
        {part}
      </code>
    ) : (
      part
    ),
  );
}
