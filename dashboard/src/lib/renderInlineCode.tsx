import type { ReactNode } from "react";

// Render only backtick-delimited code spans in finding text. Both plain
// segments and code contents remain escaped React text nodes, so PR-influenced
// model output cannot introduce HTML.
export function renderInlineCode(text: string): ReactNode {
  // The capture group puts code at odd indices; unmatched backticks stay text.
  const segments = text.split(/`([^`]+)`/g);
  return segments.map((segment, index) =>
    index % 2 === 1 ? (
      <code key={index} className="f-code">
        {segment}
      </code>
    ) : (
      segment
    ),
  );
}
