import { languageForPath } from "../lib/findingSections";

// A read-only evidence snippet. The code is a React TEXT NODE — escaped by construction, so an
// attacker-influenced finding.evidence can never inject markup (the dashboard analogue of the
// GitHub renderer's breakout-safe fence). Wide lines scroll INSIDE the block (`overflow-x: auto`);
// the page body never scrolls horizontally. `filePath` drives an optional language chip only —
// there is no client-side syntax highlighting (a highlighter would need a CSP-blocked bundle).
export function CodeBlock({ code, filePath }: { code: string; filePath?: string }) {
  const lang = filePath ? languageForPath(filePath) : "";
  return (
    <div className="codeblock">
      {lang ? <span className="codeblock-lang">{lang}</span> : null}
      <pre className="codeblock-pre">
        <code>{code}</code>
      </pre>
    </div>
  );
}
