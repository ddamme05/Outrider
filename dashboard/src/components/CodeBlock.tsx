import type { components } from "../api/schema";
import { languageForPath } from "../lib/findingSections";

// Evidence stays a React text node so supplied markup remains inert.
export function CodeBlock({ code, filePath }: { code: string; filePath?: string }) {
  const language = filePath ? languageForPath(filePath) : "";
  return (
    <div className="codeblock">
      {language ? <span className="codeblock-lang">{language}</span> : null}
      <pre className="codeblock-pre">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function ImplementationEvidence({ citation }: {
  citation: components["schemas"]["ImplementationCitation"];
}) {
  return (
    <div>
      <b>Implementation</b> <code>{citation.path}</code>
      <CodeBlock code={citation.quote} filePath={citation.path} />
    </div>
  );
}
