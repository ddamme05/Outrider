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
