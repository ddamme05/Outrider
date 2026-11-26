import type { components } from "../api/schema";

type Coverage = components["schemas"]["TriageInputCoverage"];

function patchSizeText(bytes: number | null): string {
  if (bytes === null) return "File metadata only";
  if (bytes === 0) return "Empty patch (0 bytes)";
  return `${bytes.toLocaleString()} patch ${bytes === 1 ? "byte" : "bytes"}`;
}

export function TriageInput({ coverage }: { coverage: Coverage }) {
  const { files, omitted_files: omitted } = coverage;
  const total = files.length + omitted;
  return (
    <details className="tl-triage-input">
      <summary>Triage input ({total.toLocaleString()} {total === 1 ? "file" : "files"})</summary>
      {files.length > 0 && (
        <ul>
          {files.map((file, index) => (
            <li key={index}>
              <code>{file.file_path}</code> · {patchSizeText(file.patch_bytes)}
            </li>
          ))}
        </ul>
      )}
      {total === 0 && <p>No files were listed in this triage request.</p>}
      {omitted > 0 && (
        <p>This record omits details for {omitted.toLocaleString()} {omitted === 1 ? "file" : "files"} in the triage request.</p>
      )}
    </details>
  );
}
