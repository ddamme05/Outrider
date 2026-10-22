// Dashboard labels use the API's lowercase enum values. Shared finding labels
// and LANGUAGE_BY_EXT mirror src/outrider/presentation/finding_sections.py.
// tests/unit/test_finding_sections_ts_parity.py checks shared-map equality and
// coverage of the publish/HITL enums in the dashboard-only maps.
// Label helpers fall back to the received value when a label is missing.

export const SEVERITY_LABEL: Record<string, string> = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
  info: "Info",
};

export const TYPE_LABEL: Record<string, string> = {
  sql_injection: "SQL injection",
  xss: "XSS",
  hardcoded_secret: "Hardcoded secret",
  auth_bypass: "Auth bypass",
  path_traversal: "Path traversal",
  missing_input_validation: "Missing input validation",
  n_plus_one_query: "N+1 query",
  blocking_call_in_async: "Blocking call in async",
  unused_import: "Unused import",
  missing_error_handling: "Missing error handling",
  missing_test: "Missing test",
  deprecated_api: "Deprecated API",
  command_injection: "Command injection",
  unsafe_deserialization: "Unsafe deserialization",
  tls_verify_disabled: "TLS verification disabled",
  weak_crypto: "Weak cryptography",
  weak_password_hash: "Weak password hash",
  insecure_randomness: "Insecure randomness",
  ssrf: "SSRF",
  ssrf_metadata: "SSRF (metadata endpoint)",
  open_redirect: "Open redirect",
  open_redirect_authed: "Open redirect (authenticated)",
};

export const TIER_PHRASE: Record<string, string> = {
  observed: "Structural match (OBSERVED)",
  inferred: "Traced (INFERRED)",
  judged: "Model interpretation (JUDGED)",
};

export const DEST_LABEL: Record<string, string> = {
  inline_comment: "Inline comment",
  review_body: "Review summary",
  dashboard_only: "Dashboard only",
};

export const DIMENSION_LABEL: Record<string, string> = {
  code_quality: "Code quality",
  security: "Security",
  performance: "Performance",
  test_coverage: "Test coverage",
  best_practices: "Best practices",
};

// Dashboard-only labels.

// Whether policy permits a finding to be posted; this is not a GitHub receipt.
export const ELIGIBILITY_PHRASE: Record<string, string> = {
  eligible: "Eligible to post",
  withheld: "Withheld",
};

// Why the publish gate withheld a finding.
export const ELIGIBILITY_REASON_PHRASE: Record<string, string> = {
  hitl_required_node_absent: "HITL gate not reached",
  unexpected_override_fields_present: "Malformed override fields",
  routing_emission_failed: "Routing failed",
  hitl_decision_missing: "No HITL decision",
  hitl_rejected: "Rejected by reviewer",
  hitl_suppressed: "Suppressed by reviewer",
};

// Recorded reviewer outcome for the HITL provenance chip.
export const HITL_OUTCOME_LABEL: Record<string, string> = {
  approve: "Approved",
  reject: "Rejected",
  suppress: "Suppressed",
  severity_override: "Severity overridden",
};

// Evidence language tokens mirror Python's _LANGUAGE_BY_EXT; unknown suffixes hide the chip.
export const LANGUAGE_BY_EXT: Record<string, string> = {
  ".py": "python",
  ".pyi": "python",
  ".js": "javascript",
  ".jsx": "javascript",
  ".mjs": "javascript",
  ".cjs": "javascript",
  ".ts": "typescript",
  ".tsx": "tsx",
  ".mts": "typescript",
  ".cts": "typescript",
  ".go": "go",
  ".rs": "rust",
  ".java": "java",
  ".rb": "ruby",
  ".php": "php",
  ".c": "c",
  ".h": "c",
  ".cc": "cpp",
  ".cpp": "cpp",
  ".hpp": "cpp",
  ".cs": "csharp",
  ".kt": "kotlin",
  ".swift": "swift",
  ".scala": "scala",
  ".sql": "sql",
  ".sh": "bash",
  ".bash": "bash",
  ".yaml": "yaml",
  ".yml": "yaml",
  ".json": "json",
  ".toml": "toml",
  ".html": "html",
  ".css": "css",
  ".md": "markdown",
};

export const severityLabel = (wireValue: string): string => SEVERITY_LABEL[wireValue] ?? wireValue;
export const typeLabel = (wireValue: string): string => TYPE_LABEL[wireValue] ?? wireValue;
export const tierPhrase = (wireValue: string): string => TIER_PHRASE[wireValue] ?? wireValue;
export const destLabel = (wireValue: string): string => DEST_LABEL[wireValue] ?? wireValue;
export const dimensionLabel = (wireValue: string): string => DIMENSION_LABEL[wireValue] ?? wireValue;
export const eligibilityPhrase = (wireValue: string): string => ELIGIBILITY_PHRASE[wireValue] ?? wireValue;
export const eligibilityReasonPhrase = (wireValue: string): string => ELIGIBILITY_REASON_PHRASE[wireValue] ?? wireValue;
export const hitlOutcomeLabel = (wireValue: string): string => HITL_OUTCOME_LABEL[wireValue] ?? wireValue;

// Inspect only the final filename suffix; no filesystem access is needed.
export function languageForPath(filePath: string): string {
  const basename = filePath.slice(filePath.lastIndexOf("/") + 1);
  const extensionStart = basename.lastIndexOf(".");
  if (extensionStart <= 0) return ""; // no extension, or a dotfile with no suffix
  return LANGUAGE_BY_EXT[basename.slice(extensionStart).toLowerCase()] ?? "";
}
