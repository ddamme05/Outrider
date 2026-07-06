// Humanized finding labels for the dashboard — the TypeScript mirror of the Python source
// `src/outrider/presentation/finding_sections.py`. Keys are the LOWERCASE StrEnum wire values the
// API serializes (`sql_injection`, `observed`, `inline_comment`, …); values are the display labels.
//
// The five SHARED maps (SEVERITY_LABEL, TYPE_LABEL, TIER_PHRASE, DEST_LABEL, DIMENSION_LABEL) must
// match the Python maps byte-for-byte — `tests/unit/test_finding_sections_ts_parity.py` fails CI if
// a label drifts or a new enum member ships without a label here (the same fail-closed totality the
// Python module asserts at import). ELIGIBILITY_PHRASE / ELIGIBILITY_REASON_PHRASE / HITL_OUTCOME_LABEL
// are dashboard-only (no Python counterpart; the CLI/GitHub/Slack channels never render them); the
// parity test still totality-checks them against PublishEligibility / PublishEligibilityReason /
// PerFindingOutcome so a new backend enum member can't render as a raw slug unnoticed.
//
// Runtime lookups go through the *Label()/*Phrase() helpers, which fall back to the raw wire value
// for an unknown key — the SPA degrades gracefully against a newer backend rather than rendering
// `undefined`. The parity test guarantees the shipped maps are total against the CURRENT enums, so
// that fallback only ever fires on a genuinely-ahead backend.

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

// --- Dashboard-only (no Python counterpart) -------------------------------------------------

// PublishEligibility → phrase. Whether a routed finding actually materialized on GitHub.
export const ELIGIBILITY_PHRASE: Record<string, string> = {
  eligible: "Eligible to post",
  withheld: "Withheld",
};

// PublishEligibilityReason → phrase. Why a finding was withheld (or forensically tagged).
export const ELIGIBILITY_REASON_PHRASE: Record<string, string> = {
  hitl_required_node_absent: "HITL gate not reached",
  unexpected_override_fields_present: "Malformed override fields",
  routing_emission_failed: "Routing failed",
  hitl_decision_missing: "No HITL decision",
  hitl_rejected: "Rejected by reviewer",
  hitl_suppressed: "Suppressed by reviewer",
};

// PerFindingOutcome → past-tense label for the HITL provenance chip.
export const HITL_OUTCOME_LABEL: Record<string, string> = {
  approve: "Approved",
  reject: "Rejected",
  suppress: "Suppressed",
  severity_override: "Severity overridden",
};

// Fence info-string per file extension — mirror of Python `_LANGUAGE_BY_EXT`. Drives the language
// chip on the evidence code block. "" (unknown) → no chip. Extension → highlight token only.
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

// --- Helpers (fallback to the raw wire value; never render `undefined`) ----------------------

export const severityLabel = (v: string): string => SEVERITY_LABEL[v] ?? v;
export const typeLabel = (v: string): string => TYPE_LABEL[v] ?? v;
export const tierPhrase = (v: string): string => TIER_PHRASE[v] ?? v;
export const destLabel = (v: string): string => DEST_LABEL[v] ?? v;
export const dimensionLabel = (v: string): string => DIMENSION_LABEL[v] ?? v;
export const eligibilityPhrase = (v: string): string => ELIGIBILITY_PHRASE[v] ?? v;
export const eligibilityReasonPhrase = (v: string): string => ELIGIBILITY_REASON_PHRASE[v] ?? v;
export const hitlOutcomeLabel = (v: string): string => HITL_OUTCOME_LABEL[v] ?? v;

/** Highlight token for a file path's fenced snippet, or "" when unknown. Pure suffix lookup —
 * mirror of Python `language_for_path`. No path math beyond the final `.ext`. */
export function languageForPath(filePath: string): string {
  const base = filePath.slice(filePath.lastIndexOf("/") + 1);
  const dot = base.lastIndexOf(".");
  if (dot <= 0) return ""; // no extension, or a dotfile with no suffix
  return LANGUAGE_BY_EXT[base.slice(dot).toLowerCase()] ?? "";
}
