// Humanize a raw model slug for DISPLAY. The slug itself
// ("accounts/fireworks/models/glm-5p2", "zai-org/GLM-5.2", "claude-sonnet-4-5") is the
// canonical audit / pricing / host-validation key and is stored + audited verbatim — this
// map is display-only, applied at the pipeline strip and the audit feed so one function
// governs every model label the dashboard shows.
export function prettyModel(model: string): string {
  const l = model.toLowerCase();
  if (l.includes("haiku")) return "Haiku";
  if (l.includes("sonnet")) return "Sonnet";
  if (l.includes("opus")) return "Opus";
  // GPT-5.6 family (openai host; explicit slugs only — the bare "gpt-5.6" alias is
  // rejected by the host profile, so only suffixed slugs reach the audit stream).
  const gpt = l.match(/^gpt-(\d+)\.(\d+)-(sol|terra|luna)$/);
  if (gpt?.[1] && gpt[2] && gpt[3]) {
    const tier = gpt[3];
    return `GPT-${gpt[1]}.${gpt[2]} ${tier.charAt(0).toUpperCase()}${tier.slice(1)}`;
  }
  // GLM slugs differ by host: Baseten renders "zai-org/GLM-5.2", Fireworks renders the
  // version dot as `p` → "accounts/fireworks/models/glm-5p2". Normalize both shapes to
  // "GLM-<major>.<minor>".
  const glm = l.match(/glm[-/]?(\d+)[.p](\d+)/);
  if (glm) return `GLM-${glm[1]}.${glm[2]}`;
  // An unrecognized slug (incl. a future GLM shape the pattern above doesn't match) is returned
  // VERBATIM — no lossy "GLM" guess that would collapse distinct versions into one label.
  return model;
}
