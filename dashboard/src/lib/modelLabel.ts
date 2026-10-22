// Short display labels for recorded model slugs. Audit and pricing retain
// the original slug; unknown forms are displayed verbatim.
export function prettyModel(model: string): string {
  const modelSlug = model.toLowerCase();
  if (modelSlug.includes("haiku")) return "Haiku";
  if (modelSlug.includes("sonnet")) return "Sonnet";
  if (modelSlug.includes("opus")) return "Opus";
  // Explicit GPT version/tier slugs; unsuffixed aliases fall through.
  const gptMatch = modelSlug.match(/^gpt-(\d+)\.(\d+)-(sol|terra|luna)$/);
  if (gptMatch?.[1] && gptMatch[2] && gptMatch[3]) {
    const tier = gptMatch[3];
    return `GPT-${gptMatch[1]}.${gptMatch[2]} ${tier.charAt(0).toUpperCase()}${tier.slice(1)}`;
  }
  // Recognize both dot and p version separators used by GLM hosts.
  const glmMatch = modelSlug.match(/glm[-/]?(\d+)[.p](\d+)/);
  if (glmMatch) return `GLM-${glmMatch[1]}.${glmMatch[2]}`;
  // Preserve unknown versions rather than guessing a family label.
  return model;
}
