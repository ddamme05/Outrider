// Short names for recorded model IDs. Audit and pricing keep the original ID;
// unknown forms display exactly as received.

export function prettyModel(model: string): string {
  const modelSlug = model.toLowerCase();
  if (modelSlug.includes("haiku")) return "Haiku";
  if (modelSlug.includes("sonnet")) return "Sonnet";
  if (modelSlug.includes("opus")) return "Opus";
  // Explicit GPT version/tier slugs; unsuffixed aliases fall through.
  const gptMatch = modelSlug.match(/^gpt-(\d+)(?:\.(\d+))?-(sol|terra|luna)$/);
  if (gptMatch?.[1] && gptMatch[3]) {
    const version = gptMatch[2] ? `${gptMatch[1]}.${gptMatch[2]}` : gptMatch[1];
    const tier = gptMatch[3];
    return `GPT-${version} ${tier.charAt(0).toUpperCase()}${tier.slice(1)}`;
  }
  // Recognize both dot and p version separators used by GLM hosts.
  const glmMatch = modelSlug.match(/glm[-/]?(\d+)[.p](\d+)(-flash)?/);
  if (glmMatch) {
    const variant = glmMatch[3] ? " Flash" : "";
    return `GLM-${glmMatch[1]}.${glmMatch[2]}${variant}`;
  }
  // Preserve unknown versions rather than guessing a family label.
  return model;
}
