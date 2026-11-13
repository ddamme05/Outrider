import { describe, expect, test } from "vitest";

import { prettyModel } from "./modelLabel";

describe("prettyModel", () => {
  test("Claude slugs → family label", () => {
    expect(prettyModel("claude-haiku-4-5")).toBe("Haiku");
    expect(prettyModel("claude-sonnet-4-5")).toBe("Sonnet");
    expect(prettyModel("claude-sonnet-5")).toBe("Sonnet");
    expect(prettyModel("claude-opus-4-8")).toBe("Opus");
  });

  test("GLM slugs → GLM-<major>.<minor>, both host shapes", () => {
    // Fireworks renders the version dot as `p`; Baseten uses a real dot. Both → GLM-5.2.
    expect(prettyModel("accounts/fireworks/models/glm-5p2")).toBe("GLM-5.2");
    expect(prettyModel("zai-org/GLM-5.2")).toBe("GLM-5.2");
    expect(prettyModel("accounts/fireworks/models/glm-5p3-flash")).toBe("GLM-5.3 Flash");
    expect(prettyModel("zai-org/GLM-5.3-Flash")).toBe("GLM-5.3 Flash");
  });

  test("unknown slug is returned verbatim (no lossy guess)", () => {
    expect(prettyModel("some/future-model-v9")).toBe("some/future-model-v9");
    // An unmatched version keeps its full slug.
    expect(prettyModel("accounts/fireworks/models/glm-experimental")).toBe(
      "accounts/fireworks/models/glm-experimental",
    );
  });
});

test("explicit GPT versions and tiers get short labels", () => {
  expect(prettyModel("gpt-5.6-sol")).toBe("GPT-5.6 Sol");
  expect(prettyModel("gpt-5.6-terra")).toBe("GPT-5.6 Terra");
  expect(prettyModel("gpt-5.6-luna")).toBe("GPT-5.6 Luna");
  expect(prettyModel("gpt-6.1-sol")).toBe("GPT-6.1 Sol");
  expect(prettyModel("gpt-6.1-terra")).toBe("GPT-6.1 Terra");
  expect(prettyModel("gpt-6-luna")).toBe("GPT-6 Luna");
});

test("non-family gpt shapes fall through verbatim (no lossy guess)", () => {
  expect(prettyModel("gpt-5.6")).toBe("gpt-5.6"); // outside the display pattern
  expect(prettyModel("gpt-5.6-mini")).toBe("gpt-5.6-mini");
  expect(prettyModel("gpt-4o")).toBe("gpt-4o");
});
