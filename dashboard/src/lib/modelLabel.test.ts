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
  });

  test("unknown slug is returned verbatim (no lossy guess)", () => {
    expect(prettyModel("some/future-model-v9")).toBe("some/future-model-v9");
    // A GLM-ish slug the version pattern does NOT match stays verbatim too — no collapse to a
    // bare "GLM" that would erase which variant it was.
    expect(prettyModel("accounts/fireworks/models/glm-experimental")).toBe(
      "accounts/fireworks/models/glm-experimental",
    );
  });
});
