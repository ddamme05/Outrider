import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { adoptTokenFromUrlFragment, useTokenStore } from "./token";

const reset = () => {
  sessionStorage.clear();
  useTokenStore.getState().clearToken();
  window.history.replaceState(null, "", "/");
};

describe("adoptTokenFromUrlFragment", () => {
  beforeEach(reset);
  afterEach(reset);

  test("adopts the token, stores it, and strips the fragment (keeps path)", () => {
    window.history.replaceState(null, "", "/reviews#token=demo_abc123");
    const adopted = adoptTokenFromUrlFragment();

    expect(adopted).toBe("demo_abc123");
    expect(useTokenStore.getState().token).toBe("demo_abc123");
    expect(sessionStorage.getItem("outrider.adminToken")).toBe("demo_abc123");
    expect(window.location.hash).toBe("");
    expect(window.location.pathname).toBe("/reviews");
  });

  test("trims surrounding whitespace in the token value", () => {
    window.history.replaceState(null, "", "/#token=demo_xyz%20");
    expect(adoptTokenFromUrlFragment()).toBe("demo_xyz");
    expect(useTokenStore.getState().token).toBe("demo_xyz");
  });

  test("no fragment is a no-op — normal paste flow is untouched", () => {
    window.history.replaceState(null, "", "/reviews");
    expect(adoptTokenFromUrlFragment()).toBeNull();
    expect(useTokenStore.getState().token).toBeNull();
    expect(window.location.pathname).toBe("/reviews");
  });

  test("empty or whitespace-only token value is rejected, not stored", () => {
    window.history.replaceState(null, "", "/#token=");
    expect(adoptTokenFromUrlFragment()).toBeNull();
    expect(useTokenStore.getState().token).toBeNull();

    window.history.replaceState(null, "", "/#token=%20%20");
    expect(adoptTokenFromUrlFragment()).toBeNull();
    expect(useTokenStore.getState().token).toBeNull();
  });

  test("the normal setToken paste flow still works alongside the helper", () => {
    useTokenStore.getState().setToken("pasted_token");
    expect(useTokenStore.getState().token).toBe("pasted_token");
  });
});
