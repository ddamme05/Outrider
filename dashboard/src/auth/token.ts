import { create } from "zustand";

// V1 operator auth: a single static admin API key (DECISIONS.md#011). The
// operator pastes it into the token-gate; we hold it in memory + sessionStorage
// and send it as `Authorization: Bearer`. NEVER baked into the bundle (that
// would ship the secret); sessionStorage (not localStorage) so it clears with
// the tab.
const STORAGE_KEY = "outrider.adminToken";

interface TokenState {
  token: string | null;
  setToken: (token: string) => void;
  clearToken: () => void;
}

export const useTokenStore = create<TokenState>()((set) => ({
  token: sessionStorage.getItem(STORAGE_KEY),
  setToken: (token) => {
    sessionStorage.setItem(STORAGE_KEY, token);
    set({ token });
  },
  clearToken: () => {
    sessionStorage.removeItem(STORAGE_KEY);
    set({ token: null });
  },
}));

// Non-React accessors for the openapi-fetch middleware (runs outside React).
export const getToken = (): string | null => useTokenStore.getState().token;
export const clearToken = (): void => {
  useTokenStore.getState().clearToken();
};

// One-click demo access: adopt a token passed in the URL fragment of a shared link
// (e.g. `https://demo.example/#token=demo_xxx`). The token is supplied at runtime via
// the link, NOT baked into the bundle (the "never ship the secret" rule above still
// holds), then stored via the normal setToken path so it round-trips through
// sessionStorage exactly like a pasted key. The fragment is stripped immediately so the
// token doesn't linger in the address bar or history; routing is createBrowserRouter
// (path-based), so clearing the hash never affects navigation. Returns the adopted token,
// or null when no valid (non-empty, trimmed) token is present — a no-op for normal visits.
export function adoptTokenFromUrlFragment(): string | null {
  const hash = window.location.hash.replace(/^#/, "");
  const token = new URLSearchParams(hash).get("token")?.trim();
  if (!token) {
    return null;
  }
  useTokenStore.getState().setToken(token);
  window.history.replaceState(
    null,
    "",
    window.location.pathname + window.location.search,
  );
  return token;
}
