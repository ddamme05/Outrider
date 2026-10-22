import { create } from "zustand";

// Operator auth uses the static admin key (DECISIONS.md#011). Keep it in
// memory and tab-scoped sessionStorage; API clients send it as a bearer token.
// Supply the key at runtime so the bundle never contains a credential.
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

// Shared demo links may supply #token=... at runtime. Store a non-empty,
// trimmed token through the normal auth path, then remove the fragment with
// replaceState. Path-based routing and the query string stay intact.
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
