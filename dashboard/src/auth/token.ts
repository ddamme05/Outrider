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
