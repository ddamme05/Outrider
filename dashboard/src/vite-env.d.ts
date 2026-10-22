/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional API origin; empty or absent uses the current origin. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
