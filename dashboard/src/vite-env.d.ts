/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API origin when the SPA and API are different origins. Empty/undefined =
   * same-origin (dev proxy or co-deployed). */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
