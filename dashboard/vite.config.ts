import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev backend (uvicorn). 127.0.0.1, never 0.0.0.0 (WSL2 exposure — see the
// project SESSION_RETRO). Override with VITE_API_BASE_URL in production where
// the SPA and API may be different origins.
const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // The read-API — queue / detail / findings / replay are all under /api.
      "/api": { target: BACKEND, changeOrigin: true },
      // The one legacy write path. A SPECIFIC RegExp rule (Vite treats a key
      // starting with ^ as a regex) so it matches ONLY POST /reviews/{id}/decide
      // and never swallows the SPA's own /reviews + /reviews/:id client routes.
      "^/reviews/[^/]+/decide$": { target: BACKEND, changeOrigin: true },
      // The public privacy page is backend-served HTML (FastAPI GET /privacy), NOT an SPA
      // route. Without this, the Sidebar footer link resolves against the Vite dev server
      // and React Router 404s. An exact ^…$ regex so it can't shadow a future SPA route.
      "^/privacy$": { target: BACKEND, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    // Absolute base so the openapi-fetch client issues absolute URLs that
    // node's fetch + MSW can resolve/intercept (relative URLs fail in node).
    env: { VITE_API_BASE_URL: "http://localhost" },
  },
});
