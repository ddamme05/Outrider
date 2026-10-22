import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Local development API target. Bind uvicorn to loopback (SESSION_RETRO).
// VITE_API_BASE_URL can point the client at a separate deployed API origin.
const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Dashboard reads live under /api.
      "/api": { target: BACKEND, changeOrigin: true },
      // Match only the legacy decision path, preserving the SPA review routes.
      // Vite treats a proxy key starting with ^ as a regular expression.
      "^/reviews/[^/]+/decide$": { target: BACKEND, changeOrigin: true },
      // Privacy is backend-served HTML; match its exact path.
      "^/privacy$": { target: BACKEND, changeOrigin: true },
      // Keep /setup out of this proxy (FUP-230). Live onboarding uses FastAPI
      // serving the built SPA (OUTRIDER_SERVE_SPA=1): GET /setup is the page,
      // while POST /setup and the status/callback descendants are API routes.
      // A path-only proxy would duplicate that method split. The real backend
      // must also receive GitHub callbacks and webhooks; Vite serves component
      // development, not the complete onboarding flow. api/setup.ts reports
      // invalid peer responses as SetupProtocolError.
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
