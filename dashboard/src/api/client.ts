import createFetchClient, { type Middleware } from "openapi-fetch";
import createClient from "openapi-react-query";

import { clearToken, getToken } from "../auth/token";
import type { paths } from "./schema";

// Same-origin by default (dev proxy or co-deployed); override per-origin deploys.
const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

const authMiddleware: Middleware = {
  onRequest({ request }) {
    const token = getToken();
    if (token) {
      request.headers.set("Authorization", `Bearer ${token}`);
    }
    return request;
  },
  onResponse({ response }) {
    // A 401 means the stored key is missing/stale — drop it so the token-gate
    // (which subscribes to the token store) re-prompts. The 401 still
    // propagates as the query's error.
    if (response.status === 401) {
      clearToken();
    }
    return undefined;
  },
};

// Resolve `fetch` lazily at call time rather than letting openapi-fetch capture
// globalThis.fetch at construction. The client is built at module import; a
// captured reference would bind whatever fetch existed then — which breaks test
// interceptors (MSW) that patch globalThis.fetch after import, and any future
// runtime fetch swap. Calling through globalThis on each request is free.
const fetchClient = createFetchClient<paths>({
  baseUrl,
  fetch: (...args) => globalThis.fetch(...args),
});
fetchClient.use(authMiddleware);

/** Typed TanStack-Query hooks bound to the OpenAPI schema. */
export const $api = createClient(fetchClient);
