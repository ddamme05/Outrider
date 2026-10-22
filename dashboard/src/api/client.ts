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
    // Clearing a stale key reopens the token gate; the query still receives the 401.
    if (response.status === 401) {
      clearToken();
    }
    return undefined;
  },
};

// Resolve fetch per request so interceptors installed after import still work.
const fetchClient = createFetchClient<paths>({
  baseUrl,
  fetch: (...args) => globalThis.fetch(...args),
});
fetchClient.use(authMiddleware);

/** Typed TanStack-Query hooks bound to the OpenAPI schema. */
export const $api = createClient(fetchClient);
