import { create } from "zustand";

import type { paths } from "../api/schema";

// Derive filter values from the endpoint query type, not the row status
// string. The runtime options below must still be maintained explicitly.
export type ReviewStatus = NonNullable<
  NonNullable<paths["/api/reviews"]["get"]["parameters"]["query"]>["status"]
>;

/** Select options checked for valid values; satisfies does not prove completeness. */
export const REVIEW_STATUSES = [
  "running",
  "awaiting_approval",
  "awaiting_approval_expired",
  "completed",
  "failed",
  "skipped",
] as const satisfies readonly ReviewStatus[];

// Zustand stores UI selections; TanStack Query stores the fetched API data.
interface FilterState {
  /** Include synthetic is_eval=true reviews; off by default (docs/testing.md). */
  includeEval: boolean;
  /** Server-side status filter (the queue's `?status=`), or null for all. */
  status: ReviewStatus | null;
  /** Free-text filter over the loaded review rows. */
  search: string;
  setIncludeEval: (value: boolean) => void;
  setStatus: (value: ReviewStatus | null) => void;
  setSearch: (value: string) => void;
}

export const useFilters = create<FilterState>()((set) => ({
  includeEval: false,
  status: null,
  search: "",
  setIncludeEval: (includeEval) => set({ includeEval }),
  setStatus: (status) => set({ status }),
  setSearch: (search) => set({ search }),
}));
