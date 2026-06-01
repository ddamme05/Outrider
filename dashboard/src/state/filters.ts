import { create } from "zustand";

import type { paths } from "../api/schema";

// The valid `?status=` filter values, derived from the actual query-param
// contract (the endpoint's ReviewStatusFilter literal) — not ReviewListItem.status
// (a plain string). Keeps the filter options in lockstep with the API.
export type ReviewStatus = NonNullable<
  NonNullable<paths["/api/reviews"]["get"]["parameters"]["query"]>["status"]
>;

/** Runtime list of the filterable statuses (the `<select>` options); `satisfies`
 * guards against a typo'd value drifting from the contract type. */
export const REVIEW_STATUSES = [
  "running",
  "awaiting_approval",
  "awaiting_approval_expired",
  "completed",
  "failed",
  "skipped",
] as const satisfies readonly ReviewStatus[];

// Client-only UI state (filters + the eval toggle). Server state lives in
// react-query; this is the zustand half of the split.
interface FilterState {
  /** Surface synthetic `is_eval=true` reviews (the topbar "Eval" toggle).
   * Off by default — eval-isolation default per docs/testing.md. */
  includeEval: boolean;
  /** Server-side status filter (the queue's `?status=`), or null for all. */
  status: ReviewStatus | null;
  /** Client-side free-text filter over the loaded rows (the ⌘K-ish bar). */
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
