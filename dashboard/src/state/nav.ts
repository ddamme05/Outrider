import { create } from "zustand";

// Mobile-only off-canvas nav state. On desktop the sidebar is always visible
// (CSS keeps the drawer styles behind the mobile breakpoint), so this store is
// inert there. Below the breakpoint the Topbar hamburger toggles `open`, the
// Sidebar slides in via the `.app.nav-open` class, and a scrim tap / Escape /
// nav-link tap closes it. Kept separate from the filters store: this is chrome
// state, not a review filter.
interface NavState {
  /** Whether the mobile nav drawer is open. Always false on desktop. */
  open: boolean;
  setOpen: (value: boolean) => void;
  toggle: () => void;
}

export const useNav = create<NavState>()((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
  toggle: () => set((s) => ({ open: !s.open })),
}));
