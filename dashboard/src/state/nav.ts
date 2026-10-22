import { create } from "zustand";

// Drawer state is separate from review filters. Topbar toggles it; the shell
// and sidebar close it. Desktop CSS keeps the sidebar visible regardless of
// this value, including after a resize from mobile.
interface NavState {
  /** Whether the mobile drawer is open. */
  open: boolean;
  setOpen: (value: boolean) => void;
  toggle: () => void;
}

export const useNav = create<NavState>()((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
  toggle: () => set((state) => ({ open: !state.open })),
}));
