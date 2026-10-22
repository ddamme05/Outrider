import { useEffect } from "react";
import { Outlet } from "react-router";

import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";
import { useNav } from "./state/nav";

// Shared shell: sidebar, topbar, and routed content. At the mobile breakpoint,
// CSS turns the sidebar into a drawer controlled by useNav. The close button,
// Escape, and sidebar links dismiss it; desktop CSS keeps the sidebar visible.
export function App() {
  const open = useNav((state) => state.open);
  const setOpen = useNav((state) => state.setOpen);

  // Listen for Escape only while the drawer state is open.
  useEffect(() => {
    if (!open) {
      return;
    }
    function onKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, setOpen]);

  return (
    <div className={`app${open ? " nav-open" : ""}`}>
      <Sidebar />
      {/* CSS hides this close button on desktop and disables pointer events
          while the mobile drawer is closed. */}
      <button
        type="button"
        className="nav-scrim"
        aria-label="Close navigation"
        tabIndex={open ? 0 : -1}
        onClick={() => setOpen(false)}
      />
      <div className="main-col">
        <Topbar />
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
