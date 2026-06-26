import { useEffect } from "react";
import { Outlet } from "react-router";

import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";
import { useNav } from "./state/nav";

// Signal shell: sidebar (org-mark + icon nav + footer eval toggle) + main column
// (topbar + scrolling content). Nav is Overview + Reviews only — Audit/Anomalies
// are deferred (no backing endpoint; spec non-goals).
//
// On mobile (≤768px, CSS-gated) the sidebar becomes an off-canvas drawer: the
// Topbar hamburger toggles `useNav.open`, the `.app.nav-open` class slides the
// drawer in, and the scrim button / Escape / a nav-link tap closes it. On
// desktop the drawer styles are inert and `open` stays false.
export function App() {
  const open = useNav((s) => s.open);
  const setOpen = useNav((s) => s.setOpen);

  // Escape closes the mobile drawer. No-op on desktop (open never becomes true).
  useEffect(() => {
    if (!open) {
      return;
    }
    function onKey(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  return (
    <div className={`app${open ? " nav-open" : ""}`}>
      <Sidebar />
      {/* Mobile-only scrim behind the open drawer; a tap anywhere off the drawer
          closes it. Hidden (display:none) on desktop so it never enters the
          grid; pointer-events gate it when closed on mobile. */}
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
