import { Outlet } from "react-router";

import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";

// Signal shell: sidebar (org-mark + icon nav + footer eval toggle) + main column
// (topbar + scrolling content). Nav is Overview + Reviews only — Audit/Anomalies
// are deferred (no backing endpoint; spec non-goals).
export function App() {
  return (
    <div className="app">
      <Sidebar />
      <div className="main-col">
        <Topbar />
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
