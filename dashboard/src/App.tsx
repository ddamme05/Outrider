import { NavLink, Outlet } from "react-router";

// Sidebar shell. V1 nav: Home + Reviews. Audit-log + Anomalies are deferred
// (spec non-goals), so they're intentionally absent.
export function App() {
  return (
    <div className="app">
      <nav className="sidebar">
        <div className="sidebar__brand">Outrider</div>
        <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : undefined)}>
          Home
        </NavLink>
        <NavLink to="/reviews" className={({ isActive }) => (isActive ? "active" : undefined)}>
          Reviews
        </NavLink>
      </nav>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
