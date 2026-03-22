import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { FlaskConical, PanelLeftClose } from "lucide-react";
import { useStore } from "../store";
import { NAV_ITEMS } from "@/constants.js";
import { Button } from "@/components/ui/button.jsx";

const Layout = () => {
  const activeJobId = useStore((s) => s.activeJobId);
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={`flex shrink-0 flex-col bg-sidebar border-r border-sidebar-border transition-all duration-300 ${
          expanded ? "w-60" : "w-16"
        }`}
      >
        {/* Brand & Toggle */}
        <div
          className={`flex items-center border-b border-sidebar-border py-4 transition-all duration-300 ${
            expanded ? "justify-between px-4" : "justify-center px-0"
          }`}
        >
          {expanded && (
            <div className="flex items-center gap-2.5 overflow-hidden w-auto opacity-100">
              <FlaskConical className="shrink-0 text-primary" size={20} />
              <span className="text-base font-semibold tracking-tight text-sidebar-foreground whitespace-nowrap">
                T-AutoNLP
              </span>
            </div>
          )}

          <Button
            variant="ghost"
            size="icon"
            onClick={() => setExpanded(!expanded)}
            aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
            aria-expanded={expanded}
            className="shrink-0 text-sidebar-foreground hover:bg-accent transition-all rounded-lg h-8 w-8"
            title={expanded ? "Collapse Sidebar" : "Expand Sidebar"}
          >
            {expanded ? (
              <PanelLeftClose size={18} />
            ) : (
              <FlaskConical size={22} className="text-primary" />
            )}
          </Button>
        </div>

        {expanded && (
          <div className="px-5 py-2">
            <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Multi-Objective AutoML
            </p>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 space-y-2 p-3 overflow-hidden">
          {NAV_ITEMS.map(({ to, end, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              aria-label={label}
              title={!expanded ? label : undefined}
              className={({ isActive }) =>
                [
                  "group flex items-center rounded-lg p-2 text-sm transition-colors",
                  expanded ? "gap-3" : "justify-center",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-sidebar-foreground hover:bg-accent hover:text-accent-foreground",
                ].join(" ")
              }
            >
              {({ isActive }) => (
                <>
                  <Icon
                    size={16}
                    className={`shrink-0 ${
                      isActive
                        ? "text-primary-foreground"
                        : "text-muted-foreground group-hover:text-accent-foreground"
                    }`}
                  />
                  <span className={expanded ? "truncate whitespace-nowrap" : "sr-only"}>
                    {label}
                  </span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Active job badge */}
        {activeJobId && (
          <div className="mx-3 mb-3 shrink-0">
            <div
              className={`rounded-lg bg-primary/10 border border-primary/20 ${expanded ? "px-3 py-2" : "p-2 flex justify-center"}`}
            >
              {expanded ? (
                <>
                  <p className="text-xs font-medium text-primary">Job running</p>
                  <p className="mt-0.5 truncate font-mono text-xs text-primary/60">{activeJobId}</p>
                </>
              ) : (
                <div
                  className="h-2 w-2 rounded-full bg-primary animate-pulse"
                  title={`Running: ${activeJobId}`}
                />
              )}
            </div>
          </div>
        )}
      </aside>

      <main className="flex-1 overflow-auto bg-background">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
