"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, Database, BrainCircuit, BarChart3,
  Rocket, Zap, ChevronRight, Activity, PanelLeftClose, PanelLeftOpen
} from "lucide-react";

const navItems = [
  { label: "Dashboard",    href: "/dashboard",         icon: LayoutDashboard },
  { label: "Datasets",     href: "/dashboard/datasets", icon: Database },
  { label: "Train Models", href: "/dashboard/training", icon: BrainCircuit },
  { label: "Leaderboard",  href: "/dashboard/models",   icon: BarChart3 },
  { label: "Predict",      href: "/dashboard/predict",  icon: Zap },
  { label: "Deploy",       href: "/dashboard/deploy",   icon: Rocket },
];

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

export default function Sidebar({ collapsed = false, onToggle }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside className={`fixed left-0 top-16 z-40 hidden h-[calc(100vh-4rem)] flex-shrink-0 flex-col border-r border-outline/30 bg-surface transition-all duration-300 lg:flex ${collapsed ? "w-16" : "w-72"}`}>
      <button
        onClick={onToggle}
        className={`absolute -right-3 top-4 z-50 inline-flex h-6 w-6 items-center justify-center rounded-full border border-outline/40 bg-surface text-text-muted hover:text-text-primary`}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {collapsed ? <PanelLeftOpen className="h-3.5 w-3.5" /> : <PanelLeftClose className="h-3.5 w-3.5" />}
      </button>

      {/* Logo */}
      <div className={`border-b border-outline/25 ${collapsed ? "p-3" : "p-6"}`}>
        <Link href="/" className="flex items-center gap-3 group">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary-deep transition-colors group-hover:bg-primary-deep/90">
            <Activity className="h-5 w-5 text-text-primary" />
          </div>
          {!collapsed && (
            <div>
              <div className="font-display text-sm font-semibold tracking-wide text-text-primary">Axiom Cloud</div>
              <div className="text-[10px] uppercase tracking-[0.14em] text-text-muted">ML Control Plane</div>
            </div>
          )}
        </Link>
      </div>

      {/* Nav */}
      <nav className={`flex-1 space-y-1 ${collapsed ? "p-2" : "p-4"}`}>
        {navItems.map(({ label, href, icon: Icon }) => {
          const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={`group flex items-center rounded-lg border-l-2 text-sm font-medium transition-all
                ${collapsed ? "justify-center px-2 py-2.5" : "gap-3 px-3 py-2.5"}
                ${active
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-transparent text-text-muted hover:bg-surface-variant/45 hover:text-text-primary"
                }`}
            >
              <Icon className={`h-4 w-4 flex-shrink-0 ${active ? "text-primary" : "text-outline group-hover:text-text-muted"}`} />
              {!collapsed && <span className="flex-1">{label}</span>}
              {!collapsed && active && <ChevronRight className="h-3 w-3 text-primary" />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-outline/25 p-4">
        {!collapsed && (
          <div className="text-center font-mono text-xs text-text-muted">
            axiom v1.0.0
          </div>
        )}
      </div>
    </aside>
  );
}
