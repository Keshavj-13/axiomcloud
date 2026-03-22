"use client";

import Sidebar from "@/components/layout/Sidebar";
import Link from "next/link";
import { Bell, CircleUserRound, Command, Search, LogOut } from "lucide-react";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { useRouter } from "next/navigation";
import { ProtectedRoute } from "@/components/protected-route";

export const dynamic = "force-dynamic";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { user, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    const stored = window.localStorage.getItem("dashboard.sidebar.collapsed");
    if (stored === "1") setSidebarCollapsed(true);
  }, []);

  const toggleSidebar = () => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      window.localStorage.setItem("dashboard.sidebar.collapsed", next ? "1" : "0");
      return next;
    });
  };

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-bg text-text-primary">
        <header className="fixed inset-x-0 top-0 z-50 border-b border-outline/30 bg-bg/90 backdrop-blur">
          <div className="mx-auto flex h-16 max-w-[1800px] items-center justify-between px-4 sm:px-6 lg:px-8">
            <div className="flex items-center gap-3 min-w-0">
              <form action="/dashboard/search" method="get" className="hidden md:flex">
                <label className="flex items-center gap-2 rounded-lg border border-outline/30 bg-surface-variant/40 px-3 py-1.5 text-xs text-text-muted">
                  <Search className="h-3.5 w-3.5" />
                  <input
                    type="search"
                    name="q"
                    placeholder="Search models, jobs, datasets"
                    className="w-56 bg-transparent text-text-primary placeholder:text-text-muted focus:outline-none"
                  />
                </label>
              </form>
              <div className="rounded-lg border border-outline/30 bg-surface px-2 py-1 text-[11px] uppercase tracking-[0.14em] text-text-muted">
                Production Workspace
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/dashboard/training"
                className="rounded-lg border border-outline/30 bg-surface p-2 text-text-muted hover:text-text-primary"
                aria-label="Go to training jobs"
                title="Training jobs"
              >
                <Bell className="h-4 w-4" />
              </Link>
              <Link
                href="/dashboard"
                className="rounded-lg border border-outline/30 bg-surface p-2 text-text-muted hover:text-text-primary"
                aria-label="Go to dashboard home"
                title="Dashboard home"
              >
                <Command className="h-4 w-4" />
              </Link>
              <div className="flex items-center gap-2">
                <div className="rounded-lg border border-outline/30 bg-surface px-3 py-1.5 text-xs text-text-muted">
                  {user?.email?.split("@")[0] || "User"}
                </div>
                <button
                  onClick={handleLogout}
                  className="rounded-lg border border-outline/30 bg-surface p-2 text-text-muted hover:text-red-500 hover:border-red-500/30 transition"
                  aria-label="Logout"
                  title="Logout"
                >
                  <LogOut className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </header>

        <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} />

        <main className={`pt-16 transition-all duration-300 ${sidebarCollapsed ? "lg:pl-16" : "lg:pl-72"}`}>
          <div className="mx-auto min-h-[calc(100vh-4rem)] max-w-[1800px] px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
            {children}
          </div>
        </main>
      </div>
    </ProtectedRoute>
  );
}
