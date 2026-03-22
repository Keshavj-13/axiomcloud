import Sidebar from "@/components/layout/Sidebar";
import { Bell, CircleUserRound, Command, Search } from "lucide-react";

export const dynamic = "force-dynamic";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-bg text-text-primary">
      <header className="fixed inset-x-0 top-0 z-50 border-b border-outline/30 bg-bg/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1800px] items-center justify-between px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-lg border border-outline/30 bg-surface-variant/40 px-3 py-1.5 text-xs text-text-muted md:flex">
              <Search className="h-3.5 w-3.5" />
              Search models, jobs, datasets
            </div>
            <div className="rounded-lg border border-outline/30 bg-surface px-2 py-1 text-[11px] uppercase tracking-[0.14em] text-text-muted">
              Production Workspace
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="rounded-lg border border-outline/30 bg-surface p-2 text-text-muted hover:text-text-primary">
              <Bell className="h-4 w-4" />
            </button>
            <button className="rounded-lg border border-outline/30 bg-surface p-2 text-text-muted hover:text-text-primary">
              <Command className="h-4 w-4" />
            </button>
            <button className="inline-flex items-center gap-2 rounded-lg border border-outline/30 bg-surface px-3 py-1.5 text-xs text-text-muted hover:text-text-primary">
              <CircleUserRound className="h-4 w-4" />
              Ops Account
            </button>
          </div>
        </div>
      </header>

      <Sidebar />

      <main className="pt-16 lg:pl-72">
        <div className="mx-auto min-h-[calc(100vh-4rem)] max-w-[1800px] px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
          {children}
        </div>
      </main>
    </div>
  );
}
