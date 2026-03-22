"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import Link from "next/link";

export default function HomePage() {
  const router = useRouter();
  const { user, loading } = useAuth();

  useEffect(() => {
    if (!loading && user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-surface">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-text-secondary">Loading...</p>
        </div>
      </div>
    );
  }

  if (user) return null;

  return (
    <main className="min-h-screen bg-gradient-to-br from-bg via-surface to-surface-variant text-text-primary">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center px-6 text-center">
        <p className="mb-3 text-xs uppercase tracking-[0.18em] text-primary-300">Axiom Cloud AI</p>
        <h1 className="mb-4 text-4xl font-bold md:text-5xl">Production-grade AutoML platform</h1>
        <p className="mb-8 max-w-2xl text-text-secondary">
          Upload datasets, train models, compare performance, and deploy predictions — all from one dashboard.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/login"
            className="rounded-lg bg-primary-600 px-5 py-2.5 font-semibold text-white transition hover:bg-primary-700"
          >
            Sign In
          </Link>
          <Link
            href="/dashboard"
            className="rounded-lg border border-outline px-5 py-2.5 font-semibold text-text-primary transition hover:bg-surface-hover"
          >
            Go to Dashboard
          </Link>
        </div>
      </div>
    </main>
  );
}
