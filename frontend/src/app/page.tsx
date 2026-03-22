"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import Link from "next/link";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  FlaskConical,
  Layers,
  Rocket,
  SearchCode,
  ShieldCheck,
  Upload,
  WandSparkles,
} from "lucide-react";

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
    <main className="relative min-h-screen overflow-hidden bg-bg text-text-primary">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_10%_0%,rgba(110,59,215,.22),transparent_38%),radial-gradient(circle_at_85%_12%,rgba(208,188,255,.16),transparent_32%)]" />
      <div className="pointer-events-none absolute inset-0 bg-grid-pattern bg-grid-size opacity-30" />

      <div className="relative mx-auto max-w-7xl px-6 pb-20 pt-12 lg:px-10">
        {/* Hero */}
        <section className="grid items-center gap-10 py-10 lg:grid-cols-2 lg:py-16">
          <div>
            <p className="section-kicker mb-4">Axiom Cloud AI</p>
            <h1 className="max-w-xl text-4xl font-semibold leading-tight md:text-5xl">
              From Raw Data to Deployed Models in Minutes
            </h1>
            <p className="mt-5 max-w-xl text-sm text-text-muted md:text-base">
              A production-grade AutoML platform with experiment tracking, model comparison, and real-time inference APIs.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/login" className="btn-primary">
                Start Building
              </Link>
              <Link href="#workflow" className="btn-outline">
                View Demo
              </Link>
            </div>
          </div>

          <div className="panel-glass void-shadow relative overflow-hidden rounded-2xl p-6">
            <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-primary-deep/20 blur-3xl" />
            <div className="absolute -bottom-10 -left-8 h-36 w-36 rounded-full bg-primary/15 blur-3xl" />
            <div className="relative h-[340px] rounded-xl border border-outline/25 bg-surface/70 p-4">
              <div className="absolute left-8 top-10 flex items-center gap-2 rounded-md border border-outline/30 bg-surface-variant/80 px-2 py-1 text-[11px] text-text-muted">
                <Database className="h-3.5 w-3.5 text-primary" />
                Dataset Node
              </div>
              <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-2 rounded-md border border-outline/30 bg-surface-variant/80 px-2 py-1 text-[11px] text-text-muted">
                <BrainCircuit className="h-3.5 w-3.5 text-primary" />
                AutoML Pipeline
              </div>
              <div className="absolute bottom-10 right-8 flex items-center gap-2 rounded-md border border-outline/30 bg-surface-variant/80 px-2 py-1 text-[11px] text-text-muted">
                <Rocket className="h-3.5 w-3.5 text-primary" />
                Inference API
              </div>

              <div className="absolute left-[22%] top-[20%] h-2 w-2 animate-pulse rounded-full bg-primary" />
              <div className="absolute left-[50%] top-[50%] h-2 w-2 animate-pulse rounded-full bg-primary" />
              <div className="absolute left-[78%] top-[75%] h-2 w-2 animate-pulse rounded-full bg-primary" />

              <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                <path d="M22 23 C34 35, 42 41, 50 50" fill="none" stroke="rgba(208,188,255,.45)" strokeWidth="0.6" />
                <path d="M50 50 C60 60, 68 67, 78 75" fill="none" stroke="rgba(208,188,255,.45)" strokeWidth="0.6" />
              </svg>
            </div>
          </div>
        </section>

        {/* Trust strip */}
        <section className="panel mb-12 flex flex-col items-start justify-between gap-4 rounded-xl px-5 py-4 md:flex-row md:items-center">
          <p className="text-sm text-text-muted">Inspired by Vertex AI and Kaggle AutoML</p>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="rounded-full border border-outline/30 bg-surface-variant/60 px-3 py-1 text-text-muted">Production Workflows</span>
            <span className="rounded-full border border-outline/30 bg-surface-variant/60 px-3 py-1 text-text-muted">API-First Design</span>
            <span className="rounded-full border border-outline/30 bg-surface-variant/60 px-3 py-1 text-text-muted">Secure by Default</span>
          </div>
        </section>

        {/* Core capabilities */}
        <section className="mb-14">
          <p className="section-kicker mb-3">Core capabilities</p>
          <div className="grid gap-4 md:grid-cols-3">
            {[
              {
                icon: BrainCircuit,
                title: "AutoML Engine",
                text: "Train multiple models with cross-validation and automatic task detection.",
              },
              {
                icon: FlaskConical,
                title: "Experiment Tracking",
                text: "Track runs, compare models, and manage experiments with a unified registry.",
              },
              {
                icon: Rocket,
                title: "Model Deployment",
                text: "Deploy models instantly and access them via clean REST APIs.",
              },
            ].map(({ icon: Icon, title, text }) => (
              <article key={title} className="glass rounded-xl border border-outline/30 p-5 shadow-[0_18px_40px_rgba(0,0,0,0.35)]">
                <Icon className="mb-4 h-5 w-5 text-primary" />
                <h3 className="mb-2 text-lg font-semibold">{title}</h3>
                <p className="text-sm text-text-muted">{text}</p>
              </article>
            ))}
          </div>
        </section>

        {/* Workflow */}
        <section id="workflow" className="mb-14">
          <p className="section-kicker mb-3">Visual workflow</p>
          <div className="panel rounded-xl p-5">
            <div className="grid gap-5 md:grid-cols-5 md:items-center">
              {[
                { icon: Upload, label: "Upload Dataset" },
                { icon: BrainCircuit, label: "Train Models" },
                { icon: BarChart3, label: "Compare" },
                { icon: Rocket, label: "Deploy" },
                { icon: Activity, label: "Predict" },
              ].map(({ icon: Icon, label }, idx, arr) => (
                <div key={label} className="relative">
                  <div className="flex items-center gap-3 rounded-lg border border-outline/25 bg-surface-variant/30 px-3 py-3">
                    <Icon className="h-4 w-4 text-primary" />
                    <span className="text-sm text-text-primary">{label}</span>
                  </div>
                  {idx < arr.length - 1 && (
                    <div className="absolute right-[-18px] top-1/2 hidden h-[1px] w-8 -translate-y-1/2 bg-gradient-to-r from-primary/40 to-transparent md:block" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Feature depth */}
        <section className="mb-14">
          <p className="section-kicker mb-3">Feature depth</p>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[
              { icon: Layers, title: "Interactive Leaderboards" },
              { icon: BarChart3, title: "Feature Importance Visualization" },
              { icon: Database, title: "Built-in Datasets" },
              { icon: SearchCode, title: "EDA Reports with leakage detection" },
              { icon: WandSparkles, title: "Hyperparameter tuning with time budgets" },
            ].map(({ icon: Icon, title }) => (
              <article key={title} className="panel rounded-xl p-5">
                <Icon className="mb-3 h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium text-text-primary">{title}</h3>
              </article>
            ))}
          </div>
        </section>

        {/* Tech credibility */}
        <section className="mb-14 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <div className="panel-glass rounded-xl p-6">
            <p className="section-kicker mb-3">Tech credibility</p>
            <h2 className="mb-2 text-2xl font-semibold">Built with FastAPI, Next.js, and modern ML pipelines</h2>
            <p className="text-sm text-text-muted">
              Architecture-first platform design optimized for secure, authenticated model training and low-friction inference operations.
            </p>
          </div>
          <div className="grid gap-3">
            <div className="panel rounded-lg p-4 text-sm text-text-muted"><span className="text-text-primary">Backend:</span> FastAPI, SQLAlchemy, async job orchestration</div>
            <div className="panel rounded-lg p-4 text-sm text-text-muted"><span className="text-text-primary">Frontend:</span> Next.js App Router, TypeScript, Tailwind UI system</div>
            <div className="panel rounded-lg p-4 text-sm text-text-muted"><span className="text-text-primary">ML Stack:</span> scikit-learn, XGBoost, LightGBM, explainability tooling</div>
          </div>
        </section>

        {/* Final CTA */}
        <section className="panel-glass rounded-2xl p-7 text-center">
          <div className="mx-auto max-w-2xl">
            <p className="section-kicker mb-3">Final call</p>
            <h2 className="mb-3 text-3xl font-semibold">Build smarter models, faster.</h2>
            <p className="mb-6 text-sm text-text-muted">Production-ready workflows for teams that want control, speed, and measurable model quality.</p>
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Link href="/login" className="btn-primary">Get Started</Link>
              <Link href="/dashboard" className="btn-outline">View Dashboard</Link>
            </div>
          </div>
        </section>

        <div className="mt-10 flex items-center justify-center gap-2 text-xs text-text-muted">
          <ShieldCheck className="h-3.5 w-3.5" />
          Authenticated compute · API-level protection · production posture
        </div>
      </div>
    </main>
  );
}
