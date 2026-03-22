"use client";
import { useEffect, useState } from "react";
import { metricsAPI, trainingAPI } from "@/lib/api";
import { Database, BrainCircuit, Rocket, CheckCircle, Clock, AlertTriangle, TrendingUp } from "lucide-react";
import Link from "next/link";
import MetricCard from "@/components/ui/MetricCard";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";

interface Summary {
  total_datasets: number;
  total_models: number;
  deployed_models: number;
  completed_jobs: number;
  failed_jobs: number;
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [recentJobs, setRecentJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [summaryRes, jobsRes] = await Promise.all([
          metricsAPI.getDashboardSummary(),
          trainingAPI.listJobs(),
        ]);
        setSummary(summaryRes.data);
        setRecentJobs(jobsRes.data.slice(0, 5));
      } catch { /* backend may not be running */ }
      finally { setLoading(false); }
    }
    fetchData();
  }, []);

  const statCards = [
    { label: "Datasets", value: summary?.total_datasets ?? "–", icon: Database },
    { label: "Trained Models", value: summary?.total_models ?? "–", icon: BrainCircuit },
    { label: "Deployed", value: summary?.deployed_models ?? "–", icon: Rocket },
    { label: "Completed Jobs", value: summary?.completed_jobs ?? "–", icon: CheckCircle },
  ];

  const statusIcon = (s: string) => {
    if (s === "completed") return <CheckCircle className="h-4 w-4 text-emerald-300" />;
    if (s === "running") return <Clock className="h-4 w-4 animate-spin text-amber-300" />;
    if (s === "failed") return <AlertTriangle className="h-4 w-4 text-red-300" />;
    return <Clock className="h-4 w-4 text-text-muted" />;
  };

  return (
    <div className="mx-auto max-w-7xl">
      {/* Header */}
      <div className="mb-10">
        <div className="section-kicker mb-2">Dashboard</div>
        <h1 className="font-display text-4xl font-semibold text-text-primary mb-2">
          ML Operations Overview
        </h1>
        <p className="text-text-muted">Track datasets, training, and runtime posture from one control room.</p>
      </div>

      {/* Stat Cards */}
      <div className="mb-8 grid grid-cols-2 gap-4 lg:grid-cols-4">
        {statCards.map(({ label, value, icon }) => (
          <MetricCard key={label} label={label} value={loading ? "..." : value} icon={icon} />
        ))}
      </div>

      {/* Quick Actions */}
      <div className="mb-8 grid gap-4 md:grid-cols-3">
        {[
          { href: "/dashboard/datasets", label: "Upload Dataset", desc: "Ingest CSV/XLSX to registry", icon: Database },
          { href: "/dashboard/training", label: "Train Models", desc: "Launch a new AutoML session", icon: BrainCircuit },
          { href: "/dashboard/predict", label: "Run Prediction", desc: "Validate deployed endpoint", icon: TrendingUp },
        ].map(({ href, label, desc, icon: Icon }) => (
          <Link key={href} href={href}
            className="panel flex cursor-pointer items-center gap-4 rounded-lg border-outline/20 p-5 transition-all hover:border-primary/40">
            <Icon className="h-7 w-7 flex-shrink-0 text-primary" />
            <div>
              <div className="text-sm font-semibold text-text-primary">{label}</div>
              <div className="text-xs text-text-muted">{desc}</div>
            </div>
          </Link>
        ))}
      </div>

      {/* Recent Jobs */}
      <div className="panel rounded-xl p-6">
        <PanelHeader title="Recent Training Jobs" subtitle="Latest sessions and operational status." icon={Clock} className="mb-4" />
        {recentJobs.length === 0 ? (
          <div className="py-10 text-center text-text-muted">
            <BrainCircuit className="mx-auto mb-3 h-10 w-10 opacity-30" />
            <p className="text-sm">No training jobs yet. Start by uploading a dataset!</p>
            <Link href="/dashboard/datasets" className="btn-primary text-sm mt-4 inline-flex">
              Upload Dataset
            </Link>
          </div>
        ) : (
          <div className="space-y-2">
            {recentJobs.map((job) => (
              <div key={job.job_id} className="flex items-center gap-4 rounded-lg border border-outline/15 bg-surface-variant/25 p-3 transition-colors hover:bg-surface-variant/45">
                {statusIcon(job.status)}
                <div className="flex-1 min-w-0">
                  <div className="truncate font-mono text-sm font-medium text-text-primary">{job.job_id.slice(0, 8)}…</div>
                  <div className="text-xs text-text-muted">Target: {job.target_column}</div>
                </div>
                <StatusBadge
                  label={job.status}
                  tone={
                    job.status === "completed"
                      ? "healthy"
                      : job.status === "running"
                        ? "processing"
                        : job.status === "failed"
                          ? "error"
                          : "idle"
                  }
                />
                {job.status === "completed" && (
                  <Link href={`/dashboard/models?job=${job.job_id}`}
                    className="text-xs text-primary hover:text-primary/80 underline">
                    View →
                  </Link>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
