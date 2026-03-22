"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { datasetsAPI, modelsAPI, trainingAPI } from "@/lib/api";
import { Dataset, TrainedModel, TrainingJob } from "@/types";
import { Loader2, Search, Database, Brain, BarChart3 } from "lucide-react";

function includesQuery(value: unknown, query: string) {
  if (!query) return true;
  return String(value ?? "").toLowerCase().includes(query);
}

export default function SearchPage() {
  const searchParams = useSearchParams();
  const rawQuery = (searchParams.get("q") || "").trim();
  const query = rawQuery.toLowerCase();

  const [loading, setLoading] = useState(true);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<TrainedModel[]>([]);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [jobsRes, datasetsRes, modelsRes] = await Promise.all([
          trainingAPI.listJobs(),
          datasetsAPI.list(),
          modelsAPI.list(),
        ]);

        setJobs(jobsRes.data || []);
        setDatasets(datasetsRes.data || []);
        setModels(modelsRes.data || []);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  const matchedJobs = useMemo(
    () => jobs.filter((j) => (
      includesQuery(j.job_id, query)
      || includesQuery(j.target_column, query)
      || includesQuery(j.task_type, query)
      || includesQuery(j.status, query)
    )),
    [jobs, query]
  );

  const matchedDatasets = useMemo(
    () => datasets.filter((d) => (
      includesQuery(d.name, query)
      || includesQuery(d.filename, query)
      || includesQuery(d.target_column, query)
      || includesQuery(d.task_type, query)
    )),
    [datasets, query]
  );

  const matchedModels = useMemo(
    () => models.filter((m) => (
      includesQuery(m.model_name, query)
      || includesQuery(m.model_type, query)
      || includesQuery(m.task_type, query)
      || includesQuery(m.job_id, query)
    )),
    [models, query]
  );

  const total = matchedJobs.length + matchedDatasets.length + matchedModels.length;

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-8">
        <div className="section-kicker mb-2">Search</div>
        <h1 className="font-display mb-1 text-4xl font-semibold text-text-primary">Workspace Search</h1>
        <p className="text-text-muted">Find jobs, datasets, and models from one place.</p>
      </div>

      <div className="panel mb-6 rounded-xl p-4">
        <form action="/dashboard/search" method="get" className="flex items-center gap-2">
          <Search className="h-4 w-4 text-text-muted" />
          <input
            type="search"
            name="q"
            defaultValue={rawQuery}
            placeholder="Search by id, name, target, task type..."
            className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary/50"
          />
        </form>
      </div>

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : (
        <>
          <div className="mb-4 text-sm text-text-muted">
            {rawQuery ? `Found ${total} result${total === 1 ? "" : "s"} for "${rawQuery}"` : "Enter a search term to filter results."}
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <section className="panel rounded-xl p-4">
              <div className="mb-3 flex items-center gap-2 text-text-primary">
                <BarChart3 className="h-4 w-4" />
                <h2 className="text-sm font-semibold">Training Jobs ({matchedJobs.length})</h2>
              </div>
              <div className="space-y-2">
                {matchedJobs.slice(0, 20).map((job) => (
                  <Link
                    key={job.job_id}
                    href={`/dashboard/models?job=${encodeURIComponent(job.job_id)}`}
                    className="block rounded-lg border border-outline/20 bg-surface-variant/30 px-3 py-2 text-xs hover:border-primary/40"
                  >
                    <div className="font-mono text-text-primary">{job.job_id}</div>
                    <div className="text-text-muted">{job.target_column} • {job.task_type || "unknown"} • {job.status}</div>
                  </Link>
                ))}
                {matchedJobs.length === 0 && <div className="text-xs text-text-muted">No matching jobs.</div>}
              </div>
            </section>

            <section className="panel rounded-xl p-4">
              <div className="mb-3 flex items-center gap-2 text-text-primary">
                <Database className="h-4 w-4" />
                <h2 className="text-sm font-semibold">Datasets ({matchedDatasets.length})</h2>
              </div>
              <div className="space-y-2">
                {matchedDatasets.slice(0, 20).map((dataset) => (
                  <Link
                    key={dataset.id}
                    href="/dashboard/datasets"
                    className="block rounded-lg border border-outline/20 bg-surface-variant/30 px-3 py-2 text-xs hover:border-primary/40"
                  >
                    <div className="text-text-primary">{dataset.name}</div>
                    <div className="text-text-muted">{dataset.target_column || "no target"} • {dataset.task_type || "unknown"}</div>
                  </Link>
                ))}
                {matchedDatasets.length === 0 && <div className="text-xs text-text-muted">No matching datasets.</div>}
              </div>
            </section>

            <section className="panel rounded-xl p-4">
              <div className="mb-3 flex items-center gap-2 text-text-primary">
                <Brain className="h-4 w-4" />
                <h2 className="text-sm font-semibold">Models ({matchedModels.length})</h2>
              </div>
              <div className="space-y-2">
                {matchedModels.slice(0, 20).map((model) => (
                  <Link
                    key={model.id}
                    href={`/dashboard/models?job=${encodeURIComponent(model.job_id)}`}
                    className="block rounded-lg border border-outline/20 bg-surface-variant/30 px-3 py-2 text-xs hover:border-primary/40"
                  >
                    <div className="text-text-primary">{model.model_name}</div>
                    <div className="text-text-muted">{model.model_type} • {model.task_type} • {model.job_id.slice(0, 8)}…</div>
                  </Link>
                ))}
                {matchedModels.length === 0 && <div className="text-xs text-text-muted">No matching models.</div>}
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
