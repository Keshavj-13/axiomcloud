"use client";
import { useState, useEffect, useRef } from "react";
import { datasetsAPI, trainingAPI, experimentsAPI } from "@/lib/api";
import { Dataset, TrainingJob, ExperimentRun } from "@/types";
import toast from "react-hot-toast";
import { BrainCircuit, Play, Loader2, CheckCircle, AlertTriangle, ChevronRight } from "lucide-react";
import Link from "next/link";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";

export default function TrainingPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [taskType, setTaskType] = useState("auto");
  const [testSize, setTestSize] = useState(0.2);
  const [cvFolds, setCvFolds] = useState(5);
  const [enableTuning, setEnableTuning] = useState(true);
  const [tuningTrials, setTuningTrials] = useState(16);
  const [tuningBudgetSec, setTuningBudgetSec] = useState(180);
  const [currentJob, setCurrentJob] = useState<TrainingJob | null>(null);
  const [experiments, setExperiments] = useState<ExperimentRun[]>([]);
  const [training, setTraining] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    datasetsAPI.list().then(r => setDatasets(r.data)).catch(() => {});
    experimentsAPI.list().then(r => setExperiments(r.data)).catch(() => {});
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const pollStatus = (jobId: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await trainingAPI.getStatus(jobId);
        setCurrentJob(res.data);
        if (res.data.status === "completed" || res.data.status === "failed") {
          clearInterval(pollRef.current!);
          setTraining(false);
          experimentsAPI.list().then(r => setExperiments(r.data)).catch(() => {});
          if (res.data.status === "completed") {
            toast.success("Training completed!");
          } else {
            toast.error("Training failed: " + res.data.error_message);
          }
        }
      } catch { clearInterval(pollRef.current!); setTraining(false); }
    }, 1500);
  };

  const startTraining = async () => {
    if (!selectedDataset) return toast.error("Select a dataset");
    if (!targetColumn) return toast.error("Enter target column");
    setTraining(true);
    setCurrentJob(null);
    try {
      const res = await trainingAPI.train({
        dataset_id: selectedDataset.id,
        target_column: targetColumn,
        task_type: taskType === "auto" ? undefined : taskType,
        test_size: testSize,
        cv_folds: cvFolds,
        enable_tuning: enableTuning,
        tuning_trials: tuningTrials,
        tuning_time_budget_sec: tuningBudgetSec,
      });
      const job: TrainingJob = res.data;
      setCurrentJob(job);
      toast.success("Training started!");
      pollStatus(job.job_id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Failed to start training");
      setTraining(false);
    }
  };

  const handleDatasetSelect = (ds: Dataset) => {
    setSelectedDataset(ds);
    if (ds.target_column) setTargetColumn(ds.target_column);
    if (ds.task_type) setTaskType(ds.task_type);
  };

  const progressBar = currentJob ? currentJob.progress : 0;

  return (
    <div className="mx-auto max-w-7xl">
      <div className="mb-8">
        <div className="section-kicker mb-2">Training</div>
        <h1 className="font-display mb-1 text-4xl font-semibold text-text-primary">Live Training Cockpit</h1>
        <p className="text-text-muted">Configure safe controls and monitor model training in real-time.</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Config */}
        <div className="panel p-6">
          <PanelHeader title="Session Controls" subtitle="Dataset source, target and CV setup." icon={BrainCircuit} className="mb-5" />

          {/* Dataset picker */}
          <div className="mb-4">
            <label className="mb-1.5 block text-xs text-text-muted">Dataset *</label>
            {datasets.length === 0 ? (
              <div className="rounded-lg bg-surface-variant/35 p-3 text-sm text-text-muted">
                No datasets. <Link href="/dashboard/datasets" className="text-primary underline">Upload one →</Link>
              </div>
            ) : (
              <select
                value={selectedDataset?.id ?? ""}
                onChange={e => {
                  const ds = datasets.find(d => d.id === Number(e.target.value));
                  if (ds) handleDatasetSelect(ds);
                }}
                className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary/50"
              >
                <option value="">Select dataset...</option>
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.id}>{ds.name} ({ds.num_rows} rows)</option>
                ))}
              </select>
            )}
          </div>

          {/* Target column */}
          <div className="mb-4">
            <label className="mb-1.5 block text-xs text-text-muted">Target Column *</label>
            {selectedDataset ? (
              <select
                value={targetColumn}
                onChange={e => setTargetColumn(e.target.value)}
                className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary/50"
              >
                <option value="">Select target...</option>
                {selectedDataset.columns_info?.map(c => (
                  <option key={c.name} value={c.name}>{c.name} ({c.dtype})</option>
                ))}
              </select>
            ) : (
              <input
                value={targetColumn}
                onChange={e => setTargetColumn(e.target.value)}
                placeholder="e.g. price, survived"
                className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary/50"
              />
            )}
          </div>

          {/* Task type */}
          <div className="mb-4">
            <label className="mb-1.5 block text-xs text-text-muted">Task Type</label>
            <select
              value={taskType}
              onChange={e => setTaskType(e.target.value)}
              className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary/50"
            >
              <option value="auto">Auto-detect</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>

          {/* Test size & CV */}
          <div className="grid grid-cols-2 gap-3 mb-6">
            <div>
              <label className="mb-1.5 block text-xs text-text-muted">Test Size: {(testSize * 100).toFixed(0)}%</label>
              <input type="range" min="0.1" max="0.4" step="0.05" value={testSize}
                onChange={e => setTestSize(Number(e.target.value))}
                className="w-full accent-sigma-500" />
            </div>
            <div>
              <label className="mb-1.5 block text-xs text-text-muted">CV Folds: {cvFolds}</label>
              <input type="range" min="2" max="10" step="1" value={cvFolds}
                onChange={e => setCvFolds(Number(e.target.value))}
                className="w-full accent-sigma-500" />
            </div>
          </div>

          <div className="mb-6 rounded-lg border border-outline/25 bg-surface-variant/25 p-3">
            <div className="mb-2 flex items-center justify-between">
              <label className="text-xs text-text-muted">Adaptive Hyperparameter Tuning</label>
              <button
                type="button"
                onClick={() => setEnableTuning(v => !v)}
                className={`rounded px-2 py-1 text-[11px] ${enableTuning ? "bg-primary/20 text-primary" : "bg-surface text-text-muted"}`}
              >
                {enableTuning ? "Enabled" : "Disabled"}
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1 block text-xs text-text-muted">Tuning Trials: {tuningTrials}</label>
                <input
                  type="range"
                  min="3"
                  max="60"
                  step="1"
                  value={tuningTrials}
                  disabled={!enableTuning}
                  onChange={e => setTuningTrials(Number(e.target.value))}
                  className="w-full accent-sigma-500 disabled:opacity-40"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-text-muted">Time Budget: {tuningBudgetSec}s</label>
                <input
                  type="range"
                  min="30"
                  max="900"
                  step="30"
                  value={tuningBudgetSec}
                  disabled={!enableTuning}
                  onChange={e => setTuningBudgetSec(Number(e.target.value))}
                  className="w-full accent-sigma-500 disabled:opacity-40"
                />
              </div>
            </div>
          </div>

          <button
            onClick={startTraining}
            disabled={training || !selectedDataset || !targetColumn}
            className="btn-primary w-full justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {training ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {training ? "Training..." : "Start AutoML Training"}
          </button>
        </div>

        {/* Models to train */}
        <div className="panel p-6">
          <PanelHeader title="Model Set" subtitle="Algorithms included in this run." className="mb-4" />
          <div className="space-y-2">
            {[
              { name: "Logistic / Linear Regression", tag: "baseline" },
              { name: "Random Forest", tag: "ensemble" },
              { name: "XGBoost", tag: "boosting" },
              { name: "LightGBM", tag: "boosting" },
              { name: "Gradient Boosting", tag: "ensemble" },
            ].map(({ name, tag }) => (
              <div key={name} className="flex items-center gap-3 rounded-lg bg-surface-variant/30 p-3">
                <CheckCircle className="h-4 w-4 text-emerald-300" />
                <span className="flex-1 text-sm text-text-primary">{name}</span>
                <StatusBadge label={tag} tone="idle" />
              </div>
            ))}
          </div>
          <div className="mt-4 rounded-lg border border-outline/25 bg-surface-variant/25 p-3">
            <p className="text-xs text-text-muted">
              All models include: cross-validation, feature importance, and full evaluation metrics.
            </p>
          </div>
        </div>
      </div>

      {/* Job status */}
      {currentJob && (
        <div className="panel mt-6 rounded-xl p-6">
          <h2 className="mb-4 flex items-center gap-2 font-display font-semibold text-text-primary">
            {currentJob.status === "completed" ? (
              <CheckCircle className="h-4 w-4 text-emerald-300" />
            ) : currentJob.status === "failed" ? (
              <AlertTriangle className="h-4 w-4 text-red-300" />
            ) : (
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
            )}
            Job: <span className="font-mono text-primary">{currentJob.job_id.slice(0, 8)}…</span>
          </h2>

          {/* Progress bar */}
          <div className="mb-4">
            <div className="mb-1.5 flex justify-between text-xs text-text-muted">
              <span className="capitalize">{currentJob.status}</span>
              <span>{progressBar}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-surface-variant">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary-deep to-primary transition-all duration-500"
                style={{ width: `${progressBar}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="p-2 rounded bg-sigma-900/40">
              <div className="text-sigma-500">Task</div>
              <div className="text-white font-medium">{currentJob.task_type || "detecting..."}</div>
            </div>
            <div className="p-2 rounded bg-sigma-900/40">
              <div className="text-sigma-500">Target</div>
              <div className="text-white font-medium">{currentJob.target_column}</div>
            </div>
            <div className="p-2 rounded bg-sigma-900/40">
              <div className="text-sigma-500">Status</div>
              <div className={`font-medium capitalize ${
                currentJob.status === "completed" ? "text-neon-green" :
                currentJob.status === "failed" ? "text-red-400" : "text-neon-amber"
              }`}>{currentJob.status}</div>
            </div>
          </div>

          {currentJob.status === "completed" && (
            <Link
              href={`/dashboard/models?job=${currentJob.job_id}`}
              className="btn-primary mt-4 inline-flex"
            >
              View Results <ChevronRight className="w-4 h-4" />
            </Link>
          )}

          {currentJob.error_message && (
            <div className="mt-3 p-3 rounded bg-red-900/20 border border-red-800/40 text-red-400 text-xs font-mono">
              {currentJob.error_message}
            </div>
          )}
        </div>
      )}

      <div className="panel mt-6 rounded-xl p-6">
        <h2 className="mb-3 font-display font-semibold text-text-primary">Experiment Registry</h2>
        <p className="mb-4 text-xs text-text-muted">Latest runs with config snapshots and best-model summary.</p>
        <div className="space-y-2">
          {experiments.slice(0, 8).map((run) => (
            <div key={run.run_id} className="rounded-lg border border-outline/20 bg-surface-variant/25 p-3 text-xs">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-mono text-text-primary">{run.run_id.slice(0, 8)}…</span>
                <StatusBadge label={run.status} tone={run.status === "completed" ? "healthy" : run.status === "failed" ? "error" : "idle"} />
              </div>
              <div className="mt-1 text-text-muted">
                job: {run.job_id.slice(0, 8)}… • target: {run.target_column} • task: {run.task_type || "auto"}
              </div>
              <div className="mt-1 text-text-muted">
                best: {run.best_model_name || "-"} {typeof run.best_score === "number" ? `(${run.best_score.toFixed(4)})` : ""}
              </div>
            </div>
          ))}
          {experiments.length === 0 && (
            <div className="text-xs text-text-muted">No experiment runs yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}
