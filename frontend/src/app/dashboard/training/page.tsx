"use client";
import { useState, useEffect, useRef } from "react";
import { datasetsAPI, trainingAPI, experimentsAPI } from "@/lib/api";
import {
  Dataset,
  TrainingJob,
  ExperimentRun,
  TrainingModelCatalog,
  DatasetQualityReport,
  CleanPreview,
  LeakageReport,
  EDAReport,
} from "@/types";
import toast from "react-hot-toast";
import { BrainCircuit, Play, Loader2, CheckCircle, AlertTriangle, ChevronRight, Layers } from "lucide-react";
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
  const [trainingDepth, setTrainingDepth] = useState<"quick" | "balanced" | "deep">("balanced");
  const [modelCatalog, setModelCatalog] = useState<TrainingModelCatalog>({ classification: [], regression: [], all: [] });
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisReasons, setAnalysisReasons] = useState<string[]>([]);
  const [analysisWarning, setAnalysisWarning] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<TrainingJob | null>(null);
  const [experiments, setExperiments] = useState<ExperimentRun[]>([]);
  const [training, setTraining] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    datasetsAPI.list().then(r => setDatasets(r.data)).catch(() => {});
    experimentsAPI.list().then(r => setExperiments(r.data)).catch(() => {});
    trainingAPI.modelCatalog().then(r => {
      setModelCatalog(r.data);
    }).catch(() => {});
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  useEffect(() => {
    if (trainingDepth === "quick") {
      setCvFolds(3);
      setEnableTuning(false);
      setTuningTrials(6);
      setTuningBudgetSec(60);
      return;
    }
    if (trainingDepth === "deep") {
      setCvFolds(7);
      setEnableTuning(true);
      setTuningTrials(40);
      setTuningBudgetSec(480);
      return;
    }
    setCvFolds(5);
    setEnableTuning(true);
    setTuningTrials(16);
    setTuningBudgetSec(180);
  }, [trainingDepth]);

  const visibleModelOptions = taskType === "classification"
    ? modelCatalog.classification
    : taskType === "regression"
      ? modelCatalog.regression
      : modelCatalog.all;

  useEffect(() => {
    if (!visibleModelOptions.length) return;
    setSelectedModels(prev => {
      const filtered = prev.filter(m => visibleModelOptions.includes(m));
      return filtered;
    });
  }, [taskType, modelCatalog.classification, modelCatalog.regression, modelCatalog.all]);

  const recommendModelsFromAnalysis = (
    options: string[],
    task: string,
    quality?: DatasetQualityReport,
    clean?: CleanPreview,
    leakage?: LeakageReport,
    eda?: EDAReport,
  ) => {
    const picked = new Set<string>();
    const reasons: string[] = [];

    const add = (name: string) => {
      if (options.includes(name)) picked.add(name);
    };

    if (task === "classification") {
      add("Logistic Regression");
      add("Random Forest");
      add("XGBoost");
      reasons.push("Classification detected: baseline linear + robust tree ensembles selected.");
    } else {
      add("Linear Regression");
      add("Random Forest");
      add("XGBoost");
      reasons.push("Regression detected: baseline linear + nonlinear ensembles selected.");
    }

    const rows = quality?.rows ?? eda?.overview?.rows ?? 0;
    const cols = quality?.columns ?? eda?.overview?.columns ?? 0;
    const droppedRatio = clean && quality?.rows ? Math.max(0, (quality.rows - clean.rows_after_cleaning) / quality.rows) : 0;
    const maxLeak = leakage?.leakage_risks?.length ? Math.max(...leakage.leakage_risks.map(r => r.risk_score)) : 0;

    if (rows > 60000) {
      add("LightGBM");
      add("Gradient Boosting");
      reasons.push("Large row count: preferred scalable boosting over instance-based methods.");
    } else {
      add("KNN");
      add(task === "classification" ? "Linear SVM" : "Linear SVR");
      reasons.push("Moderate row count: included local-margin methods (KNN/SVM family).");
    }

    if (cols > 80) {
      add("LightGBM");
      add("MLP Neural Net");
      reasons.push("High feature count: added neural and gradient methods for nonlinear interactions.");
    }

    if ((quality?.missing_cells ?? 0) > 0 || droppedRatio > 0.05) {
      add("Random Forest");
      add("Extra Trees");
      add("LightGBM");
      reasons.push("Cleaning/missingness detected: tree ensembles emphasized for robustness.");
    }

    if (maxLeak >= 0.8) {
      reasons.push("High leakage risk detected: review target-proxy features before trusting metrics.");
    }

    // Never auto-select experimental GNN in tabular mode.
    if (options.includes("Graph Neural Network (experimental)")) {
      reasons.push("GNN kept optional only (requires graph-structured data and longer runtime). ");
    }

    if (picked.size < 3) {
      add("Gradient Boosting");
      add("LightGBM");
    }

    return { recommended: Array.from(picked), reasons, maxLeak };
  };

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
    if (selectedModels.length === 0) return toast.error("Select at least one model");
    setTraining(true);
    setCurrentJob(null);
    try {
      const res = await trainingAPI.train({
        dataset_id: selectedDataset.id,
        target_column: targetColumn,
        task_type: taskType === "auto" ? undefined : taskType,
        test_size: testSize,
        cv_folds: cvFolds,
        models_to_train: selectedModels,
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

  const handleDatasetSelect = async (ds: Dataset) => {
    setSelectedDataset(ds);
    if (ds.target_column) setTargetColumn(ds.target_column);
    if (ds.task_type) setTaskType(ds.task_type);

    if (!modelCatalog.all.length) return;
    setAnalysisLoading(true);
    setAnalysisReasons([]);
    setAnalysisWarning(null);
    try {
      const [cleanRes, qualityRes, leakageRes, edaRes] = await Promise.all([
        datasetsAPI.cleanPreview(ds.id, 200),
        datasetsAPI.qualityReport(ds.id),
        datasetsAPI.leakageReport(ds.id),
        datasetsAPI.edaReport(ds.id),
      ]);

      const inferredTask = ds.task_type || taskType;
      const taskResolved = inferredTask === "classification" || inferredTask === "regression"
        ? inferredTask
        : "classification";
      if (taskType === "auto" || !taskType) {
        setTaskType(taskResolved);
      }

      const options = taskResolved === "classification"
        ? modelCatalog.classification
        : modelCatalog.regression;

      const recommendation = recommendModelsFromAnalysis(
        options,
        taskResolved,
        qualityRes.data,
        cleanRes.data,
        leakageRes.data,
        edaRes.data,
      );

      setSelectedModels(recommendation.recommended);
      setAnalysisReasons(recommendation.reasons);
      if (recommendation.maxLeak >= 0.8) {
        setAnalysisWarning("High leakage risk detected. Validate feature set before long/deep training.");
      }
      toast.success("Dataset cleaned + analyzed. Recommended model stack selected.");
    } catch {
      const fallbackTask = ds.task_type === "regression" ? "regression" : "classification";
      const options = fallbackTask === "classification" ? modelCatalog.classification : modelCatalog.regression;
      const fallback = options.filter((m) => ["Logistic Regression", "Linear Regression", "Random Forest", "XGBoost"].includes(m));
      setSelectedModels(fallback.slice(0, 3));
      setAnalysisReasons(["Auto-analysis unavailable; selected a conservative starter stack."]);
      toast.error("Could not run full automatic analysis, using fallback recommendation");
    } finally {
      setAnalysisLoading(false);
    }
  };

  useEffect(() => {
    if (!selectedDataset) return;
    if (!modelCatalog.all.length) return;
    if (selectedModels.length > 0) return;
    handleDatasetSelect(selectedDataset);
  }, [modelCatalog.all.length]);

  const progressBar = currentJob ? currentJob.progress : 0;
  const heavySelected = selectedModels.filter((m) => modelCatalog.details?.[m]?.cost_tier === "high");

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
            <div className="mb-3">
              <label className="mb-1 block text-xs text-text-muted">Training Depth</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { key: "quick", label: "Quick" },
                  { key: "balanced", label: "Balanced" },
                  { key: "deep", label: "Deep" },
                ].map(level => (
                  <button
                    key={level.key}
                    type="button"
                    onClick={() => setTrainingDepth(level.key as "quick" | "balanced" | "deep")}
                    className={`rounded px-2 py-1.5 text-xs ${trainingDepth === level.key ? "bg-primary/20 text-primary border border-primary/30" : "bg-surface text-text-muted border border-outline/20"}`}
                  >
                    {level.label}
                  </button>
                ))}
              </div>
            </div>
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
            disabled={training}
            className="btn-primary w-full justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {training ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {training ? "Training..." : "Start AutoML Training"}
          </button>
          {!selectedDataset || !targetColumn ? (
            <p className="mt-2 text-xs text-amber-300">
              Select a dataset and target column to start training.
            </p>
          ) : null}
        </div>

        {/* Models to train */}
        <div className="panel p-6">
          <PanelHeader title="Model Set" subtitle="Choose exactly which algorithms to run." icon={Layers} className="mb-4" />
          <div className="mb-3 rounded-lg border border-outline/20 bg-surface-variant/25 p-3 text-xs">
            <div className="mb-1 text-text-primary">Automatic first-pass analysis</div>
            <div className="text-text-muted">
              On dataset selection: cleaning preview is computed first, then quality/leakage/EDA analysis recommends the model stack.
            </div>
            {analysisLoading && (
              <div className="mt-2 flex items-center gap-2 text-text-muted"><Loader2 className="h-3.5 w-3.5 animate-spin" /> analyzing…</div>
            )}
            {!analysisLoading && analysisReasons.length > 0 && (
              <ul className="mt-2 space-y-1 text-text-muted">
                {analysisReasons.slice(0, 4).map((r, i) => <li key={`${r}-${i}`}>• {r}</li>)}
              </ul>
            )}
            {analysisWarning && <div className="mt-2 text-amber-300">Warning: {analysisWarning}</div>}
          </div>
          <div className="mb-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setSelectedModels([...visibleModelOptions])}
              className="btn-ghost px-2.5 py-1 text-xs"
            >
              Select visible
            </button>
            <button
              type="button"
              onClick={() => setSelectedModels([])}
              className="btn-ghost px-2.5 py-1 text-xs"
            >
              Clear
            </button>
            <div className="rounded border border-outline/20 px-2.5 py-1 text-xs text-text-muted">
              {selectedModels.length} selected
            </div>
          </div>
          <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
            {visibleModelOptions.map((name) => {
              const checked = selectedModels.includes(name);
              const detail = modelCatalog.details?.[name];
              const tag = detail?.family || (/Boost|XGBoost|LightGBM/i.test(name) ? "boosting" : /Forest/i.test(name) ? "ensemble" : "baseline");
              return (
                <label key={name} className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3 ${checked ? "border-primary/40 bg-primary/10" : "border-outline/20 bg-surface-variant/30"}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => setSelectedModels(prev => checked ? prev.filter(m => m !== name) : [...prev, name])}
                    className="accent-sigma-500"
                  />
                  <span className="flex-1 text-sm text-text-primary">
                    {name}
                    {detail?.warning ? <span className="ml-2 text-[11px] text-amber-300">(longer training)</span> : null}
                    {detail?.experimental ? <span className="ml-2 text-[11px] text-fuchsia-300">(experimental)</span> : null}
                  </span>
                  <StatusBadge label={tag} tone="idle" />
                </label>
              );
            })}
          </div>
          {heavySelected.length > 0 && (
            <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
              Heavy models selected ({heavySelected.length}): {heavySelected.join(", ")}. Training can take significantly longer.
            </div>
          )}
          <div className="mt-4 rounded-lg border border-outline/25 bg-surface-variant/25 p-3">
            <p className="text-xs text-text-muted">
              Model list is fully configurable per run. Current task filter: <span className="text-text-primary">{taskType}</span>.
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
