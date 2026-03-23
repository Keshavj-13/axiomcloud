"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { metricsAPI, trainingAPI, modelsAPI, datasetsAPI } from "@/lib/api";
import {
  MetricsData,
  TrainedModel,
  ShapResult,
  LimeResult,
  ModelMonitoring,
  Dataset,
  DatasetQualityReport,
  CleanPreview,
  LeakageReport,
  EDAReport,
  TrainingJob,
  AnalyticsReport,
  AnalyticsChart,
} from "@/types";
import toast from "react-hot-toast";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ScatterChart, Scatter
} from "recharts";
import { Trophy, Download, Rocket, BarChart3, Loader2, Brain, Activity } from "lucide-react";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";

export const dynamic = "force-dynamic";

const COLORS = ["#3b6ef6", "#00f5ff", "#00ff94", "#ffb700", "#7b2fff"];

function HeatCell({ value }: { value: number }) {
  const clamped = Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
  const alpha = Math.abs(clamped);
  const bg = clamped >= 0
    ? `rgba(110, 59, 215, ${0.2 + alpha * 0.5})`
    : `rgba(239, 68, 68, ${0.2 + alpha * 0.5})`;
  return <div className="h-6 w-6 rounded" style={{ background: bg }} title={clamped.toFixed(3)} />;
}

function EmptyPlotState({ message }: { message: string }) {
  return (
    <div className="flex h-[150px] items-center justify-center rounded border border-outline/20 bg-surface/40 p-3 text-center text-[11px] text-text-muted">
      {message}
    </div>
  );
}

function AnalyticsChartCard({ chart }: { chart: AnalyticsChart }) {
  const spec = (chart.spec || {}) as Record<string, unknown>;
  const rows = (spec.rows as Record<string, unknown>[] | undefined) || [];
  const xKey = (spec.xKey as string | undefined) || "label";
  const yKey = (spec.yKey as string | undefined) || "value";
  const series = (spec.series as Array<{ key: string; name: string; color?: string }> | undefined) || [];

  return (
    <div className="rounded-lg border border-outline/20 bg-surface-variant/30 p-3">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div>
          <div className="text-xs font-semibold text-text-primary">{chart.title}</div>
          <div className="text-[11px] text-text-muted">{chart.purpose}</div>
        </div>
        {chart.fallback_used ? <StatusBadge label="fallback" tone="idle" /> : null}
      </div>

      {chart.chart_type === "bar" && rows.length > 0 ? (
        <ResponsiveContainer width="100%" height={190}>
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(117,117,123,0.2)" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10, fill: "#acaab1" }} interval="preserveStartEnd" angle={-20} textAnchor="end" height={44} minTickGap={24} />
            <YAxis tick={{ fontSize: 10, fill: "#acaab1" }} />
            <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(117,117,123,0.35)", borderRadius: "8px" }} />
            {(series.length ? series : [{ key: "value", name: "Value", color: "#d0bcff" }]).map((s) => (
              <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color || "#d0bcff"} radius={[3, 3, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      ) : chart.chart_type === "line" && rows.length > 0 ? (
        <ResponsiveContainer width="100%" height={190}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(117,117,123,0.2)" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10, fill: "#acaab1" }} />
            <YAxis tick={{ fontSize: 10, fill: "#acaab1" }} />
            <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(117,117,123,0.35)", borderRadius: "8px" }} />
            {(series.length ? series : [{ key: "value", name: "Value", color: "#d0bcff" }]).map((s) => (
              <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color || "#d0bcff"} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      ) : chart.chart_type === "scatter" && rows.length > 0 ? (
        <ResponsiveContainer width="100%" height={190}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(117,117,123,0.2)" />
            <XAxis dataKey={xKey} name={xKey} tick={{ fontSize: 10, fill: "#acaab1" }} />
            <YAxis dataKey={yKey} name={yKey} tick={{ fontSize: 10, fill: "#acaab1" }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#19191d", border: "1px solid rgba(117,117,123,0.35)", borderRadius: "8px" }} />
            <Scatter data={rows} fill="#d0bcff" shape={<circle r={1.8} />} />
          </ScatterChart>
        </ResponsiveContainer>
      ) : chart.chart_type === "heatmap" ? (
        <div className="overflow-auto">
          <div className="inline-flex flex-col gap-1">
            {(((spec.matrix as number[][] | undefined) || [])).slice(0, 10).map((row, r) => (
              <div key={r} className="flex gap-1">
                {row.slice(0, 10).map((v, c) => <HeatCell key={`${r}-${c}`} value={v} />)}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="max-h-40 overflow-auto rounded border border-outline/20 bg-surface/40 p-2 text-[11px] text-text-muted">
          {rows.length > 0 ? rows.slice(0, 10).map((r, idx) => <div key={idx}>{JSON.stringify(r)}</div>) : (chart.insight || "No chart rows available.")}
        </div>
      )}

      <div className="mt-2 text-[11px] text-text-muted">Insight: {chart.insight}</div>
    </div>
  );
}

export default function ModelsPage() {
  const searchParams = useSearchParams();
  const [jobId, setJobId] = useState(searchParams.get("job") || "");
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [selectedModel, setSelectedModel] = useState<TrainedModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [compareDatasetId, setCompareDatasetId] = useState<number | "">("");
  const [shapData, setShapData] = useState<ShapResult | null>(null);
  const [limeData, setLimeData] = useState<LimeResult | null>(null);
  const [monitoring, setMonitoring] = useState<ModelMonitoring | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [monitorLoading, setMonitorLoading] = useState(false);
  const [explainTab, setExplainTab] = useState<"summary" | "shap" | "lime">("summary");
  const [sampleIndex, setSampleIndex] = useState(0);
  const [customInput, setCustomInput] = useState("{}");
  const [explainError, setExplainError] = useState<string | null>(null);
  const [qualityReport, setQualityReport] = useState<DatasetQualityReport | null>(null);
  const [cleanPreview, setCleanPreview] = useState<CleanPreview | null>(null);
  const [leakageReport, setLeakageReport] = useState<LeakageReport | null>(null);
  const [edaReport, setEdaReport] = useState<EDAReport | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [analyticsReport, setAnalyticsReport] = useState<AnalyticsReport | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);

  useEffect(() => {
    trainingAPI.listJobs().then(r => {
      const completed = (r.data as TrainingJob[]).filter((j) => j.status === "completed");
      setJobs(completed);
      if (!jobId && completed.length > 0) setJobId(completed[0].job_id);
    }).catch(() => {});
    datasetsAPI.list().then(r => setDatasets(r.data)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!jobId) return;
    setLoading(true);
    metricsAPI.getJobMetrics(jobId).then(r => {
      const payload = r.data;
      setMetrics(payload);
      const fallbackBest = payload?.models?.find((m: TrainedModel) => m.id === payload?.best_model_id)
        || payload?.best_model
        || payload?.models?.[0]
        || null;
      setSelectedModel(fallbackBest);
    }).catch(() => toast.error("Could not load metrics"))
    .finally(() => setLoading(false));
  }, [jobId]);

  useEffect(() => {
    if (!jobId || jobs.length === 0) return;
    const selectedJob = jobs.find((j) => j.job_id === jobId);
    if (!selectedJob?.dataset_id) return;

    setQualityLoading(true);
    Promise.all([
      datasetsAPI.qualityReport(selectedJob.dataset_id),
      datasetsAPI.cleanPreview(selectedJob.dataset_id, 200),
      datasetsAPI.leakageReport(selectedJob.dataset_id),
      datasetsAPI.edaReport(selectedJob.dataset_id),
    ]).then(([qualityRes, cleanRes, leakageRes, edaRes]) => {
      setQualityReport(qualityRes.data);
      setCleanPreview(cleanRes.data);
      setLeakageReport(leakageRes.data);
      setEdaReport(edaRes.data);
    }).catch(() => {
      setQualityReport(null);
      setCleanPreview(null);
      setLeakageReport(null);
      setEdaReport(null);
    }).finally(() => setQualityLoading(false));
  }, [jobId, jobs]);

  useEffect(() => {
    const activeJob = jobs.find((j) => j.job_id === jobId);
    if (!activeJob?.dataset_id || !selectedModel?.id) {
      setAnalyticsReport(null);
      return;
    }

    setAnalyticsLoading(true);
    datasetsAPI.analyticsReport(activeJob.dataset_id, {
      target_column: activeJob.target_column,
      model_type: activeJob.task_type,
      model_id: selectedModel.id,
    }).then((res) => {
      setAnalyticsReport(res.data);
    }).catch(() => {
      setAnalyticsReport(null);
    }).finally(() => {
      setAnalyticsLoading(false);
    });
  }, [jobId, jobs, selectedModel?.id]);

  const selectedJob = jobs.find((j) => j.job_id === jobId) || null;
  const droppedRows = cleanPreview && qualityReport ? Math.max(0, qualityReport.rows - cleanPreview.rows_after_cleaning) : 0;
  const leakageMax = leakageReport?.leakage_risks?.length
    ? Math.max(...leakageReport.leakage_risks.map(r => r.risk_score))
    : 0;

  const bestModel = metrics?.models?.find((m) => m.id === metrics?.best_model_id)
    || metrics?.best_model
    || metrics?.models?.[0]
    || null;

  const deployModel = async (id: number) => {
    try {
      await modelsAPI.deploy(id);
      toast.success("Model deployed!");
      if (jobId) {
        metricsAPI.getJobMetrics(jobId).then(r => {
          const payload = r.data;
          setMetrics(payload);
          const fallbackBest = payload?.models?.find((m: TrainedModel) => m.id === payload?.best_model_id)
            || payload?.best_model
            || payload?.models?.[0]
            || null;
          if (fallbackBest) setSelectedModel(fallbackBest);
        });
      }
    } catch { toast.error("Deploy failed"); }
  };

  const loadExplainability = async () => {
    if (!selectedModel) return;
    setExplainLoading(true);
    setExplainError(null);
    try {
      const parsedCustom = customInput.trim() ? JSON.parse(customInput) : undefined;
      const [shapRes, limeRes] = await Promise.all([
        modelsAPI.shap(selectedModel.id, { nsamples: 220, sample_index: sampleIndex }),
        modelsAPI.lime(selectedModel.id, { sample_index: sampleIndex, num_features: 12, custom_input: parsedCustom }),
      ]);
      setShapData(shapRes.data);
      setLimeData(limeRes.data);
    } catch (e: any) {
      const message = e?.response?.data?.detail?.error?.message || "Failed to load SHAP/LIME explanations";
      setExplainError(message);
      toast.error(message);
    } finally {
      setExplainLoading(false);
    }
  };

  const loadMonitoring = async () => {
    if (!selectedModel) return;
    setMonitorLoading(true);
    try {
      const res = await modelsAPI.monitoring(
        selectedModel.id,
        compareDatasetId === "" ? undefined : Number(compareDatasetId)
      );
      setMonitoring(res.data);
    } catch {
      toast.error("Failed to load monitoring/drift report");
    } finally {
      setMonitorLoading(false);
    }
  };

  const leaderboardData = metrics?.models.map(m => ({
    name: m.model_name,
    Accuracy: m.accuracy !== undefined && m.accuracy !== null ? +(m.accuracy * 100).toFixed(1) : undefined,
    "R²": m.r2_score !== undefined && m.r2_score !== null ? +(m.r2_score * 100).toFixed(1) : undefined,
    F1: m.f1_score !== undefined && m.f1_score !== null ? +(m.f1_score * 100).toFixed(1) : undefined,
    "ROC-AUC": m.roc_auc !== undefined && m.roc_auc !== null ? +(m.roc_auc * 100).toFixed(1) : undefined,
    "Training Time (s)": m.training_time !== undefined && m.training_time !== null ? +m.training_time.toFixed(2) : undefined,
  })) || [];

  const featureImportanceData = selectedModel?.feature_importance
    ? Object.entries(selectedModel.feature_importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 12)
        .map(([name, value]) => ({ name, value: +(value * 100).toFixed(1) }))
    : [];

  const cvData = selectedModel?.cv_scores?.map((s, i) => ({
    fold: `Fold ${i + 1}`, score: +(s * 100).toFixed(2)
  })) || [];

  const isClassification = metrics?.task_type === "classification";
  const missingByColumnData = qualityReport
    ? Object.entries(qualityReport.missing_by_column || {})
        .map(([column, missing]) => ({ column, missing }))
        .filter((r) => Number(r.missing) > 0)
        .sort((a, b) => Number(b.missing) - Number(a.missing))
        .slice(0, 6)
    : [];
  const outlierByColumnData = qualityReport
    ? Object.entries(qualityReport.outliers_by_column || {})
        .map(([column, outliers]) => ({ column, outliers }))
        .filter((r) => Number(r.outliers) > 0)
        .sort((a, b) => Number(b.outliers) - Number(a.outliers))
        .slice(0, 6)
    : [];
  const leakageRiskData = (leakageReport?.leakage_risks || [])
    .map((r) => ({
      feature: r.feature.length > 18 ? `${r.feature.slice(0, 18)}…` : r.feature,
      risk_pct: +(r.risk_score * 100).toFixed(2),
    }))
    .filter((r) => r.risk_pct > 0)
    .sort((a, b) => b.risk_pct - a.risk_pct)
    .slice(0, 6);
  const metricEntries = selectedModel?.metrics
    ? Object.entries(selectedModel.metrics)
        .filter(([, value]) => ["number", "string", "boolean"].includes(typeof value) || value === null)
        .map(([key, value]) => ({ key, value: value === null ? "null" : String(value) }))
    : [];

  const modelOriginLabel = (model?: TrainedModel | null): "local" | "server" => {
    const origin = String((model?.metrics as Record<string, unknown> | undefined)?.storage_origin || "").toLowerCase();
    if (origin === "local") return "local";
    if ((model?.model_type || "").toLowerCase().includes("local")) return "local";
    return "server";
  };

  return (
    <div className="mx-auto max-w-7xl">
      <div className="mb-8">
        <div className="section-kicker mb-2">Leaderboard</div>
        <h1 className="font-display mb-1 text-4xl font-semibold text-text-primary">Model Quality Console</h1>
        <p className="text-text-muted">Compare rank, quality, and diagnostics across completed training jobs.</p>
      </div>

      {/* Job selector */}
      <div className="panel mb-6 rounded-xl p-6">
        <div className="flex items-center gap-4">
          <label className="whitespace-nowrap text-sm text-text-muted">Training Job:</label>
          <select
            value={jobId}
            onChange={e => setJobId(e.target.value)}
            className="flex-1 rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary/50"
          >
            <option value="">Select a completed job...</option>
            {jobs.map(j => (
              <option key={j.job_id} value={j.job_id}>
                {j.job_id.slice(0, 8)}… — Target: {j.target_column} ({j.task_type})
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="panel mb-6 rounded-xl p-6">
        <PanelHeader title="Dataset Validity & Preparation" subtitle="Multi-level quality gates before trusting model performance." className="mb-4" />
        {qualityLoading ? (
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading dataset diagnostics...
          </div>
        ) : qualityReport ? (
          <div className="space-y-4">
            <div className="grid lg:grid-cols-3 gap-4">
              <div className="rounded-lg border border-outline/20 bg-surface-variant/30 p-3">
                <div className="mb-2 text-xs font-semibold text-text-primary">Missingness by column</div>
                {missingByColumnData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={missingByColumnData} margin={{ top: 6, right: 8, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(117,117,123,0.15)" />
                      <XAxis dataKey="column" tick={{ fill: "#acaab1", fontSize: 10 }} interval="preserveStartEnd" angle={-20} textAnchor="end" height={42} minTickGap={24} />
                      <YAxis tick={{ fill: "#acaab1", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(117,117,123,0.35)", borderRadius: "8px" }} />
                      <Bar dataKey="missing" fill="#d0bcff" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyPlotState message="No missing-value signal found. All columns currently have zero missing entries." />
                )}
              </div>

              <div className="rounded-lg border border-outline/20 bg-surface-variant/30 p-3">
                <div className="mb-2 text-xs font-semibold text-text-primary">Outliers by column</div>
                {outlierByColumnData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={outlierByColumnData} margin={{ top: 6, right: 8, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(117,117,123,0.15)" />
                      <XAxis dataKey="column" tick={{ fill: "#acaab1", fontSize: 10 }} interval="preserveStartEnd" angle={-20} textAnchor="end" height={42} minTickGap={24} />
                      <YAxis tick={{ fill: "#acaab1", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(117,117,123,0.35)", borderRadius: "8px" }} />
                      <Bar dataKey="outliers" fill="#6e3bd7" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyPlotState message="No outlier counts were detected for the profiled columns." />
                )}
              </div>

              <div className="rounded-lg border border-outline/20 bg-surface-variant/30 p-3">
                <div className="mb-2 text-xs font-semibold text-text-primary">Leakage risk by feature</div>
                {leakageRiskData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={leakageRiskData} margin={{ top: 6, right: 8, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(117,117,123,0.15)" />
                      <XAxis dataKey="feature" tick={{ fill: "#acaab1", fontSize: 10 }} interval="preserveStartEnd" angle={-20} textAnchor="end" height={42} minTickGap={24} />
                      <YAxis tick={{ fill: "#acaab1", fontSize: 10 }} unit="%" />
                      <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(117,117,123,0.35)", borderRadius: "8px" }} />
                      <Bar dataKey="risk_pct" fill="#ef4444" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyPlotState message="No leakage-risk values were produced. This typically means no feature crossed the leakage heuristics threshold for the selected dataset and target." />
                )}
              </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-4 text-xs">
            <div className="rounded-lg border border-outline/20 bg-surface-variant/30 p-4 space-y-2">
              <div className="text-sm font-semibold text-text-primary">Level 1 • Raw Quality</div>
              <div className="flex justify-between"><span className="text-text-muted">Dataset</span><span className="font-mono text-text-primary">{qualityReport.dataset_name}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Quality score</span><span className="font-mono text-text-primary">{qualityReport.quality_score.toFixed(1)}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Rows × cols</span><span className="font-mono text-text-primary">{qualityReport.rows} × {qualityReport.columns}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Missing cells</span><span className="font-mono text-text-primary">{qualityReport.missing_cells}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Duplicate rows</span><span className="font-mono text-text-primary">{qualityReport.duplicate_rows}</span></div>
            </div>

            <div className="rounded-lg border border-outline/20 bg-surface-variant/30 p-4 space-y-2">
              <div className="text-sm font-semibold text-text-primary">Level 2 • Cleaning Impact</div>
              <div className="flex justify-between"><span className="text-text-muted">Rows after cleaning</span><span className="font-mono text-text-primary">{cleanPreview?.rows_after_cleaning ?? "—"}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Rows dropped</span><span className="font-mono text-text-primary">{cleanPreview ? droppedRows : "—"}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Fixes applied</span><span className="font-mono text-text-primary">{cleanPreview?.applied_fixes?.length ?? 0}</span></div>
              <div className="mt-2">
                <div className="mb-1 text-text-muted">Top fixes</div>
                <ul className="space-y-1">
                  {(cleanPreview?.applied_fixes || []).slice(0, 4).map((fix, idx) => (
                    <li key={`${fix}-${idx}`} className="truncate text-text-primary">• {fix}</li>
                  ))}
                  {(cleanPreview?.applied_fixes || []).length === 0 && <li className="text-text-muted">No automatic fixes detected.</li>}
                </ul>
              </div>
            </div>

            <div className="rounded-lg border border-outline/20 bg-surface-variant/30 p-4 space-y-2">
              <div className="text-sm font-semibold text-text-primary">Level 3 • Leakage & Validity</div>
              <div className="flex justify-between"><span className="text-text-muted">Target</span><span className="font-mono text-text-primary">{selectedJob?.target_column || leakageReport?.target_column || "—"}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Leakage candidates</span><span className="font-mono text-text-primary">{leakageReport?.leakage_risks?.length ?? 0}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Max leakage risk</span><span className="font-mono text-text-primary">{(leakageMax * 100).toFixed(1)}%</span></div>
              <div className="flex justify-between"><span className="text-text-muted">High correlations</span><span className="font-mono text-text-primary">{edaReport?.high_correlations?.length ?? 0}</span></div>
              <div className="mt-2">
                <div className="mb-1 text-text-muted">Top risky features</div>
                <ul className="space-y-1">
                  {(leakageReport?.leakage_risks || []).slice(0, 4).map((risk) => (
                    <li key={risk.feature} className="truncate text-text-primary">
                      • {risk.feature} ({(risk.risk_score * 100).toFixed(1)}%)
                    </li>
                  ))}
                  {(leakageReport?.leakage_risks || []).length === 0 && <li className="text-text-muted">No strong leakage indicators found.</li>}
                </ul>
              </div>
            </div>
            </div>
          </div>
        ) : (
          <p className="text-xs text-text-muted">Select a completed job to load quality, cleaning impact, and validity diagnostics.</p>
        )}
      </div>

      {loading && (
        <div className="flex justify-center py-20">
          <Loader2 className="w-8 h-8 text-sigma-400 animate-spin" />
        </div>
      )}

      {metrics && !loading && (
        <>
          {/* Best model banner */}
          {bestModel && (
            <div className="panel-glass glow-blue mb-6 p-6">
              <div className="flex items-center gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/15">
                    <Trophy className="h-6 w-6 text-primary" />
                </div>
                <div className="flex-1">
                    <div className="mb-0.5 text-xs text-text-muted">Best Model</div>
                    <div className="font-display text-xl font-semibold text-text-primary">{bestModel.model_name}</div>
                  <div className="flex flex-wrap gap-3 mt-1">
                    <span className={`metric-badge ${modelOriginLabel(bestModel) === "local" ? "bg-emerald-900/40 text-emerald-300" : "bg-slate-800/60 text-slate-200"}`}>
                      Origin: {modelOriginLabel(bestModel)}
                    </span>
                    {bestModel.accuracy !== undefined && bestModel.accuracy !== null && (
                      <span className="metric-badge bg-blue-900/40 text-blue-300">Accuracy: {(bestModel.accuracy * 100).toFixed(1)}%</span>
                    )}
                    {bestModel.precision !== undefined && bestModel.precision !== null && (
                      <span className="metric-badge bg-fuchsia-900/40 text-fuchsia-300">Precision: {(bestModel.precision * 100).toFixed(1)}%</span>
                    )}
                    {bestModel.recall !== undefined && bestModel.recall !== null && (
                      <span className="metric-badge bg-indigo-900/40 text-indigo-300">Recall: {(bestModel.recall * 100).toFixed(1)}%</span>
                    )}
                    {bestModel.f1_score !== undefined && bestModel.f1_score !== null && (
                      <span className="metric-badge bg-cyan-900/40 text-cyan-300">F1: {(bestModel.f1_score * 100).toFixed(1)}%</span>
                    )}
                    {bestModel.roc_auc !== undefined && bestModel.roc_auc !== null && (
                      <span className="metric-badge bg-purple-900/40 text-purple-300">AUC: {(bestModel.roc_auc * 100).toFixed(1)}%</span>
                    )}
                    {bestModel.r2_score !== undefined && bestModel.r2_score !== null && (
                      <span className="metric-badge bg-green-900/40 text-green-300">R²: {bestModel.r2_score.toFixed(3)}</span>
                    )}
                    {bestModel.rmse !== undefined && bestModel.rmse !== null && (
                      <span className="metric-badge bg-red-900/40 text-red-300">RMSE: {bestModel.rmse.toFixed(3)}</span>
                    )}
                    {bestModel.mape !== undefined && bestModel.mape !== null && (
                      <span className="metric-badge bg-orange-900/40 text-orange-300">MAPE: {(bestModel.mape * 100).toFixed(2)}%</span>
                    )}
                  </div>
                </div>
                <button onClick={() => deployModel(bestModel.id)} className="btn-primary">
                  <Rocket className="w-4 h-4" /> Deploy
                </button>
              </div>
            </div>
          )}

          {/* Model cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {metrics.models.map((model, i) => (
              <div
                key={model.id}
                onClick={() => setSelectedModel(model)}
                className={`panel cursor-pointer rounded-xl p-6 transition-all ${selectedModel?.id === model.id ? "border-primary/60 glow-blue" : ""}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                    <span className="text-sm font-semibold text-text-primary">{model.model_name}</span>
                  </div>
                  <StatusBadge label={modelOriginLabel(model)} tone={modelOriginLabel(model) === "local" ? "healthy" : "idle"} />
                  {model.is_deployed && (
                    <StatusBadge label="deployed" tone="healthy" />
                  )}
                  {(bestModel?.id === model.id) && (
                    <Trophy className="w-4 h-4 text-neon-amber" />
                  )}
                </div>

                <div className="space-y-1.5">
                  {model.accuracy !== undefined && model.accuracy !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">Accuracy</span>
                      <span className="font-mono text-text-primary">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {model.f1_score !== undefined && model.f1_score !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">F1 Score</span>
                      <span className="font-mono text-text-primary">{(model.f1_score * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {model.roc_auc !== undefined && model.roc_auc !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">ROC-AUC</span>
                      <span className="font-mono text-text-primary">{(model.roc_auc * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {model.precision !== undefined && model.precision !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">Precision</span>
                      <span className="font-mono text-text-primary">{(model.precision * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {model.recall !== undefined && model.recall !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">Recall</span>
                      <span className="font-mono text-text-primary">{(model.recall * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {model.r2_score !== undefined && model.r2_score !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">R² Score</span>
                      <span className="font-mono text-text-primary">{model.r2_score.toFixed(3)}</span>
                    </div>
                  )}
                  {model.rmse !== undefined && model.rmse !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">RMSE</span>
                      <span className="font-mono text-text-primary">{model.rmse.toFixed(3)}</span>
                    </div>
                  )}
                  {model.mape !== undefined && model.mape !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">MAPE</span>
                      <span className="font-mono text-text-primary">{(model.mape * 100).toFixed(2)}%</span>
                    </div>
                  )}
                  {model.explained_variance !== undefined && model.explained_variance !== null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-muted">Explained Variance</span>
                      <span className="font-mono text-text-primary">{model.explained_variance.toFixed(3)}</span>
                    </div>
                  )}
                  <div className="flex justify-between text-xs">
                    <span className="text-text-muted">Train time</span>
                    <span className="font-mono text-text-primary">{model.training_time?.toFixed(2)}s</span>
                  </div>
                </div>

                <div className="flex gap-2 mt-4">
                  <button onClick={e => { e.stopPropagation(); deployModel(model.id); }}
                    className="flex-1 btn-outline text-xs py-1.5 justify-center">
                    <Rocket className="w-3 h-3" /> Deploy
                  </button>
                  <a href={modelsAPI.download(model.id)} download
                    className="btn-outline text-xs py-1.5 px-2.5" onClick={e => e.stopPropagation()}>
                    <Download className="w-3 h-3" />
                  </a>
                </div>
              </div>
            ))}
          </div>

          {/* Charts section */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Comparison bar chart */}
            <div className="panel rounded-xl p-6">
              <PanelHeader title="Model Comparison" icon={BarChart3} className="mb-4" />
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={leaderboardData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
                  <XAxis dataKey="name" tick={{ fill: "#6b7a9e", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#6b7a9e", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px", color: "#e8edf8" }} />
                  {isClassification ? (
                    <>
                      <Bar dataKey="Accuracy" fill="#3b6ef6" radius={[3,3,0,0]} />
                      <Bar dataKey="F1" fill="#00f5ff" radius={[3,3,0,0]} />
                    </>
                  ) : (
                    <Bar dataKey="R²" fill="#00ff94" radius={[3,3,0,0]} />
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* CV Scores */}
            {cvData.length > 0 && (
              <div className="sigma-card">
                <h3 className="font-display font-semibold text-white mb-4">
                  Cross-Validation Scores — {selectedModel?.model_name}
                </h3>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={cvData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
                    <XAxis dataKey="fold" tick={{ fill: "#6b7a9e", fontSize: 11 }} />
                    <YAxis domain={['auto', 'auto']} tick={{ fill: "#6b7a9e", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px", color: "#e8edf8" }} />
                    <Line type="monotone" dataKey="score" stroke="#3b6ef6" strokeWidth={2} dot={{ fill: "#00f5ff" }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="panel mb-6 rounded-xl p-6">
            <PanelHeader
              title="AutoML Analytics & Diagnostics"
              subtitle="Rendered directly after cross-validation scores with full-data evaluation plots and task-specific equivalents."
              className="mb-4"
            />
            {analyticsLoading ? (
              <div className="flex items-center gap-2 text-xs text-text-muted">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading analytics charts...
              </div>
            ) : analyticsReport ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
                  <span>Task: {analyticsReport.task_type}</span>
                  <span>•</span>
                  <span>Charts: {analyticsReport.chart_count}</span>
                  <span>•</span>
                  <StatusBadge label={`evaluation: ${analyticsReport.evaluation_status}`} tone={analyticsReport.evaluation_status === "ready" ? "healthy" : "idle"} />
                </div>

                <div>
                  <div className="mb-2 text-xs uppercase tracking-[0.12em] text-text-muted">1) Exploratory data analysis</div>
                  <div className="grid gap-3 md:grid-cols-1 xl:grid-cols-2">
                    {analyticsReport.exploratory_charts.map((chart) => (
                      <AnalyticsChartCard key={chart.id} chart={chart} />
                    ))}
                  </div>
                </div>

                <div>
                  <div className="mb-2 text-xs uppercase tracking-[0.12em] text-text-muted">2) Model evaluation</div>
                  <div className="grid gap-3 md:grid-cols-1 xl:grid-cols-2">
                    {analyticsReport.evaluation_charts.map((chart) => (
                      <AnalyticsChartCard key={chart.id} chart={chart} />
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-text-muted">Analytics report is not available for the selected job/model pair.</p>
            )}
          </div>

          {selectedModel && (
            <div className="panel mb-6 rounded-xl p-6">
              <PanelHeader title="Full Metric Surface" subtitle="Complete scalar metric payload for model audit and validity checks." className="mb-4" />
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-xs">
                {metricEntries.map((entry) => (
                  <div key={entry.key} className="rounded border border-outline/20 bg-surface-variant/30 px-3 py-2">
                    <div className="text-text-muted truncate">{entry.key}</div>
                    <div className="font-mono text-text-primary truncate">{entry.value}</div>
                  </div>
                ))}
                {metricEntries.length === 0 && (
                  <div className="text-text-muted">No scalar metrics found in payload.</div>
                )}
              </div>
              <div className="mt-4">
                <div className="mb-1 text-xs text-text-muted">Raw metric payload (full)</div>
                <pre className="max-h-64 overflow-auto rounded border border-outline/20 bg-surface-variant/30 p-3 text-[11px] text-text-primary">
                  {JSON.stringify(selectedModel.metrics || {}, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Feature Importance */}
          {featureImportanceData.length > 0 && (
            <div className="sigma-card mb-6">
              <h3 className="font-display font-semibold text-white mb-4">
                Feature Importance — {selectedModel?.model_name}
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportanceData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
                  <XAxis type="number" tick={{ fill: "#6b7a9e", fontSize: 11 }} unit="%" />
                  <YAxis type="category" dataKey="name" width={140} tick={{ fill: "#8b97b8", fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px", color: "#e8edf8" }} />
                  <Bar dataKey="value" fill="#3b6ef6" radius={[0,3,3,0]}>
                    {featureImportanceData.map((_, i) => (
                      <rect key={i} fill={`hsl(${220 + i * 10}, 80%, 60%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Confusion Matrix */}
          {selectedModel?.confusion_matrix && (
            <div className="sigma-card mb-6">
              <h3 className="font-display font-semibold text-white mb-4">
                Confusion Matrix — {selectedModel.model_name}
              </h3>
              <div className="overflow-x-auto">
                <table className="mx-auto text-center text-sm font-mono">
                  <thead>
                    <tr>
                      <th className="p-2 text-sigma-500 text-xs">Actual\Predicted</th>
                      {selectedModel.confusion_matrix[0].map((_, i) => (
                        <th key={i} className="p-2 text-sigma-400">Class {i}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selectedModel.confusion_matrix.map((row, i) => (
                      <tr key={i}>
                        <td className="p-2 text-sigma-400 text-xs">Class {i}</td>
                        {row.map((val, j) => {
                          const max = Math.max(...selectedModel.confusion_matrix!.flat());
                          const intensity = max > 0 ? val / max : 0;
                          return (
                            <td key={j} className="p-3 rounded text-white font-bold"
                              style={{ background: i === j
                                ? `rgba(0, 255, 148, ${0.15 + intensity * 0.4})`
                                : `rgba(239, 68, 68, ${0.05 + intensity * 0.3})` }}>
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ROC Curve */}
          {selectedModel?.roc_curve_data && (
            <div className="sigma-card">
              <h3 className="font-display font-semibold text-white mb-4">
                ROC Curve — {selectedModel.model_name}
              </h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={selectedModel.roc_curve_data.fpr.map((fpr, i) => ({
                  fpr: +fpr.toFixed(3),
                  tpr: +selectedModel.roc_curve_data!.tpr[i].toFixed(3),
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
                  <XAxis dataKey="fpr" type="number" domain={[0,1]} label={{ value: "False Positive Rate", position: "insideBottom", fill: "#6b7a9e", dy: 10 }} tick={{ fill: "#6b7a9e", fontSize: 11 }} />
                  <YAxis domain={[0,1]} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: "#6b7a9e" }} tick={{ fill: "#6b7a9e", fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px", color: "#e8edf8" }} />
                  <Line type="monotone" dataKey="tpr" stroke="#00f5ff" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Explainability */}
          <div className="sigma-card mt-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display font-semibold text-white flex items-center gap-2">
                <Brain className="w-4 h-4 text-sigma-400" /> Explainable AI (SHAP / LIME)
              </h3>
              <button onClick={loadExplainability} disabled={explainLoading || !selectedModel} className="btn-outline text-xs py-1.5 px-3">
                {explainLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : "Load Explanations"}
              </button>
            </div>

            <div className="grid md:grid-cols-4 gap-3 mb-4">
              <div>
                <label className="text-xs text-sigma-500 mb-1 block">Sample Index</label>
                <input
                  type="number"
                  min={0}
                  value={sampleIndex}
                  onChange={(e) => setSampleIndex(Math.max(0, Number(e.target.value || 0)))}
                  className="w-full bg-sigma-900/50 border border-sigma-700/50 rounded-lg px-2.5 py-1.5 text-xs text-white"
                />
              </div>
              <div className="md:col-span-2">
                <label className="text-xs text-sigma-500 mb-1 block">Custom Input JSON (optional)</label>
                <input
                  value={customInput}
                  onChange={(e) => setCustomInput(e.target.value)}
                  className="w-full bg-sigma-900/50 border border-sigma-700/50 rounded-lg px-2.5 py-1.5 text-xs text-white"
                />
              </div>
              <div>
                <label className="text-xs text-sigma-500 mb-1 block">View</label>
                <select
                  value={explainTab}
                  onChange={(e) => setExplainTab(e.target.value as "summary" | "shap" | "lime")}
                  className="w-full bg-sigma-900/50 border border-sigma-700/50 rounded-lg px-2.5 py-1.5 text-xs text-white"
                >
                  <option value="summary">Summary</option>
                  <option value="shap">SHAP</option>
                  <option value="lime">LIME</option>
                </select>
              </div>
            </div>

            {explainError && (
              <div className="mb-4 rounded-lg border border-red-700/50 bg-red-900/20 p-3 text-xs text-red-300">
                {explainError}
              </div>
            )}

            <div className="grid lg:grid-cols-2 gap-4">
              {explainTab === "summary" && (
                <>
                  <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4 text-xs space-y-1.5">
                    <div className="text-sm font-semibold text-white mb-2">SHAP Summary</div>
                    <div className="flex justify-between"><span className="text-sigma-500">Expected Value</span><span className="font-mono text-sigma-200">{String(shapData?.expected_value ?? "—")}</span></div>
                    <div className="flex justify-between"><span className="text-sigma-500">Prediction</span><span className="font-mono text-sigma-200">{String(shapData?.sample_prediction?.prediction ?? "—")}</span></div>
                    <div className="flex justify-between"><span className="text-sigma-500">Confidence</span><span className="font-mono text-sigma-200">{shapData?.sample_prediction?.confidence !== undefined ? `${(shapData.sample_prediction.confidence * 100).toFixed(2)}%` : "—"}</span></div>
                    <div className="flex justify-between"><span className="text-sigma-500">Explainer</span><span className="font-mono text-sigma-200">{shapData?.metadata?.explainer ?? "—"}</span></div>
                  </div>
                  <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4 text-xs space-y-1.5">
                    <div className="text-sm font-semibold text-white mb-2">LIME Summary</div>
                    <div className="flex justify-between"><span className="text-sigma-500">Prediction</span><span className="font-mono text-sigma-200">{String(limeData?.sample_prediction?.prediction ?? "—")}</span></div>
                    <div className="flex justify-between"><span className="text-sigma-500">Top Positive</span><span className="font-mono text-sigma-200">{limeData?.top_positive?.[0]?.feature ?? "—"}</span></div>
                    <div className="flex justify-between"><span className="text-sigma-500">Top Negative</span><span className="font-mono text-sigma-200">{limeData?.top_negative?.[0]?.feature ?? "—"}</span></div>
                    <div className="flex justify-between"><span className="text-sigma-500">Explainer</span><span className="font-mono text-sigma-200">{limeData?.metadata?.explainer ?? "—"}</span></div>
                  </div>
                </>
              )}

              {explainTab === "shap" && (
                <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4 lg:col-span-2">
                  <div className="text-sm font-semibold text-white mb-2">Global SHAP Impact</div>
                  {shapData?.global_importance?.length ? (
                    <div className="grid lg:grid-cols-2 gap-4">
                      <div className="space-y-1.5 text-xs max-h-60 overflow-auto">
                        {shapData.global_importance.slice(0, 16).map((item) => (
                          <div key={item.feature} className="flex justify-between">
                            <span className="text-sigma-400 truncate pr-3">{item.feature}</span>
                            <span className="text-sigma-200 font-mono">{item.mean_abs_contribution.toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                      <div className="space-y-1.5 text-xs max-h-60 overflow-auto">
                        <div className="text-sm font-semibold text-white mb-1">Local SHAP Contributions</div>
                        {shapData.local_contributions?.slice(0, 16).map((item) => (
                          <div key={item.feature} className="flex justify-between">
                            <span className="text-sigma-400 truncate pr-3">{item.feature}</span>
                            <span className="text-sigma-200 font-mono">{item.value.toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : <p className="text-xs text-sigma-500">Load explanations to view SHAP summary.</p>}
                </div>
              )}

              {explainTab === "lime" && (
                <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4 lg:col-span-2">
                  <div className="text-sm font-semibold text-white mb-2">LIME Feature Weights</div>
                  {limeData?.weights?.length ? (
                    <div className="space-y-1.5 text-xs max-h-60 overflow-auto">
                      {limeData.weights.slice(0, 18).map((item) => (
                        <div key={`${item.feature}-${item.rank}`} className="flex justify-between">
                          <span className="text-sigma-400 truncate pr-3">{item.feature}</span>
                          <span className={`font-mono ${item.weight >= 0 ? "text-emerald-300" : "text-red-300"}`}>{item.weight.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  ) : <p className="text-xs text-sigma-500">Load explanations to view LIME details.</p>}
                </div>
              )}
            </div>
          </div>

          {/* Monitoring + Drift */}
          <div className="sigma-card mt-6">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <h3 className="font-display font-semibold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-sigma-400" /> Model Monitoring & Drift
              </h3>
              <div className="flex items-center gap-2">
                <select
                  value={compareDatasetId}
                  onChange={(e) => setCompareDatasetId(e.target.value ? Number(e.target.value) : "")}
                  className="bg-sigma-900/50 border border-sigma-700/50 rounded-lg px-3 py-1.5 text-xs text-white"
                >
                  <option value="">No drift comparison</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </select>
                <button onClick={loadMonitoring} disabled={monitorLoading || !selectedModel} className="btn-outline text-xs py-1.5 px-3">
                  {monitorLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : "Load Monitoring"}
                </button>
              </div>
            </div>

            {monitoring ? (
              <div className="grid lg:grid-cols-2 gap-4 text-xs">
                <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4 space-y-2">
                  <div className="text-sm font-semibold text-white">Health Snapshot</div>
                  <div className="flex justify-between"><span className="text-sigma-500">Stability</span><span className="text-sigma-300 uppercase">{monitoring.monitoring.stability}</span></div>
                  <div className="flex justify-between"><span className="text-sigma-500">CV mean</span><span className="text-sigma-300 font-mono">{monitoring.monitoring.cv_mean?.toFixed(4) ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-sigma-500">CV std</span><span className="text-sigma-300 font-mono">{monitoring.monitoring.cv_std?.toFixed(4) ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-sigma-500">Primary metric</span><span className="text-sigma-300 font-mono">{monitoring.monitoring.primary_metric?.toFixed(4) ?? "—"}</span></div>
                </div>
                <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4 space-y-2">
                  <div className="text-sm font-semibold text-white">Drift Report</div>
                  {monitoring.drift ? (
                    <>
                      <div className="flex justify-between"><span className="text-sigma-500">Drift level</span><span className="text-sigma-300 uppercase">{monitoring.drift.drift_level}</span></div>
                      <div className="flex justify-between"><span className="text-sigma-500">Avg z-drift</span><span className="text-sigma-300 font-mono">{monitoring.drift.avg_z_drift.toFixed(4)}</span></div>
                      <div className="text-sigma-500 mt-2">Top drift features</div>
                      <div className="h-36">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={monitoring.drift.top_drift_features.slice(0, 6).map((f) => ({
                              feature: f.feature.length > 16 ? `${f.feature.slice(0, 16)}…` : f.feature,
                              z: f.z_drift,
                            }))}
                            margin={{ top: 6, right: 8, left: 0, bottom: 8 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(117,117,123,0.15)" />
                            <XAxis dataKey="feature" tick={{ fill: "#acaab1", fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={42} />
                            <YAxis tick={{ fill: "#acaab1", fontSize: 10 }} />
                            <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(117,117,123,0.35)", borderRadius: "8px", color: "#e7e4ec" }} />
                            <Bar dataKey="z" fill="#6e3bd7" radius={[3, 3, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </>
                  ) : (
                    <p className="text-sigma-500">Select a comparison dataset to compute drift.</p>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-xs text-sigma-500">Load monitoring to view model health and optional drift analysis.</p>
            )}
          </div>
        </>
      )}

      {!metrics && !loading && (
        <div className="text-center py-20 text-sigma-600">
          <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Select a completed training job to view model metrics.</p>
        </div>
      )}
    </div>
  );
}
