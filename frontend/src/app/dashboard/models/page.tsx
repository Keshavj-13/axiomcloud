"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { metricsAPI, trainingAPI, modelsAPI, datasetsAPI } from "@/lib/api";
import { MetricsData, TrainedModel, ShapResult, LimeResult, ModelMonitoring, Dataset } from "@/types";
import toast from "react-hot-toast";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from "recharts";
import { Trophy, Download, Rocket, BarChart3, Loader2, Brain, Activity } from "lucide-react";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";

export const dynamic = "force-dynamic";

const COLORS = ["#3b6ef6", "#00f5ff", "#00ff94", "#ffb700", "#7b2fff"];

export default function ModelsPage() {
  const searchParams = useSearchParams();
  const [jobId, setJobId] = useState(searchParams.get("job") || "");
  const [jobs, setJobs] = useState<any[]>([]);
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

  useEffect(() => {
    trainingAPI.listJobs().then(r => {
      const completed = r.data.filter((j: any) => j.status === "completed");
      setJobs(completed);
      if (!jobId && completed.length > 0) setJobId(completed[0].job_id);
    }).catch(() => {});
    datasetsAPI.list().then(r => setDatasets(r.data)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!jobId) return;
    setLoading(true);
    metricsAPI.getJobMetrics(jobId).then(r => {
      setMetrics(r.data);
      setSelectedModel(r.data.best_model || null);
    }).catch(() => toast.error("Could not load metrics"))
    .finally(() => setLoading(false));
  }, [jobId]);

  const deployModel = async (id: number) => {
    try {
      await modelsAPI.deploy(id);
      toast.success("Model deployed!");
      if (jobId) metricsAPI.getJobMetrics(jobId).then(r => setMetrics(r.data));
    } catch { toast.error("Deploy failed"); }
  };

  const loadExplainability = async () => {
    if (!selectedModel) return;
    setExplainLoading(true);
    try {
      const [shapRes, limeRes] = await Promise.all([
        modelsAPI.shap(selectedModel.id, 80),
        modelsAPI.lime(selectedModel.id, 3),
      ]);
      setShapData(shapRes.data);
      setLimeData(limeRes.data);
    } catch {
      toast.error("Failed to load SHAP/LIME explanations");
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
    Accuracy: m.accuracy ? +(m.accuracy * 100).toFixed(1) : undefined,
    "R²": m.r2_score ? +(m.r2_score * 100).toFixed(1) : undefined,
    F1: m.f1_score ? +(m.f1_score * 100).toFixed(1) : undefined,
    "ROC-AUC": m.roc_auc ? +(m.roc_auc * 100).toFixed(1) : undefined,
    "Training Time (s)": m.training_time ? +m.training_time.toFixed(2) : undefined,
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

      {loading && (
        <div className="flex justify-center py-20">
          <Loader2 className="w-8 h-8 text-sigma-400 animate-spin" />
        </div>
      )}

      {metrics && !loading && (
        <>
          {/* Best model banner */}
          {metrics.best_model && (
            <div className="panel-glass glow-blue mb-6 p-6">
              <div className="flex items-center gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/15">
                    <Trophy className="h-6 w-6 text-primary" />
                </div>
                <div className="flex-1">
                    <div className="mb-0.5 text-xs text-text-muted">Best Model</div>
                    <div className="font-display text-xl font-semibold text-text-primary">{metrics.best_model.model_name}</div>
                  <div className="flex flex-wrap gap-3 mt-1">
                    {metrics.best_model.accuracy && (
                      <span className="metric-badge bg-blue-900/40 text-blue-300">Accuracy: {(metrics.best_model.accuracy * 100).toFixed(1)}%</span>
                    )}
                    {metrics.best_model.f1_score && (
                      <span className="metric-badge bg-cyan-900/40 text-cyan-300">F1: {(metrics.best_model.f1_score * 100).toFixed(1)}%</span>
                    )}
                    {metrics.best_model.roc_auc && (
                      <span className="metric-badge bg-purple-900/40 text-purple-300">AUC: {(metrics.best_model.roc_auc * 100).toFixed(1)}%</span>
                    )}
                    {metrics.best_model.r2_score && (
                      <span className="metric-badge bg-green-900/40 text-green-300">R²: {metrics.best_model.r2_score.toFixed(3)}</span>
                    )}
                    {metrics.best_model.rmse && (
                      <span className="metric-badge bg-red-900/40 text-red-300">RMSE: {metrics.best_model.rmse.toFixed(3)}</span>
                    )}
                  </div>
                </div>
                <button onClick={() => deployModel(metrics.best_model!.id)} className="btn-primary">
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
                  {model.is_deployed && (
                    <StatusBadge label="deployed" tone="healthy" />
                  )}
                  {metrics.best_model?.id === model.id && (
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

            <div className="grid lg:grid-cols-2 gap-4">
              <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4">
                <div className="text-sm font-semibold text-white mb-2">Global SHAP Impact (sample avg)</div>
                {shapData?.error ? (
                  <p className="text-xs text-red-400">{shapData.error}</p>
                ) : (shapData?.shap_values && shapData?.feature_names && Array.isArray(shapData.shap_values) ? (
                  <div className="space-y-1.5 text-xs">
                    {shapData.feature_names.slice(0, 10).map((f, idx) => {
                      const values = (shapData.shap_values as any[])
                        .map((row: any) => Array.isArray(row) ? row[idx] : 0)
                        .filter((v: any) => typeof v === "number");
                      const meanAbs = values.length > 0
                        ? values.reduce((a: number, b: number) => a + Math.abs(b), 0) / values.length
                        : 0;
                      return (
                        <div key={f} className="flex justify-between">
                          <span className="text-sigma-400 truncate pr-3">{f}</span>
                          <span className="text-sigma-200 font-mono">{meanAbs.toFixed(4)}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : <p className="text-xs text-sigma-500">Load explanations to view SHAP summary.</p>)}
              </div>

              <div className="rounded-lg border border-sigma-800/50 bg-sigma-900/20 p-4">
                <div className="text-sm font-semibold text-white mb-2">Local LIME Explanations</div>
                {limeData?.error ? (
                  <p className="text-xs text-red-400">{limeData.error}</p>
                ) : (limeData?.lime_explanations?.length ? (
                  <div className="space-y-3 max-h-52 overflow-auto">
                    {limeData.lime_explanations.map((exp) => (
                      <div key={exp.instance} className="text-xs">
                        <div className="text-sigma-500 mb-1">Instance #{exp.instance}</div>
                        <div className="space-y-1">
                          {exp.explanation.slice(0, 4).map(([label, score], i) => (
                            <div key={i} className="flex justify-between">
                              <span className="text-sigma-400 pr-2 truncate">{label}</span>
                              <span className="font-mono text-sigma-200">{score.toFixed(4)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : <p className="text-xs text-sigma-500">Load explanations to view LIME details.</p>)}
              </div>
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
