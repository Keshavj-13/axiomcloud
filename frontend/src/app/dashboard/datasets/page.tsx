"use client";
import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { datasetsAPI } from "@/lib/api";
import { logToFile } from "@/lib/logger";
import { Dataset, ExampleDataset, DatasetQualityReport, CleanPreview, DatasetProfile, EDAReport, AnalyticsReport, AnalyticsChart } from "@/types";
import toast from "react-hot-toast";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line, ScatterChart, Scatter } from "recharts";
import {
  Upload, Database, Trash2, Table, Info,
  CloudUpload, FileSpreadsheet, Loader2, Sparkles, ShieldCheck, Wand2
} from "lucide-react";
import PanelHeader from "@/components/ui/PanelHeader";
import DataTable from "@/components/ui/DataTable";
import StatusBadge from "@/components/ui/StatusBadge";

function formatBytes(bytes?: number) {
  if (!bytes) return "–";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function HeatCell({ value }: { value: number }) {
  const clamped = Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
  const alpha = Math.abs(clamped);
  const bg = clamped >= 0
    ? `rgba(110, 59, 215, ${0.2 + alpha * 0.5})`
    : `rgba(239, 68, 68, ${0.2 + alpha * 0.5})`;
  return <div className="h-7 w-7 rounded" style={{ background: bg }} title={clamped.toFixed(3)} />;
}

function AnalyticsChartCard({ chart }: { chart: AnalyticsChart }) {
  const spec = (chart.spec || {}) as Record<string, unknown>;
  const rows = (spec.rows as Record<string, unknown>[] | undefined) || [];
  const xKey = (spec.xKey as string | undefined) || "label";
  const yKey = (spec.yKey as string | undefined) || "value";
  const series = (spec.series as Array<{ key: string; name: string; color?: string }> | undefined) || [];

  return (
    <div className="rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-text-primary">{chart.title}</div>
          <div className="text-xs text-text-muted">{chart.purpose}</div>
        </div>
        {chart.fallback_used ? <StatusBadge label="fallback" tone="idle" /> : null}
      </div>

      {chart.chart_type === "bar" && rows.length > 0 ? (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(71,71,77,0.25)" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10, fill: "#acaab1" }} interval={0} angle={-20} textAnchor="end" height={48} />
            <YAxis tick={{ fontSize: 10, fill: "#acaab1" }} />
            <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(71,71,77,0.6)", borderRadius: "8px" }} />
            {(series.length ? series : [{ key: "value", name: "Value", color: "#d0bcff" }]).map((s) => (
              <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color || "#d0bcff"} radius={[3, 3, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      ) : chart.chart_type === "line" && rows.length > 0 ? (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(71,71,77,0.25)" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10, fill: "#acaab1" }} />
            <YAxis tick={{ fontSize: 10, fill: "#acaab1" }} />
            <Tooltip contentStyle={{ background: "#19191d", border: "1px solid rgba(71,71,77,0.6)", borderRadius: "8px" }} />
            {(series.length ? series : [{ key: "value", name: "Value", color: "#d0bcff" }]).map((s) => (
              <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color || "#d0bcff"} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      ) : chart.chart_type === "scatter" && rows.length > 0 ? (
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(71,71,77,0.25)" />
            <XAxis dataKey={xKey} name={xKey} tick={{ fontSize: 10, fill: "#acaab1" }} />
            <YAxis dataKey={yKey} name={yKey} tick={{ fontSize: 10, fill: "#acaab1" }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#19191d", border: "1px solid rgba(71,71,77,0.6)", borderRadius: "8px" }} />
            <Scatter data={rows} fill="#d0bcff" />
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
        <div className="max-h-48 overflow-auto rounded border border-outline/20 bg-surface/40 p-2 text-xs">
          {rows.length > 0 ? (
            <ul className="space-y-1 text-text-muted">
              {rows.slice(0, 12).map((r, idx) => (
                <li key={idx} className="font-mono text-[11px]">{JSON.stringify(r)}</li>
              ))}
            </ul>
          ) : (
            <div className="text-text-muted">No chart rows available.</div>
          )}
        </div>
      )}

      <div className="mt-2 text-[11px] text-text-muted">Insight: {chart.insight}</div>
      <div className="mt-1 text-[10px] font-mono text-text-muted/80">artifact: {(chart.artifact_ref?.relative_path || chart.artifact_ref?.url || "n/a")}</div>
    </div>
  );
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [examples, setExamples] = useState<ExampleDataset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loadingExample, setLoadingExample] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [qualityReport, setQualityReport] = useState<DatasetQualityReport | null>(null);
  const [cleanPreview, setCleanPreview] = useState<CleanPreview | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [cleaningLoading, setCleaningLoading] = useState(false);
  const [cleanAndSaveLoading, setCleanAndSaveLoading] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [edaLoading, setEdaLoading] = useState(false);
  const [edaReport, setEdaReport] = useState<EDAReport | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [analyticsReport, setAnalyticsReport] = useState<AnalyticsReport | null>(null);

  const fetchDatasets = async () => {
    try {
      const res = await datasetsAPI.list();
      setDatasets(res.data);
    } catch (e: any) {
      toast.error("Could not connect to backend");
      logToFile(`Backend not connected when fetching datasets: ${e?.message || e}`,'error');
    }
  };

  useEffect(() => {
    fetchDatasets();
    datasetsAPI.listExamples().then(r => setExamples(r.data)).catch(() => {});
  }, []);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", file.name.replace(/\.[^/.]+$/, ""));
    if (targetColumn) formData.append("target_column", targetColumn);
    logToFile(`Upload attempt: ${file.name} (target_column=${targetColumn})`,'info');
    try {
      await datasetsAPI.upload(formData);
      toast.success(`"${file.name}" uploaded successfully`);
      logToFile(`Upload success: ${file.name}`,'info');
      fetchDatasets();
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Upload failed");
      logToFile(`Upload failed: ${file.name} | ${e?.message || e}`,'error');
    } finally { setUploading(false); }
  }, [targetColumn]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"] },
    maxFiles: 1,
  });

  const loadExample = async (key: string) => {
    setLoadingExample(key);
    try {
      await datasetsAPI.loadExample(key);
      toast.success("Example dataset loaded!");
      fetchDatasets();
    } catch { toast.error("Failed to load example"); }
    finally { setLoadingExample(null); }
  };

  const deleteDataset = async (id: number) => {
    if (!confirm("Delete this dataset?")) return;
    try {
      await datasetsAPI.delete(id);
      toast.success("Dataset deleted");
      fetchDatasets();
      if (selectedDataset?.id === id) setSelectedDataset(null);
    } catch { toast.error("Delete failed"); }
  };

  const runQualityReport = async () => {
    if (!selectedDataset) return;
    setReportLoading(true);
    try {
      const res = await datasetsAPI.qualityReport(selectedDataset.id);
      setQualityReport(res.data);
      toast.success("Quality report ready");
    } catch {
      toast.error("Failed to generate quality report");
    } finally {
      setReportLoading(false);
    }
  };

  const runCleanPreview = async () => {
    if (!selectedDataset) return;
    setCleaningLoading(true);
    try {
      const res = await datasetsAPI.cleanPreview(selectedDataset.id, 8);
      setCleanPreview(res.data);
      toast.success("Auto-clean preview ready");
    } catch {
      toast.error("Failed to generate clean preview");
    } finally {
      setCleaningLoading(false);
    }
  };

  const runCleanAndSave = async () => {
    if (!selectedDataset) return;
    setCleanAndSaveLoading(true);
    try {
      await datasetsAPI.cleanAndSave(selectedDataset.id);
      toast.success("Cleaned dataset copy created");
      await fetchDatasets();
    } catch {
      toast.error("Failed to create cleaned dataset copy");
    } finally {
      setCleanAndSaveLoading(false);
    }
  };

  const runProfile = async () => {
    if (!selectedDataset) return;
    setProfileLoading(true);
    try {
      const res = await datasetsAPI.profile(selectedDataset.id);
      setProfile(res.data);
      toast.success("Dataset profile ready");
    } catch {
      toast.error("Failed to build dataset profile");
    } finally {
      setProfileLoading(false);
    }
  };

  const runEDA = async () => {
    if (!selectedDataset) return;
    setEdaLoading(true);
    try {
      const res = await datasetsAPI.edaReport(selectedDataset.id);
      setEdaReport(res.data);
      toast.success("EDA report ready");
    } catch {
      toast.error("Failed to build EDA report");
    } finally {
      setEdaLoading(false);
    }
  };

  const runAnalyticsReport = async () => {
    if (!selectedDataset) return;
    setAnalyticsLoading(true);
    try {
      const res = await datasetsAPI.analyticsReport(selectedDataset.id, {
        target_column: selectedDataset.target_column,
      });
      setAnalyticsReport(res.data);
      toast.success(`Analytics report ready (${res.data.chart_count} charts)`);
    } catch {
      toast.error("Failed to generate analytics report");
    } finally {
      setAnalyticsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl">
      <div className="mb-8">
        <div className="section-kicker mb-2">Datasets</div>
        <h1 className="font-display mb-1 text-4xl font-semibold text-text-primary">Dataset Registry</h1>
        <p className="text-text-muted">Ingest data, inspect quality, and stage secure training sources.</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Upload zone */}
        <div className="panel-glass p-6">
          <PanelHeader title="Upload / Ingestion" subtitle="CSV and XLSX are supported for now." icon={CloudUpload} className="mb-4" />

          <div className="mb-4">
            <label className="mb-1 block text-xs text-text-muted">Target Column (optional)</label>
            <input
              value={targetColumn}
              onChange={e => setTargetColumn(e.target.value)}
              placeholder="e.g. price, survived, label"
              className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary/50"
            />
          </div>

          <div
            {...getRootProps()}
            className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-all
              ${isDragActive
                ? "border-primary/70 bg-surface-variant/70"
                : "border-outline/50 hover:border-primary/50 hover:bg-surface-variant/55"
              }`}
          >
            <input {...getInputProps()} />
            {uploading ? (
              <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-primary" />
            ) : (
              <Upload className={`mx-auto mb-3 h-8 w-8 ${isDragActive ? "text-primary" : "text-text-muted"}`} />
            )}
            <p className="mb-1 text-sm text-text-primary">
              {uploading ? "Uploading..." : isDragActive ? "Drop it here!" : "Drag & drop your CSV or Excel file"}
            </p>
            <p className="text-xs text-text-muted">or click to browse · Max 50MB</p>
          </div>
        </div>

        {/* Example datasets */}
        <div className="panel p-6">
          <PanelHeader title="Context Datasets" subtitle="Load examples to validate workflow quickly." icon={Sparkles} className="mb-4" />
          <div className="space-y-3">
            {examples.map(ex => (
              <div key={ex.key} className="flex items-center gap-3 rounded-lg border border-outline/20 bg-surface-variant/35 p-3 transition-colors hover:border-primary/35">
                <FileSpreadsheet className="h-8 w-8 flex-shrink-0 text-text-muted" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text-primary">{ex.name}</div>
                  <div className="truncate text-xs text-text-muted">{ex.description}</div>
                  <div className="flex gap-2 mt-1">
                    <StatusBadge label={ex.task_type} tone="primary" />
                    <StatusBadge label={`target: ${ex.target}`} tone="idle" />
                  </div>
                </div>
                <button
                  onClick={() => loadExample(ex.key)}
                  disabled={loadingExample === ex.key}
                  className="btn-outline text-xs py-1.5 px-3 flex-shrink-0"
                >
                  {loadingExample === ex.key ? <Loader2 className="w-3 h-3 animate-spin" /> : "Load"}
                </button>
              </div>
            ))}
            {examples.length === 0 && (
              <div className="py-6 text-center text-sm text-text-muted">Loading examples...</div>
            )}
          </div>
        </div>
      </div>

      {/* Dataset list */}
      <div className="panel mt-6 rounded-xl p-6">
        <PanelHeader title="Registry" subtitle={`Your datasets (${datasets.length})`} icon={Database} className="mb-4" />
        <DataTable
          data={datasets}
          rowKey={(ds) => ds.id}
          emptyState={
            <div className="py-12 text-center text-text-muted">
              <Database className="mx-auto mb-3 h-10 w-10 opacity-30" />
              <p className="text-sm">No datasets yet. Upload one or load an example.</p>
            </div>
          }
          columns={[
            {
              key: "name",
              header: "Name",
              render: (ds) => (
                <div className="font-medium text-text-primary">
                  {ds.is_example && <StatusBadge label="example" tone="idle" className="mr-2" />}
                  {ds.name}
                </div>
              ),
            },
            { key: "rows", header: "Rows", align: "right", className: "font-mono text-text-muted", render: (ds) => ds.num_rows?.toLocaleString() ?? "–" },
            { key: "columns", header: "Columns", align: "right", className: "font-mono text-text-muted", render: (ds) => ds.num_columns ?? "–" },
            { key: "size", header: "Size", align: "right", className: "font-mono text-text-muted", render: (ds) => formatBytes(ds.file_size) },
            { key: "target", header: "Target", className: "font-mono text-xs text-primary", render: (ds) => ds.target_column || "–" },
            {
              key: "task",
              header: "Task",
              render: (ds) => ds.task_type ? <StatusBadge label={ds.task_type} tone="primary" /> : "–",
            },
            {
              key: "actions",
              header: "Actions",
              align: "right",
              render: (ds) => (
                <div className="flex justify-end gap-2">
                  <button onClick={() => setSelectedDataset(selectedDataset?.id === ds.id ? null : ds)}
                    className="rounded p-1.5 text-text-muted transition-colors hover:bg-surface-variant hover:text-text-primary">
                    <Table className="h-3.5 w-3.5" />
                  </button>
                  <button onClick={() => deleteDataset(ds.id)}
                    className="rounded p-1.5 text-text-muted transition-colors hover:bg-red-500/20 hover:text-red-300">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ),
            },
          ]}
        />
      </div>

      {/* Preview */}
      {selectedDataset && (
        <div className="panel mt-6 rounded-xl p-6">
          <PanelHeader title={`Preview: ${selectedDataset.name}`} icon={Info} className="mb-4" />

          <div className="flex flex-wrap gap-2 mb-4">
            <button onClick={runQualityReport} disabled={reportLoading} className="btn-outline text-xs py-1.5 px-3">
              {reportLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />} Quality Report
            </button>
            <button onClick={runCleanPreview} disabled={cleaningLoading} className="btn-outline text-xs py-1.5 px-3">
              {cleaningLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />} Auto-Clean Preview
            </button>
            <button onClick={runCleanAndSave} disabled={cleanAndSaveLoading} className="btn-outline text-xs py-1.5 px-3">
              {cleanAndSaveLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />} Clean & Save Copy
            </button>
            <button onClick={runProfile} disabled={profileLoading} className="btn-outline text-xs py-1.5 px-3">
              {profileLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Table className="w-3.5 h-3.5" />} Build Profile
            </button>
            <button onClick={runEDA} disabled={edaLoading} className="btn-outline text-xs py-1.5 px-3">
              {edaLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />} EDA Report
            </button>
            <button onClick={runAnalyticsReport} disabled={analyticsLoading} className="btn-outline text-xs py-1.5 px-3">
              {analyticsLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />} Full Analytics
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-outline/20 text-left text-text-muted">
                  {selectedDataset.columns_info?.map(c => (
                    <th key={c.name} className="pb-2 pr-4 font-medium whitespace-nowrap">{c.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {selectedDataset.preview_data?.slice(0, 8).map((row, i) => (
                  <tr key={i} className="border-b border-outline/10 hover:bg-surface-variant/35">
                    {selectedDataset.columns_info?.map(c => (
                      <td key={c.name} className="max-w-[150px] whitespace-nowrap truncate py-1.5 pr-4 text-text-primary">
                        {String(row[c.name] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(qualityReport || cleanPreview) && (
            <div className="grid lg:grid-cols-2 gap-4 mt-5">
              {qualityReport && (
                <div className="rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
                  <div className="mb-2 text-sm font-semibold text-text-primary">Data Quality</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="text-text-muted">Quality score</div>
                    <div className="font-mono text-text-primary">{qualityReport.quality_score}/100</div>
                    <div className="text-text-muted">Missing cells</div>
                    <div className="font-mono text-text-primary">{qualityReport.missing_cells}</div>
                    <div className="text-text-muted">Duplicate rows</div>
                    <div className="font-mono text-text-primary">{qualityReport.duplicate_rows}</div>
                    <div className="text-text-muted">Columns with outliers</div>
                    <div className="font-mono text-text-primary">{Object.keys(qualityReport.outliers_by_column || {}).length}</div>
                  </div>
                  <ul className="mt-3 list-disc space-y-1 pl-4 text-xs text-text-muted">
                    {(qualityReport.recommendations || []).slice(0, 3).map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </div>
              )}

              {cleanPreview && (
                <div className="rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
                  <div className="mb-2 text-sm font-semibold text-text-primary">Auto-Clean Summary</div>
                  <div className="mb-2 text-xs text-text-muted">Applied fixes</div>
                  <ul className="max-h-28 list-disc space-y-1 overflow-auto pl-4 text-xs text-text-muted">
                    {cleanPreview.applied_fixes.length > 0
                      ? cleanPreview.applied_fixes.map((f, i) => <li key={i}>{f}</li>)
                      : <li>No fixes required</li>}
                  </ul>
                </div>
              )}
            </div>
          )}

          {profile && (
            <div className="mt-6 space-y-4">
              <div className="grid md:grid-cols-3 gap-3">
                {profile.summary_cards.map((card) => (
                  <div key={card.label} className="rounded-lg border border-outline/20 bg-surface-variant/30 p-3 text-xs">
                    <div className="text-text-muted">{card.label}</div>
                    <div className="font-mono text-text-primary">{String(card.value)}</div>
                  </div>
                ))}
              </div>

              <div className="grid lg:grid-cols-2 gap-4">
                <div className="rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
                  <div className="text-sm font-semibold text-text-primary mb-2">Feature Typing Intelligence</div>
                  {profile.typing_intelligence ? (
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-text-muted">Numeric</div>
                      <div className="font-mono text-text-primary">{profile.typing_intelligence.numeric}</div>
                      <div className="text-text-muted">Categorical</div>
                      <div className="font-mono text-text-primary">{profile.typing_intelligence.categorical}</div>
                      <div className="text-text-muted">Datetime-like</div>
                      <div className="font-mono text-text-primary">{profile.typing_intelligence.datetime}</div>
                      <div className="text-text-muted">Boolean</div>
                      <div className="font-mono text-text-primary">{profile.typing_intelligence.boolean}</div>
                    </div>
                  ) : (
                    <p className="text-xs text-text-muted">No typing intelligence available.</p>
                  )}
                  {(profile.typing_intelligence?.high_cardinality_candidates?.length || 0) > 0 && (
                    <div className="mt-3">
                      <div className="text-xs text-text-muted mb-1">High-cardinality candidates</div>
                      <div className="flex flex-wrap gap-1">
                        {profile.typing_intelligence!.high_cardinality_candidates.slice(0, 8).map((name) => (
                          <span key={name} className="rounded bg-surface px-2 py-1 text-[10px] text-text-primary">{name}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {profile.drift_baseline && (
                    <div className="mt-3 rounded border border-outline/20 bg-surface/40 p-2 text-xs">
                      <div className="text-text-muted">Drift baseline snapshot</div>
                      <div className="font-mono text-text-primary">
                        rows: {profile.drift_baseline.rows} • cols: {profile.drift_baseline.columns}
                      </div>
                    </div>
                  )}
                </div>

                <div className="rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
                  <div className="text-sm font-semibold text-text-primary mb-2">Target Leakage Risk</div>
                  {(profile.leakage_risks?.length || 0) > 0 ? (
                    <div className="space-y-2 text-xs">
                      {profile.leakage_risks!.slice(0, 5).map((risk) => (
                        <div key={risk.feature} className="rounded border border-outline/20 bg-surface/40 p-2">
                          <div className="flex items-center justify-between">
                            <span className="text-text-primary">{risk.feature}</span>
                            <span className="font-mono text-amber-300">score: {risk.risk_score.toFixed(2)}</span>
                          </div>
                          <div className="mt-1 text-text-muted">{risk.reasons.join("; ")}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-text-muted">No strong leakage risks detected.</p>
                  )}
                </div>
              </div>

              <div className="grid lg:grid-cols-2 gap-4">
                <div className="rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
                  <div className="text-sm font-semibold text-text-primary mb-2">Missing Values by Column</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={profile.missing_by_column.slice(0, 12)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
                      <XAxis dataKey="column" tick={{ fontSize: 10, fill: "#8a94ae" }} interval={0} angle={-20} textAnchor="end" height={45} />
                      <YAxis tick={{ fontSize: 10, fill: "#8a94ae" }} />
                      <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px" }} />
                      <Bar dataKey="missing" fill="#3b6ef6" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
                  <div className="text-sm font-semibold text-text-primary mb-2">Target Distribution</div>
                  {profile.target_distribution?.labels?.length ? (
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart
                        data={profile.target_distribution.labels.map((label, i) => ({
                          label,
                          count: profile.target_distribution!.counts[i],
                        }))}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
                        <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#8a94ae" }} interval={0} angle={-20} textAnchor="end" height={45} />
                        <YAxis tick={{ fontSize: 10, fill: "#8a94ae" }} />
                        <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px" }} />
                        <Bar dataKey="count" fill="#00f5ff" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-xs text-text-muted">No target distribution available.</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {edaReport && (
            <div className="mt-6 rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
              <div className="mb-3 text-sm font-semibold text-text-primary">Exploratory Data Analysis</div>
              <div className="grid md:grid-cols-3 gap-3 text-xs mb-4">
                <div className="rounded border border-outline/20 bg-surface/40 p-2">
                  <div className="text-text-muted">Rows / Columns</div>
                  <div className="font-mono text-text-primary">{edaReport.overview.rows} / {edaReport.overview.columns}</div>
                </div>
                <div className="rounded border border-outline/20 bg-surface/40 p-2">
                  <div className="text-text-muted">Missing / Duplicates</div>
                  <div className="font-mono text-text-primary">{edaReport.overview.missing_total} / {edaReport.overview.duplicate_rows}</div>
                </div>
                <div className="rounded border border-outline/20 bg-surface/40 p-2">
                  <div className="text-text-muted">Numeric / Categorical</div>
                  <div className="font-mono text-text-primary">{edaReport.overview.numeric_columns} / {edaReport.overview.categorical_columns}</div>
                </div>
              </div>

              <div className="grid lg:grid-cols-2 gap-4">
                <div>
                  <div className="mb-2 text-xs font-semibold text-text-primary">High Correlations</div>
                  {(edaReport.high_correlations || []).length > 0 ? (
                    <div className="space-y-2 text-xs">
                      {edaReport.high_correlations.slice(0, 6).map((c) => (
                        <div key={`${c.feature_a}-${c.feature_b}`} className="rounded border border-outline/20 bg-surface/40 p-2">
                          <div className="text-text-primary">{c.feature_a} ↔ {c.feature_b}</div>
                          <div className="font-mono text-amber-300">corr: {c.correlation.toFixed(3)}</div>
                        </div>
                      ))}
                    </div>
                  ) : <p className="text-xs text-text-muted">No strong correlations found.</p>}
                </div>

                <div>
                  <div className="mb-2 text-xs font-semibold text-text-primary">Recommendations</div>
                  <ul className="list-disc space-y-1 pl-4 text-xs text-text-muted">
                    {(edaReport.recommendations || []).map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                  <div className="mt-3 text-xs font-semibold text-text-primary">Leakage Risks: {(edaReport.leakage_risks || []).length}</div>
                </div>
              </div>
            </div>
          )}

          {analyticsReport && (
            <div className="mt-6 space-y-4 rounded-lg border border-outline/25 bg-surface-variant/30 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-text-primary">AutoML Analytics & Diagnostics</div>
                  <div className="text-xs text-text-muted">
                    Task: {analyticsReport.task_type} • Charts: {analyticsReport.chart_count} (min {analyticsReport.minimum_required_charts})
                  </div>
                </div>
                <StatusBadge label={`evaluation: ${analyticsReport.evaluation_status}`} tone={analyticsReport.evaluation_status === "ready" ? "healthy" : "idle"} />
              </div>

              <div>
                <div className="mb-2 text-xs uppercase tracking-[0.12em] text-text-muted">1) Exploratory data analysis</div>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {analyticsReport.exploratory_charts.map((chart) => (
                    <AnalyticsChartCard key={chart.id} chart={chart} />
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-xs uppercase tracking-[0.12em] text-text-muted">2) Model evaluation</div>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {analyticsReport.evaluation_charts.map((chart) => (
                    <AnalyticsChartCard key={chart.id} chart={chart} />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
